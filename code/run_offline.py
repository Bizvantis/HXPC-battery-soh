"""Leave-one-domain-out experiment: fits every model once per (held-out domain, seed) and stores
predictions so that all calibration variants / online simulations can be evaluated cheaply.

usage: python run_offline.py --seeds 0 1 2 [--protocol lodo|lodso]
"""
import argparse
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from conformal import conformal_quantile
from data import FEATURE_SETS, add_causal_features, load_all
from models import (DomainSubsampleEnsemble, fit_gp, fit_lgb, predict_mlp, thin, train_mlp)

warnings.filterwarnings("ignore")
K_REF = 5
ALPHAS = [0.05, 0.1, 0.2]
F = FEATURE_SETS["causal"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "offline")


def prepare():
    d = add_causal_features(load_all(), k_ref=K_REF)
    ref = d[d["cycle"] < K_REF].groupby("cell")["soh"].median()
    d["soh_ref"] = ref.reindex(d["cell"]).values
    d["dsoh"] = d["soh"] - d["soh_ref"]
    return d


def split_cells(S, frac, rng):
    """Hold out `frac` of the cells of every source domain for calibration."""
    cal = []
    for _, g in S.groupby("domain"):
        cells = g["cell"].unique()
        rng.shuffle(cells)
        cal.extend(cells[: max(1, int(round(frac * len(cells))))])
    return S["cell"].isin(cal).values


def run_fold(d, target_doms, seed):
    rng = np.random.default_rng(seed)
    S = d[~d["domain"].isin(target_doms)]
    Tg = d[d["domain"].isin(target_doms) & (d["cycle"] >= K_REF)].copy()
    S_eval = S[S["cycle"] >= K_REF]
    timings = {}
    res = Tg[["cell", "domain", "dataset", "cycle", "life_frac", "soh", "soh_ref", "dsoh"]].copy()

    # ---------------------------------------------------------------- proposed: cross-domain scores
    t = time.time()
    inner = []
    for k in sorted(S["domain"].unique()):
        ens = DomainSubsampleEnsemble(F, seed=seed).fit(thin(S[S["domain"] != k]))
        Dk = S_eval[S_eval["domain"] == k]
        mu, sd = ens.predict(Dk)
        inner.append(pd.DataFrame({"domain": k, "cell": Dk["cell"].values, "cycle": Dk["cycle"].values,
                                   "life_frac": Dk["life_frac"].values, "y": Dk["dsoh"].values,
                                   "mu": mu, "sd": sd}))
    inner = pd.concat(inner, ignore_index=True)
    ens = DomainSubsampleEnsemble(F, seed=seed).fit(thin(S))
    res["mu_dse"], res["sd_dse"] = ens.predict(Tg)
    timings["proposed"] = time.time() - t

    # ---------------------------------------------------------------- split CP / CQR / QR / weighted CP
    t = time.time()
    is_cal = split_cells(S, 0.2, rng)
    Tr, Ca = thin(S[~is_cal]), S[is_cal & (S["cycle"] >= K_REF)]
    m = fit_lgb(Tr[F], Tr["dsoh"], seed)
    res["mu_split"] = m.predict(Tg[F])
    r_cal = np.abs(Ca["dsoh"].values - m.predict(Ca[F]))
    for a in ALPHAS:
        res[f"q_split_{a}"] = conformal_quantile(r_cal, a)
    timings["split"] = time.time() - t

    # weighted CP under covariate shift (Tibshirani et al. 2019): domain classifier source-cal vs target
    t = time.time()
    Xs = Ca[F].sample(min(len(Ca), 20000), random_state=seed)
    Xt = Tg[F].sample(min(len(Tg), 20000), random_state=seed)
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, verbose=-1,
                             n_jobs=8, random_state=seed)
    clf.fit(pd.concat([Xs, Xt]), np.r_[np.zeros(len(Xs)), np.ones(len(Xt))])
    prior = len(Xs) / len(Xt)

    def lr(X):
        p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(p / (1 - p) * prior, 1e-4, 1e4)

    w_cal, w_te = lr(Ca[F]), lr(Tg[F])
    o = np.argsort(r_cal)
    s_sorted, cw = r_cal[o], np.cumsum(w_cal[o])
    for a in ALPHAS:
        thr = (1 - a) * (cw[-1] + w_te)
        i = np.searchsorted(cw, thr)
        res[f"q_wcp_{a}"] = np.where(i >= len(s_sorted), np.inf, s_sorted[np.minimum(i, len(s_sorted) - 1)])
    timings["wcp"] = time.time() - t + timings["split"]

    # CQR (Romano et al. 2019) and raw LightGBM quantile regression
    t = time.time()
    for a in ALPHAS:
        ql = fit_lgb(Tr[F], Tr["dsoh"], seed, objective="quantile", alpha=a / 2)
        qh = fit_lgb(Tr[F], Tr["dsoh"], seed, objective="quantile", alpha=1 - a / 2)
        lo_c, hi_c = ql.predict(Ca[F]), qh.predict(Ca[F])
        s = np.maximum(lo_c - Ca["dsoh"].values, Ca["dsoh"].values - hi_c)
        qc = conformal_quantile(s, a)
        lo_t, hi_t = ql.predict(Tg[F]), qh.predict(Tg[F])
        res[f"lo_qr_{a}"], res[f"hi_qr_{a}"] = lo_t, hi_t
        res[f"lo_cqr_{a}"], res[f"hi_cqr_{a}"] = lo_t - qc, hi_t + qc
    timings["cqr"] = time.time() - t

    # ---------------------------------------------------------------- neural + GP baselines
    St = thin(S)
    mean, std = St[F].mean(), St[F].std().replace(0, 1)
    Xn, yn = ((St[F] - mean) / std).values, St["dsoh"].values
    Xte = ((Tg[F] - mean) / std).values
    t = time.time()
    mus, sig2 = [], []
    for j in range(5):
        net = train_mlp(Xn, yn, seed * 10 + j)
        mu, sg = predict_mlp(net, Xte)
        mus.append(mu)
        sig2.append(sg ** 2)
    mus = np.array(mus)
    res["mu_de"] = mus.mean(0)
    res["sd_de"] = np.sqrt(np.mean(sig2, 0) + mus.var(0))
    timings["deep_ensemble"] = time.time() - t
    t = time.time()
    net = train_mlp(Xn, yn, seed * 10 + 7, p=0.1)
    res["mu_mcd"], res["sd_mcd"] = predict_mlp(net, Xte, mc=50)
    timings["mc_dropout"] = time.time() - t
    t = time.time()
    gp = fit_gp(Xn, yn, seed)
    res["mu_gp"], res["sd_gp"] = gp.predict(Xte, return_std=True)
    timings["gp"] = time.time() - t
    return res, inner, timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--protocol", default="lodo", choices=["lodo", "lodso"])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    d = prepare()
    out = os.path.join(OUT, args.protocol)
    os.makedirs(out, exist_ok=True)
    if args.protocol == "lodo":
        folds = {T: [T] for T in sorted(d["domain"].unique())}
    else:
        folds = {ds: sorted(d.loc[d["dataset"] == ds, "domain"].unique()) for ds in ["XJTU", "TJU", "MIT", "HUST"]}
    if args.only:
        folds = {k: v for k, v in folds.items() if k in args.only}
    for seed in args.seeds:
        for name, doms in folds.items():
            f = os.path.join(out, f"{name}_s{seed}.pkl")
            if os.path.exists(f):
                continue
            t = time.time()
            res, inner, timings = run_fold(d, doms, seed)
            pd.to_pickle({"target": res, "inner": inner, "timings": timings}, f)
            rm = lambda c: np.sqrt(np.mean((res[c] - res["dsoh"]) ** 2)) * 100
            print(f"[seed {seed}] {name:12s} {time.time() - t:6.1f}s  RMSE% dse={rm('mu_dse'):.2f} "
                  f"split={rm('mu_split'):.2f} de={rm('mu_de'):.2f} mcd={rm('mu_mcd'):.2f} gp={rm('mu_gp'):.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
