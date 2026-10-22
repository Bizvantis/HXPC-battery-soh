"""Further analyses for the journal version:
(1) coverage by life stage, (2) adaptivity (width vs. |error| rank correlation),
(3) online threshold trajectories, (4) label budget of every in-service / recalibration setting."""
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import evaluate as E

BB = "hdse"
RES = E.RES
STAGES = [(0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "mid"), (2 / 3, 1.01, "late")]


def fold_analysis(name, seed, alpha=0.1, L=100):
    tg, inner0, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    d0b = E.day0_baselines(tg, alpha)
    d0p, beta = E.day0_proposed(mu, sd, inner, alpha)
    insb = E.inservice_baselines(tg, inner0, alpha, L)
    insp, _ = E.inservice_proposed(tg, mu, sd, inner, alpha, L, beta)
    meth = {"split": d0b["split"], "cqr": d0b["cqr"], "gp": d0b["gp"], "hxcp": d0p["hxcp"],
            "aci": insb["aci_split"], "persist_online": insb["persist_online"], "hxcp+": insp["hxcp+"]}
    y = tg["dsoh"].values
    ev = ~E.checkup_mask(tg, L)
    stage_rows, adapt_rows = [], []
    for m, (lo, hi, c) in meth.items():
        lo, hi, c = np.asarray(lo), np.asarray(hi), np.asarray(c)
        mask = ev if m in ("aci", "persist_online", "hxcp+") else np.ones(len(tg), bool)
        cov = (y >= lo) & (y <= hi)
        for a, b, lab in STAGES:
            s = mask & (tg["life_frac"].values >= a) & (tg["life_frac"].values < b)
            stage_rows.append(dict(fold=name, seed=int(seed), method=m, stage=lab, cov=cov[s].mean(),
                                   width=np.mean(hi[s] - lo[s]) * 100))
        w = (hi - lo)[mask]
        err = np.abs(y - c)[mask]
        rho = spearmanr(w, err).correlation if np.std(w) > 1e-12 else np.nan
        adapt_rows.append(dict(fold=name, seed=int(seed), method=m, spearman=rho))
    return stage_rows, adapt_rows


def q_trajectory(name, seed=0, alpha=0.1, L=100):
    """Threshold q after each fleet check-up (proposed, normalised scale)."""
    tg, _, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    _, beta = E.day0_proposed(mu, sd, inner, alpha)
    kappa, gamma = E.select_kappa_gamma(inner, L, beta, alpha)
    inn = inner.rename(columns={"y": "dsoh"})
    c_t, _, st_t = E.anchored(tg, mu, L, kappa)
    c_i, _, st_i = E.anchored(inn, inner["mu"].values, L, kappa)
    sc_t = E._scale(sd, st_t, L, beta, gamma)
    sc_i = E._scale(inner["sd"].values, st_i, L, beta, gamma)
    q0 = E.hq(inner, np.abs(inner["y"].values - c_i) / sc_i, alpha)
    is_cu = E.checkup_mask(tg, L)
    s = np.abs(tg["dsoh"].values - c_t) / sc_t
    ev = np.where(is_cu)[0]
    ev = ev[np.lexsort((tg["cell"].values[ev], tg["cycle"].values[ev]))]
    tr = E.QuantileTracker(q0, alpha, 0.05 * q0)
    qs, errs = [q0], []
    for k in ev:
        errs.append(tr.update(s[k]))
        qs.append(tr.q)
    return dict(fold=name, q=np.array(qs), err=np.array(errs), q0=q0, n=len(ev))


def label_budget():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "offline", "lodo", "*_s0.pkl"))):
        name = os.path.basename(f)[:-7]
        tg = pd.read_pickle(f)["target"]
        n_cell = tg["cell"].nunique()
        life = tg.groupby("cell").size()
        for L in (25, 50, 100, 200):
            rows.append(dict(fold=name, setting=f"L={L}", labels_per_domain=int(E.checkup_mask(tg, L).sum()),
                             needs_lab_ageing=False))
        for m in (1, 2, 3, 5):
            rows.append(dict(fold=name, setting=f"recal m={m}", labels_per_domain=int(m * life.median()),
                             needs_lab_ageing=True))
    return pd.DataFrame(rows)


def main():
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in sorted(glob.glob(os.path.join(RES, "offline", "lodo", "*.pkl")))]
    out = Parallel(n_jobs=6)(delayed(fold_analysis)(n, s) for n, s in jobs)
    st = pd.DataFrame([r for a, _ in out for r in a])
    ad = pd.DataFrame([r for _, b in out for r in b])
    st.to_csv(os.path.join(RES, "life_stage.csv"), index=False)
    ad.to_csv(os.path.join(RES, "adaptivity.csv"), index=False)
    print(st.groupby(["method", "stage", "fold"]).mean(numeric_only=True).groupby(["method", "stage"])[["cov", "width"]].mean().unstack().round(3))
    print(ad.groupby(["method", "fold"]).spearman.mean().groupby("method").agg(["mean", "min", "max"]).round(3))
    qt = [q_trajectory(n) for n in ["HUST", "TJU-NCM+NCA", "MIT-b1", "XJTU-3C"]]
    pd.to_pickle(qt, os.path.join(RES, "q_traj.pkl"))
    for q in qt:
        print(q["fold"], "n check-ups", q["n"], "q0 %.2f -> q_end %.2f" % (q["q0"], q["q"][-1]), "miss rate %.3f" % q["err"].mean())
    lb = label_budget()
    lb.to_csv(os.path.join(RES, "label_budget.csv"), index=False)
    print(lb.groupby("setting").labels_per_domain.mean().round(0))


if __name__ == "__main__":
    main()
