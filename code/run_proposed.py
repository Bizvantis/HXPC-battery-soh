"""Proposed method with a heterogeneous domain-subsample ensemble backbone (GBDT + MLP members).
Only computes the cross-domain (inner) and target predictions needed by the proposed calibration;
baselines come from run_offline.py.

usage: python run_proposed.py --seeds 0 1 2 [--protocol lodo|lodso]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

from models import fit_lgb, predict_mlp, thin, train_mlp
from run_offline import F, K_REF, OUT, prepare

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


class HeteroDSE:
    """Members: `n_gb` LightGBM + `n_nn` MLPs, each trained on a random 70% subset of source domains."""

    def __init__(self, n_gb=2, n_nn=2, frac=0.7, seed=0):
        self.n_gb, self.n_nn, self.frac, self.seed = n_gb, n_nn, frac, seed

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        doms = np.array(sorted(train["domain"].unique()))
        k = max(2, int(round(self.frac * len(doms)))) if len(doms) > 2 else len(doms)
        self.members = []
        for m in range(self.n_gb + self.n_nn):
            sub = train[train["domain"].isin(rng.choice(doms, size=k, replace=False))]
            if m < self.n_gb:
                self.members.append(("gb", fit_lgb(sub[F], sub["dsoh"], self.seed * 100 + m, n_jobs=4), None))
            else:
                mean, std = sub[F].mean(), sub[F].std().replace(0, 1)
                net = train_mlp(((sub[F] - mean) / std).values, sub["dsoh"].values, self.seed * 100 + m, epochs=40)
                self.members.append(("nn", net, (mean, std)))
        return self

    def predict(self, X):
        P = []
        for kind, mdl, norm in self.members:
            if kind == "gb":
                P.append(mdl.predict(X[F]))
            else:
                P.append(predict_mlp(mdl, ((X[F] - norm[0]) / norm[1]).values)[0])
        P = np.stack(P)
        return P.mean(0), P.std(0)


def run_fold(d, target_doms, seed):
    S = d[~d["domain"].isin(target_doms)]
    Tg = d[d["domain"].isin(target_doms) & (d["cycle"] >= K_REF)]
    S_eval = S[S["cycle"] >= K_REF]
    t = time.time()
    inner = []
    for k in sorted(S["domain"].unique()):
        ens = HeteroDSE(seed=seed).fit(thin(S[S["domain"] != k]))
        Dk = S_eval[S_eval["domain"] == k]
        mu, sd = ens.predict(Dk)
        inner.append(pd.DataFrame({"domain": k, "cell": Dk["cell"].values, "cycle": Dk["cycle"].values,
                                   "life_frac": Dk["life_frac"].values, "y": Dk["dsoh"].values, "mu": mu, "sd": sd}))
    inner = pd.concat(inner, ignore_index=True)
    ens = HeteroDSE(seed=seed).fit(thin(S))
    mu, sd = ens.predict(Tg)
    return pd.DataFrame({"mu_hdse": mu, "sd_hdse": sd}, index=Tg.index), inner, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--protocol", default="lodo", choices=["lodo", "lodso"])
    args = ap.parse_args()
    d = prepare()
    out = os.path.join(OUT, args.protocol + "_hdse")
    os.makedirs(out, exist_ok=True)
    if args.protocol == "lodo":
        folds = {T: [T] for T in sorted(d["domain"].unique())}
    else:
        folds = {ds: sorted(d.loc[d["dataset"] == ds, "domain"].unique()) for ds in ["XJTU", "TJU", "MIT", "HUST"]}
    for seed in args.seeds:
        for name, doms in folds.items():
            f = os.path.join(out, f"{name}_s{seed}.pkl")
            if os.path.exists(f):
                continue
            tg, inner, sec = run_fold(d, doms, seed)
            pd.to_pickle({"target": tg, "inner": inner, "seconds": sec}, f)
            y = d.loc[tg.index, "dsoh"].values
            print(f"[seed {seed}] {name:12s} {sec:6.1f}s RMSE% hdse={np.sqrt(np.mean((tg.mu_hdse - y) ** 2)) * 100:.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
