"""Evaluate the domain-adaptation backbone (run_da.py) with exactly the calibration, online simulation,
source-only hyper-parameter selection and metrics used for every other method (evaluate.py).

Rows per (fold, seed): split CP with a DANN point predictor (day zero), HX-CP and HX-CP+ with the
HDSE-DA backbone, and, for a paired comparison, HX-CP and HX-CP+ with the HDSE backbone.
usage: python analysis_da.py [lodo|lodso]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from conformal import hierarchical_weights
from evaluate import RES, checkup_mask, day0_proposed, inservice_proposed, load_fold, metrics

ALPHA, LS = 0.1, (25, 50, 100, 200)


def eval_fold(protocol, name, seed):
    tg, _, backbones = load_fold(protocol, name, seed)
    blob = pd.read_pickle(os.path.join(RES, "offline", protocol + "_hdse_da", f"{name}_s{seed}.pkl"))
    base = pd.read_pickle(os.path.join(RES, "offline", protocol, f"{name}_s{seed}.pkl"))["target"]
    da = blob["target"].reindex(base.index).reset_index(drop=True)
    inn = blob["inner"]
    inn["w_h"] = hierarchical_weights(inn["domain"].values, inn["cell"].values)
    backbones = {"hdse": backbones["hdse"], "hdse_da": (da["mu_hdse_da"].values, da["sd_hdse_da"].values, inn)}
    rows, b0 = [], dict(fold=name, seed=int(seed), alpha=ALPHA)
    mu, q = da["mu_dann"].values, da[f"q_dann_{ALPHA}"].values
    rows.append(dict(b0, stage="day0", L=0, method="split@dann", **metrics(tg, mu - q, mu + q, ALPHA, mu)))
    betas = {}
    for bb, (m, s, inner) in backbones.items():
        d0, betas[bb] = day0_proposed(m, s, inner, ALPHA)
        lo, hi, c = d0["hxcp"]
        rows.append(dict(b0, stage="day0", L=0, method=f"hxcp@{bb}", beta=betas[bb], **metrics(tg, lo, hi, ALPHA, c)))
    for L in LS:
        ev = ~checkup_mask(tg, L)
        sub = tg[ev]
        for bb, (m, s, inner) in backbones.items():
            ins, (kp, gm) = inservice_proposed(tg, m, s, inner, ALPHA, L, betas[bb])
            lo, hi, c = ins["hxcp+"]
            rows.append(dict(b0, stage="inservice", L=L, method=f"hxcp+@{bb}", beta=betas[bb], kappa=kp, gamma=gm,
                             **metrics(sub, np.asarray(lo)[ev], np.asarray(hi)[ev], ALPHA, np.asarray(c)[ev])))
    return rows


def main(protocol="lodo"):
    from joblib import Parallel, delayed
    files = sorted(glob.glob(os.path.join(RES, "offline", protocol + "_hdse_da", "*.pkl")))
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in files]
    res = Parallel(n_jobs=4)(delayed(eval_fold)(protocol, n, s) for n, s in jobs)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RES, f"da_{protocol}.csv"), index=False)
    return df


if __name__ == "__main__":
    df = main(sys.argv[1] if len(sys.argv) > 1 else "lodo")
    dom = df.groupby(["stage", "L", "method", "fold"])[["cov", "width", "winkler", "cell_ok", "rmse"]].mean()
    s = dom.groupby(["stage", "L", "method"]).mean()
    s["worst"] = dom["cov"].groupby(["stage", "L", "method"]).min()
    pd.set_option("display.width", 200)
    print(s.round(3))
