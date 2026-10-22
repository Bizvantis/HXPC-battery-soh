"""Extra analyses: step-size sensitivity, trajectory certificate, alpha sweep summary, label budget."""
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import evaluate as E
from conformal import conformal_quantile

BB = "hdse"
RES = E.RES


def eta_fold(name, seed, L=100, alpha=0.1, etas=(0.01, 0.02, 0.05, 0.1, 0.2)):
    tg, _, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    _, beta = E.day0_proposed(mu, sd, inner, alpha)
    ev = ~E.checkup_mask(tg, L)
    rows = []
    for eta in etas:
        ins, _ = E.inservice_proposed(tg, mu, sd, inner, alpha, L, beta, lr_frac=eta)
        lo, hi, m = ins["hxcp+"]
        rows.append(dict(fold=name, seed=int(seed), eta=eta,
                         **E.metrics(tg[ev], np.asarray(lo)[ev], np.asarray(hi)[ev], alpha, np.asarray(m)[ev])))
    return rows


def traj_fold(name, seed, alpha=0.1, delta=0.1):
    """Trajectory certificate (Prop. 1) at the cell level: calibration cells = cross-domain cells."""
    tg, _, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    _, beta = E.day0_proposed(mu, sd, inner, alpha)
    b = 0.0 if np.isinf(beta) else beta
    scale_i = np.ones(len(inner)) if np.isinf(beta) else inner["sd"].values + b + E.EPS
    scale_t = np.ones(len(tg)) if np.isinf(beta) else sd + b + E.EPS
    s = np.abs(inner["y"].values - inner["mu"].values) / scale_i
    S = pd.Series(s).groupby(inner["cell"].values).quantile(1 - delta)
    # domain-balanced cell scores: resample so each domain contributes equally (weighted quantile)
    dom_of = inner.groupby("cell")["domain"].first().reindex(S.index)
    w = 1.0 / dom_of.map(dom_of.value_counts()).values
    q_traj = E.weighted_quantile(S.values, w, alpha)
    q_cell = conformal_quantile(S.values, alpha)  # exchangeable-cell version (Prop. 1 as stated)
    out = []
    for nm, q in [("traj_domainbal", q_traj), ("traj_cell", q_cell)]:
        cov = np.abs(tg["dsoh"].values - mu) <= q * scale_t
        cc = pd.Series(cov).groupby(tg["cell"].values).mean()
        out.append(dict(fold=name, seed=int(seed), method=nm, frac_cells_certified=(cc >= 1 - delta).mean(),
                        cov=cov.mean(), width=2 * np.mean(q * scale_t) * 100))
    return out


def main():
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in sorted(glob.glob(os.path.join(RES, "offline", "lodo", "*.pkl")))]
    eta = Parallel(n_jobs=6)(delayed(eta_fold)(n, s) for n, s in jobs)
    eta = pd.DataFrame([r for rr in eta for r in rr])
    eta.to_csv(os.path.join(RES, "sens_eta.csv"), index=False)
    print(eta.groupby(["eta", "fold"]).mean(numeric_only=True).groupby("eta")[["cov", "width", "winkler", "cell_p10"]].mean().round(3))
    tr = Parallel(n_jobs=6)(delayed(traj_fold)(n, s) for n, s in jobs)
    tr = pd.DataFrame([r for rr in tr for r in rr])
    tr.to_csv(os.path.join(RES, "traj_cert.csv"), index=False)
    print(tr.groupby(["method", "fold"]).mean(numeric_only=True).round(3))
    print(tr.groupby(["method", "fold"]).mean(numeric_only=True).groupby("method").mean().round(3))


if __name__ == "__main__":
    main()
