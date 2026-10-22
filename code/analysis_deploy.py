"""Deployment-realism analyses of HX-CP+ (answers to likely reviewer questions):
(1) noisy check-ups, (2) irregular check-up schedules, (3) per-cell vs fleet-level tracking,
(4) fleet size. All use the stored LODO predictions; alpha = 0.1, nominal L = 100."""
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import evaluate as E
from conformal import QuantileTracker

BB, ALPHA, L = "hdse", 0.1, 100
RES = E.RES


def irregular_mask(df, L, seed):
    """Per cell: first check-up at U(0.5L, 1.5L) after commissioning, then gaps ~ U(0.5L, 1.5L)."""
    rng = np.random.default_rng(seed)
    cu = np.zeros(len(df), bool)
    cyc = df["cycle"].values
    for _, idx in df.groupby("cell", sort=False).indices.items():
        c_of = dict(zip(cyc[idx], idx))
        t, tmax = 5, cyc[idx].max()
        while True:
            t += int(rng.uniform(0.5 * L, 1.5 * L))
            if t > tmax:
                break
            # nearest available cycle at or after t
            avail = [c for c in range(t, min(t + 10, tmax + 1)) if c in c_of]
            if avail:
                cu[c_of[avail[0]]] = True
    return cu


def hxcp_plus(tg_obs, tg_true, mu, sd, inner_obs, inner, beta, per_cell=False):
    """HX-CP+ with observed (possibly noisy) labels at check-ups; evaluated against true labels."""
    kappa, gamma = E.select_kappa_gamma(inner_obs, L, beta, ALPHA)
    inn = inner_obs.rename(columns={"y": "dsoh"})
    c_t, _, st_t = E.anchored(tg_obs, mu, L, kappa)
    c_i, _, st_i = E.anchored(inn, inner_obs["mu"].values, L, kappa)
    sc_t, sc_i = E._scale(sd, st_t, L, beta, gamma), E._scale(inner_obs["sd"].values, st_i, L, beta, gamma)
    q0 = E.hq(inner_obs, np.abs(inner_obs["y"].values - c_i) / sc_i, ALPHA)
    if not per_cell:
        hw = E.online_fleet(tg_obs, c_t, sc_t, q0, ALPHA, L)
    else:
        hw = np.empty(len(tg_obs))
        is_cu = E.checkup_mask(tg_obs, L)
        s = np.abs(tg_obs["dsoh"].values - c_t) / sc_t
        cyc = tg_obs["cycle"].values
        for _, idx in tg_obs.groupby("cell", sort=False).indices.items():
            idx = idx[np.argsort(cyc[idx])]
            tr = QuantileTracker(q0, ALPHA, 0.05 * q0)
            for i in idx:
                hw[i] = max(tr.q, 0.0) * sc_t[i]
                if is_cu[i]:
                    tr.update(s[i])
    ev = ~E.checkup_mask(tg_obs, L)
    return E.metrics(tg_true[ev], (c_t - hw)[ev], (c_t + hw)[ev], ALPHA, c_t[ev])


def with_noise(df, ycol, sigma, seed, L):
    """Copy of df whose labels at check-up rows carry Gaussian measurement noise (sigma in SOH units)."""
    out = df.copy()
    if sigma > 0:
        rng = np.random.default_rng(seed)
        cu = E.checkup_mask(out if ycol == "dsoh" else out.rename(columns={"y": "dsoh"}), L)
        out.loc[cu, ycol] = out.loc[cu, ycol].values + rng.normal(0, sigma, cu.sum())
    return out


def fold(name, seed):
    tg, inner0, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    _, beta = E.day0_proposed(mu, sd, inner, ALPHA)
    rows = []
    base = dict(fold=name, seed=int(seed))
    # (1) noisy check-ups (noise in both the target fleet and the replay on source domains)
    for sig in (0.0, 0.005, 0.01, 0.02):
        tg_o = with_noise(tg, "dsoh", sig, 100 + int(seed), L)
        in_o = with_noise(inner, "y", sig, 200 + int(seed), L)
        rows.append(dict(base, exp="noise", level=sig * 100, **hxcp_plus(tg_o, tg, mu, sd, in_o, inner, beta)))
    # (2) irregular schedule (target and replay)
    tg_i, in_i = tg.copy(), inner.copy()
    tg_i["cu_override"] = irregular_mask(tg_i, L, 300 + int(seed))
    in_i["cu_override"] = irregular_mask(in_i.rename(columns={"y": "dsoh"}), L, 400 + int(seed))
    rows.append(dict(base, exp="schedule", level="irregular", **hxcp_plus(tg_i, tg_i, mu, sd, in_i, in_i, beta)))
    rows.append(dict(base, exp="schedule", level="regular", **hxcp_plus(tg, tg, mu, sd, inner, inner, beta)))
    # (3) per-cell vs fleet tracking
    rows.append(dict(base, exp="tracking", level="per-cell", **hxcp_plus(tg, tg, mu, sd, inner, inner, beta, per_cell=True)))
    rows.append(dict(base, exp="tracking", level="fleet", **hxcp_plus(tg, tg, mu, sd, inner, inner, beta)))
    # (4) fleet size (domains with >= 30 cells)
    cells = tg["cell"].unique()
    if len(cells) >= 30:
        rng = np.random.default_rng(500 + int(seed))
        for n in (3, 5, 10, 20):
            for rep in range(3):
                sub = rng.choice(cells, n, replace=False)
                m = tg["cell"].isin(sub).values
                t2 = tg[m].reset_index(drop=True)
                rows.append(dict(base, exp="fleet", level=n, rep=rep,
                                 **hxcp_plus(t2, t2, mu[m], sd[m], inner, inner, beta)))
        rows.append(dict(base, exp="fleet", level=len(cells), rep=0, **hxcp_plus(tg, tg, mu, sd, inner, inner, beta)))
    return rows


def main():
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in sorted(glob.glob(os.path.join(RES, "offline", "lodo", "*.pkl")))]
    out = Parallel(n_jobs=5)(delayed(fold)(n, s) for n, s in jobs)
    df = pd.DataFrame([r for rr in out for r in rr])
    df.to_csv(os.path.join(RES, "deploy.csv"), index=False)
    g = df.groupby(["exp", "level", "fold"]).mean(numeric_only=True).groupby(["exp", "level"])[
        ["cov", "width", "winkler", "cell_p10", "rmse"]].mean()
    pd.set_option("display.width", 200)
    print(g.round(3))
    print(df[df.exp == "fleet"].groupby(["level", "fold"])["cov"].mean().groupby("level").agg(["mean", "min"]).round(3))


if __name__ == "__main__":
    main()
