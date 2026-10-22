"""Calibration variants, online (check-up) simulation and metrics on stored offline predictions.

Day-0 (label-free on the target domain):
  baselines: split CP, weighted CP (covariate shift), CQR, LightGBM-QR, deep ensemble, MC dropout, GP
  proposed : XD-CP  = cross-domain scores + hierarchical (domain->cell->cycle) pooling
             XD-CP-n = same with ensemble-normalised scores, beta chosen by nested source CV
In service (sparse capacity check-ups every L cycles):
  baselines: ACI on split CP (Gibbs & Candes 2021); target recalibration with m fully labelled cells
  proposed : residual-anchored correction + fleet-level online quantile tracking on XD-CP-n
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm

from conformal import QuantileTracker, conformal_quantile, hierarchical_weights, weighted_quantile

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BETAS = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, np.inf]
EPS = 1e-4


# ------------------------------------------------------------------------------------ metrics
def metrics(df, lo, hi, alpha, mu=None):
    y = df["dsoh"].values
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cov = (y >= lo) & (y <= hi)
    width = hi - lo
    # Winkler / interval score (Gneiting & Raftery 2007), in SOH %
    pen = (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))
    cell_cov = pd.Series(cov).groupby(df["cell"].values).mean()
    out = {"cov": cov.mean(), "width": np.nanmean(np.where(np.isfinite(width), width, np.nan)) * 100,
           "winkler": np.nanmean(np.where(np.isfinite(width), width + pen, np.nan)) * 100,
           "cell_p10": cell_cov.quantile(0.1), "cell_ok": (cell_cov >= 1 - alpha - 0.05).mean(),
           "inf_frac": np.mean(~np.isfinite(width))}
    if mu is not None:
        e = np.asarray(mu) - y
        out["rmse"], out["mae"] = np.sqrt(np.mean(e ** 2)) * 100, np.mean(np.abs(e)) * 100
    return out


# ------------------------------------------------------------------------------- day-0 methods
def hq(inner, scores, alpha):
    if "w_h" not in inner.columns:
        w = hierarchical_weights(inner["domain"].values, inner["cell"].values)
    else:
        w = inner["w_h"].values
    return weighted_quantile(scores, w, alpha)


def select_beta(inner, alpha):
    """Nested leave-one-source-domain-out choice of the normalisation offset beta: minimise the
    mean Winkler score on held-out source domains (no target information is used)."""
    best, best_b = np.inf, np.inf
    r = np.abs(inner["y"].values - inner["mu"].values)
    for b in BETAS:
        s = r if np.isinf(b) else r / (inner["sd"].values + b + EPS)
        sc = []
        for k in inner["domain"].unique():
            m = (inner["domain"] == k).values
            q = hq(inner[~m], s[~m], alpha)
            hw = q if np.isinf(b) else q * (inner["sd"].values[m] + b + EPS)
            y, mu = inner["y"].values[m], inner["mu"].values[m]
            pen = (2 / alpha) * np.maximum(np.abs(y - mu) - hw, 0)
            # domain-averaged Winkler score
            sc.append(np.mean(2 * hw + pen))
        if np.mean(sc) < best:
            best, best_b = np.mean(sc), b
    return best_b


def day0_baselines(tg, alpha):
    """Label-free baselines: {method: (lo, hi, mu)}."""
    z = norm.ppf(1 - alpha / 2)
    out = {}
    for m in ["de", "mcd", "gp"]:
        out[m] = (tg[f"mu_{m}"] - z * tg[f"sd_{m}"], tg[f"mu_{m}"] + z * tg[f"sd_{m}"], tg[f"mu_{m}"])
    out["split"] = (tg["mu_split"] - tg[f"q_split_{alpha}"], tg["mu_split"] + tg[f"q_split_{alpha}"], tg["mu_split"])
    out["wcp"] = (tg["mu_split"] - tg[f"q_wcp_{alpha}"], tg["mu_split"] + tg[f"q_wcp_{alpha}"], tg["mu_split"])
    out["qr"] = (tg[f"lo_qr_{alpha}"], tg[f"hi_qr_{alpha}"], tg["mu_split"])
    out["cqr"] = (tg[f"lo_cqr_{alpha}"], tg[f"hi_cqr_{alpha}"], tg["mu_split"])
    return out


def day0_proposed(mu, sd, inner, alpha):
    """HX-CP and its ablations for one backbone (target mean/sd arrays + inner cross-domain table)."""
    out = {}
    r = np.abs(inner["y"] - inner["mu"]).values
    q = conformal_quantile(r, alpha)                       # sample-pooled cross-domain scores
    out["xd_pool"] = (mu - q, mu + q, mu)
    q = hq(inner, r, alpha)                                # + hierarchical pooling
    out["xd"] = (mu - q, mu + q, mu)
    b = select_beta(inner, alpha)                          # + ensemble normalisation (nested CV)
    if np.isinf(b):
        out["hxcp"] = out["xd"]
    else:
        q = hq(inner, r / (inner["sd"].values + b + EPS), alpha)
        hw = q * (sd + b + EPS)
        out["hxcp"] = (mu - hw, mu + hw, mu)
    return out, b


# ------------------------------------------------------------------------- online simulation
def checkup_mask(df, L):
    """Capacity check-ups every L cycles after commissioning (cycle index relative to K_REF=5).
    A boolean column `cu_override` (irregular schedules, see analysis_deploy.py) takes precedence."""
    if "cu_override" in df.columns:
        return df["cu_override"].values.astype(bool)
    c = df["cycle"].values - 5
    return (c > 0) & (c % L == 0)


def _anchor_state(df, L):
    """Per row: index (into df) of the most recent check-up strictly before it in the same cell,
    or -1 for commissioning. Cached on (id(df), L) because it is reused for many (kappa, mu)."""
    key = (id(df), len(df), L, "cu_override" in df.columns, int(df["cycle"].values[:200].sum()))
    if _anchor_state.cache.get("key") == key:
        return _anchor_state.cache["val"]
    is_cu = checkup_mask(df, L)
    cell = pd.factorize(df["cell"].values)[0]
    cyc = df["cycle"].values
    order = np.lexsort((cyc, cell))
    idx_sorted = np.where(is_cu[order], order, -1)
    # previous check-up within the cell: shift by one inside each cell, then forward-fill
    prev = np.empty(len(df), dtype=np.int64)
    s = pd.Series(idx_sorted)
    g = pd.Series(cell[order])
    shifted = s.groupby(g.values).shift(1)
    shifted = shifted.where(shifted >= 0)
    filled = shifted.groupby(g.values).ffill().fillna(-1).astype(np.int64).values
    prev[order] = filled
    val = (prev, is_cu)
    _anchor_state.cache = {"key": key, "val": val}
    return val


_anchor_state.cache = {}


def anchored(df, mu, L, kappa=1.0):
    """Anchored estimate between capacity check-ups:
        centre_t = y_anchor + kappa * (mu_t - mu_anchor),
    i.e. the last measured SOH plus (a shrunk) model-predicted degradation since that check-up.
    kappa = 1 is residual anchoring, kappa = 0 is persistence. Commissioning (cycle K_REF, label
    0 by construction, mu_anchor := 0) is the first anchor, so with no check-ups centre = mu.
    Returns the centre, the last observed label and the staleness (cycles since the anchor)."""
    prev, _ = _anchor_state(df, L)
    has = prev >= 0
    pi = np.where(has, prev, 0)
    y = df["dsoh"].values
    cyc = df["cycle"].values
    ya = np.where(has, y[pi], 0.0)
    ma = np.where(has, np.asarray(mu)[pi], 0.0)
    lc = np.where(has, cyc[pi], 5)
    return ya + kappa * (np.asarray(mu) - ma), ya, np.maximum(cyc - lc, 1)


def online_fleet(df, center, scale, q0, alpha, L, lr_frac=0.05):
    """Fleet-level online quantile tracking. All cells of the target domain age in parallel;
    at each check-up the score of the (pre-update) prediction updates the shared threshold q.
    A row at cycle c uses the threshold after all check-ups at cycles < c.
    Returns per-row half-widths (threshold in force times the row's scale)."""
    is_cu = checkup_mask(df, L)
    s = np.abs(df["dsoh"].values - center) / scale
    cyc = df["cycle"].values
    ev = np.where(is_cu)[0]
    ev = ev[np.lexsort((df["cell"].values[ev], cyc[ev]))]
    tr = QuantileTracker(q0, alpha, lr_frac * q0 if np.isfinite(q0) and q0 > 0 else 0.01)
    qs = np.empty(len(ev) + 1)
    qs[0] = tr.q
    for j, k in enumerate(ev):
        tr.update(s[k])
        qs[j + 1] = tr.q
    n_before = np.searchsorted(cyc[ev], cyc, side="left")
    return np.maximum(qs[n_before], 0.0) * scale  # an empty set is reported as zero width


GAMMAS = [0.0, 0.5, 1.0]


def _scale(sd, stale, L, beta, gamma):
    base = np.ones_like(stale, dtype=float) if np.isinf(beta) else sd + beta + EPS
    return base * (stale / L) ** gamma


KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def select_kappa_gamma(inner, L, beta, alpha):
    """Nested leave-one-source-domain-out choice of (kappa, gamma) minimising the mean Winkler
    score on held-out source domains (no target information)."""
    inn = inner.rename(columns={"y": "dsoh"})
    y = inner["y"].values
    best, arg = np.inf, (1.0, 0.0)
    doms = inner["domain"].values
    for kp in KAPPAS:
        c, _, st = anchored(inn, inner["mu"].values, L, kp)
        r = np.abs(y - c)
        for g in GAMMAS:
            sc = _scale(inner["sd"].values, st, L, beta, g)
            s = r / sc
            win = []
            for k in np.unique(doms):
                m = doms == k
                hw = hq(inner[~m], s[~m], alpha) * sc[m]
                win.append(np.mean(2 * hw + (2 / alpha) * np.maximum(r[m] - hw, 0)))
            if np.mean(win) < best:
                best, arg = np.mean(win), (kp, g)
    return arg


def inservice_baselines(tg, inner, alpha, L, lr_frac=0.05):
    out = {}
    inn = inner.rename(columns={"y": "dsoh"})
    mu = tg["mu_split"].values                             # ACI on split CP, same check-ups
    q0 = tg[f"q_split_{alpha}"].values[0]
    hw = online_fleet(tg, mu, np.ones(len(tg)), q0, alpha, L, lr_frac)
    out["aci_split"] = (mu - hw, mu + hw, mu)
    _, ly_t, _ = anchored(tg, np.zeros(len(tg)), L)       # persistence of the last measurement
    _, ly_i, _ = anchored(inn, np.zeros(len(inner)), L)
    q0 = hq(inner, np.abs(inner["y"].values - ly_i), alpha)
    out["persist_cp"] = (ly_t - q0, ly_t + q0, ly_t)
    hw = online_fleet(tg, ly_t, np.ones(len(tg)), q0, alpha, L, lr_frac)
    out["persist_online"] = (ly_t - hw, ly_t + hw, ly_t)
    return out


def inservice_proposed(tg, mu, sd, inner, alpha, L, beta, lr_frac=0.05):
    out = {}
    inn = inner.rename(columns={"y": "dsoh"})
    y_i = inner["y"].values
    kappa, gamma = select_kappa_gamma(inner, L, beta, alpha)
    c_t, _, st_t = anchored(tg, mu, L, kappa)
    c_i, _, st_i = anchored(inn, inner["mu"].values, L, kappa)
    sc_t, sc_i = _scale(sd, st_t, L, beta, gamma), _scale(inner["sd"].values, st_i, L, beta, gamma)
    q0 = hq(inner, np.abs(y_i - c_i) / sc_i, alpha)
    out["hxcp+static"] = (c_t - q0 * sc_t, c_t + q0 * sc_t, c_t)
    hw = online_fleet(tg, c_t, sc_t, q0, alpha, L, lr_frac)
    out["hxcp+"] = (c_t - hw, c_t + hw, c_t)
    # ablation: no anchoring (centre = mu), online tracking only
    sc0_t = _scale(sd, np.full(len(tg), float(L)), L, beta, 0.0)
    sc0_i = _scale(inner["sd"].values, np.full(len(inner), float(L)), L, beta, 0.0)
    q00 = hq(inner, np.abs(y_i - inner["mu"].values) / sc0_i, alpha)
    hw = online_fleet(tg, mu, sc0_t, q00, alpha, L, lr_frac)
    out["abl_noanchor"] = (mu - hw, mu + hw, mu)
    # ablation: unnormalised scores (beta = inf, gamma = 0)
    q1 = hq(inner, np.abs(y_i - c_i), alpha)
    hw = online_fleet(tg, c_t, np.ones(len(tg)), q1, alpha, L, lr_frac)
    out["abl_abs"] = (c_t - hw, c_t + hw, c_t)
    # ablation: full residual anchoring (kappa = 1)
    c1_t, _, s1_t = anchored(tg, mu, L, 1.0)
    c1_i, _, s1_i = anchored(inn, inner["mu"].values, L, 1.0)
    sct1, sci1 = _scale(sd, s1_t, L, beta, gamma), _scale(inner["sd"].values, s1_i, L, beta, gamma)
    q2 = hq(inner, np.abs(y_i - c1_i) / sci1, alpha)
    hw = online_fleet(tg, c1_t, sct1, q2, alpha, L, lr_frac)
    out["abl_kappa1"] = (c1_t - hw, c1_t + hw, c1_t)
    return out, (kappa, gamma)


def target_recal(tg, alpha, m, seed):
    """Recalibrate split-CP on m fully labelled target cells (lab ageing tests on the new type);
    evaluated on the remaining target cells."""
    rng = np.random.default_rng(seed)
    cells = tg["cell"].unique()
    if len(cells) <= m:
        return None
    lab = rng.choice(cells, size=m, replace=False)
    msk = tg["cell"].isin(lab).values
    r = np.abs(tg["dsoh"].values[msk] - tg["mu_split"].values[msk])
    w = 1.0 / pd.Series(r).groupby(tg["cell"].values[msk]).transform("size").values
    q = weighted_quantile(r, w, alpha)
    mu = tg["mu_split"].values[~msk]
    return tg[~msk], mu - q, mu + q, mu


# ------------------------------------------------------------------------------------- driver
def load_fold(protocol, name, seed):
    blob = pd.read_pickle(os.path.join(RES, "offline", protocol, f"{name}_s{seed}.pkl"))
    tg = blob["target"]
    backbones = {"dse": (tg["mu_dse"].values, tg["sd_dse"].values, blob["inner"])}
    fh = os.path.join(RES, "offline", protocol + "_hdse", f"{name}_s{seed}.pkl")
    if os.path.exists(fh):
        h = pd.read_pickle(fh)
        ht = h["target"].reindex(tg.index)
        backbones["hdse"] = (ht["mu_hdse"].values, ht["sd_hdse"].values, h["inner"])
    for _, (_, _, inn) in backbones.items():
        inn["w_h"] = hierarchical_weights(inn["domain"].values, inn["cell"].values)
    return tg.reset_index(drop=True), blob["inner"], backbones


def evaluate_fold(protocol, name, seed, alphas=(0.05, 0.1, 0.2), Ls=(25, 50, 100, 200)):
    rows = []
    tg, inner0, backbones = load_fold(protocol, name, seed)
    base = dict(fold=name, seed=int(seed))
    for a in alphas:
        for meth, (lo, hi, mu) in day0_baselines(tg, a).items():
            rows.append(dict(base, alpha=a, stage="day0", L=0, method=meth, **metrics(tg, lo, hi, a, mu)))
        betas = {}
        for bb, (mu_b, sd_b, inner) in backbones.items():
            d0, betas[bb] = day0_proposed(mu_b, sd_b, inner, a)
            for meth, (lo, hi, mu) in d0.items():
                rows.append(dict(base, alpha=a, stage="day0", L=0, method=f"{meth}@{bb}", beta=betas[bb],
                                 **metrics(tg, lo, hi, a, mu)))
        for L in Ls:
            ev = ~checkup_mask(tg, L)
            sub = tg[ev]
            for meth, (lo, hi, mu) in inservice_baselines(tg, inner0, a, L).items():
                rows.append(dict(base, alpha=a, stage="inservice", L=L, method=meth,
                                 **metrics(sub, np.asarray(lo)[ev], np.asarray(hi)[ev], a, np.asarray(mu)[ev])))
            for bb, (mu_b, sd_b, inner) in backbones.items():
                ins, (kp, gm) = inservice_proposed(tg, mu_b, sd_b, inner, a, L, betas[bb])
                for meth, (lo, hi, mu) in ins.items():
                    rows.append(dict(base, alpha=a, stage="inservice", L=L, method=f"{meth}@{bb}", beta=betas[bb],
                                     kappa=kp, gamma=gm,
                                     **metrics(sub, np.asarray(lo)[ev], np.asarray(hi)[ev], a, np.asarray(mu)[ev])))
        for m in (1, 2, 3, 5):
            rc = target_recal(tg, a, m, int(seed))
            if rc is not None:
                sub, lo, hi, mu = rc
                rows.append(dict(base, alpha=a, stage="recal", L=m, method=f"recal_m{m}", **metrics(sub, lo, hi, a, mu)))
    return rows


def evaluate(protocol="lodo", n_jobs=4, **kw):
    from joblib import Parallel, delayed
    files = sorted(glob.glob(os.path.join(RES, "offline", protocol, "*.pkl")))
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in files]
    res = Parallel(n_jobs=n_jobs)(delayed(evaluate_fold)(protocol, n, s, **kw) for n, s in jobs)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(os.path.join(RES, f"metrics_{protocol}.csv"), index=False)
    return df


if __name__ == "__main__":
    import sys
    df = evaluate(sys.argv[1] if len(sys.argv) > 1 else "lodo")
    s = df[(df.alpha == 0.1)].groupby(["stage", "L", "method"])[
        ["cov", "width", "winkler", "cell_p10", "cell_ok", "rmse"]].mean()
    pd.set_option("display.width", 200)
    print(s.round(3))
