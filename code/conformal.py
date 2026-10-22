"""Conformal calibration utilities: (weighted) quantiles, hierarchical weights, online tracking."""
import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha):
    """Standard split-conformal quantile with the (n+1) finite-sample correction."""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else s[k - 1]


def weighted_quantile(scores, weights, alpha, w_test=0.0):
    """Quantile of sum_i p_i delta_{s_i} + p_test delta_{+inf} (Tibshirani et al., 2019).

    `w_test` is the (unnormalised) weight of the test point; with w_test = 0 this is the plain
    weighted empirical quantile used for CDF pooling (Dunn et al., 2023).
    """
    scores = np.asarray(scores, float)
    weights = np.asarray(weights, float)
    o = np.argsort(scores)
    s, w = scores[o], weights[o]
    total = w.sum() + w_test
    cw = np.cumsum(w) / total
    i = np.searchsorted(cw, 1 - alpha)
    return np.inf if i >= len(s) else s[i]


def hierarchical_weights(domain, cell):
    """Weights making every domain count equally, every cell equally within its domain and
    every cycle equally within its cell (domain -> cell -> cycle pooling). Each domain's weights
    sum to one, so the weights of any subset of whole domains are unchanged."""
    df = pd.DataFrame({"d": np.asarray(domain), "c": np.asarray(cell)})
    n_cyc = df.groupby("c")["c"].transform("size").values
    n_cell = df.groupby("d")["c"].transform("nunique").values
    return 1.0 / (n_cyc * n_cell)


def cell_weights(cell):
    """Each calibration cell counts equally (cell -> cycle pooling)."""
    _, inv, cnt = np.unique(np.asarray(cell), return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]


def trajectory_threshold(scores, cell, alpha, delta):
    """(delta, alpha) trajectory-level calibration.

    For each calibration cell take the (1-delta) within-cell quantile of its scores; the
    split-conformal quantile of these cell-level scores gives a threshold q such that, for an
    exchangeable new cell, P( fraction of its cycles covered >= 1-delta ) >= 1-alpha.
    """
    scores = np.asarray(scores)
    cell = np.asarray(cell)
    per_cell = [np.quantile(scores[cell == c], 1 - delta, method="higher") for c in np.unique(cell)]
    return conformal_quantile(per_cell, alpha)


class QuantileTracker:
    """Online conformal quantile tracking (P-control of Angelopoulos, Candes & Tibshirani, 2023;
    equivalently online gradient descent on the pinball loss). Long-run miscoverage -> alpha for
    any (bounded) score sequence."""

    def __init__(self, q0, alpha, lr):
        self.q, self.alpha, self.lr = float(q0), alpha, lr

    def update(self, score):
        err = float(score > self.q)
        self.q = self.q + self.lr * (err - self.alpha)  # no clipping: keeps the telescoping bound exact
        return err
