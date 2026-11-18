"""Domain-adaptation backbone: domain-adversarial neural networks (DANN, Ganin et al. 2016).

Two uses are evaluated on the same folds as every other method:
  (1) split CP with a DANN point predictor, calibrated on held-out source cells exactly like split CP;
  (2) HDSE-DA: the heterogeneous domain-subsample ensemble of run_proposed.py with its two MLP members
      replaced by DANN members. Each member is adapted to the unlabelled features of the domain it
      predicts: the held-out source domain for the cross-domain scores, the target domain for deployment.
      The LightGBM members and the domain subsets are identical to those of the HDSE.
Target labels are never used; only target features, which are available on day zero.

usage: python run_da.py --seeds 0 1 2 [--protocol lodo|lodso] [--threads 3]
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from conformal import conformal_quantile
from models import fit_lgb, thin
from run_offline import ALPHAS, F, K_REF, OUT, prepare, split_cells

warnings.filterwarnings("ignore")


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class _DANN(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.head = nn.Linear(h, 1)
        self.dom = nn.Sequential(nn.Linear(h, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, lam=0.0):
        z = self.body(x)
        return self.head(z).squeeze(-1), self.dom(_GradReverse.apply(z, lam)).squeeze(-1)


def train_dann(Xs, ys, Xt, seed, epochs=40, lr=2e-3, bs=512):
    """Regression on standardised source labels plus a source-vs-target domain classifier behind a
    gradient-reversal layer; adaptation weight lambda_p = 2/(1+exp(-10p)) - 1 (Ganin et al. 2016).
    Early stopping on a 10% source validation split (MSE), as for the MLP members."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_m, y_s = ys.mean(), ys.std() + 1e-8
    n = len(Xs)
    perm = rng.permutation(n)
    va, tr = perm[: n // 10], perm[n // 10:]
    Xs_t = torch.tensor(Xs, dtype=torch.float32)
    ys_t = torch.tensor((ys - y_m) / y_s, dtype=torch.float32)
    Xt_t = torch.tensor(Xt, dtype=torch.float32)
    net = _DANN(Xs.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    nb = max(1, int(np.ceil(len(tr) / bs)))
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(tr)
        for b in range(nb):
            p = (ep * nb + b) / (epochs * nb)
            lam = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
            j = torch.tensor(idx[b * bs:(b + 1) * bs])
            jt = torch.tensor(rng.integers(0, len(Xt), size=len(j)))
            mu, ds = net(Xs_t[j], lam)
            _, dt = net(Xt_t[jt], lam)
            loss = ((ys_t[j] - mu) ** 2).mean() + bce(torch.cat([ds, dt]),
                                                      torch.cat([torch.zeros(len(ds)), torch.ones(len(dt))]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            v = ((ys_t[va] - net(Xs_t[va])[0]) ** 2).mean().item()
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32))[0].numpy() * y_s + y_m
    return predict


class HeteroDSE_DA:
    """As run_proposed.HeteroDSE (same random domain subsets and LightGBM members), with the MLP
    members replaced by DANN members adapted to the unlabelled features `X_adapt`."""

    def __init__(self, n_gb=2, n_nn=2, frac=0.7, seed=0, n_jobs=4):
        self.n_gb, self.n_nn, self.frac, self.seed, self.n_jobs = n_gb, n_nn, frac, seed, n_jobs

    def fit(self, train, X_adapt):
        rng = np.random.default_rng(self.seed)
        doms = np.array(sorted(train["domain"].unique()))
        k = max(2, int(round(self.frac * len(doms)))) if len(doms) > 2 else len(doms)
        self.members = []
        for m in range(self.n_gb + self.n_nn):
            sub = train[train["domain"].isin(rng.choice(doms, size=k, replace=False))]
            if m < self.n_gb:
                self.members.append(("gb", fit_lgb(sub[F], sub["dsoh"], self.seed * 100 + m, n_jobs=self.n_jobs), None))
            else:
                mean, std = sub[F].mean(), sub[F].std().replace(0, 1)
                pred = train_dann(((sub[F] - mean) / std).values, sub["dsoh"].values,
                                  ((X_adapt[F] - mean) / std).values, self.seed * 100 + m)
                self.members.append(("da", pred, (mean, std)))
        return self

    def predict(self, X):
        P = []
        for kind, mdl, norm in self.members:
            P.append(mdl.predict(X[F]) if kind == "gb" else mdl(((X[F] - norm[0]) / norm[1]).values))
        P = np.stack(P)
        return P.mean(0), P.std(0)


def run_fold(d, target_doms, seed, n_jobs):
    S = d[~d["domain"].isin(target_doms)]
    Tg = d[d["domain"].isin(target_doms) & (d["cycle"] >= K_REF)]
    S_eval = S[S["cycle"] >= K_REF]
    t0 = time.time()
    res = pd.DataFrame(index=Tg.index)

    # (1) split CP with a DANN point predictor (same calibration cells as split CP in run_offline.py)
    rng = np.random.default_rng(seed)
    is_cal = split_cells(S, 0.2, rng)
    Tr, Ca = thin(S[~is_cal]), S[is_cal & (S["cycle"] >= K_REF)]
    mean, std = Tr[F].mean(), Tr[F].std().replace(0, 1)
    pred = train_dann(((Tr[F] - mean) / std).values, Tr["dsoh"].values,
                      ((thin(Tg)[F] - mean) / std).values, seed * 10 + 3)
    res["mu_dann"] = pred(((Tg[F] - mean) / std).values)
    r_cal = np.abs(Ca["dsoh"].values - pred(((Ca[F] - mean) / std).values))
    for a in ALPHAS:
        res[f"q_dann_{a}"] = conformal_quantile(r_cal, a)

    # (2) HDSE-DA: cross-domain scores, each inner ensemble adapted to the held-out source domain
    inner = []
    for k in sorted(S["domain"].unique()):
        Dk = S_eval[S_eval["domain"] == k]
        ens = HeteroDSE_DA(seed=seed, n_jobs=n_jobs).fit(thin(S[S["domain"] != k]), thin(Dk))
        mu, sd = ens.predict(Dk)
        inner.append(pd.DataFrame({"domain": k, "cell": Dk["cell"].values, "cycle": Dk["cycle"].values,
                                   "life_frac": Dk["life_frac"].values, "y": Dk["dsoh"].values, "mu": mu, "sd": sd}))
    inner = pd.concat(inner, ignore_index=True)
    ens = HeteroDSE_DA(seed=seed, n_jobs=n_jobs).fit(thin(S), thin(Tg))
    res["mu_hdse_da"], res["sd_hdse_da"] = ens.predict(Tg)
    return res, inner, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--protocol", default="lodo", choices=["lodo", "lodso"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    d = prepare()
    out = os.path.join(OUT, args.protocol + "_hdse_da")
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
            tg, inner, sec = run_fold(d, doms, seed, args.threads)
            pd.to_pickle({"target": tg, "inner": inner, "seconds": sec}, f)
            y = d.loc[tg.index, "dsoh"].values
            rm = lambda c: np.sqrt(np.mean((tg[c].values - y) ** 2)) * 100
            print(f"[seed {seed}] {name:12s} {sec:6.1f}s RMSE% dann={rm('mu_dann'):.2f} hdse_da={rm('mu_hdse_da'):.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
