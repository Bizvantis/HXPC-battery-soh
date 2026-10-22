"""Point / probabilistic regressors used by the proposed method and the baselines (CPU only)."""
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

torch.set_num_threads(8)

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=20, verbose=-1, n_jobs=8)


def thin(df, per_cell=150):
    """Keep ~per_cell evenly spaced cycles of every cell so long-lived cells do not dominate."""
    idx = []
    for _, g in df.groupby("cell", sort=False):
        step = max(1, len(g) // per_cell)
        idx.append(g.index.values[::step])
    return df.loc[np.concatenate(idx)]


def fit_lgb(X, y, seed, **kw):
    p = dict(LGB_PARAMS, random_state=seed)
    p.update(kw)
    return lgb.LGBMRegressor(**p).fit(X, y)


class DomainSubsampleEnsemble:
    """M gradient-boosting members, each trained on a random subset (fraction `frac`) of the
    source domains. The spread of the members is an epistemic signal for domain novelty."""

    def __init__(self, features, M=5, frac=0.7, seed=0):
        self.F, self.M, self.frac, self.seed = features, M, frac, seed

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        doms = np.array(sorted(train["domain"].unique()))
        self.members = []
        for m in range(self.M):
            k = max(2, int(round(self.frac * len(doms)))) if len(doms) > 2 else len(doms)
            keep = rng.choice(doms, size=k, replace=False)
            sub = train[train["domain"].isin(keep)]
            self.members.append(fit_lgb(sub[self.F], sub["dsoh"], self.seed * 100 + m))
        return self

    def predict(self, X):
        P = np.stack([m.predict(X[self.F]) for m in self.members])
        return P.mean(0), P.std(0)


# --------------------------------------------------------------------------- neural baselines
class _MLP(nn.Module):
    def __init__(self, d, h=128, p=0.0, hetero=True):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Dropout(p),
                                  nn.Linear(h, h), nn.SiLU(), nn.Dropout(p),
                                  nn.Linear(h, h), nn.SiLU(), nn.Dropout(p))
        self.mu = nn.Linear(h, 1)
        self.ls = nn.Linear(h, 1) if hetero else None

    def forward(self, x):
        z = self.body(x)
        mu = self.mu(z).squeeze(-1)
        ls = self.ls(z).squeeze(-1).clamp(-12, 4) if self.ls is not None else torch.zeros_like(mu)
        return mu, ls


def train_mlp(X, y, seed, p=0.0, epochs=60, lr=2e-3, bs=512):
    """Gaussian-NLL MLP with a 10% cell-agnostic validation split for early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    perm = rng.permutation(n)
    va, tr = perm[: n // 10], perm[n // 10:]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    net = _MLP(X.shape[1], p=p)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        idx = torch.tensor(rng.permutation(tr))
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            mu, ls = net(Xt[j])
            loss = (0.5 * ((yt[j] - mu) ** 2) * torch.exp(-2 * ls) + ls).mean() if ep >= 5 \
                else ((yt[j] - mu) ** 2).mean()  # MSE warm-up stabilises NLL training
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            mu, ls = net(Xt[va])
            v = (0.5 * ((yt[va] - mu) ** 2) * torch.exp(-2 * ls) + ls).mean().item()
        if ep >= 5:
            if v < best - 1e-5:
                best, best_state, bad = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def predict_mlp(net, X, mc=0):
    Xt = torch.tensor(X, dtype=torch.float32)
    if mc:
        net.train()  # keep dropout active
        with torch.no_grad():
            outs = [net(Xt) for _ in range(mc)]
        mus = torch.stack([o[0] for o in outs]).numpy()
        sig2 = torch.stack([torch.exp(2 * o[1]) for o in outs]).numpy()
        return mus.mean(0), np.sqrt(sig2.mean(0) + mus.var(0))
    net.eval()
    with torch.no_grad():
        mu, ls = net(Xt)
    return mu.numpy(), torch.exp(ls).numpy()


def fit_gp(X, y, seed, n=2500):
    rng = np.random.default_rng(seed)
    j = rng.choice(len(X), size=min(n, len(X)), replace=False)
    k = ConstantKernel(1e-3, (1e-6, 1.0)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-5, (1e-8, 1e-2))
    return GaussianProcessRegressor(k, normalize_y=True, random_state=seed, n_restarts_optimizer=0).fit(X[j], y[j])
