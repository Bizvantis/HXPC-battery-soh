"""Check whether the GP baseline is under-tuned (near-constant predictive std)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as G
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from run_offline import prepare, F
from models import thin

d = prepare()
T = sys.argv[1] if len(sys.argv) > 1 else "XJTU-2C"
S = thin(d[d.domain != T]); te = d[(d.domain == T) & (d.cycle >= 5)].sample(4000, random_state=0)
m, s = S[F].mean(), S[F].std().replace(0, 1)
X, y, Xt, yt = ((S[F] - m) / s).values, S.dsoh.values, ((te[F] - m) / s).values, te.dsoh.values
j = np.random.default_rng(0).choice(len(X), 1500, replace=False)
cfgs = {
    "old": Ck(1e-3, (1e-6, 1.0)) * RBF(np.ones(X.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-5, (1e-8, 1e-2)),
    "new": Ck(1.0, (1e-2, 1e2)) * RBF(np.full(X.shape[1], 5.0), (1e-1, 1e3)) + WhiteKernel(1e-2, (1e-5, 1.0)),
}
for name, k in cfgs.items():
    t = time.time()
    g = G(k, normalize_y=True, n_restarts_optimizer=0, random_state=0).fit(X[j], y[j])
    mu, sd = g.predict(Xt, return_std=True)
    print(T, name, "rmse %.2f sd_mean %.3f sd_cv %.3f cov90 %.3f  %.0fs" % (
        np.sqrt(np.mean((mu - yt) ** 2)) * 100, sd.mean() * 100, sd.std() / sd.mean(),
        np.mean(np.abs(mu - yt) <= 1.645 * sd), time.time() - t), flush=True)
    print("   ", g.kernel_)
