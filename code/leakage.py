"""In-domain evaluation under the protocol of Wang et al. (2024): quantify the optimism of
look-ahead (whole-life per-cell) normalisation versus the causal representation."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from data import load_all, add_causal_features, wang_leaky_features, FEATURES, FEATURE_SETS
from models import thin

def wang_3sigma(df):
    # 3-sigma row removal per cell computed over the whole life (as in the released code)
    keep = np.ones(len(df), bool)
    cols = FEATURES + ["soh"]
    for _, g in df.groupby("cell"):
        z = (g[cols] - g[cols].mean()).abs() > 3 * g[cols].std()
        keep[df.index.get_indexer(g.index[z.any(axis=1)])] = False
    return df[keep]

def split(dom_df, seed):
    cells = sorted(dom_df.cell.unique())
    if dom_df.dataset.iloc[0] == "XJTU":   # official split: batteries 4 and 8 for testing
        te = [c for c in cells if c.endswith("-4") or c.endswith("-8")]
    else:
        rng = np.random.default_rng(seed); te = list(rng.choice(cells, max(2, len(cells)//5), replace=False))
    return dom_df.cell.isin(te)

base = load_all()
variants = {
    "leaky (Wang 2024 preprocessing)": (wang_leaky_features(wang_3sigma(base.copy())), FEATURES, "soh"),
    "causal, absolute SOH": (add_causal_features(base), FEATURE_SETS["causal"], "soh"),
}
c = add_causal_features(base); ref = c[c.cycle < 5].groupby("cell").soh.median()
c["dsoh"] = c.soh - ref.reindex(c.cell).values
variants["causal, commissioning-referenced"] = (c, FEATURE_SETS["causal"], "dsoh")
rows = []
for name, (df, F, tgt) in variants.items():
    for dom, g in df.groupby("domain"):
        for seed in range(3):
            te = split(g, seed); tr = thin(g[~te])
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=2, random_state=seed).fit(tr[F], tr[tgt])
            T = g[te & (g.cycle >= 5)]
            p = m.predict(T[F]); y = T[tgt].values
            if tgt == "dsoh":
                p, y = p + (T.soh - T.dsoh).values, T.soh.values
            rows.append(dict(variant=name, domain=dom, seed=seed, mape=np.mean(np.abs(p-y)/y)*100, rmse=np.sqrt(np.mean((p-y)**2))*100))
r = pd.DataFrame(rows); r.to_csv("../results/leakage.csv", index=False)
print(r.groupby(["variant"])[["mape", "rmse"]].mean().round(3))
print(r.groupby(["domain", "variant"]).mape.mean().unstack().round(2))
