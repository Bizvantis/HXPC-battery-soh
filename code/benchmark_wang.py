"""In-domain point-accuracy benchmark on the exact test cells used by Wang et al. (Nat. Commun.
2024), compared with the per-batch results released in their repository (PINN, MLP, CNN)."""
import json, os, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings("ignore")
from data import load_all, add_causal_features, FEATURE_SETS
from models import thin

W = json.load(open("../results/wang2024_released_results.json"))
d = add_causal_features(load_all())
ref = d[d.cycle < 5].groupby("cell").soh.median(); d["soh_ref"] = ref.reindex(d.cell).values; d["dsoh"] = d.soh - d.soh_ref
d["name"] = d.cell.str.split("/").str[1]

def experiments():
    xj = ["2C", "3C", "R2.5", "R3", "RW", "Sim"]
    for i, b in enumerate(xj):
        g = d[d.domain == f"XJTU-{b}"]
        te = g.name.str.contains("4") | g.name.str.contains("8")
        yield f"XJTU:{i}", f"XJTU-{b}", g, te
    for i, dom in enumerate(["TJU-NCA", "TJU-NCM", "TJU-NCM+NCA"]):
        g = d[d.domain == dom]
        yield f"TJU:{i}", dom, g, g.name.isin(W[f"TJU:experiment_mean_{i}"])
    g = d[d.dataset == "HUST"]
    yield "HUST:0", "HUST", g, g.name.isin([t.replace("--", "-") for t in W["HUST:experiment_mean_0"]])
    g = d[d.dataset == "MIT"]
    yield "MIT:0", "MIT (all)", g, g.name.str.split("-").str[-1].astype(int) % 5 == 0

rows = []
for key, label, g, te in experiments():
    for tgt in ["dsoh", "soh"]:
        for seed in range(5):
            tr = thin(g[~te]); F = FEATURE_SETS["causal"]
            m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, verbose=-1, n_jobs=2, random_state=seed).fit(tr[F], tr[tgt])
            T = g[te & (g.cycle >= 5)]
            p = m.predict(T[F]) + (T.soh_ref.values if tgt == "dsoh" else 0)
            e = pd.DataFrame({"c": T.cell.values, "ae": np.abs(p - T.soh.values), "ape": np.abs(p - T.soh.values) / T.soh.values, "se": (p - T.soh.values) ** 2})
            pc = e.groupby("c").agg(mae=("ae", "mean"), mape=("ape", "mean"), rmse=("se", lambda s: np.sqrt(s.mean())))
            rows.append(dict(exp=key, batch=label, target=tgt, seed=seed, n_test=te.groupby(g.cell).first().sum(),
                             MAE=pc.mae.mean() * 100, MAPE=pc.mape.mean() * 100, RMSE=pc.rmse.mean() * 100))
r = pd.DataFrame(rows)
s = r.groupby(["exp", "batch", "target"])[["MAE", "MAPE", "RMSE"]].mean().unstack("target")
for mdl in ["Ours", "MLP", "CNN"]:
    for k in s.index.get_level_values(0):
        ds, i = k.split(":")
        x = pd.read_excel(f"../data/PINN4SOH/results analysis/processed results/{mdl}-{ds}-results.xlsx", sheet_name=f"battery_mean_{i}")
        s.loc[k, (f"MAPE", f"Wang-{mdl}")] = x.MAPE.mean() * 100
        s.loc[k, (f"RMSE", f"Wang-{mdl}")] = x.RMSE.mean() * 100
s = s.sort_index(axis=1)
pd.set_option("display.width", 250)
print(s.round(3)); print(s.mean().round(3))
s.to_csv("../results/benchmark_wang.csv"); r.to_csv("../results/benchmark_wang_raw.csv", index=False)
