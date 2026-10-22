"""Deployment-realism table (paper/tab_deploy.tex) from results/deploy.csv."""
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BS = "\\\\"
d = pd.read_csv(os.path.join(RES, "deploy.csv"))
d["level"] = d["level"].astype(str)


def agg(sub):
    g = sub.groupby("fold").mean(numeric_only=True)
    return g["cov"].mean(), g["cov"].min(), g["width"].mean(), g["winkler"].mean(), g["rmse"].mean()


rows = []
for lv, lab in [("0.0", "none (exact)"), ("0.5", "$\\sigma=0.5$~pp"), ("1.0", "$\\sigma=1$~pp"), ("2.0", "$\\sigma=2$~pp")]:
    rows.append(("Check-up noise", lab, agg(d[(d.exp == "noise") & (d.level == lv)])))
for lv, lab in [("regular", "regular, every $L$"), ("irregular", "irregular, $U(0.5L,1.5L)$ gaps")]:
    rows.append(("Schedule", lab, agg(d[(d.exp == "schedule") & (d.level == lv)])))
for lv, lab in [("fleet", "fleet-level (shared $q$)"), ("per-cell", "per-cell tracker")]:
    rows.append(("Tracking", lab, agg(d[(d.exp == "tracking") & (d.level == lv)])))
fl = d[d.exp == "fleet"].copy()
full_levels = sorted(set(fl.level) - {"3", "5", "10", "20"}, key=int)
for lv in ["3", "5", "10", "20"]:
    rows.append(("Fleet size$^\\dagger$", f"{lv} cells", agg(fl[fl.level == lv])))
rows.append(("Fleet size$^\\dagger$", "all cells", agg(fl[fl.level.isin(full_levels)])))

lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrr}", r"\toprule",
         "Factor & Setting & Cov & Worst & W & IS & RMSE" + BS, r" & & (\%) & (\%) & (pp) & (pp) & (pp)" + BS]
prev = None
for fac, lab, (c, w0, w, i, r) in rows:
    if fac != prev:
        lines.append(r"\midrule")
    lines.append(f"{fac if fac != prev else ''} & {lab} & {100 * c:.1f} & {100 * w0:.1f} & {w:.2f} & {i:.2f} & {r:.2f}" + BS)
    prev = fac
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(RES, "..", "paper", "tab_deploy.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
