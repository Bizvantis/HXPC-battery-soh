"""Decision-level table (paper/tab_decision.tex) from results/decision.csv."""
import os

import pandas as pd

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BS = "\\\\"
df = pd.read_csv(os.path.join(RES, "decision.csv"))
g = df.groupby(["threshold", "method", "fold"]).mean(numeric_only=True).groupby(["threshold", "method"]).mean()
order = [("point (no UQ)", "Point estimate (no UQ)", "0"), ("Split CP", "Split CP", "0"), ("CQR", "CQR", "0"),
         ("GP", "GP", "0"), ("HX-CP", "HX-CP (ours)", "0"), ("ACI", "ACI", "S"),
         ("Persist.+online", "Persist.+online", "S"), ("HX-CP+", "HX-CP+ (ours)", "S")]
lines = [r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llrrrrrrrr}", r"\toprule",
         r" & & \multicolumn{4}{c}{Threshold 90\% SOH} & \multicolumn{4}{c}{Threshold 85\% SOH}" + BS,
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         r"Stage & Method & Miss & FA & $\le L$ & Early & Miss & FA & $\le L$ & Early" + BS, r"\midrule"]
prev = None
for k, nm, st in order:
    if prev is not None and st != prev:
        lines.append(r"\midrule")
    cells = []
    for th in (0.9, 0.85):
        r = g.loc[(th, k)]
        cells += [f"{100 * r['miss']:.1f}", f"{100 * r['false_alarm']:.1f}", f"{100 * r['within_L']:.0f}",
                  f"{100 * r['early_100']:.0f}"]
    lines.append(("Day 0" if st == "0" and prev != "0" else ("$L=100$" if st == "S" and prev != "S" else "")) +
                 f" & {nm} & " + " & ".join(cells) + BS)
    prev = st
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(RES, "..", "paper", "tab_decision.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
