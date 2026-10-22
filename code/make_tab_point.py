"""In-domain benchmark table vs Wang et al. (paper/tab_point.tex)."""
import pandas as pd

BS = "\\\\"
s = pd.read_csv("../results/benchmark_wang.csv", header=[0, 1], index_col=[0, 1])
m = s["MAPE"]
COLS = ["dsoh", "soh", "Wang-Ours", "Wang-MLP", "Wang-CNN"]


def row(label, vals):
    best = min(vals)
    return label + " & " + " & ".join(("\\textbf{%.2f}" % v) if v == best else "%.2f" % v for v in vals) + BS


lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Ours (causal)} & \multicolumn{3}{c}{Wang et al.\ \cite{wang2024pinn}}" + BS,
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
         r"Test batch & $\Delta$SOH & SOH & PINN & MLP & CNN" + BS, r"\midrule"]
for (exp, b), r in m.iterrows():
    lines.append(row(b, [r[c] for c in COLS]))
mm = m.mean()
lines += [r"\midrule", row("Mean", [mm[c] for c in COLS]), r"\bottomrule", r"\end{tabular}"]
open("../paper/tab_point.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("better than PINN on", int((m["dsoh"] < m["Wang-Ours"]).sum()), "of", len(m))
