"""Generate the dataset/domain summary table (paper/tab_data.tex)."""
from data import DOMAIN_INFO, load_all

BS = "\\\\"
d = load_all()
life = d.groupby("cell")["cycle"].max() + 1
dom_of = d.groupby("cell")["domain"].first()
order = ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW", "XJTU-Sim", "TJU-NCA", "TJU-NCM",
         "TJU-NCM+NCA", "HUST", "MIT-b1", "MIT-b2", "MIT-b3"]
proto = {"XJTU-2C": "2C CC discharge", "XJTU-3C": "3C CC discharge", "XJTU-R2.5": "random 2.5\\,A",
         "XJTU-R3": "random 3\\,A", "XJTU-RW": "random walk", "XJTU-Sim": "satellite profile",
         "TJU-NCA": "C-rate/temp. grid", "TJU-NCM": "C-rate/temp. grid", "TJU-NCM+NCA": "C-rate/temp. grid",
         "HUST": "77 discharge prot.", "MIT-b1": "fast charge", "MIT-b2": "fast charge", "MIT-b3": "fast charge (CLO)"}


def fmt(n):
    return f"{n:,}".replace(",", "{,}")


lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lllrrr}", r"\toprule",
         "Domain & Chem. & Protocol & Cells & Cycles & Med.\\ life" + BS, r"\midrule"]
for o in order:
    ds, ch, _ = DOMAIN_INFO[o]
    sub = d[d["domain"] == o]
    lines.append(f"{o} & {ch} & {proto[o]} & {sub['cell'].nunique()} & {fmt(len(sub))} & "
                 f"{int(life[dom_of == o].median())}" + BS)
lines += [r"\midrule", f"Total & & & {d['cell'].nunique()} & {fmt(len(d))} & " + BS, r"\bottomrule",
          r"\end{tabular}"]
open("../paper/tab_data.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
