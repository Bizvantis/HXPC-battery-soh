"""Tables, headline numbers (paper/numbers.tex) and statistical tests from results/metrics_*.csv."""
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
BB = os.environ.get("MAIN_BACKBONE", "hdse")
LMAIN = int(os.environ.get("LMAIN", "100"))
BS = "\\\\"

DAY0 = [("split", "Split CP \\cite{lei2018}"), ("wcp", "Weighted CP \\cite{tibshirani2019}"),
        ("cqr", "CQR \\cite{romano2019cqr}"), ("qr", "LightGBM QR"), ("de", "Deep ensemble \\cite{lakshminarayanan2017}"),
        ("mcd", "MC dropout \\cite{gal2016}"), ("gp", "GP \\cite{richardson2019}"),
        (f"xd_pool@{BB}", "XD-CP, pooled (abl.)"), (f"xd@{BB}", "XD-CP, hierarchical (abl.)"),
        (f"hxcp@{BB}", "\\textbf{HX-CP (ours)}")]
INS = [("aci_split", "ACI on split CP \\cite{gibbs2021aci}"), ("persist_cp", "Persistence + XD-CP"),
       ("persist_online", "Persistence + online"), (f"abl_noanchor@{BB}", "HX-CP + online, no anchor (abl.)"),
       (f"abl_abs@{BB}", "HX-CP+, unnormalised (abl.)"), (f"abl_kappa1@{BB}", "HX-CP+, $\\kappa{=}1$ (abl.)"),
       (f"hxcp+static@{BB}", "HX-CP+, static $q_0$ (abl.)"), (f"hxcp+@{BB}", "\\textbf{HX-CP+ (ours)}")]


def load(protocol="lodo"):
    df = pd.read_csv(os.path.join(RES, f"metrics_{protocol}.csv"))
    return df


def per_domain(df):
    """average over seeds -> one row per (stage, L, method, alpha, fold)"""
    return df.groupby(["stage", "L", "method", "alpha", "fold"], as_index=False).mean(numeric_only=True)


def summarise(pdm, methods, stage, L, alpha=0.1):
    sub = pdm[(pdm.stage == stage) & (pdm.L == L) & (pdm.alpha == alpha)]
    rows = []
    for key, name in methods:
        g = sub[sub.method == key]
        if g.empty:
            continue
        rows.append(dict(key=key, name=name, cov=g["cov"].mean(), worst=g["cov"].min(),
                         width=g["width"].mean(), winkler=g["winkler"].mean(), p10=g["cell_p10"].mean(),
                         ok=g["cell_ok"].mean(), rmse=g["rmse"].mean(), inf=g["inf_frac"].mean(), n=len(g)))
    return pd.DataFrame(rows)


def fmt_table(t, bold_best=("winkler",)):
    lines = []
    best = {c: t[c].min() for c in bold_best}
    for _, r in t.iterrows():
        cells = [r["name"]]
        cells.append(f"{100 * r['cov']:.1f}")
        cells.append(f"{100 * r['worst']:.1f}")
        cells.append(f"{r['p10'] * 100:.1f}")
        cells.append(f"{r['ok'] * 100:.0f}")
        w = "$\\infty$" if r["inf"] > 0.5 else f"{r['width']:.2f}"
        cells.append(w)
        wk = "$\\infty$" if r["inf"] > 0.5 else f"{r['winkler']:.2f}"
        if r["inf"] <= 0.5 and np.isclose(r["winkler"], best["winkler"]):
            wk = "\\textbf{" + wk + "}"
        cells.append(wk)
        cells.append(f"{r['rmse']:.2f}")
        lines.append(" & ".join(cells) + BS)
    return "\n".join(lines)


HEADER = ("Method & Cov. & Worst & P10 & Rel. & Width & IS & RMSE" + BS)


def write_table(path, t):
    body = [r"\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{lrrrrrrr}", r"\toprule", HEADER,
            r" & (\%) & (\%) & (\%) & (\%) & (pp) & (pp) & (pp)" + BS, r"\midrule", fmt_table(t), r"\bottomrule",
            r"\end{tabular}"]
    open(path, "w").write("\n".join(body) + "\n")


def friedman(pdm, keys, stage, L, metric="winkler", alpha=0.1):
    sub = pdm[(pdm.stage == stage) & (pdm.L == L) & (pdm.alpha == alpha) & pdm.method.isin(keys)]
    piv = sub.pivot(index="fold", columns="method", values=metric)[keys].dropna()
    piv = piv.replace([np.inf], np.nan).dropna()
    stat, p = friedmanchisquare(*[piv[k].values for k in keys])
    ranks = piv.rank(axis=1).mean()
    return stat, p, ranks, piv


def main(protocol="lodo"):
    df = load(protocol)
    pdm = per_domain(df)
    t0 = summarise(pdm, DAY0, "day0", 0)
    t1 = summarise(pdm, INS, "inservice", LMAIN)
    write_table(os.path.join(PAP, f"tab_day0_{protocol}.tex"), t0)
    write_table(os.path.join(PAP, f"tab_inservice_{protocol}.tex"), t1)
    pd.set_option("display.width", 220)
    print(t0.drop(columns="name").round(3).to_string())
    print(t1.drop(columns="name").round(3).to_string())
    return pdm, t0, t1


if __name__ == "__main__":
    main()
