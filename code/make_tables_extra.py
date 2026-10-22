"""Additional LaTeX tables for the journal version (written to paper/)."""
import os

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")
BB = "hdse"
BS = "\\\\"
ORDER = ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW", "XJTU-Sim", "TJU-NCA", "TJU-NCM",
         "TJU-NCM+NCA", "HUST", "MIT-b1", "MIT-b2", "MIT-b3"]


def pdm(protocol="lodo"):
    df = pd.read_csv(os.path.join(RES, f"metrics_{protocol}.csv"))
    return df.groupby(["stage", "L", "method", "alpha", "fold"], as_index=False).mean(numeric_only=True)


def sel(p, method, stage, L, alpha=0.1):
    return p[(p.method == method) & (p.stage == stage) & (p.L == L) & (p.alpha == alpha)].set_index("fold")


def write(name, lines):
    open(os.path.join(PAP, name), "w").write("\n".join(lines) + "\n")


def per_domain(p):
    """Coverage / IS per domain for main methods (day 0 and in service)."""
    cols = [("split", "day0", 0, "Split"), ("cqr", "day0", 0, "CQR"), ("gp", "day0", 0, "GP"),
            (f"hxcp@{BB}", "day0", 0, "HX-CP"), ("aci_split", "inservice", 100, "ACI"),
            ("persist_online", "inservice", 100, "Pers."), (f"hxcp+@{BB}", "inservice", 100, "HX-CP+")]
    head = " & ".join(c[3] for c in cols)
    lines = [r"\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule",
             r" & \multicolumn{8}{c}{Day zero} & \multicolumn{6}{c}{In service, $L=100$}" + BS,
             r"\cmidrule(lr){2-9}\cmidrule(lr){10-15}",
             "Domain & " + " & ".join(r"\multicolumn{2}{c}{" + c[3] + "}" for c in cols) + BS,
             " & " + " & ".join(["Cov", "IS"] * len(cols)) + BS, r"\midrule"]
    data = {c[0]: sel(p, c[0], c[1], c[2]) for c in cols}
    for d in ORDER:
        cells = []
        for c in cols:
            r = data[c[0]].loc[d]
            cells += [f"{100 * r['cov']:.0f}", f"{r['winkler']:.1f}"]
        lines.append(d + " & " + " & ".join(cells) + BS)
    lines.append(r"\midrule")
    cells = []
    for c in cols:
        cells += [f"{100 * data[c[0]]['cov'].mean():.1f}", f"{data[c[0]]['winkler'].mean():.1f}"]
    lines += ["Mean & " + " & ".join(cells) + BS, r"\bottomrule", r"\end{tabular}"]
    write("tab_perdomain.tex", lines)


def alpha_L(p):
    """Coverage / width / IS of the main methods for every alpha and L."""
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{$\alpha=0.05$} & \multicolumn{3}{c}{$\alpha=0.10$} & \multicolumn{3}{c}{$\alpha=0.20$}" + BS,
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
             "Setting & Method & Cov & W & IS & Cov & W & IS & Cov & W & IS" + BS, r"\midrule"]
    rows = [("Day 0", 0, "day0", [("split", "Split CP"), ("cqr", "CQR"), ("gp", "GP"), (f"hxcp@{BB}", "HX-CP")])]
    for L in (25, 50, 100, 200):
        rows.append((f"$L={L}$", L, "inservice", [("aci_split", "ACI"), ("persist_online", "Persist.+online"),
                                                   (f"hxcp+@{BB}", "HX-CP+")]))
    for lab, L, st, meths in rows:
        for j, (k, nm) in enumerate(meths):
            cells = [lab if j == 0 else "", nm]
            for a in (0.05, 0.1, 0.2):
                g = sel(p, k, st, L, a)
                cells += [f"{100 * g['cov'].mean():.1f}", f"{g['width'].mean():.2f}", f"{g['winkler'].mean():.2f}"]
            lines.append(" & ".join(cells) + BS)
        lines.append(r"\midrule" if lab != "$L=200$" else r"\bottomrule")
    lines.append(r"\end{tabular}")
    write("tab_alphaL.tex", lines)


def stats(p):
    """Wilcoxon p-values (IS) of the proposed method against each baseline, per setting."""
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrr}", r"\toprule",
             "Setting & Comparison & $\\Delta$IS (\\%) & wins/13 & $p$" + BS, r"\midrule"]
    d0 = sel(p, f"hxcp@{BB}", "day0", 0)["winkler"]
    for k, nm in [("split", "Split CP"), ("cqr", "CQR"), ("qr", "LGB-QR"), ("de", "Deep ens."), ("mcd", "MC dropout"), ("gp", "GP")]:
        b = sel(p, k, "day0", 0)["winkler"].reindex(d0.index)
        lines.append(f"Day 0 & HX-CP vs {nm} & {100 * (1 - d0.mean() / b.mean()):.1f} & {(d0 < b).sum()} & {wilcoxon(d0, b).pvalue:.3f}" + BS)
    keys = [f"hxcp@{BB}", "split", "cqr", "qr", "de", "mcd", "gp"]
    piv = pd.concat([sel(p, k, "day0", 0)["winkler"].rename(k) for k in keys], axis=1)
    fr = friedmanchisquare(*[piv[k] for k in keys])
    lines.append(f"Day 0 & Friedman (7 methods) & & & {fr.pvalue:.3f}" + BS)
    lines.append(r"\midrule")
    for L in (25, 50, 100, 200):
        o = sel(p, f"hxcp+@{BB}", "inservice", L)["winkler"]
        for k, nm in [("persist_online", "Persist.+online"), ("aci_split", "ACI")]:
            b = sel(p, k, "inservice", L)["winkler"].reindex(o.index)
            lines.append(f"$L={L}$ & HX-CP+ vs {nm} & {100 * (1 - o.mean() / b.mean()):.1f} & {(o < b).sum()} & {wilcoxon(o, b).pvalue:.3f}" + BS)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_stats.tex", lines)


def backbone(p):
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llrrrr}", r"\toprule",
             "Stage & Backbone & Cov & Worst & W & IS" + BS, r"\midrule"]
    for st, L, k, lab in [("day0", 0, "hxcp", "Day 0 (HX-CP)"), ("inservice", 100, "hxcp+", "In service, $L=100$")]:
        for bb, nm in [("dse", "GBDT-only (DSE)"), ("hdse", "Heterogeneous (HDSE)")]:
            g = sel(p, f"{k}@{bb}", st, L)
            lines.append(f"{lab if bb == 'dse' else ''} & {nm} & {100 * g['cov'].mean():.1f} & {100 * g['cov'].min():.1f} & {g['width'].mean():.2f} & {g['winkler'].mean():.2f}" + BS)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_backbone.tex", lines)


def lodso():
    p = pdm("lodso")
    meths = [("split", "day0", 0, "Split CP"), ("cqr", "day0", 0, "CQR"), ("gp", "day0", 0, "GP"),
             ("mcd", "day0", 0, "MC dropout"), (f"hxcp@dse", "day0", 0, "HX-CP (DSE)"), (f"hxcp@{BB}", "day0", 0, "HX-CP (HDSE)"),
             ("aci_split", "inservice", 100, "ACI"), ("persist_online", "inservice", 100, "Persist.+online"),
             ("hxcp+@dse", "inservice", 100, "HX-CP+ (DSE)"), (f"hxcp+@{BB}", "inservice", 100, "HX-CP+ (HDSE)")]
    folds = ["XJTU", "TJU", "HUST", "MIT"]
    lines = [r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{l" + "rr" * 5 + "}", r"\toprule",
             "Method & " + " & ".join(r"\multicolumn{2}{c}{" + f + "}" for f in folds + ["Mean"]) + BS,
             " & " + " & ".join(["Cov", "IS"] * 5) + BS, r"\midrule"]
    for k, st, L, nm in meths:
        g = sel(p, k, st, L)
        cells = []
        for f in folds:
            cells += [f"{100 * g.loc[f, 'cov']:.0f}", f"{g.loc[f, 'winkler']:.1f}"]
        cells += [f"{100 * g['cov'].mean():.1f}", f"{g['winkler'].mean():.1f}"]
        lines.append(nm + " & " + " & ".join(cells) + BS)
        if k == f"hxcp@{BB}":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_lodso.tex", lines)


def leakage():
    r = pd.read_csv(os.path.join(RES, "leakage.csv"))
    g = r.groupby(["variant", "domain"])[["mape", "rmse"]].mean().groupby("variant").agg(["mean", "std"])
    order = [("leaky (Wang 2024 preprocessing)", "Whole-life min--max + $3\\sigma$ filter \\cite{wang2024pinn}", "no"),
             ("causal, absolute SOH", "Causal features, absolute SOH", "yes"),
             ("causal, commissioning-referenced", "Causal features, $\\Delta$SOH (ours)", "yes")]
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcrr}", r"\toprule",
             "Preprocessing & Deployable & MAPE (\\%) & RMSE (pp)" + BS, r"\midrule"]
    for k, nm, dep in order:
        lines.append(f"{nm} & {dep} & {g.loc[k, ('mape', 'mean')]:.2f} & {g.loc[k, ('rmse', 'mean')]:.2f}" + BS)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_leakage.tex", lines)


def hyper_sel():
    df = pd.read_csv(os.path.join(RES, "metrics_lodo.csv"))
    i = df[(df.alpha == 0.1) & (df.method == f"hxcp+@{BB}")]
    d = df[(df.alpha == 0.1) & (df.method == f"hxcp@{BB}")]
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
             "$L$ & mean $\\kappa$ & mode $\\kappa$ & mean $\\gamma$ & share $\\gamma{=}0$" + BS, r"\midrule"]
    for L in (25, 50, 100, 200):
        g = i[i.L == L]
        lines.append(f"{L} & {g.kappa.mean():.2f} & {g.kappa.mode().iloc[0]:.2f} & {g.gamma.mean():.2f} & {(g.gamma == 0).mean():.2f}" + BS)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_hypersel.tex", lines)
    return d.beta.value_counts().to_dict()


def life_stage():
    ls = pd.read_csv(os.path.join(RES, "life_stage.csv")).groupby(["method", "stage", "fold"]).mean(numeric_only=True)
    ls = ls.groupby(["method", "stage"])[["cov", "width"]].mean()
    meths = [("split", "Split CP (day 0)"), ("cqr", "CQR (day 0)"), ("gp", "GP (day 0)"), ("hxcp", "HX-CP (day 0)"),
             ("aci", "ACI ($L=100$)"), ("persist_online", "Persist.+online ($L=100$)"), ("hxcp+", "HX-CP+ ($L=100$)")]
    lines = [r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{early} & \multicolumn{2}{c}{mid} & \multicolumn{2}{c}{late}" + BS,
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             "Method & Cov & W & Cov & W & Cov & W" + BS, r"\midrule"]
    for k, nm in meths:
        cells = []
        for s in ("early", "mid", "late"):
            cells += [f"{100 * ls.loc[(k, s), 'cov']:.1f}", f"{ls.loc[(k, s), 'width']:.2f}"]
        lines.append(nm + " & " + " & ".join(cells) + BS)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_lifestage.tex", lines)


if __name__ == "__main__":
    p = pdm()
    per_domain(p)
    alpha_L(p)
    stats(p)
    backbone(p)
    lodso()
    leakage()
    print("beta selection:", hyper_sel())
    life_stage()
    print("written")
