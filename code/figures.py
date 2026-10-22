"""Publication figures (IEEE column 3.5 in, page 7.16 in). Run after evaluate.py."""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_all

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "figs")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(FIG, exist_ok=True)

# validated reference categorical palette (dataviz skill), fixed order
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
DS_COL = {"XJTU": C["blue"], "TJU": C["orange"], "HUST": C["aqua"], "MIT": C["violet"]}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.0,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

ORDER = ["XJTU-2C", "XJTU-3C", "XJTU-R2.5", "XJTU-R3", "XJTU-RW", "XJTU-Sim", "TJU-NCA", "TJU-NCM",
         "TJU-NCM+NCA", "HUST", "MIT-b1", "MIT-b2", "MIT-b3"]


def fig_data():
    d = load_all()
    fig, axes = plt.subplots(2, 7, figsize=(7.16, 2.3))
    axes = axes.ravel()
    for ax, dom in zip(axes, ORDER):
        sub = d[d.domain == dom]
        col = DS_COL[sub.dataset.iloc[0]]
        for _, g in sub.groupby("cell"):
            ax.plot(g.cycle.values, g.soh.values * 100, color=col, lw=0.4, alpha=0.55)
        ax.set_title(f"{dom} ({sub.cell.nunique()})", pad=2)
        ax.set_ylim(63, 113)
        ax.tick_params(length=2, pad=1)
        if ax not in (axes[0], axes[7]):
            ax.set_yticklabels([])
    for ax in axes[len(ORDER):]:
        ax.axis("off")
    axes[0].set_ylabel("SOH (%)")
    axes[7].set_ylabel("SOH (%)")
    for ax in axes[7:13]:
        ax.set_xlabel("cycle")
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)
    # PCA of causal features coloured by dataset (feature-space shift)
    ax = axes[13]
    ax.axis("on")
    from sklearn.decomposition import PCA
    from data import FEATURE_SETS, add_causal_features
    c = add_causal_features(d)
    F = FEATURE_SETS["causal"]
    s = c.groupby("cell", group_keys=False)[c.columns.tolist()].apply(lambda g: g.iloc[::max(1, len(g) // 20)])
    X = (s[F] - s[F].mean()) / s[F].std().replace(0, 1)
    Z = PCA(2, random_state=0).fit_transform(X.fillna(0).values)
    for ds, col in DS_COL.items():
        m = (s.dataset == ds).values
        ax.scatter(Z[m, 0], Z[m, 1], s=1.5, color=col, alpha=0.5, lw=0, label=ds, rasterized=True)
    ax.set_title("feature PCA", pad=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2", labelpad=1)
    ax.set_xlim(np.percentile(Z[:, 0], [0.5, 99.5]))
    ax.set_ylim(np.percentile(Z[:, 1], [0.5, 99.5]))
    for ds, col in DS_COL.items():  # direct labels at cluster medians (text in ink, mark carries colour)
        m = (s.dataset == ds).values
        cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
        ax.annotate(ds, (cx, cy), fontsize=5.5, color=INK, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.6, alpha=0.9))
    fig.savefig(os.path.join(FIG, "fig_data.pdf"))
    fig.savefig(os.path.join(FIG, "fig_data.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------- result figures
BB = os.environ.get("MAIN_BACKBONE", "hdse")
NAMES = {"split": "Split CP", "wcp": "Weighted CP", "cqr": "CQR", "qr": "LGB-QR", "de": "Deep ens.",
         "mcd": "MC dropout", "gp": "GP", f"xd@{BB}": "XD-CP (abl.)", f"hxcp@{BB}": "HX-CP (ours)",
         "aci_split": "ACI", "persist_online": "Persistence+online", f"hxcp+@{BB}": "HX-CP+ (ours)"}


def _pdm():
    df = pd.read_csv(os.path.join(RES, "metrics_lodo.csv"))
    return df.groupby(["stage", "L", "method", "alpha", "fold"], as_index=False).mean(numeric_only=True)


def fig_coverage():
    """Per-domain day-0 coverage heat-map (diverging around the 90% target) + width."""
    p = _pdm()
    keys = ["split", "wcp", "cqr", "qr", "de", "mcd", "gp", f"xd@{BB}", f"hxcp@{BB}"]
    sub = p[(p.stage == "day0") & (p.alpha == 0.1) & p.method.isin(keys)]
    cov = sub.pivot(index="method", columns="fold", values="cov").loc[keys, ORDER] * 100
    inf = sub.pivot(index="method", columns="fold", values="inf_frac").loc[keys, ORDER]
    fig, ax = plt.subplots(figsize=(7.16, 2.15))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    vals = np.where(inf.values > 0.5, np.nan, cov.values)
    cmap.set_bad("#d9d8d4")
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=TwoSlopeNorm(vcenter=90, vmin=20, vmax=100), aspect="auto")
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            v = cov.values[i, j]
            txt = "∞" if inf.values[i, j] > 0.5 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=5.5,
                    color="white" if (v < 45) else INK)
    ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys])
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=30, ha="right")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("coverage (%), target 90", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.savefig(os.path.join(FIG, "fig_coverage.pdf"))
    fig.savefig(os.path.join(FIG, "fig_coverage.png"))
    plt.close(fig)


def fig_L():
    """In-service performance vs check-up interval L (alpha = 0.1)."""
    p = _pdm()
    keys = [("aci_split", C["orange"], "--"), ("persist_online", C["aqua"], "-."),
            (f"hxcp+@{BB}", C["blue"], "-")]
    sub = p[(p.stage == "inservice") & (p.alpha == 0.1)]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.65))
    for ax, (met, lab) in zip(axes, [("cov", "coverage (%)"), ("width", "mean width (pp)"), ("winkler", "interval score (pp)")]):
        for k, col, ls in keys:
            g = sub[sub.method == k].groupby("L")[met].mean()
            vals = g.values * (100 if met == "cov" else 1)
            ax.plot(g.index, vals, color=col, ls=ls, marker="o", ms=3, label=NAMES[k])
        if met == "cov":
            ax.axhline(90, color=INK2, lw=0.6, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks([25, 50, 100, 200], ["25", "50", "100", "200"])
        ax.set_xlabel("check-up interval $L$ (cycles)")
        ax.set_ylabel(lab)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIG, "fig_L.pdf"))
    fig.savefig(os.path.join(FIG, "fig_L.png"))
    plt.close(fig)


def fig_traj(doms=("TJU-NCM+NCA", "HUST", "XJTU-3C"), seed=0, L=100):
    """One representative (median-life) cell per hard domain: day-0 split CP vs HX-CP vs in-service HX-CP+."""
    import evaluate as E
    fig, axes = plt.subplots(1, len(doms), figsize=(7.16, 1.9))
    for ax, dom in zip(np.atleast_1d(axes), doms):
        tg, _, bbs = E.load_fold("lodo", dom, seed)
        mu, sd, inner = bbs[BB]
        d0, beta = E.day0_proposed(mu, sd, inner, 0.1)
        ins, _ = E.inservice_proposed(tg, mu, sd, inner, 0.1, L, beta)
        sp = E.day0_baselines(tg, 0.1)["split"]
        life = tg.groupby("cell").size().sort_values()
        c = life.index[len(life) // 2]
        m = (tg.cell == c).values
        x, ref, y = tg.cycle.values[m], tg.soh_ref.values[m] * 100, tg.soh.values[m] * 100
        cu = E.checkup_mask(tg, L)[m]
        lo, hi, _ = sp
        ax.fill_between(x, ref + np.asarray(lo)[m] * 100, ref + np.asarray(hi)[m] * 100, color=C["orange"], alpha=0.30, lw=0, label="Split CP, day 0")
        lo, hi, _ = ins["hxcp+"]
        ax.fill_between(x, ref + np.asarray(lo)[m] * 100, ref + np.asarray(hi)[m] * 100, color=C["blue"], alpha=0.35, lw=0, label=f"HX-CP+, $L$={L}")
        lo, hi, _ = d0["hxcp"]
        ax.plot(x, ref + np.asarray(lo)[m] * 100, color=C["violet"], lw=0.7, ls="--", label="HX-CP, day 0")
        ax.plot(x, ref + np.asarray(hi)[m] * 100, color=C["violet"], lw=0.7, ls="--")
        ax.plot(x, y, color=INK, lw=0.9, label="measured SOH")
        ax.plot(x[cu], y[cu], "o", ms=2.5, mfc="white", mec=INK, mew=0.6, label="check-up")
        ax.set_title(f"{dom}: cell {c.split('/')[1]}", pad=2)
        ax.set_xlabel("cycle")
    np.atleast_1d(axes)[0].set_ylabel("SOH (%)")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(pad=0.3, rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_traj.pdf"))
    fig.savefig(os.path.join(FIG, "fig_traj.png"))
    plt.close(fig)


def _save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def fig_extra():
    """(a) calibration across alpha, (b) coverage by life stage, (c) label efficiency, (d) online threshold."""
    p = _pdm()
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75))
    # (a) empirical vs nominal coverage
    ax = axes[0]
    series = [("split", "day0", 0, C["orange"], "--", "Split CP (d0)"), ("gp", "day0", 0, C["yellow"], ":", "GP (d0)"),
              (f"hxcp@{BB}", "day0", 0, C["violet"], "-.", "HX-CP (d0)"),
              ("aci_split", "inservice", 100, C["red"], "--", "ACI"),
              (f"hxcp+@{BB}", "inservice", 100, C["blue"], "-", "HX-CP+")]
    for k, st, L, col, ls, lab in series:
        g = p[(p.method == k) & (p.stage == st) & (p.L == L)].groupby("alpha")["cov"].mean()
        ax.plot(100 * (1 - g.index), 100 * g.values, color=col, ls=ls, marker="o", ms=2.5, label=lab)
    ax.plot([75, 100], [75, 100], color=INK2, lw=0.6)
    ax.set_xlabel("nominal coverage (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title("(a) calibration", pad=2)
    ax.set_ylim(50, 108)
    ax.legend(frameon=False, fontsize=4.6, loc="upper left", ncol=2, columnspacing=0.5, handlelength=1.2)
    # (b) life stage
    ax = axes[1]
    ls_ = pd.read_csv(os.path.join(RES, "life_stage.csv")).groupby(["method", "stage", "fold"]).mean(numeric_only=True)
    ls_ = ls_.groupby(["method", "stage"])["cov"].mean().unstack()[["early", "mid", "late"]]
    for k, col, lab in [("split", C["orange"], "Split CP"), ("hxcp", C["violet"], "HX-CP"),
                        ("persist_online", C["aqua"], "Persist.+onl."), ("hxcp+", C["blue"], "HX-CP+")]:
        ax.plot(range(3), 100 * ls_.loc[k].values, color=col, marker="o", ms=2.5, label=lab)
    ax.axhline(90, color=INK2, lw=0.6, ls=":")
    ax.set_xticks(range(3), ["early", "mid", "late"])
    ax.set_ylabel("coverage (%)")
    ax.set_title("(b) life stage", pad=2)
    ax.legend(frameon=False, fontsize=5, loc="lower left")
    # (c) label efficiency: interval score vs labelled cycles per domain
    ax = axes[2]
    lb = pd.read_csv(os.path.join(RES, "label_budget.csv")).groupby("setting").labels_per_domain.mean()
    xs, ys = [], []
    for L in (200, 100, 50, 25):
        xs.append(lb[f"L={L}"])
        ys.append(p[(p.method == f"hxcp+@{BB}") & (p.stage == "inservice") & (p.L == L) & (p.alpha == 0.1)].winkler.mean())
    ax.plot(xs, ys, color=C["blue"], marker="o", ms=3, label="HX-CP+ (check-ups)")
    for L, x, y in zip((200, 100, 50, 25), xs, ys):
        ax.annotate(f"L={L}", (x, y), fontsize=5, xytext=(2, 3), textcoords="offset points", color=INK2)
    xs2, ys2 = [], []
    for m in (1, 2, 3, 5):
        xs2.append(lb[f"recal m={m}"])
        ys2.append(p[(p.method == f"recal_m{m}") & (p.alpha == 0.1)].winkler.mean())
    ax.plot(xs2, ys2, color=C["orange"], marker="s", ms=3, ls="--", label="recalibration (lab cells)")
    ax.set_xscale("log")
    ax.set_xlabel("labelled cycles per domain")
    ax.set_ylabel("interval score (pp)")
    ax.set_title("(c) label efficiency", pad=2)
    ax.legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    # (d) online threshold trajectories
    ax = axes[3]
    qt = pd.read_pickle(os.path.join(RES, "q_traj.pkl"))
    for q, col in zip(qt, [C["aqua"], C["orange"], C["violet"], C["blue"]]):
        ax.plot(np.arange(len(q["q"])) / max(len(q["q"]) - 1, 1), q["q"] / q["q0"], color=col, lw=0.9,
                label=f"{q['fold']} (N={q['n']})")
    ax.axhline(1, color=INK2, lw=0.6, ls=":")
    ax.set_xlabel("fraction of fleet check-ups")
    ax.set_ylabel("$q/q_0$")
    ax.set_title("(d) online threshold", pad=2)
    ax.set_ylim(0.15, 2.0)
    ax.legend(frameon=False, fontsize=4.8, loc="upper center", ncol=2, columnspacing=0.6, handlelength=1.2)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    _save(fig, "fig_extra")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["data"]
    for w in which:
        globals()["fig_" + w]()
        print("saved", w)
