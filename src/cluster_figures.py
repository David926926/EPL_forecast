"""Step 4 figures (static PNGs for the report). Run after src.clustering: python -m src.cluster_figures"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.clustering import SEGMENTS, SEGMENTS_PATH, TABLES, load, wilson
from src.config import FIGURES
from src.features import GAP_COLS

# Reference palette (dataviz skill): categorical slots in fixed order, neutral ink for text.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
OUTCOME = {"H": BLUE, "D": ORANGE, "A": AQUA}
SEG_COLOR = {"even": BLUE, "middle": ORANGE, "lopsided": AQUA}
SEG_NAME = {"even": "Even", "middle": "Middle", "lopsided": "Lopsided"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"abs_gd38_diff": "|38-match goal diff|", "abs_xi_rating_diff": "|XI rating diff|",
          "abs_form_pts_diff": "|form PPG diff|"}
PCT = matplotlib.ticker.PercentFormatter(1, decimals=0)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "font.size": 9, "legend.frameon": False,
    "lines.linewidth": 2, "axes.axisbelow": True,
})


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stacked_outcomes(ax, rows, names):
    left = np.zeros(len(rows))
    for o, lab in [("H", "Home win"), ("D", "Draw"), ("A", "Away win")]:
        ax.barh(names, rows[o], left=left, color=OUTCOME[o], edgecolor=SURFACE, linewidth=2, height=0.62, label=lab)
        for i, (l, v) in enumerate(zip(left, rows[o])):
            ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += rows[o].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PCT)
    ax.grid(False)


def fig_draw_vs_gap(d):
    q = pd.qcut(d.abs_gd38_diff, 10)
    g = d.groupby(q, observed=True).agg(n=("y", "size"), k=("y", lambda s: (s == "D").sum()),
                                        mkt=("p_mkt_D", "mean"), mid=("abs_gd38_diff", "median"))
    lo, hi = wilson(g.k.values, g.n.values)
    x = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline((d.y == "D").mean(), color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(len(g) - 0.5, (d.y == "D").mean() + 0.004, f"all matches {(d.y == 'D').mean():.1%}",
            ha="right", va="bottom", color=INK2, fontsize=8)
    ax.vlines(x, lo, hi, color=BLUE, lw=1.2, alpha=0.6)
    ax.plot(x, g.k / g.n, "o-", color=BLUE, ms=6, mec=SURFACE, mew=1.5, label="Actual draw rate (95% CI)")
    ax.plot(x, g.mkt, "s--", color=ORANGE, ms=6, mec=SURFACE, mew=1.5, lw=1.6, label="Market implied P(draw)")
    ax.set_xticks(x, [f"{m:.2f}" for m in g.mid])
    ax.set_xlabel("|38-match goal-difference gap| decile (median, goals per match), closest → widest")
    ax.set_ylabel("Draw share")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Draws get rarer as the long-run strength gap widens (2009/10–2015/16)")
    ax.legend(loc="lower left")
    save(fig, "cluster_draw_vs_strength_gap.png")


def fig_gap_features(d):
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.25]})
    for ax, c in zip(axes[:3], GAP_COLS):
        if c == "abs_form_pts_diff":  # discrete: multiples of 0.2 points per game
            vc = d[c].round(1).value_counts().sort_index()
            ax.bar(vc.index, vc.values, width=0.17, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        else:
            ax.hist(d[c], bins=30, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.set_title(LABELS[c], fontsize=9)
        ax.set_yticks([])
        ax.grid(False)
    corr = d[GAP_COLS + ["mkt_gap"]].corr("spearman").values
    names = [LABELS[c].strip("|").replace(" diff", "") for c in GAP_COLS] + ["market gap"]
    ax = axes[3]
    ax.imshow(corr, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if corr[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.grid(False)
    ax.set_title("Spearman correlation", fontsize=9)
    fig.suptitle("Clustering inputs: distributions and rank correlations (2009/10–2015/16)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "cluster_gap_features.png")


def fig_model_selection(grid):
    g = grid[grid.fold == 4]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    for cov, col in zip(["full", "diag", "tied", "spherical"], [BLUE, ORANGE, AQUA, YELLOW]):
        s = g[(g.method == "gmm") & (g["cov"] == cov)]
        a1.plot(s.k, s.bic, "o-", color=col, ms=6, mec=SURFACE, mew=1.5, label=cov)
    best = g[g.method == "gmm"].sort_values("bic").iloc[0]
    a1.annotate(f"min BIC: k={int(best.k)}, {best['cov']}", (best.k, best.bic), xytext=(-110, 30),
                textcoords="offset points", fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED})
    a1.set(xlabel="k", ylabel="BIC (lower is better)", title="GMM: BIC by k and covariance type")
    a1.legend(title="covariance", fontsize=8, title_fontsize=8)
    gm = g[g.method == "gmm"].groupby("k").silhouette.max()
    km = g[g.method == "kmeans"].set_index("k").silhouette
    a2.axhspan(0, 0.25, color=GRID, alpha=0.6, lw=0)
    a2.axhspan(0.25, 0.5, color=GRID, alpha=0.25, lw=0)
    a2.text(6.1, 0.49, "0.25–0.5: weak structure", ha="right", va="top", fontsize=8, color=INK2)
    a2.text(6.1, 0.24, "< 0.25: no substantial structure", ha="right", va="top", fontsize=8, color=INK2)
    a2.plot(gm.index, gm.values, "o-", color=BLUE, ms=6, mec=SURFACE, mew=1.5, label="GMM (best cov per k)")
    a2.plot(km.index, km.values, "s-", color=ORANGE, ms=6, mec=SURFACE, mew=1.5, label="K-means")
    a2.plot(best.k, best.silhouette, "D", color=INK, ms=7, mec=SURFACE, mew=1.5,
            label=f"chosen GMM (k={int(best.k)}, {best['cov']}): {best.silhouette:.2f}")
    a2.set(xlabel="k", ylabel="Silhouette", title="Silhouette: weak structure at best", ylim=(0, 0.6))
    a2.legend(fontsize=8, loc="lower left")
    for a in (a1, a2):
        a.set_xticks(range(2, 7))
    fig.suptitle("Model selection on fold-4 training seasons (2009/10–2013/14)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "cluster_model_selection.png")


def fig_cluster_outcomes(prof):
    prof = prof.assign(o=prof.segment.map({s: i for i, s in enumerate(SEGMENTS)})).sort_values(["o", "abs_gd38_diff"])
    names = [f"{SEG_NAME[s].upper()} · cluster {c}\n|gd38| {r:.2f}, n={n}"
             for c, s, r, n in zip(prof.index, prof.segment, prof.abs_gd38_diff, prof.n)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    stacked_outcomes(ax, prof, names)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    ax.set_title("Fold-4 GMM clusters and the type each is merged into (all seasons 2009/10–2015/16)", pad=24)
    save(fig, "cluster_outcomes.png")


def fig_segment_overview(outcomes, pred):
    o = outcomes[(outcomes["sample"].str.startswith("pooled")) & (outcomes.definition == "GMM")].set_index("segment").loc[SEGMENTS]
    p = pred.set_index("segment").loc[SEGMENTS]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    stacked_outcomes(a1, o, [f"{SEG_NAME[s]}\n(n={n})" for s, n in zip(SEGMENTS, o.n)])
    a1.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    a1.set_title(f"Outcomes by match type (χ² p = {o.p_chi2.iloc[0]:.1g})", pad=24)
    y = np.arange(3)
    a2.barh(y, p.market_skill, color=BLUE, height=0.55, edgecolor=SURFACE)
    for yi, v, f, m in zip(y, p.market_skill, p.ll_frequency, p.ll_market):
        a2.text(v + 0.004, yi, f"{v:.1%}  (log loss {f:.3f} → {m:.3f})", va="center", fontsize=8, color=INK2)
    a2.set_yticks(y, [SEG_NAME[s] for s in SEGMENTS])
    a2.invert_yaxis()
    a2.set_xlim(0, 0.32)
    a2.xaxis.set_major_formatter(PCT)
    a2.set_xlabel("Market skill vs in-type frequency baseline")
    a2.set_title("How predictable each type is", pad=24)
    fig.suptitle("Pooled test seasons 2012/13–2015/16", x=0.01, ha="left", fontsize=9, color=INK2, y=1.0)
    fig.tight_layout()
    save(fig, "cluster_segment_overview.png")


def fig_definitions(outcomes):
    o = outcomes[outcomes["sample"].str.startswith("pooled")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(3)
    for off, name, col in [(-0.2, "GMM", BLUE), (0.0, "|gd38| quantile rule", ORANGE), (0.2, "K-means", AQUA)]:
        c = o[o.definition == name].set_index("segment").reindex(SEGMENTS)
        ok = c.n.notna().to_numpy()
        ax.hlines((y + off)[ok], c.D_ci_lo[ok], c.D_ci_hi[ok], color=col, lw=1.4)
        ax.plot(c.D[ok], (y + off)[ok], "o", color=col, ms=7, mec=SURFACE, mew=1.5,
                label=f"{name} (χ² p = {c.p_chi2.dropna().iloc[0]:.1g})")
    ax.set_yticks(y, [SEG_NAME[s] for s in SEGMENTS])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Draw rate in the type, pooled test seasons (95% Wilson CI)")
    ax.set_title("GMM types vs a plain |gd38| quantile rule (K-means k=2 has no middle type)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "cluster_definition_comparison.png")


def fig_test_seasons(per):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    seasons = sorted(per.season.unique())
    x = np.arange(len(seasons))
    for s in SEGMENTS:
        v = per[per.segment == s].set_index("season").loc[seasons]
        ax.plot(x, v.D, "o-", color=SEG_COLOR[s], ms=7, mec=SURFACE, mew=1.5, label=SEG_NAME[s])
    ax.set_xticks(x, [f"{s}\n(fold {i + 1})" for i, s in enumerate(seasons)], fontsize=8)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("Draw rate")
    ax.set_title("Out-of-sample draw rate by type and test season", pad=22)
    ax.legend(fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "cluster_test_seasons.png")


def fig_scatter(lab4, d):
    x = lab4.merge(d[["match_api_id", "abs_gd38_diff", "abs_xi_rating_diff"]], on="match_api_id")
    fig, ax = plt.subplots(figsize=(5.8, 4))
    for s in ["middle", "lopsided", "even"]:
        g = x[x.segment == s]
        ax.scatter(g.abs_gd38_diff, g.abs_xi_rating_diff, s=9, color=SEG_COLOR[s], lw=0.3, edgecolor=SURFACE,
                   alpha=0.85, label=SEG_NAME[s])
    ax.set(xlabel="|38-match goal diff| (goals per match)", ylabel="|XI rating diff|",
           title="Match types in two of the three gap dimensions (fold-4 model)")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(SEG_NAME[s]) for s in SEGMENTS]
    ax.legend([h[i] for i in order], [l[i] for i in order], fontsize=8, markerscale=2)
    save(fig, "cluster_segment_scatter.png")


def main():
    d = load()
    grid = pd.read_csv(TABLES / "cluster_model_selection.csv")
    prof = pd.read_csv(TABLES / "cluster_outcomes_fold4.csv", index_col=0)
    outcomes = pd.read_csv(TABLES / "cluster_segment_outcomes.csv")
    pred = pd.read_csv(TABLES / "cluster_predictability.csv")
    per = pd.read_csv(TABLES / "cluster_test_season_outcomes.csv")
    lab = pd.read_parquet(SEGMENTS_PATH)
    fig_draw_vs_gap(d)
    fig_gap_features(d)
    fig_model_selection(grid)
    fig_cluster_outcomes(prof)
    fig_segment_overview(outcomes, pred)
    fig_definitions(outcomes)
    fig_test_seasons(per)
    fig_scatter(lab[lab.fold == 4], d)
    print("saved:", sorted(p.name for p in FIGURES.glob("cluster_*.png")))


if __name__ == "__main__":
    main()
