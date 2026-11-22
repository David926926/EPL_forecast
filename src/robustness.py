"""Step 7: rolling-backtest robustness — per-season results, round-block bootstrap, calibration.

Uses the stored predictions of Steps 5-6 (no refitting). Key models per type:
  frequency (in-type), market, gd38-only logit (per-type), logit F1 = player ratings only (per-type),
  logit T = team performance only (per-type, global, and the 50/50 blend of the two).
Outputs
  - per-season log loss of the key models, and 2015/16 (the proposal's original test) on its own;
  - overall results on all 1,520 test matches (each type's predictions concatenated);
  - bootstrap CIs for key contrasts: match-level vs round-block (resample whole matchweeks
    (season, stage) to respect within-round correlation);
  - calibration: reliability curves per type x outcome, calibration-in-the-large, ECE, and the
    calibration slope (one-vs-rest logistic regression of the outcome on logit(p); 1 = ideal,
    < 1 = over-confident, > 1 = under-confident).
Writes report/tables/robustness_*.csv and figures/robustness_*.png. Run after src.training_regimes: python -m src.robustness
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.clustering import SEGMENTS, TABLES, wilson
from src.config import CLASSES, FEATURES_PATH, MARKET_PATH
from src.evaluate import score_all
from src.cluster_figures import AQUA, BLUE, INK, INK2, MUTED, ORANGE, SEG_NAME, SURFACE, YELLOW, save
from src.training_regimes import PRED_PATH as STEP6_PRED

KEY = {  # label -> (regime, model, layer)
    "frequency": ("-", "frequency", "-"),
    "gd38-only (per-type)": ("segment", "strength_baseline", "-"),
    "logit F1 ratings (per-type)": ("segment", "logit", "F1"),
    "logit T (per-type)": ("segment", "logit", "T"),
    "logit T (global)": ("global", "logit", "T"),
    "logit T (blend)": ("blend", "logit", "T"),
    "market": ("-", "market", "-"),
}
KEY_STYLE = {"frequency": (MUTED, (0, (1, 2)), "s"), "gd38-only (per-type)": ("#8a8984", (0, (5, 3)), "^"),
             "logit F1 ratings (per-type)": (YELLOW, (0, (3, 1.5)), "v"),
             "logit T (per-type)": (ORANGE, "-", "o"), "logit T (global)": (BLUE, "-", "o"),
             "logit T (blend)": (AQUA, "-", "o"), "market": (INK, "-", "D")}
CAL_MODELS = ["logit T (per-type)", "logit T (global)", "market"]
N_BOOT = 2000
P_COLS = ["p_H", "p_D", "p_A"]


def load():
    p = pd.read_parquet(STEP6_PRED)
    stage = pd.read_parquet(FEATURES_PATH)[["match_api_id", "stage"]]
    p = p.merge(stage, on="match_api_id")
    p["block"] = p.season + "-" + p.stage.astype(str)
    frames = []
    for label, (reg, model, layer) in KEY.items():
        frames.append(p[(p.regime == reg) & (p.model == model) & (p.layer == layer)].assign(key=label))
    return pd.concat(frames, ignore_index=True)


def per_match_ll(df):
    P = df[P_COLS].to_numpy()
    return -np.log(np.clip(P[np.arange(len(P)), df.y_int.to_numpy()], 1e-15, 1))


# ------------------------------------------------------------------ per season / overall
def season_table(k):
    rows = []
    for (seg, key, season), g in k.groupby(["segment", "key", "season"]):
        rows.append({"segment": seg, "key": key, "season": season, **score_all(g[P_COLS].to_numpy(), g.y_int.to_numpy())})
    return pd.DataFrame(rows)


def overall_table(k):
    """All 1,520 test matches: each type's predictions concatenated. 'frequency' here is the in-type
    frequency; the global frequency baseline (all-match shares) is added for reference."""
    rows = []
    for key, g in k.groupby("key"):
        for season, gg in [("pooled", g)] + list(g.groupby("season")):
            rows.append({"key": key, "season": season, **score_all(gg[P_COLS].to_numpy(), gg.y_int.to_numpy())})
    mk = pd.read_parquet(MARKET_PATH)
    test = mk[mk.match_api_id.isin(k.match_api_id)]
    for season, gg in [("pooled", test)] + list(test.groupby("season")):
        rows.append({"key": "frequency (all matches)", "season": season,
                     **score_all(gg[["p_freq_H", "p_freq_D", "p_freq_A"]].to_numpy(), gg.y_int.to_numpy())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ bootstrap
CONTRASTS = [("logit T (per-type)", "gd38-only (per-type)"), ("logit T (per-type)", "market"),
             ("logit T (global)", "logit T (per-type)"), ("logit T (global)", "market"),
             ("logit T (blend)", "logit T (global)"), ("logit T (blend)", "market"),
             ("logit F1 ratings (per-type)", "gd38-only (per-type)"),
             ("gd38-only (per-type)", "frequency"), ("market", "frequency")]


def boot_ci(d, blocks, rng):
    """Match-level and round-block bootstrap CIs for the mean of per-match differences d."""
    n = len(d)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    m_boot = d[idx].mean(axis=1)
    codes, uniq = pd.factorize(blocks)
    sums = np.bincount(codes, weights=d)
    cnts = np.bincount(codes)
    bidx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    b_boot = sums[bidx].sum(axis=1) / cnts[bidx].sum(axis=1)
    return (np.percentile(m_boot, [2.5, 97.5]), np.percentile(b_boot, [2.5, 97.5]), len(uniq))


def bootstrap(k):
    rng = np.random.default_rng(0)
    rows = []
    groups = [(s, k[k.segment == s]) for s in SEGMENTS] + [("all", k)]
    for seg, g in groups:
        keyed = {key: gg.sort_values(["segment", "fold", "match_api_id"]) for key, gg in g.groupby("key")}
        for a, b in CONTRASTS:
            d = per_match_ll(keyed[a]) - per_match_ll(keyed[b])
            (mlo, mhi), (blo, bhi), nb = boot_ci(d, keyed[a].block.to_numpy(), rng)
            rows.append({"segment": seg, "contrast": f"{a} − {b}", "diff": d.mean(), "match_ci_lo": mlo, "match_ci_hi": mhi,
                         "block_ci_lo": blo, "block_ci_hi": bhi, "n_matches": len(d), "n_rounds": nb,
                         "width_ratio": (bhi - blo) / (mhi - mlo)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ calibration
def calibration(k, n_bins=5):
    curves, stats = [], []
    for (seg, key), g in k[k.key.isin(CAL_MODELS)].groupby(["segment", "key"]):
        for j, c in enumerate(CLASSES):
            p = g[P_COLS[j]].to_numpy()
            y = (g.y_int.to_numpy() == j).astype(int)
            bins = pd.qcut(p, n_bins, labels=False, duplicates="drop")
            ece = 0.0
            for b in np.unique(bins):
                m = bins == b
                lo, hi = wilson(y[m].sum(), m.sum())
                curves.append({"segment": seg, "key": key, "outcome": c, "bin": b, "n": m.sum(),
                               "mean_pred": p[m].mean(), "obs": y[m].mean(), "obs_lo": lo, "obs_hi": hi})
                ece += m.mean() * abs(p[m].mean() - y[m].mean())
            z = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
            stats.append({"segment": seg, "key": key, "outcome": c, "mean_pred": p.mean(), "obs_rate": y.mean(),
                          "ece": ece, "cal_slope": lr.coef_[0, 0], "cal_intercept": lr.intercept_[0],
                          "pred_sd": p.std()})
    return pd.DataFrame(curves), pd.DataFrame(stats)


# ------------------------------------------------------------------ figures
def fig_seasons(st):
    seasons = sorted(st.season.unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, seg in zip(axes, SEGMENTS):
        x = np.arange(len(seasons))
        for key in KEY:
            v = st[(st.segment == seg) & (st.key == key)].set_index("season").loc[seasons].log_loss
            col, ls, mk = KEY_STYLE[key]
            ax.plot(x, v, marker=mk, color=col, ls=ls, ms=6, mec=SURFACE, mew=1.2, lw=1.8, label=key)
        ax.set_xticks(x, [s.replace("20", "", 1).replace("/20", "/") for s in seasons])
        ax.set_title(SEG_NAME[seg])
    axes[0].set_ylabel("Log loss (lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=7, loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("Rolling backtest: log loss per test season and type (each panel has its own y-axis)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "robustness_season_logloss.png")


def fig_calibration(cv):
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    for r, seg in enumerate(SEGMENTS):
        for c, out in enumerate(CLASSES):
            ax = axes[r, c]
            q = cv[(cv.segment == seg) & (cv.outcome == out)]
            lo = min(q.mean_pred.min(), q.obs_lo.min()) - 0.02
            hi = max(q.mean_pred.max(), q.obs_hi.max()) + 0.02
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls=(0, (4, 3)))
            for i, key in enumerate(CAL_MODELS):
                g = q[q.key == key].sort_values("mean_pred")
                col = KEY_STYLE[key][0]
                off = (i - 1) * 0.003
                ax.vlines(g.mean_pred + off, g.obs_lo, g.obs_hi, color=col, lw=1, alpha=0.5)
                ax.plot(g.mean_pred + off, g.obs, "o-", color=col, ms=5, mec=SURFACE, mew=1, lw=1.6, label=key)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
            ax.set_title(f"{SEG_NAME[seg]} · P({out})", fontsize=9.5)
            if r == 2:
                ax.set_xlabel("mean predicted probability (quintile bins)")
            if c == 0:
                ax.set_ylabel("observed frequency (95% CI)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Calibration on pooled test seasons: points on the dashed diagonal = honest probabilities",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "robustness_calibration.png")


def fig_bootstrap(bs):
    b = bs.copy()
    b["label"] = b.segment.map({**SEG_NAME, "all": "All types"}) + ": " + b.contrast
    keep = b.contrast.isin(["logit T (per-type) − gd38-only (per-type)", "logit T (per-type) − market",
                            "logit T (global) − logit T (per-type)", "logit T (global) − market"])
    b = b[keep]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    y = np.arange(len(b))
    ax.axvline(0, color=INK2, lw=1)
    ax.hlines(y - 0.15, b.match_ci_lo, b.match_ci_hi, color=MUTED, lw=2, label="match-level bootstrap")
    ax.hlines(y + 0.15, b.block_ci_lo, b.block_ci_hi, color=BLUE, lw=2, label="round-block bootstrap")
    ax.plot(b["diff"], y, "o", color=INK, ms=5, mec=SURFACE, label="point estimate")
    ax.set_yticks(y, b.label, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ log loss (negative = first model better), 95% CI")
    ax.set_title("Match-level vs round-block bootstrap CIs for the key contrasts")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "robustness_bootstrap_ci.png")


def main():
    k = load()
    st = season_table(k)
    st.to_csv(TABLES / "robustness_season_scores.csv", index=False)
    ov = overall_table(k)
    ov.to_csv(TABLES / "robustness_overall.csv", index=False)
    bs = bootstrap(k)
    bs.to_csv(TABLES / "robustness_bootstrap.csv", index=False)
    cv, cs = calibration(k)
    cv.to_csv(TABLES / "robustness_calibration_curves.csv", index=False)
    cs.to_csv(TABLES / "robustness_calibration_stats.csv", index=False)
    fig_seasons(st)
    fig_calibration(cv)
    fig_bootstrap(bs)

    pd.set_option("display.width", 220)
    print("per season log loss:\n", st.pivot_table(index=["segment", "key"], columns="season", values="log_loss").round(4).to_string())
    print("\noverall (all 1,520 test matches):\n", ov.pivot_table(index="key", columns="season", values="log_loss").round(4).to_string())
    print("\nbootstrap:\n", bs.round(4).to_string(index=False))
    print("\ncalibration:\n", cs.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
