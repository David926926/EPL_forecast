"""Step 6: per-type vs global training, and two simple between-model ensembles (research question 3).

For every match type, the same 6 models x 4 feature sets (F1/F2/F3 + T; plus the gd38-only logit) are
ALWAYS scored on that type's test matches. Regimes:
  segment   trained on that type's train(+val) matches only          (Step 5 predictions, reused)
  global    trained on all matches, weight 1                         (fitted here)
  blend     blend ensemble: 0.5 x segment + 0.5 x global prediction of the same model and features.
            The weight is fixed in advance (no tuning, so no selection on test results).
Plus model_avg: the 6-model average, the equal-weight average of the 6 models' predictions, for every
regime x feature set (also fixed in advance).
Hyper-parameters are tuned on that type's val matches in every regime. Paired match-level bootstrap
(2,000 draws) for all contrasts.
Writes data/processed/regime_predictions.parquet, report/tables/regimes_*.csv, figures/regimes_*.png.
Run after src.per_type_models: python -m src.training_regimes
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.clustering import SEGMENTS, TABLES
from src.config import PROCESSED
from src.cluster_figures import AQUA, BLUE, INK, INK2, MUTED, ORANGE, SEG_NAME, SURFACE, save
from src.per_type_models import (ALL_LAYERS, INCREMENTS, LAYER_LABEL, LAYERS, MODEL_COLORS, MODEL_ORDER,
                                 PRED_PATH as STEP5_PRED, fit_segment, per_match_logloss, score)

PRED_PATH = PROCESSED / "regime_predictions.parquet"
REGIMES = ["segment", "global", "blend"]
REGIME_NAME = {"segment": "per-type", "global": "global", "blend": "blend (per-type + global)"}
REGIME_COLOR = {"segment": ORANGE, "global": BLUE, "blend": AQUA}
MODEL_SET = MODEL_ORDER + ["model_avg"]
COLORS = {**MODEL_COLORS, "model_avg": INK}
KEY = ["segment", "fold", "match_api_id", "season", "y_int"]
P = ["p_H", "p_D", "p_A"]
N_BOOT = 2000


def fit_all():
    out = Parallel(n_jobs=len(SEGMENTS))(delayed(fit_segment)(s, "global") for s in SEGMENTS)
    glob = pd.concat([p[~p.model.isin(["frequency", "market"])] for p, _ in out], ignore_index=True).assign(regime="global")
    metas = pd.concat([m for _, m in out], ignore_index=True)
    step5 = pd.read_parquet(STEP5_PRED)
    base = step5[step5.model.isin(["frequency", "market"])].assign(regime="-")
    seg = step5[~step5.model.isin(["frequency", "market"])].assign(regime="segment")

    # blend: average of the per-type and the global prediction of the same model and features
    m = seg.merge(glob, on=KEY + ["model", "layer"], suffixes=("_s", "_g"))
    assert len(m) == len(seg) == len(glob)
    blend = m[KEY + ["model", "layer"]].copy()
    for c in P:
        blend[c] = 0.5 * m[f"{c}_s"] + 0.5 * m[f"{c}_g"]
    blend["regime"] = "blend"

    # 6-model average: equal weights, for every regime and feature set
    allm = pd.concat([seg, glob, blend], ignore_index=True)
    six = allm[allm.model.isin(MODEL_ORDER)]
    assert six.groupby(["regime"] + KEY + ["layer"]).size().eq(len(MODEL_ORDER)).all()
    avg = six.groupby(["regime"] + KEY + ["layer"], as_index=False)[P].mean().assign(model="model_avg")
    return pd.concat([base, allm, avg], ignore_index=True), metas


def scores(preds):
    out = []
    for reg, g in preds.groupby("regime"):
        out.append(score(g).assign(regime=reg))
    sc = pd.concat(out, ignore_index=True)
    p = sc[sc.season == "pooled"].copy()
    ref = p[p.regime == "-"].pivot(index="segment", columns="model", values="log_loss")
    p["ll_frequency"] = p.segment.map(ref.frequency)
    p["ll_market"] = p.segment.map(ref.market)
    p["skill"] = 1 - p.log_loss / p.ll_frequency
    p["gap_closed"] = (p.ll_frequency - p.log_loss) / (p.ll_frequency - p.ll_market)
    return sc, p


def bootstrap(preds):
    rows = []
    for seg, gs in preds.groupby("segment"):
        keyed = {k: g.sort_values(["fold", "match_api_id"]) for k, g in gs.groupby(["regime", "model", "layer"])}
        ll = {k: per_match_logloss(g) for k, g in keyed.items()}
        n = len(next(iter(ll.values())))
        idx = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))

        def add(kind, model, layer, a, b, label):
            d = ll[a] - ll[b]
            boots = d[idx].mean(axis=1)
            rows.append({"segment": seg, "kind": kind, "model": model, "layer": layer, "contrast": label,
                         "diff": d.mean(), "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                         "p_better": (boots < 0).mean(), "n": n})

        cells = [("strength_baseline", "-")] + [(m, l) for m in MODEL_SET for l in ALL_LAYERS]
        for m, l in cells:
            add("regime", m, l, ("segment", m, l), ("global", m, l), "per-type vs global")
            add("regime", m, l, ("blend", m, l), ("segment", m, l), "blend vs per-type")
            add("regime", m, l, ("blend", m, l), ("global", m, l), "blend vs global")
            for reg in REGIMES:
                add("vs_market", m, l, (reg, m, l), ("-", "market", "-"), f"{REGIME_NAME[reg]} vs market")
                add("vs_strength", m, l, (reg, m, l), ("segment", "strength_baseline", "-"),
                    f"{REGIME_NAME[reg]} vs per-type gd38-only")
        for reg in REGIMES:
            for l in ALL_LAYERS:
                add("model_avg", "model_avg", l, (reg, "model_avg", l), (reg, "logit", l),
                    f"{REGIME_NAME[reg]}: 6-model average vs logit")
            for m in MODEL_SET:
                for a, b, label in INCREMENTS:
                    add("layer_increment", m, a, (reg, m, a), (reg, m, b), f"{REGIME_NAME[reg]}: {label}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_regimes_logit(sm):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, seg in zip(axes, SEGMENTS):
        q = sm[sm.segment == seg]
        ref = q[q.regime == "-"].set_index("model").log_loss
        for name, ls in [("frequency", (0, (1, 2))), ("market", "-")]:
            ax.axhline(ref[name], color=INK2 if name == "market" else MUTED, lw=1.2, ls=ls)
            ax.text(4.45, ref[name], f"{name} {ref[name]:.3f}", va="bottom", ha="right", fontsize=7.5, color=INK2)
        for i, reg in enumerate(REGIMES):
            r = q[q.regime == reg]
            lg = r[r.model == "logit"].set_index("layer").log_loss
            off = (i - 1) * 0.05
            ax.plot(0 + off, r[r.model == "strength_baseline"].log_loss.iloc[0], "s", color=REGIME_COLOR[reg],
                    ms=6, mec=SURFACE, mew=1.5)
            ax.plot(np.arange(1, 4) + off, lg.loc[LAYERS], "o-", color=REGIME_COLOR[reg], ms=6, mec=SURFACE, mew=1.5,
                    label=REGIME_NAME[reg])
            ax.plot(4 + off, lg.loc["T"], "D", color=REGIME_COLOR[reg], ms=6, mec=SURFACE, mew=1.5)
        for xv in [0.5, 3.5]:
            ax.axvline(xv, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(np.arange(5), ["gd38 only", "F1 ratings", "F1+F2", "F1+F2+F3", "T (team)"], fontsize=8)
        ax.set_xlim(-0.3, 4.5)
        ax.set_title(f"{SEG_NAME[seg]} (test n={int(q.n.iloc[0])})")
    axes[0].set_ylabel("Log loss (lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("Logistic regression: per-type, global and blended training. Scored on each type's test matches "
                 "(each panel has its own y-axis)", x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "regimes_logit_by_regime.png")


def _contrast_rows():
    order = [("strength_baseline", "-")] + [(m, l) for m in MODEL_SET for l in ALL_LAYERS]
    name = lambda m: "6-model average" if m == "model_avg" else m.replace("_", " ")
    labels = ["gd38-only logit"] + [f"{name(m)} {LAYER_LABEL[l]}" for m, l in order[1:]]
    colors = [MUTED] + [COLORS[m] for m, _ in order[1:]]
    return order, labels, colors


def fig_contrast(bs):
    b = bs[(bs.kind == "regime") & (bs.contrast == "per-type vs global")]
    order, labels, colors = _contrast_rows()
    fig, axes = plt.subplots(1, 3, figsize=(13, 9.4), sharey=True)
    for ax, seg in zip(axes, SEGMENTS):
        q = b[b.segment == seg].set_index(["model", "layer"]).loc[order]
        y = np.arange(len(order))
        ax.axvline(0, color=INK2, lw=1)
        for yi, (_, r), c in zip(y, q.iterrows(), colors):
            ax.hlines(yi, r.ci_lo, r.ci_hi, color=c, lw=1.6)
            ax.plot(r["diff"], yi, "o", color=c, ms=6, mec=SURFACE, mew=1.2)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(SEG_NAME[seg])
        ax.set_xlabel("log loss: per-type − global (positive = global better)")
    fig.suptitle("Per-type training vs global training, same model and features; 95% bootstrap CIs",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "regimes_pertype_vs_global.png")


def fig_blend(bs):
    """The blend against each of its two members: blend − per-type (circles) and blend − global (squares)."""
    b = bs[bs.kind == "regime"]
    order, labels, colors = _contrast_rows()
    fig, axes = plt.subplots(1, 3, figsize=(13, 9.4), sharey=True)
    for ax, seg in zip(axes, SEGMENTS):
        y = np.arange(len(order))
        ax.axvline(0, color=INK2, lw=1)
        for contrast, mk, dy in [("blend vs per-type", "o", -0.17), ("blend vs global", "s", 0.17)]:
            q = b[(b.segment == seg) & (b.contrast == contrast)].set_index(["model", "layer"]).loc[order]
            for yi, (_, r), c in zip(y, q.iterrows(), colors):
                ax.hlines(yi + dy, r.ci_lo, r.ci_hi, color=c, lw=1.3)
                ax.plot(r["diff"], yi + dy, mk, color=c, ms=5, mec=SURFACE, mew=1)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(SEG_NAME[seg])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_xlabel("Δ log loss of the blend (negative = blend better)")
    axes[0].plot([], [], "o", color=INK2, label="blend − per-type")
    axes[0].plot([], [], "s", color=INK2, label="blend − global")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Blend ensemble (0.5 × per-type + 0.5 × global) against each of its two members; 95% bootstrap CIs",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "regimes_blend_vs_members.png")


def fig_best(sm):
    """Best model of each regime per type (lowest log loss), as skill vs in-type frequency."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(3)
    for i, reg in enumerate(REGIMES):
        best = sm[sm.regime == reg].sort_values("log_loss").groupby("segment").head(1).set_index("segment").loc[SEGMENTS]
        ax.plot(best.skill, y + (i - 1) * 0.15, "o", color=REGIME_COLOR[reg], ms=8, mec=SURFACE, mew=1.2,
                label=f"{REGIME_NAME[reg]} (best model)", ls="none")
        for yi, (_, r) in zip(y, best.iterrows()):
            ax.text(r.skill + 0.003, yi + (i - 1) * 0.15, f"{r.model.replace('_', ' ')} {r.layer}".replace(" -", ""),
                    va="center", fontsize=7, color=INK2)
    mk = sm[sm.model == "market"].set_index("segment").loc[SEGMENTS]
    ax.plot(mk.skill, y + 0.32, "D", color=INK, ms=7, mec=SURFACE, label="market", ls="none")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [SEG_NAME[s] for s in SEGMENTS])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Skill = 1 − log loss / in-type frequency log loss")
    ax.set_title("Best model under each training regime vs the market")
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "regimes_best_by_regime.png")


def main():
    t0 = time.time()
    preds, meta = fit_all()
    preds.to_parquet(PRED_PATH, index=False)
    meta.to_csv(TABLES / "regimes_tuning.csv", index=False)
    sc, sm = scores(preds)
    sc.to_csv(TABLES / "regimes_scores.csv", index=False)
    sm.to_csv(TABLES / "regimes_summary.csv", index=False)
    bs = bootstrap(preds)
    bs.to_csv(TABLES / "regimes_bootstrap.csv", index=False)
    fig_regimes_logit(sm)
    fig_contrast(bs)
    fig_blend(bs)
    fig_best(sm)

    pd.set_option("display.width", 220)
    for seg in SEGMENTS:
        t = sm[sm.segment == seg].pivot_table(index=["model", "layer"], columns="regime", values="log_loss")
        print(f"\n=== {seg} (log loss) ===\n", t.round(4).to_string())
    r = bs[(bs.kind == "regime")]
    print("\nregime contrasts:\n", r[["segment", "model", "layer", "contrast", "diff", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
