"""Step 5: supervised models within each match type (research question 2).

For every fold and every type (even / middle / lopsided), models learn from that type's train(+val)
matches and are scored on that type's test matches ("segment" regime). Rows per type: 3 baselines
(frequency of that type, single-strength logit on gd38_diff, market) + 6 models x 4 feature sets:
the nested layers F1 (player ratings) -> F2 (+ team performance) -> F3 (+ recent / situational), and
T (team performance only) as the reverse-order reference.
Main results are pooled over the 4 test seasons with a match-level bootstrap (2,000 draws).
Cross-type comparison uses relative metrics:
  skill       = 1 - LL_model / LL_frequency(type)
  gap_closed  = (LL_frequency - LL_model) / (LL_frequency - LL_market)
Also: layer increments with bootstrap CIs, in both orders (F2 vs F1 = team performance beyond ratings;
F2 vs T = ratings beyond team performance; F3 vs F2), and fold-4 F3 logit coefficients per type with
residualisation that follows the layer order (team variables on ratings; form on ratings + team).
Writes data/processed/per_type_predictions.parquet, report/tables/models_*.csv, figures/models_*.png.
Run: python -m src.per_type_models
"""
import ast
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.clustering import SEGMENTS, TABLES
from src.config import FOLDS, PROCESSED
from src.evaluate import score_all
from src.features import BINARY_COLS
from src.modeling import FEATURE_SETS, LR, baselines, load_fold, run
from src.cluster_figures import (AQUA, BLUE, INK, INK2, MUTED, ORANGE, SEG_NAME, SEQ, SURFACE, YELLOW, save)

PRED_PATH = PROCESSED / "per_type_predictions.parquet"
MODEL_ORDER = ["logit", "cart", "random_forest", "lightgbm", "xgboost", "mlp"]
MODEL_COLORS = dict(zip(MODEL_ORDER, [BLUE, "#7a5af5", ORANGE, AQUA, "#c43e7a", YELLOW]))
LAYERS = ["F1", "F2", "F3"]          # nested layers
ALL_LAYERS = LAYERS + ["T"]           # + reverse-order reference (team performance only)
LAYER_LABEL = {"F1": "F1", "F2": "F1+F2", "F3": "F1+F2+F3", "T": "T"}
INCREMENTS = [("F2", "F1", "F2 vs F1"), ("F3", "F2", "F3 vs F2"), ("F2", "T", "F2 vs T")]
N_BOOT = 2000
# residualisation follows the layer order: (variables, regressed on)
RESIDUALISE = [(["gd38_diff", "sot38_diff"], ["gk_diff", "def_diff", "mid_diff", "att_diff"]),
               (["form_pts_diff", "form_gd_diff", "form_sot_diff"],
                ["gk_diff", "def_diff", "mid_diff", "att_diff", "gd38_diff", "sot38_diff"])]


# ------------------------------------------------------------------ fitting
def fit_segment(segment, regime="segment"):
    preds, metas = [], []
    for fold in FOLDS:
        d = load_fold(fold)
        for name, pred in baselines(d, segment):
            preds.append(pred.assign(fold=fold["fold"], segment=segment, model=name, layer="-"))
        for model, layer in [("logit", "Strength")] + [(m, l) for m in MODEL_ORDER for l in ALL_LAYERS]:
            t0 = time.time()
            pred, meta, _ = run(d, model, layer, regime, segment)
            label = "strength_baseline" if layer == "Strength" else model
            lay = "-" if layer == "Strength" else layer
            preds.append(pred.assign(fold=fold["fold"], segment=segment, model=label, layer=lay))
            metas.append({"fold": fold["fold"], **meta, "model": label, "layer": lay, "secs": time.time() - t0})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metas)


def fit_all(regime="segment"):
    out = Parallel(n_jobs=len(SEGMENTS))(delayed(fit_segment)(s, regime) for s in SEGMENTS)
    return pd.concat([p for p, _ in out], ignore_index=True), pd.concat([m for _, m in out], ignore_index=True)


# ------------------------------------------------------------------ scoring
def per_match_logloss(p):
    P = p[["p_H", "p_D", "p_A"]].to_numpy()
    return -np.log(np.clip(P[np.arange(len(P)), p.y_int.to_numpy()], 1e-15, 1))


def score(preds):
    rows = []
    for (seg, model, layer), g in preds.groupby(["segment", "model", "layer"]):
        for key, gg in [("pooled", g)] + list(g.groupby("season")):
            rows.append({"segment": seg, "model": model, "layer": layer, "season": key,
                         **score_all(gg[["p_H", "p_D", "p_A"]].to_numpy(), gg.y_int.to_numpy())})
    return pd.DataFrame(rows)


def summary(sc):
    """Relative metrics per type (pooled test): skill vs in-type frequency and share of gap closed."""
    p = sc[sc.season == "pooled"].copy()
    ref = p[p.layer == "-"].pivot(index="segment", columns="model", values="log_loss")
    p["ll_frequency"] = p.segment.map(ref.frequency)
    p["ll_market"] = p.segment.map(ref.market)
    p["skill"] = 1 - p.log_loss / p.ll_frequency
    p["gap_closed"] = (p.ll_frequency - p.log_loss) / (p.ll_frequency - p.ll_market)
    return p


def bootstrap(preds):
    """Paired match-level bootstrap within each type: model vs baselines, and layer increments."""
    rows = []
    for seg, gs in preds.groupby("segment"):
        keyed = {(m, l): g.sort_values(["fold", "match_api_id"]) for (m, l), g in gs.groupby(["model", "layer"])}
        ll = {k: per_match_logloss(g) for k, g in keyed.items()}
        n = len(next(iter(ll.values())))
        idx = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n))

        def add(kind, model, a, b, label):
            d = ll[a] - ll[b]
            boots = d[idx].mean(axis=1)
            rows.append({"segment": seg, "kind": kind, "model": model, "contrast": label, "diff": d.mean(),
                         "ci_lo": np.percentile(boots, 2.5), "ci_hi": np.percentile(boots, 97.5),
                         "p_better": (boots < 0).mean(), "n": n})

        for (m, l) in ll:
            for ref in ["strength_baseline", "market", "frequency"]:
                if m != ref:
                    add("vs_baseline", m, (m, l), (ref, "-"), f"{l} vs {ref}" if l != "-" else f"vs {ref}")
        for m in MODEL_ORDER:
            for a, b, label in INCREMENTS:
                add("layer_increment", m, (m, a), (m, b), label)
            add("layer_increment", m, (m, "T"), ("strength_baseline", "-"), "T vs gd38-only")
    return pd.DataFrame(rows)


def residualise(df):
    """Sequential residualisation (uses the raw values of the earlier layers as regressors)."""
    raw = df.copy()
    df = df.copy()
    for targets, anchor in RESIDUALISE:
        A = np.column_stack([np.ones(len(raw)), raw[anchor].to_numpy()])
        for c in targets:
            beta, *_ = np.linalg.lstsq(A, raw[c].to_numpy(), rcond=None)
            df[c] = raw[c] - A @ beta
    return df


def max_vif(df, cols):
    cont = [c for c in cols if c not in BINARY_COLS]
    R = np.corrcoef(((df[cont] - df[cont].mean()) / df[cont].std()).to_numpy(), rowvar=False)
    return float(np.diag(np.linalg.inv(R)).max())


def coefficients(meta, n_boot=500):
    """Fold-4 F3 logit per type, refit on that type's train+val with the tuned C after residualising
    team variables on ratings and form variables on ratings + team. Coefficients vs the H class
    (per 1 sd; binary per unit)."""
    d = load_fold(FOLDS[3])
    cols = FEATURE_SETS["F3"]
    names = [c for c in cols if c not in BINARY_COLS] + [c for c in cols if c in BINARY_COLS]
    out = []
    for seg in SEGMENTS:
        trva = d[d.split.isin(["train", "val"]) & (d.segment == seg)].reset_index(drop=True)
        C = ast.literal_eval(meta.query("fold == 4 and model == 'logit' and layer == 'F3' and segment == @seg").params.iloc[0])["C"]

        def coefs(df):
            df = residualise(df)
            m = LR().fit(df[cols], df.y_int.to_numpy(), np.ones(len(df)), {"C": C}).m[-1].coef_
            return np.vstack([m[1] - m[0], m[2] - m[0]])

        point = coefs(trva)
        rng = np.random.default_rng(0)
        boots = np.stack([coefs(trva.iloc[rng.integers(0, len(trva), len(trva))]) for _ in range(n_boot)])
        for i, contrast in enumerate(["D_vs_H", "A_vs_H"]):
            for j, name in enumerate(names):
                lo, hi = np.percentile(boots[:, i, j], [2.5, 97.5])
                out.append({"segment": seg, "contrast": contrast, "feature": name, "coef": point[i, j],
                            "ci_lo": lo, "ci_hi": hi, "C": C, "n": len(trva),
                            "max_vif_raw": max_vif(trva, cols), "max_vif_residualised": max_vif(residualise(trva), cols)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
def fig_logloss(sc):
    p = sc[sc.season == "pooled"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, seg in zip(axes, SEGMENTS):
        q = p[p.segment == seg]
        ref = q[q.layer == "-"].set_index("model").log_loss
        for name, ls, lab in [("frequency", (0, (1, 2)), "frequency"), ("strength_baseline", (0, (5, 3)), "gd38-only"),
                              ("market", "-", "market")]:
            ax.axhline(ref[name], color=INK2 if name == "market" else MUTED, lw=1.2, ls=ls)
            below = name == "market" and ref["market"] < ref["strength_baseline"]
            ax.text(2.45, ref[name], f"{lab} {ref[name]:.3f}", va="top" if below else "bottom", ha="right",
                    fontsize=7.5, color=INK2)
        x = np.arange(3)
        for i, m in enumerate(MODEL_ORDER):
            v = q[q.model == m].set_index("layer").log_loss
            off = (i - (len(MODEL_ORDER) - 1) / 2) * 0.035
            ax.plot(x + off, v.loc[LAYERS], "o-", color=MODEL_COLORS[m], ms=6, mec=SURFACE, mew=1.5,
                    label=m.replace("_", " "))
            ax.plot(3 + off, v.loc["T"], "D", color=MODEL_COLORS[m], ms=6, mec=SURFACE, mew=1.5)
        ax.axvline(2.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_xticks(np.arange(4), ["F1 ratings", "F1+F2", "F1+F2+F3", "T (team only)"])
        ax.set_xlim(-0.3, 3.5)
        ax.set_title(f"{SEG_NAME[seg]} (test n={int(q.n.iloc[0])})")
    axes[0].set_ylabel("Log loss (lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, ncol=6, loc="upper left", bbox_to_anchor=(0.01, 0.93))
    fig.suptitle("Per-type models, pooled test seasons 2012/13–2015/16 (each panel has its own y-axis)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save(fig, "models_logloss_by_layer.png")


def fig_skill(sm):
    rows = [("strength_baseline", "-", "gd38-only logit", MUTED, "s"), ("logit", "F1", "logit F1 (ratings)", SEQ[2], "o"),
            ("logit", "F2", "logit F1+F2", SEQ[4], "o"), ("logit", "F3", "logit F1+F2+F3", SEQ[6], "o"),
            ("logit", "T", "logit T (team only)", ORANGE, "^"), ("market", "-", "market", INK, "D")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(3)
    for i, (m, l, lab, col, mk) in enumerate(rows):
        v = sm[(sm.model == m) & (sm.layer == l)].set_index("segment").loc[SEGMENTS].skill
        ax.plot(v, y + (i - 2.5) * 0.11, mk, color=col, ms=7, mec=SURFACE, mew=1.2, label=lab, ls="none")
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [SEG_NAME[s] for s in SEGMENTS])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Skill = 1 − log loss / in-type frequency log loss (higher is better)")
    ax.set_title("Skill vs the in-type frequency baseline, pooled test seasons")
    ax.legend(fontsize=7.5, loc="upper right")
    save(fig, "models_skill_by_type.png")


def fig_increments(bs):
    b = bs[(bs.kind == "layer_increment") & (bs.contrast != "T vs gd38-only")]
    names = {"F2 vs F1": "team perf. on ratings (F2 vs F1)", "F3 vs F2": "recent on F2 (F3 vs F2)",
             "F2 vs T": "ratings on team perf. (F2 vs T)"}
    marks = {"F2 vs F1": "o", "F3 vs F2": "s", "F2 vs T": "D"}
    order = [(m, c) for m in MODEL_ORDER for c in names]
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.6), sharey=True)
    for ax, seg in zip(axes, SEGMENTS):
        q = b[b.segment == seg].set_index(["model", "contrast"]).loc[order]
        y = np.arange(len(order))
        ax.axvline(0, color=INK2, lw=1)
        for yi, ((m, c), r) in zip(y, q.iterrows()):
            ax.hlines(yi, r.ci_lo, r.ci_hi, color=MODEL_COLORS[m], lw=1.6)
            ax.plot(r["diff"], yi, marks[c], color=MODEL_COLORS[m], ms=6, mec=SURFACE, mew=1.2)
        ax.set_yticks(y, [f"{m.replace('_', ' ')}: {names[c]}" for m, c in order], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_title(f"{SEG_NAME[seg]}")
        ax.set_xlabel("Δ log loss from adding the layer (negative = helps)")
    fig.suptitle("Layer increments in both orders: 95% bootstrap CIs, pooled test seasons",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "models_layer_increments.png")


def fig_coefs(cf):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
    order = cf[(cf.contrast == "D_vs_H") & (cf.segment == "even")].sort_values("coef").feature.tolist()
    for r, (contrast, title) in enumerate([("D_vs_H", "log[P(D)/P(H)]"), ("A_vs_H", "log[P(A)/P(H)]")]):
        for c, seg in enumerate(SEGMENTS):
            ax = axes[r, c]
            q = cf[(cf.contrast == contrast) & (cf.segment == seg)].set_index("feature").loc[order]
            y = np.arange(len(q))
            sig = ((q.ci_lo > 0) | (q.ci_hi < 0)).to_numpy()
            ax.axvline(0, color=INK2, lw=1)
            ax.hlines(y, q.ci_lo, q.ci_hi, color=np.where(sig, BLUE, MUTED), lw=1.5)
            ax.scatter(q.coef, y, color=np.where(sig, BLUE, MUTED), s=30, edgecolor=SURFACE, lw=1.2, zorder=3)
            ax.set_yticks(y, order, fontsize=7.5)
            ax.set_title(f"{SEG_NAME[seg]} · {title} (n={q.n.iloc[0]}, C={q.C.iloc[0]})", fontsize=9.5)
            if r == 1:
                ax.set_xlabel("coefficient (per 1 sd; binary per unit)")
    fig.suptitle("Fold-4 F1+F2+F3 logit per type; team vars residualised on ratings, form on ratings+team. "
                 "95% bootstrap CI, blue = excludes 0", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "models_coefficients.png")


def main():
    t0 = time.time()
    preds, meta = fit_all("segment")
    preds.to_parquet(PRED_PATH, index=False)
    meta.to_csv(TABLES / "models_tuning.csv", index=False)
    sc = score(preds)
    sc.to_csv(TABLES / "models_scores.csv", index=False)
    sm = summary(sc)
    sm.to_csv(TABLES / "models_summary.csv", index=False)
    bs = bootstrap(preds)
    bs.to_csv(TABLES / "models_bootstrap.csv", index=False)
    cf = coefficients(meta)
    cf.to_csv(TABLES / "models_coefficients.csv", index=False)
    fig_logloss(sc)
    fig_skill(sm)
    fig_increments(bs)
    fig_coefs(cf)

    pd.set_option("display.width", 220)
    for seg in SEGMENTS:
        print(f"\n=== {seg} ===")
        print(sm[sm.segment == seg].sort_values("log_loss")[["model", "layer", "log_loss", "rps", "brier", "n", "skill",
                                                             "gap_closed"]].round(4).to_string(index=False))
    inc = bs[bs.kind == "layer_increment"]
    print("\nlayer increments:\n", inc[["segment", "model", "contrast", "diff", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    vs = bs[(bs.kind == "vs_baseline") & bs.contrast.str.contains("strength_baseline|market")]
    print("\nvs baselines (logit + market):\n",
          vs[vs.model.isin(["logit", "market"])][["segment", "model", "contrast", "diff", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
    sig = cf[(cf.ci_lo > 0) | (cf.ci_hi < 0)]
    print("\nsignificant coefficients:\n", sig[["segment", "contrast", "feature", "coef", "ci_lo", "ci_hi"]].round(3).to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
