"""Step 4: unsupervised match types — even / middle / lopsided (research question 1).

Pre-specified procedure, run separately for every backtest fold (fit on that fold's TRAIN seasons
only, then applied to its val and test seasons):
  1. Standardise the three odds-free gap features (GAP_COLS) with train mean / sd.
  2. GaussianMixture, (k, covariance_type) chosen by minimum BIC over k = 2..6 x
     {full, diag, tied, spherical}. KMeans (k by maximum silhouette) is the comparison method.
  3. Merge clusters into three match types, using the cluster centres in standardised space:
       even      the cluster with the smallest summed gap;
       lopsided  starting from the largest-gap cluster, add clusters until they hold >= 25% of the
                 train matches;
       middle    every other cluster.
     Each match gets `segment` (argmax cluster -> type) and p_even / p_middle / p_lopsided
     (summed posterior of the clusters in each type; used for weighting in Step 6).
Comparisons (thresholds from the fold's train seasons):
  rule_segment  |gd38_diff| cut at the train quantiles that reproduce the GMM type shares
  km_segment    the same merging rule applied to the KMeans clusters
  rule_mkt20    market |p_H - p_A| in the lowest 20% of train (odds used ONLY for this comparison)

Writes data/processed/segments.parquet and report/tables/cluster_*.csv.
Run: python -m src.clustering
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.config import CLASSES, FEATURES_PATH, FOLDS, MARKET_PATH, PROCESSED, ROOT
from src.evaluate import log_loss
from src.features import GAP_COLS

warnings.filterwarnings("ignore", category=UserWarning)
TABLES = ROOT / "report" / "tables"
SEGMENTS_PATH = PROCESSED / "segments.parquet"
SEGMENTS = ["even", "middle", "lopsided"]
LOPSIDED_SHARE = 0.25
KS = range(2, 7)
COV_TYPES = ["full", "diag", "tied", "spherical"]
N_INIT = 10
SEED = 0


def load():
    f = pd.read_parquet(FEATURES_PATH)
    mk = pd.read_parquet(MARKET_PATH)[["match_api_id", "p_mkt_H", "p_mkt_D", "p_mkt_A"]]
    d = f.merge(mk, on="match_api_id")
    d["mkt_gap"] = (d.p_mkt_H - d.p_mkt_A).abs()
    return d[d.is_warmup == 0].reset_index(drop=True)


def select_models(X):
    rows = []
    for k, cov in itertools.product(KS, COV_TYPES):
        g = GaussianMixture(k, covariance_type=cov, n_init=N_INIT, random_state=SEED).fit(X)
        lab = g.predict(X)
        sil = silhouette_score(X, lab) if len(set(lab)) > 1 else np.nan
        rows.append({"method": "gmm", "k": k, "cov": cov, "bic": g.bic(X), "silhouette": sil})
    for k in KS:
        km = KMeans(k, n_init=N_INIT, random_state=SEED).fit(X)
        rows.append({"method": "kmeans", "k": k, "cov": None, "bic": np.nan,
                     "silhouette": silhouette_score(X, km.labels_), "inertia": km.inertia_})
    return pd.DataFrame(rows)


def cluster_to_segment(centers, train_labels):
    """Map cluster id -> type with the pre-specified rule (see module docstring)."""
    k = len(centers)
    score = np.asarray(centers).sum(axis=1)
    order = np.argsort(score)
    shares = np.bincount(train_labels, minlength=k) / len(train_labels)
    seg = {c: "middle" for c in range(k)}
    seg[order[0]] = "even"
    cum = 0.0
    for c in order[::-1]:
        if c == order[0] or cum >= LOPSIDED_SHARE:
            break
        seg[c] = "lopsided"
        cum += shares[c]
    return seg


def fit_fold(d, fold):
    tr_mask = (d.season.isin(fold["train"])).to_numpy()
    scaler = StandardScaler().fit(d.loc[tr_mask, GAP_COLS])
    X_all = scaler.transform(d[GAP_COLS])
    X_tr = X_all[tr_mask]

    grid = select_models(X_tr).assign(fold=fold["fold"])
    best_g = grid[grid.method == "gmm"].sort_values("bic").iloc[0]
    best_k = grid[grid.method == "kmeans"].sort_values("silhouette", ascending=False).iloc[0]
    gmm = GaussianMixture(int(best_g.k), covariance_type=best_g["cov"], n_init=N_INIT, random_state=SEED).fit(X_tr)
    km = KMeans(int(best_k.k), n_init=N_INIT, random_state=SEED).fit(X_tr)

    seg_map = cluster_to_segment(gmm.means_, gmm.predict(X_tr))
    km_map = cluster_to_segment(km.cluster_centers_, km.labels_)

    out = d[["match_api_id", "season", "y", "abs_gd38_diff", "mkt_gap"]].copy()
    out["fold"] = fold["fold"]
    out["split"] = np.select([tr_mask, d.season == fold["val"], d.season == fold["test"]],
                             ["train", "val", "test"], "unused")
    out["cluster"] = gmm.predict(X_all)
    out["segment"] = out.cluster.map(seg_map)
    post = gmm.predict_proba(X_all)
    for s in SEGMENTS:
        cols = [c for c, v in seg_map.items() if v == s]
        out[f"p_{s}"] = post[:, cols].sum(axis=1) if cols else 0.0
    out["km_segment"] = pd.Series(km.predict(X_all)).map(km_map).to_numpy()

    tr = out[tr_mask]
    sh = tr.segment.value_counts(normalize=True).reindex(SEGMENTS, fill_value=0)
    lo, hi = tr.abs_gd38_diff.quantile(sh["even"]), tr.abs_gd38_diff.quantile(1 - sh["lopsided"])
    out["rule_segment"] = np.select([out.abs_gd38_diff <= lo, out.abs_gd38_diff > hi], ["even", "lopsided"], "middle")
    out["rule_mkt20"] = (out.mkt_gap <= tr.mkt_gap.quantile(0.20)).astype(int)

    spec = {"fold": fold["fold"], "gmm_k": int(best_g.k), "gmm_cov": best_g["cov"], "gmm_silhouette": best_g.silhouette,
            "kmeans_k": int(best_k.k), "kmeans_silhouette": best_k.silhouette,
            **{f"share_{s}": sh[s] for s in SEGMENTS},
            **{f"n_train_{s}": int((tr.segment == s).sum()) for s in SEGMENTS}}
    profile = pd.DataFrame(scaler.inverse_transform(gmm.means_), columns=GAP_COLS)
    profile["cluster"] = range(len(profile))
    profile["segment"] = profile.cluster.map(seg_map)
    profile["train_share"] = np.bincount(gmm.predict(X_tr), minlength=len(profile)) / len(X_tr)
    return out.drop(columns=["abs_gd38_diff", "mkt_gap"]), grid, spec, (X_tr, best_g, out.loc[tr_mask, "segment"].to_numpy(), profile)


def stability(X_tr, best_g, ref_segment, n_seeds=10):
    """Refit the chosen GMM with 10 seeds (one init each): ARI between cluster labelings, and how
    often the resulting match type agrees with the main fit."""
    labels, agree = [], []
    for s in range(n_seeds):
        g = GaussianMixture(int(best_g.k), covariance_type=best_g["cov"], n_init=1, random_state=s).fit(X_tr)
        lab = g.predict(X_tr)
        seg = pd.Series(lab).map(cluster_to_segment(g.means_, lab)).to_numpy()
        labels.append(lab)
        agree.append((seg == ref_segment).mean())
    ari = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
    return {"ari_mean": np.mean(ari), "ari_min": np.min(ari),
            "segment_agreement_mean": np.mean(agree), "segment_agreement_min": np.min(agree)}


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def outcome_table(x, col, sample, definition):
    """Outcome shares per type for one labeling; chi-square test of type x outcome (3 x 3)."""
    tab = pd.crosstab(x[col], x.y).reindex(index=SEGMENTS, columns=CLASSES, fill_value=0)
    chi2, p, _, _ = stats.chi2_contingency(tab[tab.sum(axis=1) > 0])
    rows = []
    for s in SEGMENTS:
        g = x[x[col] == s]
        if len(g) == 0:
            continue
        lo, hi = wilson((g.y == "D").sum(), len(g))
        rows.append({"sample": sample, "definition": definition, "segment": s, "n": len(g), "share": len(g) / len(x),
                     "H": (g.y == "H").mean(), "D": (g.y == "D").mean(), "A": (g.y == "A").mean(),
                     "D_ci_lo": lo, "D_ci_hi": hi, "mkt_D": g.p_mkt_D.mean(), "chi2": chi2, "p_chi2": p})
    return rows


def segment_frequency(lab, fold, seg):
    tv = lab[(lab.fold == fold) & lab.split.isin(["train", "val"]) & (lab.segment == seg)]
    return np.bincount(tv.y.map({c: i for i, c in enumerate(CLASSES)}), minlength=3) / len(tv)


def predictability(lab, mk):
    """Per type, pooled test seasons: in-type frequency baseline vs market log loss."""
    x = lab[lab.split == "test"].merge(mk, on="match_api_id")
    rows = []
    for s in SEGMENTS:
        g = x[x.segment == s]
        P_freq = np.vstack([np.tile(segment_frequency(lab, f, s), (len(gg), 1)) for f, gg in g.groupby("fold")])
        g = pd.concat([gg for _, gg in g.groupby("fold")])
        y = g.y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        ll_f, ll_m = log_loss(P_freq, y), log_loss(g[["p_mkt_H", "p_mkt_D", "p_mkt_A"]].to_numpy(), y)
        rows.append({"segment": s, "n_test": len(g), "ll_frequency": ll_f, "ll_market": ll_m,
                     "market_skill": 1 - ll_m / ll_f, "ll_uniform": np.log(3)})
    return pd.DataFrame(rows)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    d = load()
    mk = d[["match_api_id", "p_mkt_H", "p_mkt_D", "p_mkt_A"]]
    labels, grids, specs, profiles = [], [], [], []
    for fold in FOLDS:
        lab, grid, spec, (X_tr, best_g, ref_seg, profile) = fit_fold(d, fold)
        spec.update(stability(X_tr, best_g, ref_seg))
        labels.append(lab); grids.append(grid); specs.append(spec); profiles.append(profile.assign(fold=fold["fold"]))
        print(f"  fold {fold['fold']}: GMM k={spec['gmm_k']} {spec['gmm_cov']}, shares "
              f"{spec['share_even']:.2f}/{spec['share_middle']:.2f}/{spec['share_lopsided']:.2f}", flush=True)

    lab = pd.concat(labels, ignore_index=True)
    lab.to_parquet(SEGMENTS_PATH, index=False)
    specs = pd.DataFrame(specs)
    pd.concat(grids, ignore_index=True).to_csv(TABLES / "cluster_model_selection.csv", index=False)
    specs.to_csv(TABLES / "cluster_fold_specs.csv", index=False)

    # cluster profiles + outcomes (fold 4 model, all seasons 2009/10-2015/16)
    f4 = lab[lab.fold == 4]
    out4 = f4.groupby("cluster").agg(n=("y", "size"), H=("y", lambda s: (s == "H").mean()),
                                     D=("y", lambda s: (s == "D").mean()), A=("y", lambda s: (s == "A").mean()))
    prof = pd.concat(profiles, ignore_index=True)
    prof.to_csv(TABLES / "cluster_profiles.csv", index=False)
    prof[prof.fold == 4].set_index("cluster").join(out4).to_csv(TABLES / "cluster_outcomes_fold4.csv")

    # outcome distribution per type: GMM, |gd38| quantile rule, KMeans; in-sample and out-of-sample
    xm = lab.merge(mk, on="match_api_id")
    rows = []
    for sample, x in [("fold-4 train 2009/10-2013/14", xm[(xm.fold == 4) & (xm.split == "train")]),
                      ("pooled test 2012/13-2015/16", xm[xm.split == "test"])]:
        for col, name in [("segment", "GMM"), ("rule_segment", "|gd38| quantile rule"), ("km_segment", "K-means")]:
            rows += outcome_table(x, col, sample, name)
    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(TABLES / "cluster_segment_outcomes.csv", index=False)

    # agreement between GMM types and the comparisons (pooled test)
    t = xm[xm.split == "test"]
    agree = pd.DataFrame([{"comparison": name, "agreement": (t.segment == t[col]).mean(),
                           "ari": adjusted_rand_score(t.segment, t[col])}
                          for col, name in [("rule_segment", "|gd38| quantile rule"), ("km_segment", "K-means")]])
    agree.to_csv(TABLES / "cluster_agreement.csv", index=False)

    # even type vs market-closest-20% (pooled test)
    ev_rows = []
    for name, mask in [("GMM even", t.segment == "even"), ("|gd38| rule even", t.rule_segment == "even"),
                       ("Market closest 20%", t.rule_mkt20 == 1)]:
        g, rest = t[mask], t[~mask]
        lo, hi = wilson((g.y == "D").sum(), len(g))
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(mask, t.y))
        ev_rows.append({"definition": name, "n": len(g), "D": (g.y == "D").mean(), "D_ci_lo": lo, "D_ci_hi": hi,
                        "D_rest": (rest.y == "D").mean(), "mkt_D": g.p_mkt_D.mean(), "p_chi2_vs_rest": p,
                        "overlap_with_gmm_even": (mask & (t.segment == "even")).sum() / (t.segment == "even").sum()})
    pd.DataFrame(ev_rows).to_csv(TABLES / "cluster_even_vs_market.csv", index=False)

    pred = predictability(lab, mk)
    pred.to_csv(TABLES / "cluster_predictability.csv", index=False)

    per_season = t.groupby(["fold", "season", "segment"]).agg(
        n=("y", "size"), D=("y", lambda s: (s == "D").mean()), H=("y", lambda s: (s == "H").mean())).reset_index()
    per_season.to_csv(TABLES / "cluster_test_season_outcomes.csv", index=False)
    f4.groupby("season").segment.value_counts(normalize=True).unstack().reindex(columns=SEGMENTS) \
        .to_csv(TABLES / "cluster_segment_share_by_season.csv")

    pd.set_option("display.width", 220)
    print("\nfold specs:\n", specs.round(3).to_string(index=False))
    print("\noutcomes per type:\n", outcomes.round(3).to_string(index=False))
    print("\nagreement:\n", agree.round(3).to_string(index=False))
    print("\neven vs market:\n", pd.DataFrame(ev_rows).round(3).to_string(index=False))
    print("\npredictability:\n", pred.round(4).to_string(index=False))
    print("\nper test season:\n", per_season.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
