# Technical notes: pipeline, outputs and data decisions

The data is the EPL subset of `database.sqlite`, 2008/09–2015/16. Treat `database.sqlite` as read-only. Every file in `data/processed/` can be rebuilt from it.

## Run

```bash
pip install -r requirements.txt
bash run_all.sh          # run from the repository root, or run the modules below in order (about 20 minutes in total)
```

| Order | Module | Output (`data/processed/` unless noted) |
|---|---|---|
| 1 | `python -m src.build_matches` | `matches.parquet`: 3,040 matches with teams, score, label `y` (H/D/A) and `y_int` (0/1/2), the 22 starters with X/Y coordinates, odds, and per-match XML stats (post-match, so they may only be used lagged). Also `missing_starters.csv`. |
| 2 | `python -m src.features` | `features.parquet`: F1 / F2 / F3 features, clustering gap features and raw home/away columns |
| 3 | `python -m src.market` | `market.parquet` (`p_mkt_*` consensus, `p_b365_*`, `p_freq_*`, `p_unif_*`) and `baseline_scores.csv` |
| 4 | `python -m src.checks` | Assert-based checkpoints: counts, leakage, and a brute-force recomputation of the 38- and 5-match rolling features and h2h |
| 5 | `python -m src.clustering` | `segments.parquet` (long format: one row per fold × match) and `report/tables/cluster_*.csv` |
| 6 | `python -m src.cluster_figures` | `figures/cluster_*.png` |
| 7 | `python -m src.per_type_models` | Per-type models: `per_type_predictions.parquet`, `report/tables/models_*.csv`, `figures/models_*.png` (6 models: logit, CART, random forest, LightGBM, XGBoost, MLP; about 5 minutes, the three types run in parallel) |
| 8 | `python -m src.training_regimes` | Per-type vs global training, plus two fixed-weight ensembles (a blend of the two: 0.5 × per-type + 0.5 × global; and the average of the 6 models): `regime_predictions.parquet`, `report/tables/regimes_*.csv`, `figures/regimes_*.png` (about 6 minutes) |
| 9 | `python -m src.robustness` | Robustness: per-season scores, round-block bootstrap, calibration (`report/tables/robustness_*.csv`, `figures/robustness_*.png`; seconds) |
| 10 | `python -m src.make_output` | `report/all_results.md`: every result table in one file, generated from the outputs |

**Features** (nested layers, from the most basic to the most situational). F1 player ability: FIFA ratings of the starting XI by position (`gk_diff`, `def_diff`, `mid_diff`, `att_diff`). F2 adds team performance: `gd38_diff` and `sot38_diff` (per-match goal difference and shots on target over each club's last 38 EPL matches, home minus away), plus `home_promoted` and `away_promoted`. F3 adds 5-match form, goal environment, h2h, rest days, season stage and `xi_vs_usual_diff`. T (team performance only, the four F2 additions) is the reverse-order reference, so increments are reported in both directions (F2 vs F1 and F2 vs T).

### `segments.parquet` (input to Steps 5–6)

Each fold's scaler and GMM are fit **only on that fold's train seasons**. `split` ∈ {train, val, test, unused}.
Clusters are merged into three match types by a pre-specified rule: the smallest-gap cluster is `even`; starting from the largest-gap cluster, clusters are added to `lopsided` until they hold at least 25% of the train matches; the rest are `middle`.
Columns: `segment` (even / middle / lopsided), `p_even` / `p_middle` / `p_lopsided` (summed GMM posteriors; not used by the models), `cluster`, `km_segment` (the same rule applied to K-means), `rule_segment` (|gd38_diff| cut at the quantiles that reproduce the GMM type shares) and `rule_mkt20` (market closest 20%; comparison only).

```python
lab = pd.read_parquet("data/processed/segments.parquet")
fold4 = lab[lab.fold == 4]          # labels to use with FOLDS[3]
```

## Using the features

```python
import pandas as pd
from src.features import FEATURE_SETS, GAP_COLS, BINARY_COLS   # FEATURE_SETS["F1"|"F2"|"F3"]
from src.config import FOLDS
from src.evaluate import score_all                               # log_loss / rps / brier
f = pd.read_parquet("data/processed/features.parquet")
mk = pd.read_parquet("data/processed/market.parquet")
df = f.merge(mk[["match_api_id", "p_mkt_H", "p_mkt_D", "p_mkt_A", "p_freq_H", "p_freq_D", "p_freq_A"]], on="match_api_id")
df = df[df.is_warmup == 0]                                       # 2008/09 is warm-up only
```

Odds columns (`p_mkt_*`, `p_b365_*`) are **baselines only**. Never use them as model features or clustering inputs.

## Decisions taken in Steps 1–3 (beyond the plan)

- **Shots on target**: the XML `shoton` events leave out goals, so `sot = XML shoton + goals scored`. The result averages 4.8 per team per match. The raw count is kept as `*_sot_xml`.
- **Missing starters**: 78 matches are each missing 1–2 starter ids (80 ids in total; the list is in `missing_starters.csv`). No goalkeeper is missing. Positional means use the starters that are available. A positional group left empty falls back to that side's XI mean.
- **Positions from Y**: 1 = GK, 3 = DEF, 5–8 = MID, 9–11 = ATT. Each side averages 1 / 4.0 / 4.3 / 1.7 players per group.
- **Player ratings**: each starter gets the latest FIFA `overall_rating` dated **strictly before** the match date.
- **Rolling windows (38 matches for F1, 5 for F2)** stay inside a club's current EPL spell: the current season plus earlier seasons played in the EPL without a break. Earlier spells of returning clubs are dropped as stale. When a promoted club has fewer matches than the window, the empty slots are filled with the per-match average of past promoted clubs in their first season, using earlier seasons only. For goal difference this is about −0.5 to −0.64 per match.
- **Coefficient interpretation (Step 5)**: residualisation follows the layer order: `gd38_diff` and `sot38_diff` are residualised on the four ratings, and the 5-match form variables on ratings + `gd38_diff` + `sot38_diff`, so each coefficient reads as "beyond the earlier layers". Prediction models use the raw features.
- **h2h**: the home team's average points over the last 5 meetings at either venue. With no meetings, it is set to the warm-up season's mean points per team-match (1.37) and `h2h_missing = 1`.
- **Rest days**: each team's rest is capped at 14 before taking the difference. A team's first match counts as 14.
- **Frequency baseline** for season *s*: H/D/A shares over 2009/10…*s*−1, which is that backtest fold's train + val.
