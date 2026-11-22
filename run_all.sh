#!/usr/bin/env bash
# Rebuild everything from database.sqlite (about 20 minutes, mostly the model fitting).
set -euo pipefail
cd "$(dirname "$0")"
python -m src.build_matches      # Step 1 -> matches.parquet
python -m src.features           # Step 2 -> features.parquet
python -m src.market             # Step 3 -> market.parquet, baseline_scores.csv
python -m src.checks             # leakage / recomputation checkpoints (asserts)
python -m src.clustering         # Step 4 -> segments.parquet, report/tables/cluster_*.csv (~20 s)
python -m src.cluster_figures    # Step 4 -> figures/cluster_*.png
python -m src.per_type_models    # Step 5 -> per_type_predictions.parquet, report/tables/models_*.csv, figures/models_*.png
python -m src.training_regimes   # Step 6 -> regime_predictions.parquet, report/tables/regimes_*.csv, figures/regimes_*.png
python -m src.robustness         # Step 7 -> report/tables/robustness_*.csv, figures/robustness_*.png (seconds; reuses predictions)
python -m src.make_output        # report/all_results.md (every result table in one file)
