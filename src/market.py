"""Step 3: market baseline (odds are a yardstick, never a model feature).

Per bookmaker: p_k = (1/odds_k) / sum_j (1/odds_j)  (basic normalisation removes the margin).
Market consensus = mean of the normalised probabilities of B365, BW, IW, LB, WH, VC over the
books available for that match, renormalised. B365 is also reported alone.

Also scores the non-model baselines per season: uniform (1/3 each) and frequency (H/D/A shares of
2009/10 .. season-1, i.e. exactly the train+val seasons of the backtest fold whose test season it
is; 2009/10 uses 2008/09). Writes data/processed/market.parquet and baseline_scores.csv.
Run: python -m src.market
"""
import numpy as np
import pandas as pd

from src.config import CLASSES, CONSENSUS_BOOKS, MARKET_PATH, MATCHES_PATH, PROCESSED, SEASONS
from src.evaluate import score_table


def implied(m, book):
    odds = m[[f"{book}{o}" for o in "HDA"]].to_numpy(dtype=float)
    inv = 1.0 / odds
    over = inv.sum(axis=1)
    return inv / over[:, None], over - 1.0


def build():
    m = pd.read_parquet(MATCHES_PATH)
    out = m[["match_api_id", "season", "date", "y", "y_int"]].copy()
    stack = []
    for b in CONSENSUS_BOOKS:
        p, margin = implied(m, b)
        stack.append(p)
        out[f"margin_{b}"] = margin
        if b == "B365":
            out[["p_b365_H", "p_b365_D", "p_b365_A"]] = p
    stack = np.stack(stack)                           # (books, n, 3)
    cons = np.nanmean(stack, axis=0)
    cons /= cons.sum(axis=1, keepdims=True)
    out[["p_mkt_H", "p_mkt_D", "p_mkt_A"]] = cons
    out["n_books"] = (~np.isnan(stack[:, :, 0])).sum(axis=0)

    out[["p_unif_H", "p_unif_D", "p_unif_A"]] = 1 / 3
    for i, s in enumerate(SEASONS[1:], start=1):
        past = m[m.season.isin(SEASONS[1:i] or SEASONS[:1])]
        freq = past.y.value_counts(normalize=True).reindex(CLASSES).to_numpy()
        out.loc[out.season == s, ["p_freq_H", "p_freq_D", "p_freq_A"]] = freq
    return out


PROB_SETS = {
    "uniform": ["p_unif_H", "p_unif_D", "p_unif_A"],
    "frequency": ["p_freq_H", "p_freq_D", "p_freq_A"],
    "market_B365": ["p_b365_H", "p_b365_D", "p_b365_A"],
    "market_consensus": ["p_mkt_H", "p_mkt_D", "p_mkt_A"],
}

if __name__ == "__main__":
    mk = build()
    assert np.allclose(mk[["p_mkt_H", "p_mkt_D", "p_mkt_A"]].sum(axis=1), 1)
    assert (mk.n_books >= 5).all(), mk.n_books.value_counts()
    mk.to_parquet(MARKET_PATH, index=False)

    scored = mk[mk.season != SEASONS[0]]
    tab = score_table(scored, PROB_SETS)
    tab.to_csv(PROCESSED / "baseline_scores.csv", index=False)
    print(f"saved {MARKET_PATH} {mk.shape}")
    print("mean margin:", mk[[c for c in mk if c.startswith("margin_")]].mean().round(4).to_dict())
    for metric in ["log_loss", "rps", "brier"]:
        print(f"\n{metric}:\n", tab.pivot(index="season", columns="model", values=metric)
              [list(PROB_SETS)].round(4).to_string())
