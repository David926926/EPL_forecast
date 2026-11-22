"""Probability scoring rules for H/D/A forecasts. P is (n, 3) in CLASSES order [H, D, A];
y is an int array of class indices (0=H, 1=D, 2=A). All functions return the mean over matches."""
import numpy as np
import pandas as pd

EPS = 1e-15


def _onehot(y, k=3):
    y = np.asarray(y, dtype=int)
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def log_loss(P, y):
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return float(-np.log(P[np.arange(len(P)), np.asarray(y, dtype=int)]).mean())


def brier(P, y):
    return float(((np.asarray(P, dtype=float) - _onehot(y)) ** 2).sum(axis=1).mean())


def rps(P, y):
    """Ranked probability score over the ordered outcomes H > D > A (divided by K-1 = 2)."""
    cp = np.cumsum(np.asarray(P, dtype=float), axis=1)[:, :2]
    co = np.cumsum(_onehot(y), axis=1)[:, :2]
    return float(((cp - co) ** 2).sum(axis=1).mean() / 2)


def score_all(P, y):
    return {"log_loss": log_loss(P, y), "rps": rps(P, y), "brier": brier(P, y), "n": len(y)}


def score_table(df, prob_sets, y_col="y_int", by="season"):
    """prob_sets: {name: [colH, colD, colA]}. Returns long table of scores per group and name."""
    rows = []
    groups = list(df.groupby(by)) + [("ALL", df)] if by else [("ALL", df)]
    for g, d in groups:
        for name, cols in prob_sets.items():
            dd = d.dropna(subset=cols)
            rows.append({by or "group": g, "model": name, **score_all(dd[cols].values, dd[y_col].values)})
    return pd.DataFrame(rows)
