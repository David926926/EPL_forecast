"""Unified train -> tune on val -> refit on train+val -> predict test, shared by Steps 5-6.

Every run targets one match type `segment` (even / middle / lopsided); evaluation is ALWAYS on the
matches of that type. Training regimes (which matches a model learns from):
  "segment"   only matches of that type in the fold's train (and val) seasons  [Step 5, per-type model]
  "global"    all matches, weight 1                                             [Step 6 global model]
Hyper-parameters are chosen by log loss on the fold's VAL matches of that type, then the model is
refit on train+val with those settings and predicts the TEST matches of that type exactly once.
"""
import itertools
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

from src.clustering import SEGMENTS_PATH
from src.config import FEATURES_PATH, MARKET_PATH
from src.evaluate import log_loss
from src.features import BINARY_COLS, FEATURE_SETS

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
FEATURE_SETS = {"Strength": ["gd38_diff"], **FEATURE_SETS}
SEED = 0


def load_fold(fold):
    f = pd.read_parquet(FEATURES_PATH)
    mk = pd.read_parquet(MARKET_PATH)[["match_api_id", "p_mkt_H", "p_mkt_D", "p_mkt_A"]]
    lab = pd.read_parquet(SEGMENTS_PATH)
    lab = lab[lab.fold == fold["fold"]][["match_api_id", "split", "segment", "p_even", "p_middle", "p_lopsided"]]
    return f.merge(mk, on="match_api_id").merge(lab, on="match_api_id")


def regime_rows(d, split, regime, segment):
    s = d[d.split.isin(split)]
    if regime == "segment":
        s = s[s.segment == segment]
        return s, np.ones(len(s))
    if regime == "global":
        return s, np.ones(len(s))
    raise ValueError(regime)


# ------------------------------------------------------------------ models
def _scaler(cols):
    cont = [c for c in cols if c not in BINARY_COLS]
    binr = [c for c in cols if c in BINARY_COLS]
    return ColumnTransformer([("num", StandardScaler(), cont), ("bin", "passthrough", binr)])


class LR:
    name = "logit"
    grid = [{"C": c} for c in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]]

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        self.m = make_pipeline(_scaler(X.columns), LogisticRegression(C=p["C"], max_iter=5000))
        self.m.fit(X, y, logisticregression__sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class RF:
    """Random forest + sigmoid calibration (5-fold internal CV, so it works after the train+val refit)."""
    name = "random_forest"
    grid = [{"max_depth": d, "min_samples_leaf": l} for d, l in itertools.product([3, 5, 8], [20, 50])]

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        rf = RandomForestClassifier(n_estimators=300, max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                    max_features="sqrt", n_jobs=-1, random_state=SEED)
        self.m = CalibratedClassifierCV(rf, method="sigmoid", cv=5).fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


class LGBM:
    """Early stopping on val picks the number of rounds; the refit on train+val reuses it."""
    name = "lightgbm"
    grid = [{"num_leaves": n, "min_data_in_leaf": m, "lambda_l2": l}
            for n, m, l in itertools.product([4, 8, 15], [30, 60], [1.0, 10.0])]
    base = {"objective": "multiclass", "num_class": 3, "learning_rate": 0.02, "feature_fraction": 0.8,
            "verbose": -1, "seed": SEED, "deterministic": True, "num_threads": 4}

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        params = {**self.base, **{k: v for k, v in p.items() if k != "n_rounds"}}
        dtr = lgb.Dataset(X, y, weight=w)
        if X_val is not None:
            dva = lgb.Dataset(X_val, y_val, reference=dtr)
            self.m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
            self.n_rounds = self.m.best_iteration
        else:
            self.m = lgb.train(params, dtr, num_boost_round=p["n_rounds"])
            self.n_rounds = p["n_rounds"]
        return self

    def predict_proba(self, X):
        return self.m.predict(X, num_iteration=self.n_rounds)


class MLP:
    """One hidden layer (16), L2 weight decay; lbfgs instead of early stopping because the even-group
    val split is too small to carve out an extra stopping set. Probabilities averaged over 5 seeds.
    (All training rows have weight 1; sklearn's MLP takes no sample weights.)"""
    name = "mlp"
    grid = [{"alpha": a} for a in [0.1, 1.0, 3.0, 10.0, 30.0]]

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        self.ms = []
        for s in range(5):
            m = make_pipeline(_scaler(X.columns), MLPClassifier((16,), alpha=p["alpha"], solver="lbfgs",
                                                                  max_iter=2000, random_state=s))
            self.ms.append(m.fit(X, y))
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


class CART:
    """A single decision tree (the base learner that random forest and boosting combine).
    Leaf frequencies are Laplace-smoothed ((count + 1) / (n + 3)) so no outcome gets probability 0."""
    name = "cart"
    grid = [{"max_depth": d, "min_samples_leaf": l} for d, l in itertools.product([2, 3, 4, 6], [20, 50, 100])]

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        self.m = DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                                        random_state=SEED)
        self.m.fit(X, y, sample_weight=w)
        return self

    def predict_proba(self, X):
        full = np.zeros((len(X), 3))
        full[:, self.m.classes_] = self.m.predict_proba(X)
        n = self.m.tree_.weighted_n_node_samples[self.m.apply(X)][:, None]
        return (full * n + 1) / (n + 3)


class XGB:
    """XGBoost; like LGBM, early stopping on val picks the number of rounds and the refit reuses it."""
    name = "xgboost"
    grid = [{"max_depth": d, "min_child_weight": m, "reg_lambda": l}
            for d, m, l in itertools.product([2, 3, 4], [5, 20], [1.0, 10.0])]
    base = {"objective": "multi:softprob", "learning_rate": 0.02, "subsample": 0.8, "colsample_bytree": 0.8,
            "tree_method": "hist", "n_jobs": 4, "random_state": SEED, "eval_metric": "mlogloss"}

    def fit(self, X, y, w, p, X_val=None, y_val=None):
        params = {**self.base, **{k: v for k, v in p.items() if k != "n_rounds"}}
        if X_val is not None:
            self.m = xgb.XGBClassifier(n_estimators=3000, early_stopping_rounds=100, **params)
            self.m.fit(X, y, sample_weight=w, eval_set=[(X_val, y_val)], verbose=False)
            self.n_rounds = self.m.best_iteration + 1
        else:
            self.n_rounds = p["n_rounds"]
            self.m = xgb.XGBClassifier(n_estimators=self.n_rounds, **params)
            self.m.fit(X, y, sample_weight=w, verbose=False)
        return self

    def predict_proba(self, X):
        return self.m.predict_proba(X)


MODELS = {"logit": LR, "cart": CART, "random_forest": RF, "lightgbm": LGBM, "xgboost": XGB, "mlp": MLP}


# ------------------------------------------------------------------ one run
def run(d, model_name, layer, regime, segment):
    """Tune on the val matches of `segment`, refit on train+val, predict its test matches."""
    cols = FEATURE_SETS[layer]
    Model = MODELS[model_name]
    tr, w_tr = regime_rows(d, ["train"], regime, segment)
    va = d[(d.split == "val") & (d.segment == segment)]
    te = d[(d.split == "test") & (d.segment == segment)]
    trva, w_trva = regime_rows(d, ["train", "val"], regime, segment)

    best, best_loss, best_extra = None, np.inf, {}
    for p in Model.grid:
        m = Model().fit(tr[cols], tr.y_int.to_numpy(), w_tr, p, va[cols], va.y_int.to_numpy())
        loss = log_loss(m.predict_proba(va[cols]), va.y_int)
        if loss < best_loss:
            best, best_loss = p, loss
            best_extra = {"n_rounds": m.n_rounds} if hasattr(m, "n_rounds") else {}
    final_p = {**best, **best_extra}
    m = Model().fit(trva[cols], trva.y_int.to_numpy(), w_trva, final_p)
    P = m.predict_proba(te[cols])
    pred = te[["match_api_id", "season", "y_int"]].copy()
    pred[["p_H", "p_D", "p_A"]] = P
    meta = {"model": model_name, "layer": layer, "regime": regime, "segment": segment, "params": str(final_p),
            "val_log_loss": best_loss, "n_train": len(trva), "eff_n_train": float(w_trva.sum()), "n_val": len(va)}
    return pred, meta, m


def baselines(d, segment):
    """Frequency (train+val shares of that type) and market predictions on the test matches of that type."""
    trva = d[d.split.isin(["train", "val"]) & (d.segment == segment)]
    te = d[(d.split == "test") & (d.segment == segment)]
    freq = np.bincount(trva.y_int, minlength=3) / len(trva)
    out = []
    for name, P in [("frequency", np.tile(freq, (len(te), 1))),
                    ("market", te[["p_mkt_H", "p_mkt_D", "p_mkt_A"]].to_numpy())]:
        pred = te[["match_api_id", "season", "y_int"]].copy()
        pred[["p_H", "p_D", "p_A"]] = P
        out.append((name, pred))
    return out
