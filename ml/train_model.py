"""Train the SafeRoute Random Forest on the synthetic dataset.

Run:  python -m ml.train_model
Outputs (in ml/model/):  saferoute_rf.joblib, model_meta.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import (CATEGORICAL_FEATURES, FEATURES, HIGH_RISK_MAX, LOW_RISK_MIN, NUMERIC_FEATURES,
                         TARGET, TRAVEL_MODES)
from ml.generate_dataset import DATA_PATH, main as generate_dataset

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "saferoute_rf.joblib"
META_PATH = MODEL_DIR / "model_meta.json"


def _pipeline(estimator, scale: bool = False) -> Pipeline:
    num = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", pre), ("model", estimator)])


def _metrics(pipe, X, y) -> dict:
    pred = np.clip(pipe.predict(X), 0, 100)
    return {
        "r2": round(float(r2_score(y, pred)), 4),
        "mae": round(float(mean_absolute_error(y, pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
    }


def _aggregate_importance(pipe) -> dict:
    """Sum Random Forest importances of one-hot columns back to the original feature."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    out = {f: 0.0 for f in FEATURES}
    for name, value in zip(names, imp):
        kind, rest = name.split("__", 1)
        if kind == "num":
            out[rest] += float(value)
        else:
            col = next(c for c in CATEGORICAL_FEATURES if rest.startswith(c + "_"))
            out[col] += float(value)
    return out


def train(n_estimators: int = 100, seed: int = 42) -> dict:
    if not DATA_PATH.exists():
        generate_dataset()
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)

    t0 = time.time()
    rf = _pipeline(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=4, max_features=0.8,
                                         n_jobs=-1, random_state=seed))
    rf.fit(X_tr, y_tr)
    train_seconds = round(time.time() - t0, 1)

    # Baselines for the viva: a linear model and gradient boosting.
    ridge = _pipeline(Ridge(alpha=1.0), scale=True).fit(X_tr, y_tr)
    hgb = _pipeline(HistGradientBoostingRegressor(random_state=seed)).fit(X_tr, y_tr)
    comparison = {
        "Random Forest (used)": _metrics(rf, X_te, y_te),
        "Gradient Boosting": _metrics(hgb, X_te, y_te),
        "Linear (Ridge) baseline": _metrics(ridge, X_te, y_te),
    }

    sub = X_te.sample(min(2500, len(X_te)), random_state=seed)
    perm = permutation_importance(rf, sub, y_te.loc[sub.index], n_repeats=3, random_state=seed, n_jobs=1)
    permutation = {f: round(float(max(v, 0)), 5) for f, v in zip(FEATURES, perm.importances_mean)}
    total = sum(permutation.values()) or 1.0
    permutation = {k: round(v / total, 5) for k, v in permutation.items()}

    impurity = _aggregate_importance(rf)

    # Reference values used by the explainability step ("typical" conditions).
    references = {
        "numeric_mean": {f: round(float(X_tr[f].mean()), 3) for f in NUMERIC_FEATURES},
        "categorical_mode": {f: str(X_tr[f].mode().iloc[0]) for f in CATEGORICAL_FEATURES},
        "by_mode_median": {
            m: {"distance": round(float(X_tr.loc[X_tr.travel_mode == m, "distance"].median()), 3),
                "route_duration": round(float(X_tr.loc[X_tr.travel_mode == m, "route_duration"].median()), 3)}
            for m in TRAVEL_MODES
        },
    }

    # Refit on all data for the deployed model (the metrics above are from the held-out split).
    final = _pipeline(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=4, max_features=0.8,
                                            n_jobs=-1, random_state=seed)).fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH, compress=3)

    meta = {
        "model": "RandomForestRegressor",
        "n_estimators": n_estimators,
        "trained_rows": int(len(df)),
        "test_rows": int(len(X_te)),
        "train_seconds": train_seconds,
        "sklearn_version": sklearn.__version__,
        "metrics_holdout": comparison["Random Forest (used)"],
        "model_comparison": comparison,
        "feature_importance": {k: round(v, 5) for k, v in impurity.items()},
        "permutation_importance": permutation,
        "references": references,
        "thresholds": {"low_risk_min": LOW_RISK_MIN, "high_risk_max": HIGH_RISK_MAX},
        "data_note": "Synthetic demo data. Scores are NOT real-world safety measurements.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"Trained in {train_seconds}s | R2={meta['metrics_holdout']['r2']} "
          f"MAE={meta['metrics_holdout']['mae']} | model size {size_mb:.1f} MB")
    return meta


if __name__ == "__main__":
    train()
