"""Prediction and explainability on top of the trained Random Forest.

Explainability = grouped occlusion. For every group of related features we replace the
route's values with "typical" reference values and measure how many score points the
prediction moves. Positive = this factor lifts the score, negative = it pulls it down.
It is an approximation (like a simple SHAP), easy to explain in a viva.
"""
from __future__ import annotations

import json
import threading
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml.features import FEATURES, night_factor, risk_level  # noqa: F401  (re-exported)
from ml.train_model import META_PATH, MODEL_PATH, train

# group key -> (display label, features, kind)
GROUPS = {
    "street_lighting": ("Street lighting", ["street_lighting"], "route"),
    "accident_frequency": ("Accident history", ["accident_frequency"], "route"),
    "crime_rate": ("Crime index", ["crime_rate"], "route"),
    "pedestrian_density": ("Pedestrian presence", ["pedestrian_density"], "route"),
    "road_condition": ("Road condition", ["road_condition"], "route"),
    "intersection_density": ("Intersections", ["intersection_density"], "route"),
    "traffic": ("Traffic", ["traffic_density", "vehicle_density"], "route"),
    "weather": ("Weather and visibility", ["weather_condition", "rainfall", "visibility"], "route"),
    "time": ("Time of day", ["time_of_day", "day_of_week"], "route"),
    "trip_length": ("Trip length", ["distance", "route_duration"], "route"),
    "road_type": ("Road type", ["road_type"], "route"),
    "profile": ("Traveler profile and mode", ["traveler_type", "travel_mode"], "profile"),
}
NEUTRAL_POINTS = 0.5  # |effect| below this many points is shown as neutral

_lock = threading.Lock()
_model = None
_meta = None


def get_model():
    """Load the trained model, training it first if it is missing or built with another sklearn."""
    global _model, _meta
    if _model is not None:
        return _model, _meta
    with _lock:
        if _model is not None:
            return _model, _meta
        loaded = False
        if MODEL_PATH.exists() and META_PATH.exists():
            try:
                meta = json.loads(META_PATH.read_text())
                if meta.get("sklearn_version") == sklearn.__version__:
                    _model = joblib.load(MODEL_PATH)
                    _meta = meta
                    loaded = True
            except Exception:  # corrupt or incompatible file -> retrain below
                loaded = False
        if not loaded:
            _meta = train()
            _model = joblib.load(MODEL_PATH)
        try:
            _model.named_steps["model"].set_params(n_jobs=1)  # small batches: threads only add overhead
        except Exception:
            pass
    return _model, _meta


def model_info() -> dict:
    _, meta = get_model()
    return {k: meta[k] for k in ("model", "n_estimators", "trained_rows", "test_rows", "metrics_holdout",
                                 "model_comparison", "feature_importance", "permutation_importance",
                                 "thresholds", "data_note")}


def _frame(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[FEATURES]


def score(rows: List[dict]) -> np.ndarray:
    """Predicted safety scores (0-100) for a list of feature dicts."""
    model, _ = get_model()
    return np.clip(model.predict(_frame(rows)), 0, 100)


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #
def _reference_row(row: dict, meta: dict) -> dict:
    ref = dict(meta["references"]["numeric_mean"])
    ref.update(meta["references"]["categorical_mode"])
    ref["time_of_day"] = 12.0  # a typical daytime hour
    ref["day_of_week"] = 2
    ref["weather_condition"] = "clear"
    ref["rainfall"] = 0.0
    # "Clear" visibility for the row's own hour and lighting (night is naturally darker),
    # so the weather factor is not confused with time of day.
    night = float(night_factor(row["time_of_day"]))
    ref["visibility"] = 10.0 * (1 - night) + (4.0 + 4.0 * row["street_lighting"] / 100.0) * night
    ref["traveler_type"] = "general"
    med = meta["references"]["by_mode_median"].get(row["travel_mode"])
    if med:
        ref["distance"] = med["distance"]
        ref["route_duration"] = med["route_duration"]
    return ref


def _phrase(group: str, row: dict, effect: float) -> str:
    v = row
    if group == "street_lighting":
        x = v["street_lighting"]
        return "Good street lighting" if x >= 70 else "Poor street lighting" if x <= 45 else "Moderate street lighting"
    if group == "accident_frequency":
        x = v["accident_frequency"]
        return "Low accident frequency" if x <= 2.5 else "High accident frequency" if x >= 5.5 else "Moderate accident frequency"
    if group == "crime_rate":
        x = v["crime_rate"]
        return "Low crime index" if x <= 28 else "High crime index" if x >= 48 else "Moderate crime index"
    if group == "pedestrian_density":
        x = v["pedestrian_density"]
        return "High pedestrian presence" if x >= 55 else "Low pedestrian presence" if x <= 28 else "Moderate pedestrian presence"
    if group == "road_condition":
        x = v["road_condition"]
        return "Good road condition" if x >= 68 else "Poor road condition" if x <= 48 else "Average road condition"
    if group == "intersection_density":
        x = v["intersection_density"]
        return "Few intersections" if x <= 6 else "High intersection density" if x >= 12 else "Moderate intersection density"
    if group == "traffic":
        x = v["traffic_density"]
        return "Light traffic" if x <= 30 else "Heavy traffic" if x >= 62 else "Moderate traffic"
    if group == "weather":
        w = v["weather_condition"]
        if w in ("light_rain", "heavy_rain"):
            return "Rain and reduced visibility" if effect < 0 else "Rain (limited effect)"
        if w == "fog":
            return "Fog and low visibility"
        return "Reduced visibility" if effect < 0 else "Clear conditions and good visibility"
    if group == "time":
        n = float(night_factor(v["time_of_day"]))
        return "Night-time travel" if n >= 0.8 else "Dusk or dawn travel" if n >= 0.2 else "Daytime travel"
    if group == "trip_length":
        mins = round(v["route_duration"])
        return f"Longer exposure time ({mins} min)" if effect < 0 else f"Short trip ({mins} min)"
    if group == "road_type":
        return "Road type: " + str(v["road_type"]).replace("_", " ")
    return "Traveler profile and mode"


def explain(rows: List[dict]) -> List[List[dict]]:
    """Grouped-occlusion factors for each row (sorted by absolute effect)."""
    model, meta = get_model()
    group_keys = list(GROUPS)
    frames = []
    for row in rows:
        ref = _reference_row(row, meta)
        frames.append(dict(row))
        for g in group_keys:
            alt = dict(row)
            for f in GROUPS[g][1]:
                alt[f] = ref[f]
            frames.append(alt)
    preds = np.clip(model.predict(_frame(frames)), 0, 100)
    per_row = len(group_keys) + 1
    out = []
    for i, row in enumerate(rows):
        base = preds[i * per_row]
        factors = []
        for j, g in enumerate(group_keys, start=1):
            effect = float(base - preds[i * per_row + j])
            label, _, kind = GROUPS[g]
            impact = "neutral" if abs(effect) < NEUTRAL_POINTS else ("positive" if effect > 0 else "negative")
            factors.append({"key": g, "label": label, "text": _phrase(g, row, effect),
                            "effect": round(effect, 2), "impact": impact, "kind": kind})
        factors.sort(key=lambda f: -abs(f["effect"]))
        out.append(factors)
    return out


# Groups used when explaining a CHANGE between two situations (exact Shapley values).
ATTR_GROUPS = {
    "time": ("Time of day", ["time_of_day", "day_of_week"]),
    "visibility": ("Visibility", ["visibility"]),
    "weather": ("Weather", ["weather_condition", "rainfall"]),
    "traffic": ("Traffic", ["traffic_density", "vehicle_density"]),
    "pedestrian_density": ("Pedestrian presence", ["pedestrian_density"]),
    "trip_length": ("Trip length", ["distance", "route_duration"]),
    "profile": ("Traveler profile and mode", ["traveler_type", "travel_mode"]),
    "other": ("Other route features", ["street_lighting", "accident_frequency", "crime_rate", "road_condition",
                                       "intersection_density", "road_type"]),
}


def attribute_change(base_row: dict, new_row: dict) -> List[dict]:
    """Which changed condition moved the score, and by how much?

    Exact Shapley values over the groups that actually changed (at most 8 groups, so the
    128 coalitions are one small batch). The effects add up to the total change.
    """
    from itertools import combinations
    from math import factorial

    model, _ = get_model()
    changed = [(g, lab, feats) for g, (lab, feats) in ATTR_GROUPS.items()
               if any(base_row[f] != new_row[f] for f in feats)]
    n = len(changed)
    if n == 0:
        return []
    subsets = [c for r in range(n + 1) for c in combinations(range(n), r)]
    frames = []
    for sub in subsets:
        row = dict(base_row)
        for i in sub:
            for f in changed[i][2]:
                row[f] = new_row[f]
        frames.append(row)
    preds = dict(zip(subsets, np.clip(model.predict(_frame(frames)), 0, 100)))
    out = []
    for i, (g, lab, _) in enumerate(changed):
        phi = 0.0
        others = [j for j in range(n) if j != i]
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for sub in combinations(others, r):
                phi += w * (preds[tuple(sorted(sub + (i,)))] - preds[sub])
        out.append({"key": g, "label": lab, "effect": round(float(phi), 2)})
    out.sort(key=lambda d: -abs(d["effect"]))
    return out
