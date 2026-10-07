"""Generate the SYNTHETIC SafeRoute dataset.

Run:  python -m ml.generate_dataset
The output (data/saferoute_dataset.csv) is demo data. It is NOT real safety data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import (FEATURES, TARGET, TRAVEL_MODES, TRAVELER_TYPES, WEATHER, dynamic_context,
                         estimate_duration, sample_static, synthetic_safety_score)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "saferoute_dataset.csv"


def generate(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    static = sample_static(rng, n)
    hour = rng.uniform(0, 24, n)
    dow = rng.integers(0, 7, n)
    weather = rng.choice(WEATHER, size=n, p=[0.50, 0.22, 0.15, 0.06, 0.07])
    traveler = rng.choice(TRAVELER_TYPES, size=n)
    mode = rng.choice(TRAVEL_MODES, size=n)

    dyn = dynamic_context(static, hour, dow, weather, noise_rng=rng)
    distance = np.clip(rng.lognormal(1.0, 0.7, n), 0.3, 25)
    duration = estimate_duration(distance, mode, dyn["traffic_density"])

    df = pd.DataFrame({
        "traffic_density": dyn["traffic_density"],
        "accident_frequency": static["accident_frequency"],
        "street_lighting": static["street_lighting"],
        "pedestrian_density": dyn["pedestrian_density"],
        "road_condition": static["road_condition"],
        "crime_rate": static["crime_rate"],
        "rainfall": dyn["rainfall"],
        "visibility": dyn["visibility"],
        "intersection_density": static["intersection_density"],
        "vehicle_density": dyn["vehicle_density"],
        "time_of_day": hour,
        "day_of_week": dow,
        "distance": distance,
        "route_duration": duration,
        "weather_condition": weather,
        "road_type": static["road_type"],
        "traveler_type": traveler,
        "travel_mode": mode,
    })
    df[TARGET] = synthetic_safety_score(df, rng)
    df = df[FEATURES + [TARGET]].round(3)
    return df


def main() -> pd.DataFrame:
    df = generate()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(df):,} synthetic rows to {DATA_PATH}")
    return df


if __name__ == "__main__":
    main()
