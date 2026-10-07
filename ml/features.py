"""Shared feature definitions and the synthetic "ground truth" for SafeRoute AI.

IMPORTANT: everything in this file produces SYNTHETIC / DEMO data. The numbers are
generated from hand-written rules plus noise so the ML pipeline can be demonstrated.
They do not describe real streets, real crime, or real accidents.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Feature lists
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES = [
    "traffic_density",      # 0-100 index
    "accident_frequency",   # 0-10 (incidents per year per km, synthetic)
    "street_lighting",      # 0-100 index
    "pedestrian_density",   # 0-100 index
    "road_condition",       # 0-100 index (100 = excellent)
    "crime_rate",           # 0-100 synthetic index
    "rainfall",             # mm per hour
    "visibility",           # km (0.2 - 10)
    "intersection_density", # intersections per km
    "vehicle_density",      # 0-100 index
    "time_of_day",          # hour, 0-23.99
    "day_of_week",          # 0 = Monday ... 6 = Sunday
    "distance",             # km
    "route_duration",       # minutes
]
CATEGORICAL_FEATURES = ["weather_condition", "road_type", "traveler_type", "travel_mode"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "safety_score"

WEATHER = ["clear", "cloudy", "light_rain", "heavy_rain", "fog"]
ROAD_TYPES = ["arterial", "commercial", "residential", "service_lane", "highway"]
TRAVELER_TYPES = ["student", "general", "senior_citizen", "cyclist", "pedestrian"]
TRAVEL_MODES = ["walking", "cycling", "public_transport", "car"]

# Risk bands (documented in the README and shown in the UI)
LOW_RISK_MIN = 70   # 70-100 -> Low risk
HIGH_RISK_MAX = 40  # 0-39   -> High risk, 40-69 -> Medium


def risk_level(score: float) -> str:
    if score >= LOW_RISK_MIN:
        return "Low"
    if score >= HIGH_RISK_MAX:
        return "Medium"
    return "High"


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def night_factor(hour):
    """0 = full daylight, 1 = full night (smooth dusk / dawn ramps)."""
    h = np.asarray(hour, dtype=float) % 24
    return np.interp(h, [0, 5, 7.5, 17.5, 20, 24], [1, 1, 0, 0, 1, 1])


def _traffic_profile(hour, dow):
    h = np.asarray(hour, dtype=float) % 24
    m = np.interp(h, [0, 5, 7, 9, 11, 16, 18, 20, 22, 24],
                  [0.30, 0.30, 0.90, 1.40, 1.00, 1.10, 1.45, 1.00, 0.60, 0.35])
    return np.where(np.asarray(dow) >= 5, m * 0.85, m)


def _pedestrian_profile(hour, dow):
    h = np.asarray(hour, dtype=float) % 24
    m = np.interp(h, [0, 5, 7, 10, 16, 18, 21, 23, 24],
                  [0.15, 0.15, 0.60, 0.90, 1.00, 1.25, 0.80, 0.30, 0.15])
    return np.where(np.asarray(dow) >= 5, m * 1.1, m)


WEATHER_RAINFALL = {"clear": 0.0, "cloudy": 0.0, "light_rain": 3.0, "heavy_rain": 15.0, "fog": 0.0}
WEATHER_VIS_MULT = {"clear": 1.0, "cloudy": 0.95, "light_rain": 0.7, "heavy_rain": 0.4, "fog": 0.2}
WEATHER_PED_MULT = {"clear": 1.0, "cloudy": 1.0, "light_rain": 0.8, "heavy_rain": 0.55, "fog": 0.85}

# --------------------------------------------------------------------------- #
# Static (infrastructure) attributes by road type
# --------------------------------------------------------------------------- #
#                  lighting, road_cond, accidents, crime, intersections, base_traffic, base_ped
_ROAD_PROFILE = {
    "arterial":     (75, 70, 5.0, 35, 10, 65, 45),
    "commercial":   (80, 65, 4.0, 40, 14, 60, 75),
    "residential":  (55, 55, 2.0, 30, 8, 25, 40),
    "service_lane": (35, 40, 1.5, 50, 4, 12, 15),
    "highway":      (65, 75, 6.0, 20, 2, 80, 8),
}
_ROAD_PROBS = [0.30, 0.20, 0.30, 0.10, 0.10]


def sample_static(rng: np.random.Generator, n: int, road_types=None, quality_shift=None) -> dict:
    """Sample infrastructure attributes (do not change with time or weather).

    quality_shift (points) lifts or lowers lighting/road condition and nudges
    accidents/crime, so individual routes differ from the road-type average.
    """
    if road_types is None:
        road_types = rng.choice(ROAD_TYPES, size=n, p=_ROAD_PROBS)
    road_types = np.asarray(road_types)
    if quality_shift is None:
        quality_shift = rng.normal(0, 8, size=n)
    quality_shift = np.broadcast_to(np.asarray(quality_shift, dtype=float), (n,))

    prof = np.array([_ROAD_PROFILE[r] for r in road_types], dtype=float)
    light = np.clip(prof[:, 0] + quality_shift + rng.normal(0, 12, n), 5, 100)
    road_cond = np.clip(prof[:, 1] + 0.8 * quality_shift + rng.normal(0, 12, n), 5, 100)
    acc = np.clip(rng.gamma(2.5, prof[:, 2] / 2.5, n) - 0.04 * quality_shift, 0, 10)
    crime = np.clip(prof[:, 3] + (60 - light) * 0.2 - 0.3 * quality_shift + rng.normal(0, 12, n), 2, 100)
    inter = np.clip(prof[:, 4] + rng.normal(0, 2.5, n), 0.5, 25)
    base_traffic = np.clip(prof[:, 5] + rng.normal(0, 10, n), 3, 100)
    base_ped = np.clip(prof[:, 6] + rng.normal(0, 10, n), 2, 100)
    return {
        "road_type": road_types,
        "street_lighting": light,
        "road_condition": road_cond,
        "accident_frequency": acc,
        "crime_rate": crime,
        "intersection_density": inter,
        "base_traffic": base_traffic,
        "base_pedestrian": base_ped,
    }


def dynamic_context(static: dict, hour, dow, weather, traffic_multiplier=1.0, noise_rng=None) -> dict:
    """Turn static attributes + time + weather into the live features.

    Works for scalars and numpy arrays. `weather` is a name or an array of names.
    """
    hour = np.asarray(hour, dtype=float)
    dow = np.asarray(dow)
    weather_arr = np.atleast_1d(np.asarray(weather))
    night = night_factor(hour)

    traffic = np.asarray(static["base_traffic"], dtype=float) * _traffic_profile(hour, dow) * traffic_multiplier
    ped = np.asarray(static["base_pedestrian"], dtype=float) * _pedestrian_profile(hour, dow)
    ped = ped * np.array([WEATHER_PED_MULT[w] for w in weather_arr]).reshape(ped.shape or (1,))[...]
    vis_night = 4.0 + 4.0 * np.asarray(static["street_lighting"], dtype=float) / 100.0
    vis = 10.0 * (1 - night) + vis_night * night
    vis = vis * np.array([WEATHER_VIS_MULT[w] for w in weather_arr]).reshape(vis.shape or (1,))[...]
    rain = np.array([WEATHER_RAINFALL[w] for w in weather_arr], dtype=float).reshape(vis.shape or (1,))[...]
    veh_share = {"arterial": 1.0, "commercial": 0.9, "residential": 0.7, "service_lane": 0.5, "highway": 1.25}
    share = np.array([veh_share[r] for r in np.atleast_1d(static["road_type"])]).reshape(traffic.shape or (1,))[...]
    vehicle = traffic * share

    if noise_rng is not None:
        shape = np.shape(traffic)
        traffic = traffic + noise_rng.normal(0, 4, shape)
        ped = ped + noise_rng.normal(0, 4, shape)
        vehicle = vehicle + noise_rng.normal(0, 4, shape)
        rain = np.where(rain > 0, np.clip(rain * noise_rng.uniform(0.4, 1.6, shape), 0.2, 40), 0)
        vis = vis * noise_rng.uniform(0.9, 1.1, shape)

    return {
        "traffic_density": np.clip(traffic, 0, 100),
        "pedestrian_density": np.clip(ped, 0, 100),
        "vehicle_density": np.clip(vehicle, 0, 100),
        "visibility": np.clip(vis, 0.2, 10),
        "rainfall": np.clip(rain, 0, 40),
    }


# --------------------------------------------------------------------------- #
# Trip length
# --------------------------------------------------------------------------- #
SPEED_KMH = {"walking": 4.8, "cycling": 14.0, "public_transport": 20.0, "car": 30.0}


def estimate_duration(distance_km, mode, traffic_density):
    """Minutes for the trip. Motor modes slow down with traffic."""
    distance_km = np.asarray(distance_km, dtype=float)
    traffic = np.asarray(traffic_density, dtype=float)
    mode = np.asarray(mode)
    speed = np.array([SPEED_KMH[m] for m in np.atleast_1d(mode)], dtype=float).reshape(np.shape(distance_km) or (1,))
    motor = np.isin(mode, ["car", "public_transport"])
    speed = np.where(motor, speed * (1 - 0.5 * traffic / 100.0), speed)
    minutes = distance_km / np.maximum(speed, 1.0) * 60.0
    minutes = minutes + np.where(mode == "public_transport", 6.0, 0.0)
    return minutes


# --------------------------------------------------------------------------- #
# Synthetic ground truth: how risk is generated for the demo dataset
# --------------------------------------------------------------------------- #
# term order: light, pedestrian, accident, crime, traffic, road, intersections, weather, road type
_TRAVELER_WEIGHTS = {
    "student":        (20, 12, 12, 15, 7, 4, 5, 8, 3),
    "general":        (14, 9, 13, 11, 12, 9, 8, 9, 3),
    "senior_citizen": (11, 8, 12, 9, 14, 17, 15, 10, 5),
    "cyclist":        (11, 3, 13, 6, 22, 19, 10, 12, 6),
    "pedestrian":     (15, 13, 11, 10, 13, 8, 13, 8, 4),
}
_MODE_MULT = {
    "walking":          (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    "cycling":          (1.0, 0.6, 1.0, 0.7, 1.3, 1.3, 1.0, 1.2, 1.0),
    "public_transport": (0.5, 0.5, 0.7, 0.6, 0.5, 0.4, 0.4, 0.6, 0.3),
    "car":              (0.3, 0.2, 1.2, 0.3, 1.2, 1.0, 1.0, 1.2, 0.0),
}
_ROAD_TYPE_EXTRA = {"highway": 1.0, "arterial": 0.4, "commercial": 0.2, "residential": 0.0, "service_lane": 0.3}
RISK_SCALE = 1.75


def synthetic_safety_score(df, rng=None):
    """Hand-written rule that turns conditions into a 0-100 safety score (synthetic)."""
    night = night_factor(df["time_of_day"].to_numpy())
    light = (100 - df["street_lighting"].to_numpy()) / 100 * (0.45 + 0.95 * night)
    ped = (100 - df["pedestrian_density"].to_numpy()) / 100 * (0.30 + 0.60 * night)
    acc = df["accident_frequency"].to_numpy() / 10
    crime = df["crime_rate"].to_numpy() / 100 * (0.60 + 0.60 * night)
    traffic = (0.6 * df["traffic_density"].to_numpy() + 0.4 * df["vehicle_density"].to_numpy()) / 100
    road = (100 - df["road_condition"].to_numpy()) / 100
    inter = np.clip(df["intersection_density"].to_numpy() / 20, 0, 1.3)
    weather = (0.55 * np.minimum(df["rainfall"].to_numpy() / 25, 1)
               + 0.45 * (1 - np.minimum(df["visibility"].to_numpy(), 10) / 10))
    rt = np.array([_ROAD_TYPE_EXTRA[r] for r in df["road_type"]])
    terms = np.column_stack([light, ped, acc, crime, traffic, road, inter, weather, rt])

    w = np.array([_TRAVELER_WEIGHTS[t] for t in df["traveler_type"]], dtype=float)
    m = np.array([_MODE_MULT[x] for x in df["travel_mode"]], dtype=float)
    base = (terms * w * m).sum(axis=1) * RISK_SCALE
    exposure = 1 + 0.10 * np.clip(df["route_duration"].to_numpy() / 60, 0, 1.5)
    risk = base * exposure
    score = 100 - risk
    if rng is not None:
        score = score + rng.normal(0, 2.0, len(score))
    return np.clip(score, 3, 99)
