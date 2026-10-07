"""Turns a plan (origin, destination, time, profile...) into scored, explained routes."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from backend.models.schemas import PredictRiskRequest, RouteRequest, Scenario
from backend.services import route_service
from backend.utils.config import DISCLAIMER, SCORE_NOTE
from ml import predict as ml
from ml.features import FEATURES, dynamic_context, estimate_duration, risk_level

# Minutes of extra travel that cost one point of "recommendation utility" (see README)
TIME_PENALTY = {"walking": 0.35, "cycling": 0.40, "public_transport": 0.50, "car": 0.60}
LOCATION_FACTORS = {"street_lighting", "accident_frequency", "crime_rate", "pedestrian_density",
                    "road_condition", "intersection_density", "traffic", "road_type"}

LABELS = {
    "mode": {"walking": "Walking", "cycling": "Cycling", "public_transport": "Public Transport", "car": "Car"},
    "traveler": {"student": "Student", "general": "General", "senior_citizen": "Senior Citizen",
                 "cyclist": "Cyclist", "pedestrian": "Pedestrian"},
    "weather": {"clear": "Clear", "cloudy": "Cloudy", "light_rain": "Light rain", "heavy_rain": "Heavy rain",
                "fog": "Fog"},
}
PRIORITIES = {
    "student": "street lighting, pedestrian activity, accident history and night-time conditions",
    "senior_citizen": "road condition, traffic, intersection density and route complexity",
    "cyclist": "traffic, road condition and cycling suitability",
    "pedestrian": "pedestrian density, lighting, crossings and traffic",
    "general": "a balanced mix of lighting, traffic, accident history and road condition",
}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt_hour(hour: float) -> str:
    h = int(hour) % 24
    m = int(round((hour - int(hour)) * 60))
    if m == 60:
        h, m = (h + 1) % 24, 0
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}:{m:02d} {suffix}"


@dataclass(frozen=True)
class Context:
    hour: float
    day: date
    weather: str
    traffic_mult: float
    mode: str
    traveler: str
    origin: str
    destination: str

    @property
    def dow(self) -> int:
        return self.day.weekday()


def _parse_time(t: str) -> float:
    h, m = t.split(":")
    return int(h) + int(m) / 60.0


def make_context(req: RouteRequest, scenario: Optional[Scenario] = None) -> Context:
    ctx = Context(hour=_parse_time(req.travel_time), day=req.travel_date or date.today(), weather=req.weather,
                  traffic_mult=1.0, mode=req.travel_mode, traveler=req.traveler_type,
                  origin=req.origin, destination=req.destination)
    if scenario is not None:
        if scenario.travel_time:
            ctx = replace(ctx, hour=_parse_time(scenario.travel_time))
        if scenario.travel_date:
            ctx = replace(ctx, day=scenario.travel_date)
        if scenario.weather:
            ctx = replace(ctx, weather=scenario.weather)
        if scenario.travel_mode:
            ctx = replace(ctx, mode=scenario.travel_mode)
        if scenario.traveler_type:
            ctx = replace(ctx, traveler=scenario.traveler_type)
        ctx = replace(ctx, traffic_mult=max(0.2, 1 + scenario.traffic_change_pct / 100.0))
    return ctx


def context_summary(ctx: Context) -> dict:
    return {"origin": ctx.origin, "destination": ctx.destination, "time": fmt_hour(ctx.hour),
            "hour": round(ctx.hour, 2), "date": ctx.day.isoformat(), "day": DAYS[ctx.dow],
            "weather": LABELS["weather"][ctx.weather], "weather_key": ctx.weather,
            "traffic_change_pct": round((ctx.traffic_mult - 1) * 100),
            "mode": LABELS["mode"][ctx.mode], "mode_key": ctx.mode,
            "traveler": LABELS["traveler"][ctx.traveler], "traveler_key": ctx.traveler,
            "priorities": PRIORITIES[ctx.traveler]}


def feature_row(static: dict, ctx: Context, distance_km: float) -> dict:
    dyn = dynamic_context(static, ctx.hour, ctx.dow, ctx.weather, ctx.traffic_mult)
    dyn = {k: float(np.ravel(v)[0]) for k, v in dyn.items()}
    duration = float(np.ravel(estimate_duration(distance_km, ctx.mode, dyn["traffic_density"]))[0])
    return {
        "traffic_density": dyn["traffic_density"], "accident_frequency": static["accident_frequency"],
        "street_lighting": static["street_lighting"], "pedestrian_density": dyn["pedestrian_density"],
        "road_condition": static["road_condition"], "crime_rate": static["crime_rate"],
        "rainfall": dyn["rainfall"], "visibility": dyn["visibility"],
        "intersection_density": static["intersection_density"], "vehicle_density": dyn["vehicle_density"],
        "time_of_day": ctx.hour, "day_of_week": ctx.dow, "distance": distance_km, "route_duration": duration,
        "weather_condition": ctx.weather, "road_type": static["road_type"],
        "traveler_type": ctx.traveler, "travel_mode": ctx.mode,
    }


def _rounded(row: dict) -> dict:
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}


def _main_factor(factors: List[dict]) -> str:
    for f in factors:
        if f["kind"] == "route" and f["key"] in LOCATION_FACTORS and f["impact"] == "negative":
            return f["text"]
    for f in factors:
        if f["kind"] == "route" and f["impact"] == "negative":
            return f["text"]
    return "No single dominant factor"


def evaluate(req: RouteRequest, scenario: Optional[Scenario] = None, geometry: bool = True) -> dict:
    """Score and explain every route alternative for a plan (optionally under a scenario)."""
    ctx = make_context(req, scenario)
    routes, meta = route_service.base_routes(req.origin, req.destination)

    route_rows = [feature_row(r["static"], ctx, r["distance_km"]) for r in routes]
    junction_rows, junction_ref = [], []
    for r in routes:
        for j in r["junctions"]:
            junction_rows.append(feature_row(j["static"], ctx, 0.3))
            junction_ref.append((r["id"], j))

    scores = ml.score(route_rows + junction_rows)
    explanations = ml.explain(route_rows + junction_rows)
    r_scores, j_scores = scores[:len(routes)], scores[len(routes):]
    r_expl, j_expl = explanations[:len(routes)], explanations[len(routes):]

    fastest_minutes = min(row["route_duration"] for row in route_rows)
    lam = TIME_PENALTY[ctx.mode]
    out, all_junctions = [], []
    for r, row, sc, ex in zip(routes, route_rows, r_scores, r_expl):
        factors = [f for f in ex if f["kind"] == "route"]
        pos = [f["text"] for f in factors if f["impact"] == "positive"][:3]
        neg = [f["text"] for f in factors if f["impact"] == "negative"][:3]
        utility = float(sc) - lam * (row["route_duration"] - fastest_minutes)
        out.append({"id": r["id"], "name": r["name"], "label": r["label"], "distance_km": r["distance_km"],
                    "duration_min": int(round(row["route_duration"])), "safety_score": round(float(sc), 1),
                    "risk_level": risk_level(sc), "recommended": False, "utility": round(utility, 1),
                    "features": _rounded(row), "factors": factors, "strengths": pos, "concerns": neg,
                    "coords": [[round(a, 5), round(b, 5)] for a, b in r["coords"]] if geometry else [],
                    "junctions": []})
    for (rid, j), sc, ex in zip(junction_ref, j_scores, j_expl):
        item = {"name": j["name"], "route_id": rid, "lat": round(j["lat"], 5), "lon": round(j["lon"], 5),
                "safety_score": round(float(sc), 1), "risk_level": risk_level(sc),
                "main_factor": _main_factor(ex)}
        all_junctions.append(item)
        if geometry:
            next(o for o in out if o["id"] == rid)["junctions"].append(item)

    best = max(out, key=lambda o: o["utility"])
    best["recommended"] = True
    fastest = min(out, key=lambda o: o["duration_min"])
    safest = max(out, key=lambda o: o["safety_score"])
    if best["id"] == fastest["id"]:
        basis = (f"{best['name']} is the fastest option and also has the best balance of predicted safety and time.")
    else:
        extra = best["duration_min"] - fastest["duration_min"]
        gain = best["safety_score"] - fastest["safety_score"]
        basis = (f"{best['name']} takes about {extra} min longer than {fastest['name']} but its predicted "
                 f"safety score is {gain:+.0f} points. Recommendation score = safety score minus "
                 f"{lam} point per extra minute.")
    hotspots = sorted([j for j in all_junctions if j["safety_score"] < 70], key=lambda j: j["safety_score"])[:3]
    return {"context": {**context_summary(ctx), "origin_known": meta["origin_known"],
                        "destination_known": meta["destination_known"], "straight_km": meta["straight_km"],
                        "origin_xy": list(meta["origin_xy"]), "destination_xy": list(meta["destination_xy"])},
            "routes": out, "recommended_id": best["id"], "fastest_id": fastest["id"], "safest_id": safest["id"],
            "recommendation_basis": basis, "hotspots": hotspots, "demo_mode": True, "disclaimer": DISCLAIMER}


def time_profile(req: RouteRequest) -> dict:
    """Predicted score of each route for every hour of the day (same day, weather, profile)."""
    base_ctx = make_context(req)
    routes, _ = route_service.base_routes(req.origin, req.destination)
    rows, index = [], []
    for h in range(24):
        ctx = replace(base_ctx, hour=float(h))
        for r in routes:
            rows.append(feature_row(r["static"], ctx, r["distance_km"]))
            index.append((r["id"], h))
    scores = ml.score(rows)
    series: Dict[str, List[float]] = {r["id"]: [0.0] * 24 for r in routes}
    for (rid, h), s in zip(index, scores):
        series[rid][h] = round(float(s), 1)
    return {"hours": list(range(24)), "scores": series, "context": context_summary(base_ctx)}


def predict_risk(body: PredictRiskRequest) -> dict:
    row = body.model_dump()
    score = float(ml.score([row])[0])
    factors = [f for f in ml.explain([row])[0] if f["kind"] == "route"]
    return {"safety_score": round(score, 1), "risk_level": risk_level(score), "factors": factors,
            "note": SCORE_NOTE, "disclaimer": DISCLAIMER}
