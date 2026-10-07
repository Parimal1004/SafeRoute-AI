"""What-If analysis: re-run the whole pipeline with changed conditions and compare."""
from __future__ import annotations

from typing import List

from backend.models.schemas import RouteRequest, Scenario
from backend.services import risk_service as rs
from backend.utils.config import DISCLAIMER
from ml import predict as ml


def describe_changes(req: RouteRequest, scenario: Scenario) -> List[str]:
    base, new = rs.make_context(req), rs.make_context(req, scenario)
    out = []
    if new.hour != base.hour:
        out.append(f"Time: {rs.fmt_hour(base.hour)} → {rs.fmt_hour(new.hour)}")
    if new.day != base.day:
        out.append(f"Day: {rs.DAYS[base.dow]} → {rs.DAYS[new.dow]}")
    if new.weather != base.weather:
        out.append(f"Weather: {rs.LABELS['weather'][base.weather]} → {rs.LABELS['weather'][new.weather]}")
    if scenario.traffic_change_pct:
        out.append(f"Traffic: {scenario.traffic_change_pct:+.0f}%")
    if new.mode != base.mode:
        out.append(f"Mode: {rs.LABELS['mode'][base.mode]} → {rs.LABELS['mode'][new.mode]}")
    if new.traveler != base.traveler:
        out.append(f"Traveler: {rs.LABELS['traveler'][base.traveler]} → {rs.LABELS['traveler'][new.traveler]}")
    return out


def run(req: RouteRequest, scenario: Scenario) -> dict:
    """Returns the comparison (without the agent text, which the route layer adds)."""
    before = rs.evaluate(req, geometry=False)
    after = rs.evaluate(req, scenario, geometry=False)
    after_by_id = {r["id"]: r for r in after["routes"]}
    routes = []
    for b in before["routes"]:
        a = after_by_id[b["id"]]
        drivers = ml.attribute_change(b["features"], a["features"])
        routes.append({"id": b["id"], "name": b["name"], "baseline_score": b["safety_score"],
                       "scenario_score": a["safety_score"], "delta": round(a["safety_score"] - b["safety_score"], 1),
                       "baseline_risk": b["risk_level"], "scenario_risk": a["risk_level"],
                       "baseline_duration": b["duration_min"], "scenario_duration": a["duration_min"],
                       "drivers": drivers})
    changes = describe_changes(req, scenario)
    return {"changes": changes, "baseline": before, "scenario": after, "routes": routes,
            "recommended_before": before["recommended_id"], "recommended_after": after["recommended_id"],
            "recommendation_changed": before["recommended_id"] != after["recommended_id"],
            "disclaimer": DISCLAIMER}
