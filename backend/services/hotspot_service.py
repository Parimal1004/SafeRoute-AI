"""Demo-city safety hotspots.

A fixed set of 12 SYNTHETIC junctions on an abstract 8 km x 6 km grid (not real places).
Their predicted score changes with time of day, weather and traveler profile.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from backend.models.schemas import RouteRequest
from backend.services import risk_service as rs
from ml import predict as ml
from ml.features import risk_level, sample_static

NAMES = "ABCDEFGHIJKL"


@lru_cache(maxsize=1)
def _junctions() -> tuple:
    rng = np.random.default_rng(11)
    road_types = rng.choice(["arterial", "commercial", "residential", "service_lane", "highway"], size=12)
    st = sample_static(rng, 12, road_types=road_types)
    # 4 x 3 grid of cells with jitter, so junctions never overlap on the map
    cells = np.array([(2 * c + 1, 2 * r + 1) for r in range(3) for c in range(4)], dtype=float)
    xy = cells + rng.uniform(-0.55, 0.55, size=(12, 2))
    xy = xy[rng.permutation(12)]
    out = []
    for i in range(12):
        static = {k: (str(v[i]) if k == "road_type" else float(v[i])) for k, v in st.items()}
        static["intersection_density"] = float(np.clip(static["intersection_density"] + 4, 1, 25))
        out.append({"name": f"Junction {NAMES[i]}", "x": round(float(xy[i][0]), 2),
                    "y": round(float(xy[i][1]), 2), "static": static})
    return tuple(out)


def city_hotspots(hour: float, traveler: str, mode: str, weather: str) -> dict:
    req = RouteRequest(origin="demo a", destination="demo b", travel_mode=mode, traveler_type=traveler,
                       travel_time=f"{int(hour) % 24}:{int(round((hour - int(hour)) * 60)) % 60:02d}",
                       weather=weather)
    ctx = rs.make_context(req)
    js = _junctions()
    rows = [rs.feature_row(j["static"], ctx, 0.3) for j in js]
    scores = ml.score(rows)
    expl = ml.explain(rows)
    items = [{"name": j["name"], "x": j["x"], "y": j["y"], "safety_score": round(float(s), 1),
              "risk_level": risk_level(s), "main_factor": rs._main_factor(e)}
             for j, s, e in zip(js, scores, expl)]
    items.sort(key=lambda i: i["safety_score"])
    return {"context": rs.context_summary(ctx), "junctions": items,
            "note": "Synthetic demo junctions on an abstract grid. Not real locations."}
