"""Demo route simulation (no external map API needed).

Origins and destinations are looked up in a small list of Hyderabad place names to get
approximate coordinates. Unknown text gets a stable pseudo-location near the city centre.
The three route alternatives are SIMULATED curves with SYNTHETIC attributes, not real roads.
Results are deterministic for the same origin/destination pair.
"""
from __future__ import annotations

import math
import re
import zlib
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from ml.features import sample_static

# name -> (lat, lon), approximate
PLACES: Dict[str, Tuple[float, float]] = {
    "CBIT": (17.3913, 78.3190),
    "Gachibowli": (17.4401, 78.3489),
    "HITEC City": (17.4435, 78.3772),
    "Madhapur": (17.4486, 78.3908),
    "Kondapur": (17.4600, 78.3640),
    "Financial District": (17.4172, 78.3426),
    "Kokapet": (17.3930, 78.3340),
    "Narsingi": (17.3880, 78.3620),
    "Manikonda": (17.4041, 78.3870),
    "Mehdipatnam": (17.3950, 78.4400),
    "Jubilee Hills": (17.4326, 78.4071),
    "Banjara Hills": (17.4156, 78.4347),
    "Ameerpet": (17.4375, 78.4483),
    "Begumpet": (17.4440, 78.4676),
    "Secunderabad": (17.4399, 78.4983),
    "Kukatpally": (17.4948, 78.3996),
    "Koti": (17.3850, 78.4867),
    "Charminar": (17.3616, 78.4747),
    "Uppal": (17.4058, 78.5591),
    "LB Nagar": (17.3457, 78.5522),
}
_ALIASES = {
    "chaitanya bharathi institute of technology": "CBIT",
    "hitech city": "HITEC City",
    "hi-tech city": "HITEC City",
    "hitec city": "HITEC City",
    "fin district": "Financial District",
    "l b nagar": "LB Nagar",
    "lb nagar": "LB Nagar",
}
_BY_LOWER = {k.lower(): k for k in PLACES}
CITY_CENTRE = (17.40, 78.45)

# id, label, preferred road types, circuity (road distance / straight line),
# bend offset, quality shift range
ARCHETYPES = [
    {"id": "A", "label": "Balanced route", "road_types": ["arterial", "commercial"], "circuity": 1.22,
     "offset": 0.12, "shift": (-8, 8)},
    {"id": "B", "label": "Main-road corridor", "road_types": ["commercial", "arterial"], "circuity": 1.38,
     "offset": -0.22, "shift": (-4, 12)},
    {"id": "C", "label": "Direct shortcut", "road_types": ["residential", "service_lane"], "circuity": 1.10,
     "offset": 0.04, "shift": (-18, 4)},
]
JUNCTIONS_PER_ROUTE = 4


def known_places() -> List[str]:
    return list(PLACES)


def resolve_place(text: str) -> Tuple[float, float, bool]:
    """Return (lat, lon, is_known). Accepts a known name, an alias or 'lat, lon'."""
    t = re.sub(r"\s+", " ", text.strip()).lower()
    if t in _BY_LOWER:
        lat, lon = PLACES[_BY_LOWER[t]]
        return lat, lon, True
    if t in _ALIASES:
        lat, lon = PLACES[_ALIASES[t]]
        return lat, lon, True
    m = re.match(r"^\s*(-?\d{1,2}\.\d+)\s*[, ]\s*(-?\d{1,3}\.\d+)\s*$", t)
    if m:
        return float(m.group(1)), float(m.group(2)), True
    h = zlib.crc32(t.encode())
    rng = np.random.default_rng(h)
    return (CITY_CENTRE[0] + rng.uniform(-0.07, 0.07), CITY_CENTRE[1] + rng.uniform(-0.09, 0.09), False)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _curve(p0, p2, offset, rng, n=48):
    """Quadratic Bezier between two (lat, lon) points, plus a small seeded wiggle."""
    cos = math.cos(math.radians((p0[0] + p2[0]) / 2))
    x0, y0 = p0[1] * cos, p0[0]
    x2, y2 = p2[1] * cos, p2[0]
    dx, dy = x2 - x0, y2 - y0
    cx, cy = (x0 + x2) / 2 - offset * dy, (y0 + y2) / 2 + offset * dx
    t = np.linspace(0, 1, n)
    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x2
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y2
    length = math.hypot(dx, dy)
    phase = rng.uniform(0, 2 * math.pi)
    wig = 0.015 * length * np.sin(7 * math.pi * t + phase) * np.sin(math.pi * t)
    nx, ny = (-dy / length if length else 0), (dx / length if length else 0)
    x, y = x + wig * nx, y + wig * ny
    return [(float(yy), float(xx / cos)) for xx, yy in zip(x, y)]


def _scalar_static(rng, road_type, shift) -> dict:
    s = sample_static(rng, 1, road_types=[road_type], quality_shift=[shift])
    return {k: (str(v[0]) if k == "road_type" else float(v[0])) for k, v in s.items()}


def _junction_static(rng, base: dict) -> dict:
    j = dict(base)
    j["street_lighting"] = float(np.clip(base["street_lighting"] + rng.normal(0, 14), 5, 100))
    j["road_condition"] = float(np.clip(base["road_condition"] + rng.normal(0, 12), 5, 100))
    j["accident_frequency"] = float(np.clip(base["accident_frequency"] * rng.uniform(0.5, 1.9), 0, 10))
    j["crime_rate"] = float(np.clip(base["crime_rate"] + rng.normal(0, 10), 2, 100))
    j["intersection_density"] = float(np.clip(base["intersection_density"] + rng.uniform(2, 7), 0.5, 25))
    j["base_traffic"] = float(np.clip(base["base_traffic"] * rng.uniform(0.8, 1.3), 3, 100))
    j["base_pedestrian"] = float(np.clip(base["base_pedestrian"] * rng.uniform(0.7, 1.3), 2, 100))
    return j


@lru_cache(maxsize=256)
def base_routes(origin: str, destination: str) -> tuple:
    """Three simulated alternatives with static attributes. Same input -> same output."""
    o_lat, o_lon, o_known = resolve_place(origin)
    d_lat, d_lon, d_known = resolve_place(destination)
    straight = haversine_km(o_lat, o_lon, d_lat, d_lon)
    if straight < 0.3:
        raise ValueError("Origin and destination are too close together. Pick two different places.")
    seed = zlib.crc32(f"{origin.strip().lower()}|{destination.strip().lower()}".encode())
    routes = []
    for i, arch in enumerate(ARCHETYPES):
        rng = np.random.default_rng(seed + 101 * (i + 1))
        road_type = arch["road_types"][int(rng.integers(0, len(arch["road_types"])))]
        shift = float(rng.uniform(*arch["shift"]))
        static = _scalar_static(rng, road_type, shift)
        coords = _curve((o_lat, o_lon), (d_lat, d_lon), arch["offset"], rng)
        distance = round(straight * arch["circuity"] * float(rng.uniform(0.97, 1.03)), 1)
        junctions = []
        for k in range(JUNCTIONS_PER_ROUTE):
            idx = int(round((k + 1) / (JUNCTIONS_PER_ROUTE + 1) * (len(coords) - 1)))
            lat, lon = coords[idx]
            junctions.append({"name": f"Route {arch['id']} · Junction {k + 1}", "lat": lat, "lon": lon,
                              "static": _junction_static(rng, static)})
        routes.append({"id": arch["id"], "label": arch["label"], "name": f"Route {arch['id']}",
                       "distance_km": max(distance, 0.4), "static": static, "coords": coords,
                       "junctions": junctions})
    return tuple(routes), {"origin_known": o_known, "destination_known": d_known,
                           "straight_km": round(straight, 2),
                           "origin_xy": (o_lat, o_lon), "destination_xy": (d_lat, d_lon)}
