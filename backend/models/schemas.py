"""Pydantic request / response models."""
from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

TravelMode = Literal["walking", "cycling", "public_transport", "car"]
TravelerType = Literal["student", "general", "senior_citizen", "cyclist", "pedestrian"]
WeatherType = Literal["clear", "cloudy", "light_rain", "heavy_rain", "fog"]
RoadType = Literal["arterial", "commercial", "residential", "service_lane", "highway"]

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def _check_time(v: Optional[str]) -> Optional[str]:
    if v is not None and not _TIME_RE.match(v):
        raise ValueError("time must look like HH:MM (24-hour), e.g. 22:00")
    return v


# ----------------------------- requests ----------------------------------- #
class RouteRequest(BaseModel):
    origin: str = Field(..., min_length=2, max_length=80, examples=["CBIT"])
    destination: str = Field(..., min_length=2, max_length=80, examples=["Gachibowli"])
    travel_mode: TravelMode = "walking"
    traveler_type: TravelerType = "student"
    travel_date: Optional[date] = None
    travel_time: str = "22:00"
    weather: WeatherType = "clear"
    include_agent: bool = True

    @field_validator("travel_time")
    @classmethod
    def _time(cls, v):
        return _check_time(v)


class Scenario(BaseModel):
    """Changes applied on top of the original plan. Fields left empty stay unchanged."""
    travel_time: Optional[str] = None
    travel_date: Optional[date] = None
    weather: Optional[WeatherType] = None
    traffic_change_pct: float = Field(0, ge=-80, le=200)
    travel_mode: Optional[TravelMode] = None
    traveler_type: Optional[TravelerType] = None

    @field_validator("travel_time")
    @classmethod
    def _time(cls, v):
        return _check_time(v)


class WhatIfRequest(BaseModel):
    plan: RouteRequest
    scenario: Scenario
    include_agent: bool = True


class PredictRiskRequest(BaseModel):
    traffic_density: float = Field(45, ge=0, le=100)
    accident_frequency: float = Field(3, ge=0, le=10)
    street_lighting: float = Field(70, ge=0, le=100)
    pedestrian_density: float = Field(50, ge=0, le=100)
    road_condition: float = Field(65, ge=0, le=100)
    crime_rate: float = Field(30, ge=0, le=100)
    rainfall: float = Field(0, ge=0, le=60)
    visibility: float = Field(8, ge=0.1, le=10)
    intersection_density: float = Field(8, ge=0, le=30)
    vehicle_density: float = Field(40, ge=0, le=100)
    time_of_day: float = Field(22, ge=0, lt=24)
    day_of_week: int = Field(0, ge=0, le=6)
    distance: float = Field(2.0, gt=0, le=50)
    route_duration: float = Field(25, gt=0, le=300)
    weather_condition: WeatherType = "clear"
    road_type: RoadType = "arterial"
    traveler_type: TravelerType = "student"
    travel_mode: TravelMode = "walking"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    plan: RouteRequest
    question: str = Field(..., min_length=1, max_length=500)
    history: List[ChatTurn] = []


class TimeProfileRequest(BaseModel):
    plan: RouteRequest


# ----------------------------- responses ---------------------------------- #
class Factor(BaseModel):
    key: str
    label: str
    text: str
    effect: float
    impact: Literal["positive", "negative", "neutral"]
    kind: str


class Junction(BaseModel):
    name: str
    route_id: str
    lat: float
    lon: float
    safety_score: float
    risk_level: str
    main_factor: str


class RouteOption(BaseModel):
    id: str
    name: str
    label: str
    distance_km: float
    duration_min: int
    safety_score: float
    risk_level: str
    recommended: bool
    utility: float
    features: Dict[str, object]
    factors: List[Factor]
    strengths: List[str]
    concerns: List[str]
    coords: List[List[float]] = []
    junctions: List[Junction] = []


class AgentMessage(BaseModel):
    text: str
    mode: Literal["llm", "offline"]
    model: Optional[str] = None


class RecommendResponse(BaseModel):
    context: Dict[str, object]
    routes: List[RouteOption]
    recommended_id: str
    fastest_id: str
    safest_id: str
    recommendation_basis: str
    hotspots: List[Junction]
    agent: Optional[AgentMessage] = None
    demo_mode: bool = True
    disclaimer: str


class PredictRiskResponse(BaseModel):
    safety_score: float
    risk_level: str
    factors: List[Factor]
    note: str
    disclaimer: str


class RouteChange(BaseModel):
    id: str
    name: str
    baseline_score: float
    scenario_score: float
    delta: float
    baseline_risk: str
    scenario_risk: str
    baseline_duration: int
    scenario_duration: int
    drivers: List[Dict[str, object]]


class WhatIfResponse(BaseModel):
    changes: List[str]
    baseline: Dict[str, object]
    scenario: Dict[str, object]
    routes: List[RouteChange]
    recommended_before: str
    recommended_after: str
    recommendation_changed: bool
    agent: Optional[AgentMessage] = None
    disclaimer: str


class ToolCall(BaseModel):
    tool: str
    args: Dict[str, object]
    summary: str


class ChatResponse(BaseModel):
    answer: str
    mode: Literal["llm", "offline"]
    model: Optional[str] = None
    intent: str
    tools_used: List[ToolCall]
    disclaimer: str


class TimeProfileResponse(BaseModel):
    hours: List[int]
    scores: Dict[str, List[float]]
    context: Dict[str, object]
