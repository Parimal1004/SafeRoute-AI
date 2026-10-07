from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import (RecommendResponse, RouteRequest, TimeProfileRequest, TimeProfileResponse,
                                    WhatIfRequest, WhatIfResponse, TravelMode, TravelerType, WeatherType)
from backend.services import agent_service, hotspot_service, risk_service, whatif_service

router = APIRouter(tags=["routing"])


@router.post("/recommend-route", response_model=RecommendResponse)
def recommend_route(req: RouteRequest):
    try:
        result = risk_service.evaluate(req)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result["agent"] = agent_service.recommendation_message(result) if req.include_agent else None
    return result


@router.post("/what-if", response_model=WhatIfResponse)
def what_if(body: WhatIfRequest):
    try:
        result = whatif_service.run(body.plan, body.scenario)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    result["agent"] = (agent_service.whatif_message(result, result["scenario"])
                       if body.include_agent else None)
    return result


@router.post("/time-profile", response_model=TimeProfileResponse)
def time_profile(body: TimeProfileRequest):
    try:
        return risk_service.time_profile(body.plan)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/hotspots")
def hotspots(hour: float = Query(22, ge=0, lt=24), traveler_type: TravelerType = "general",
             travel_mode: TravelMode = "walking", weather: WeatherType = "clear"):
    return hotspot_service.city_hotspots(hour, traveler_type, travel_mode, weather)
