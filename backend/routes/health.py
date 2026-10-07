from fastapi import APIRouter

from backend.services import llm_client, route_service
from backend.utils import config
from ml import predict as ml

router = APIRouter(tags=["system"])


@router.get("/")
def root():
    return {"name": config.APP_NAME, "version": config.APP_VERSION,
            "tagline": "The shortest route isn't always the safest route.",
            "docs": "/docs", "disclaimer": config.DISCLAIMER,
            "endpoints": ["GET /health", "POST /predict-risk", "POST /recommend-route", "POST /what-if",
                          "POST /agent/chat", "POST /time-profile", "GET /hotspots", "GET /model-info",
                          "GET /places"]}


@router.get("/health")
def health():
    _, meta = ml.get_model()
    return {"status": "ok", "version": config.APP_VERSION, "model_loaded": True,
            "model": meta["model"], "llm": llm_client.status(),
            "agent_mode": "llm" if llm_client.configured() else "offline"}


@router.get("/model-info")
def model_info():
    return ml.model_info()


@router.get("/places")
def places():
    return {"places": route_service.known_places(),
            "note": "Demo coordinates for Hyderabad. Any other text is placed at a stable simulated location."}
