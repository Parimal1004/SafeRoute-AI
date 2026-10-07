from fastapi import APIRouter

from backend.models.schemas import PredictRiskRequest, PredictRiskResponse
from backend.services import risk_service

router = APIRouter(tags=["ml"])


@router.post("/predict-risk", response_model=PredictRiskResponse)
def predict_risk(body: PredictRiskRequest):
    """Predict the safety score for explicit feature values and explain it."""
    return risk_service.predict_risk(body)
