from fastapi import APIRouter, HTTPException

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import agent_service

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    try:
        return agent_service.chat(body.plan, body.question, body.history)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
