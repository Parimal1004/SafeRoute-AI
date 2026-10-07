"""LLM path with a fake provider: no network and no key needed."""
from backend.models.schemas import RouteRequest
from backend.services import agent_service, llm_client, risk_service

PLAN = RouteRequest(origin="CBIT", destination="Kokapet", travel_time="22:00")


def test_llm_answer_is_used_when_configured(monkeypatch):
    monkeypatch.setattr(llm_client, "configured", lambda: True)
    monkeypatch.setattr(llm_client, "chat", lambda messages, **kw: "Route B has a lower predicted risk.")
    agent_service._cache.clear()
    ev = risk_service.evaluate(PLAN, geometry=False)
    out = agent_service.recommendation_message(ev)
    assert out["mode"] == "llm" and "lower predicted risk" in out["text"]


def test_unsafe_llm_wording_is_replaced_by_the_offline_answer(monkeypatch):
    monkeypatch.setattr(llm_client, "configured", lambda: True)
    monkeypatch.setattr(llm_client, "chat", lambda messages, **kw: "Route B is definitely safe, go ahead.")
    agent_service._cache.clear()
    out = agent_service.recommendation_message(risk_service.evaluate(PLAN, geometry=False))
    assert out["mode"] == "offline" and "definitely safe" not in out["text"].lower()


def test_llm_failure_falls_back(monkeypatch):
    def boom(messages, **kw):
        raise llm_client.LLMError("down")
    monkeypatch.setattr(llm_client, "configured", lambda: True)
    monkeypatch.setattr(llm_client, "chat", boom)
    agent_service._cache.clear()
    out = agent_service.recommendation_message(risk_service.evaluate(PLAN, geometry=False))
    assert out["mode"] == "offline" and out["text"]
