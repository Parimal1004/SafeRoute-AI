import pytest
from fastapi.testclient import TestClient

from backend.main import app

PLAN = {"origin": "CBIT", "destination": "Kokapet", "travel_mode": "walking", "traveler_type": "student",
        "travel_time": "22:00", "weather": "clear"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_and_health(client):
    assert client.get("/").json()["name"] == "SafeRoute AI"
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True


def test_recommend_route(client):
    r = client.post("/recommend-route", json=PLAN)
    assert r.status_code == 200
    j = r.json()
    assert len(j["routes"]) == 3
    assert sum(x["recommended"] for x in j["routes"]) == 1
    assert j["recommended_id"] in {"A", "B", "C"}
    assert all(0 <= x["safety_score"] <= 100 for x in j["routes"])
    assert j["agent"]["text"] and "guarantee" in j["agent"]["text"].lower()
    assert "synthetic" in j["disclaimer"].lower()


def test_same_input_gives_same_output(client):
    a = client.post("/recommend-route", json=PLAN).json()
    b = client.post("/recommend-route", json=PLAN).json()
    assert [r["safety_score"] for r in a["routes"]] == [r["safety_score"] for r in b["routes"]]


def test_validation_errors(client):
    assert client.post("/recommend-route", json={**PLAN, "destination": "CBIT"}).status_code == 422
    assert client.post("/recommend-route", json={**PLAN, "travel_time": "99:00"}).status_code == 422
    assert client.post("/recommend-route", json={**PLAN, "travel_mode": "rocket"}).status_code == 422


def test_what_if_night_is_riskier_than_day(client):
    day = {**PLAN, "travel_time": "12:00"}
    r = client.post("/what-if", json={"plan": day, "scenario": {"travel_time": "23:00"}}).json()
    assert all(x["delta"] < 0 for x in r["routes"])
    assert r["changes"] and r["agent"]["text"]


def test_what_if_without_changes_changes_nothing(client):
    r = client.post("/what-if", json={"plan": PLAN, "scenario": {}}).json()
    assert r["changes"] == [] and all(abs(x["delta"]) < 0.05 for x in r["routes"])


def test_predict_risk(client):
    r = client.post("/predict-risk", json={}).json()
    assert 0 <= r["safety_score"] <= 100 and r["risk_level"] in {"Low", "Medium", "High"}
    assert client.post("/predict-risk", json={"street_lighting": 500}).status_code == 422


def test_time_profile_and_hotspots(client):
    p = client.post("/time-profile", json={"plan": PLAN}).json()
    assert len(p["hours"]) == 24 and set(p["scores"]) == {"A", "B", "C"}
    h = client.get("/hotspots", params={"hour": 22}).json()
    assert len(h["junctions"]) == 12
    scores = [j["safety_score"] for j in h["junctions"]]
    assert scores == sorted(scores)


def test_agent_chat_uses_real_numbers_and_tools(client):
    base = client.post("/recommend-route", json=PLAN).json()
    r = client.post("/agent/chat", json={"plan": PLAN, "question": "What happens if I travel at 11 PM?"}).json()
    assert r["intent"] == "scenario"
    assert {t["tool"] for t in r["tools_used"]} == {"score_routes", "run_what_if"}
    assert r["mode"] == "offline"
    q = client.post("/agent/chat", json={"plan": PLAN, "question": "Why did you recommend this route?"}).json()
    top = next(x for x in base["routes"] if x["recommended"])
    assert f"{top['safety_score']:.0f}/100" in q["answer"]


def test_agent_never_claims_certainty(client):
    for question in ["Is Route B definitely safe?", "Why is Route C risky?", "hello"]:
        text = client.post("/agent/chat", json={"plan": PLAN, "question": question}).json()["answer"].lower()
        assert "definitely safe" not in text and "guaranteed safe" not in text
