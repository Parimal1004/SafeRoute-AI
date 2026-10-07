"""Thin client for the SafeRoute FastAPI backend."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests
import streamlit as st

TIMEOUT = 90  # free hosting can take ~1 minute to wake up after idling


class BackendError(RuntimeError):
    pass


def backend_url() -> str:
    url = os.getenv("BACKEND_URL", "")
    if not url:
        try:
            url = st.secrets.get("BACKEND_URL", "")
        except Exception:  # no secrets file
            url = ""
    return (url or "http://localhost:8000").rstrip("/")

_LOCAL = requests.Session()
_LOCAL.trust_env = False

def _session_for(url: str):
    host = url.split("//", 1)[-1].split("/", 1)[0].split(":")[0]
    return _LOCAL if host in ("127.0.0.1", "localhost", "::1") else requests

def _request(method: str, path: str, payload: Optional[dict] = None, params: Optional[dict] = None,
             timeout: float = TIMEOUT) -> Any:
    url = backend_url() + path
    try:
        resp = _session_for(url).request(method, url, json=payload, params=params, timeout=timeout)
        
    except requests.exceptions.ConnectionError:
        raise BackendError(f"Cannot reach the backend at {backend_url()}. Start it with "
                           "`uvicorn backend.main:app --port 8000`, or if it is hosted on a free plan, wait "
                           "a minute for it to wake up and try again.")
    except requests.exceptions.Timeout:
        raise BackendError("The backend took too long to respond. Free hosting may be waking up; try again.")
    if resp.status_code == 422:
        try:
            detail = resp.json().get("detail")
            msg = detail if isinstance(detail, str) else "; ".join(
                f"{'.'.join(str(x) for x in d.get('loc', [])[1:])}: {d.get('msg')}" for d in detail)
        except Exception:
            msg = "invalid input"
        raise BackendError(msg)
    if resp.status_code >= 400:
        raise BackendError(f"Backend error {resp.status_code}")
    return resp.json()


def health(timeout: float = 6) -> Dict:
    return _request("GET", "/health", timeout=timeout)


@st.cache_data(ttl=3600, show_spinner=False)
def model_info() -> Dict:
    return _request("GET", "/model-info")


@st.cache_data(ttl=3600, show_spinner=False)
def places() -> list:
    return _request("GET", "/places")["places"]


def recommend(plan: dict) -> Dict:
    return _request("POST", "/recommend-route", plan)


def what_if(plan: dict, scenario: dict) -> Dict:
    return _request("POST", "/what-if", {"plan": plan, "scenario": scenario})


def time_profile(plan: dict) -> Dict:
    return _request("POST", "/time-profile", {"plan": plan})


def hotspots(hour: float, traveler_type: str, travel_mode: str, weather: str) -> Dict:
    return _request("GET", "/hotspots", params={"hour": hour, "traveler_type": traveler_type,
                                                "travel_mode": travel_mode, "weather": weather})


def predict_risk(features: dict) -> Dict:
    return _request("POST", "/predict-risk", features)


def chat(plan: dict, question: str, history: list) -> Dict:
    return _request("POST", "/agent/chat", {"plan": plan, "question": question, "history": history})
