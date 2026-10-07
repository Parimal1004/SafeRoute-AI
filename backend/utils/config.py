"""Settings read from environment variables (and a local .env file if present)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

APP_NAME = "SafeRoute AI"
APP_VERSION = "1.0.0"

DISCLAIMER = ("This system provides decision support and should not be treated as a guarantee of "
              "real-world safety. Scores are predicted from synthetic demo data.")
SCORE_NOTE = "Predicted safety score based on available historical and contextual factors."


def llm_api_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip()


def llm_base_url() -> str:
    return os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip().rstrip("/")


def llm_model() -> str:
    return os.getenv("LLM_MODEL", "llama-3.1-8b-instant").strip()


def llm_timeout() -> float:
    try:
        return float(os.getenv("LLM_TIMEOUT", "25"))
    except ValueError:
        return 25.0


def cors_origins() -> list:
    raw = os.getenv("CORS_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]
