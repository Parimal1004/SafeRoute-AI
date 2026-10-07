"""Minimal OpenAI-compatible chat client (works with Groq, OpenRouter, Gemini's OpenAI endpoint,
OpenAI, Ollama...). Uses plain HTTP so no vendor SDK is needed. If no key is set or a call
fails, the caller falls back to the offline agent, so the app never breaks."""
from __future__ import annotations

from typing import Dict, List
from urllib.parse import urlparse

import requests

from backend.utils import config


class LLMError(RuntimeError):
    pass


def configured() -> bool:
    return bool(config.llm_api_key())


def status() -> dict:
    host = urlparse(config.llm_base_url()).netloc
    return {"configured": configured(), "provider_host": host if configured() else None,
            "model": config.llm_model() if configured() else None}


def chat(messages: List[Dict[str, str]], max_tokens: int = 450, temperature: float = 0.3) -> str:
    if not configured():
        raise LLMError("LLM_API_KEY is not set")
    try:
        resp = requests.post(
            f"{config.llm_base_url()}/chat/completions",
            headers={"Authorization": f"Bearer {config.llm_api_key()}", "Content-Type": "application/json"},
            json={"model": config.llm_model(), "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature},
            timeout=config.llm_timeout(),
        )
        if resp.status_code != 200:
            raise LLMError(f"LLM provider returned HTTP {resp.status_code}")
        text = resp.json()["choices"][0]["message"]["content"]
    except LLMError:
        raise
    except Exception as exc:  # network error, bad JSON, missing keys...
        raise LLMError(f"LLM call failed: {type(exc).__name__}") from exc
    text = (text or "").strip()
    if not text:
        raise LLMError("LLM returned an empty answer")
    return text
