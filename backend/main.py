"""SafeRoute AI backend.   Run:  uvicorn backend.main:app --reload"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `ml` and `backend` importable

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from backend.routes import agent, health, risk, routing  # noqa: E402
from backend.utils import config  # noqa: E402
from ml import predict as ml  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.get_model()  # load (or train on first run) so the first request is fast
    yield


app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION, lifespan=lifespan,
              description="Route safety prediction with ML, explainability and an AI agent. "
                          + config.DISCLAIMER)
app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins(), allow_methods=["*"], allow_headers=["*"])
for r in (health.router, risk.router, routing.router, agent.router):
    app.include_router(r)
