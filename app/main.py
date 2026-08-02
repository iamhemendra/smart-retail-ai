"""
FastAPI entrypoint. Wires up routers and exposes /health + /dashboard/stats.

Run with:
    uvicorn app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from app.config import require_api_key
from app.schemas import DashboardStats
from app.routers import vision, nlp, chatbot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-retail-ai")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm-load models once at startup so the first request isn't slow.
    Failures are logged, not fatal — individual endpoints raise a clear 503
    if their specific model is missing, which is expected before you've run
    the training notebooks."""
    from app.services import cv_service, nlp_service

    for name, loader in [
        ("product classifier", cv_service.load_product_classifier),
        ("face db", cv_service.load_face_db),
        ("sentiment model", nlp_service.load_sentiment_model),
    ]:
        try:
            loader()
            logger.info("Loaded %s", name)
        except FileNotFoundError as e:
            logger.warning("Skipping %s: %s", name, e)

    yield  # app runs here


app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform",
    description=(
        "Face recognition, product classification, sentiment analysis, "
        "and a FAQ chatbot behind one API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(vision.router)
app.include_router(nlp.router)
app.include_router(chatbot.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/dashboard/stats", response_model=DashboardStats, dependencies=[Depends(require_api_key)])
async def dashboard_stats():
    """
    Aggregate stats for a simple frontend chart. This is stubbed with an
    in-memory shape — wire it up to real visit/sentiment logs (e.g. a
    customer_visits table) once you're persisting requests somewhere.
    """
    from app.services import cv_service

    db = cv_service.load_face_db()
    total_visits = sum(len(v) for v in db.values())

    return {
        "total_visits": total_visits,
        "unique_customers": len(db),
        "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
        "top_intents": [],
    }
