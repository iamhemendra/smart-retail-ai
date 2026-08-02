"""Pydantic models for request/response validation across all routers."""
from typing import Optional
from pydantic import BaseModel, Field


# ---------- Vision ----------

class FaceRecognitionResponse(BaseModel):
    recognized: bool
    customer_id: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    visit_logged_at: Optional[str] = None


class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top_k: list[dict] = Field(default_factory=list)


# ---------- NLP ----------

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SentimentResponse(BaseModel):
    label: str  # "positive" | "negative" | "neutral"
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------- Chatbot ----------

class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = None


class ChatbotResponse(BaseModel):
    reply: str
    matched_intent: Optional[str] = None
    source: str  # "rule" | "ml_fallback" | "default"


# ---------- Dashboard ----------

class DashboardStats(BaseModel):
    total_visits: int
    unique_customers: int
    sentiment_breakdown: dict
    top_intents: list[dict]
