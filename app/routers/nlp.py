from fastapi import APIRouter, Depends, HTTPException

from app.config import require_api_key
from app.schemas import SentimentRequest, SentimentResponse
from app.services import nlp_service

router = APIRouter(tags=["nlp"], dependencies=[Depends(require_api_key)])


@router.post("/analyze-sentiment", response_model=SentimentResponse)
async def analyze_sentiment(payload: SentimentRequest):
    try:
        result = nlp_service.analyze_sentiment(payload.text)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    return result
