from fastapi import APIRouter, Depends

from app.config import require_api_key
from app.schemas import ChatbotRequest, ChatbotResponse
from app.services import chatbot_service

router = APIRouter(tags=["chatbot"], dependencies=[Depends(require_api_key)])


@router.post("/chatbot", response_model=ChatbotResponse)
async def chatbot(payload: ChatbotRequest):
    result = chatbot_service.get_reply(payload.message)
    return result
