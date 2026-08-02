from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from app.config import require_api_key
from app.schemas import FaceRecognitionResponse, ProductClassificationResponse
from app.services import cv_service

router = APIRouter(tags=["vision"], dependencies=[Depends(require_api_key)])

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB


async def _read_image(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpeg/png/etc).")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large (max 8MB).")
    return data


@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await _read_image(file)
    try:
        result = cv_service.recognize_face(image_bytes)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


@router.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(file: UploadFile = File(...)):
    image_bytes = await _read_image(file)
    try:
        result = cv_service.classify_product(image_bytes)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result
