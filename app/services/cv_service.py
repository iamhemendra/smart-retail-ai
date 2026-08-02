"""
Computer Vision service.

Two responsibilities:
  1. Product image classification (transfer learning, MobileNetV2/Keras).
  2. Face recognition for returning-customer detection.

Both loaders are lazy + cached so `app/main.py` can call `load_all()` once
at startup and every request reuses the same in-memory model.
"""
import os
import pickle
import datetime
from typing import Optional

import numpy as np
import cv2

from app.config import MODELS_DIR

# ---------------------------------------------------------------------------
# Product image classifier (Keras / MobileNetV2 transfer learning)
# ---------------------------------------------------------------------------

_product_model = None
_class_names: list[str] = ["shoes", "bags", "electronics", "clothing", "groceries"]

PRODUCT_MODEL_PATH = os.path.join(MODELS_DIR, "product_classifier.h5")
IMG_SIZE = (224, 224)  # MobileNetV2 default input


def load_product_classifier():
    """Load the trained Keras model once. Train it via
    notebooks/01_image_classifier_training.ipynb first."""
    global _product_model
    if _product_model is None:
        if not os.path.exists(PRODUCT_MODEL_PATH):
            raise FileNotFoundError(
                f"{PRODUCT_MODEL_PATH} not found. Run "
                "notebooks/01_image_classifier_training.ipynb first."
            )
        # Imported lazily so the whole app doesn't pay TF's import cost
        # unless this module is actually used.
        from tensorflow import keras
        _product_model = keras.models.load_model(PRODUCT_MODEL_PATH)
    return _product_model


def preprocess_image_bytes(image_bytes: bytes, size=IMG_SIZE) -> np.ndarray:
    """Decode raw upload bytes -> resized RGB float32 array ready for the model."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image bytes — is this a valid image file?")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    return img


def classify_product(image_bytes: bytes) -> dict:
    model = load_product_classifier()
    img = preprocess_image_bytes(image_bytes)
    batch = np.expand_dims(img, axis=0)
    preds = model.predict(batch, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    top_k_idx = np.argsort(preds)[::-1][:3]
    return {
        "category": _class_names[top_idx],
        "confidence": float(preds[top_idx]),
        "top_k": [
            {"category": _class_names[i], "confidence": float(preds[i])}
            for i in top_k_idx
        ],
    }


# ---------------------------------------------------------------------------
# OpenCV preprocessing utilities (Module A1 — reusable helpers)
# ---------------------------------------------------------------------------

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def canny_edges(img: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = to_grayscale(img) if img.ndim == 3 else img
    return cv2.Canny(gray, low, high)


_haar_face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Haar cascade face bounding boxes: [(x, y, w, h), ...]. Cheap and fast;
    good for a live webcam demo. For actual recognition we use encodings
    (see below), not this."""
    gray = to_grayscale(img)
    faces = _haar_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return [tuple(f) for f in faces]


# ---------------------------------------------------------------------------
# Face recognition (returning-customer detection)
# ---------------------------------------------------------------------------

FACE_DB_PATH = os.path.join(MODELS_DIR, "face_db.pkl")
MATCH_TOLERANCE = 0.6  # lower = stricter match (face_recognition default is 0.6)

_face_db: Optional[dict] = None  # {customer_id: [128-d encoding, ...]}


def load_face_db() -> dict:
    global _face_db
    if _face_db is None:
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, "rb") as f:
                _face_db = pickle.load(f)
        else:
            _face_db = {}
    return _face_db


def save_face_db() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(_face_db or {}, f)


def enroll_customer(customer_id: str, image_bytes: bytes) -> bool:
    """Add a face encoding for a new/existing customer. Returns False if no
    face was detected in the image."""
    import face_recognition  # lazy import: heavy, dlib-based

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return False

    db = load_face_db()
    db.setdefault(customer_id, []).append(encodings[0])
    save_face_db()
    return True


def recognize_face(image_bytes: bytes) -> dict:
    """Compare an incoming face against the enrolled DB. Returns a dict
    matching schemas.FaceRecognitionResponse."""
    import face_recognition  # lazy import: heavy, dlib-based

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return {"recognized": False, "customer_id": None, "confidence": 0.0,
                 "visit_logged_at": None}

    probe = encodings[0]
    db = load_face_db()

    best_id, best_distance = None, 1.0
    for customer_id, known_encodings in db.items():
        distances = face_recognition.face_distance(known_encodings, probe)
        min_dist = float(np.min(distances)) if len(distances) else 1.0
        if min_dist < best_distance:
            best_distance, best_id = min_dist, customer_id

    if best_id is not None and best_distance <= MATCH_TOLERANCE:
        confidence = round(1.0 - best_distance, 4)
        return {
            "recognized": True,
            "customer_id": best_id,
            "confidence": confidence,
            "visit_logged_at": datetime.datetime.utcnow().isoformat(),
        }

    return {"recognized": False, "customer_id": None, "confidence": 0.0,
             "visit_logged_at": None}
