"""
Hybrid FAQ chatbot: fast rule-based keyword matching first, ML intent
classifier as a fallback, canned default reply if neither is confident.
"""
import os
import json
import joblib

from app.config import MODELS_DIR
from app.services.nlp_service import clean_text

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "intents.json")
CHATBOT_MODEL_PATH = os.path.join(MODELS_DIR, "chatbot_model.pkl")
CHATBOT_VECTORIZER_PATH = os.path.join(MODELS_DIR, "chatbot_vectorizer.pkl")

_intents: dict | None = None
_ml_model = None
_ml_vectorizer = None

ML_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_REPLY = (
    "I'm not sure I understood that. You can ask about order status, "
    "returns, shipping, store hours, or say 'talk to a human'."
)


def load_intents() -> dict:
    global _intents
    if _intents is None:
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            _intents = json.load(f)
    return _intents


def _rule_based_match(message: str) -> tuple[str | None, str | None]:
    """Keyword-overlap match against each intent's patterns.
    Returns (intent_tag, reply) or (None, None)."""
    intents = load_intents()
    cleaned_tokens = set(clean_text(message).split())

    best_tag, best_score = None, 0
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_tokens = set(clean_text(pattern).split())
            overlap = len(cleaned_tokens & pattern_tokens)
            if overlap > best_score:
                best_score, best_tag = overlap, intent["tag"]

    if best_tag and best_score >= 1:
        intent = next(i for i in intents["intents"] if i["tag"] == best_tag)
        import random
        return best_tag, random.choice(intent["responses"])
    return None, None


def _ml_fallback(message: str) -> tuple[str | None, str | None, float]:
    """TF-IDF + classifier fallback, trained in a future
    notebooks/04_chatbot_intent_training.ipynb (not required for MVP —
    the rule-based path covers the required intents.json out of the box).
    Returns (tag, reply, confidence)."""
    global _ml_model, _ml_vectorizer
    if not (os.path.exists(CHATBOT_MODEL_PATH) and os.path.exists(CHATBOT_VECTORIZER_PATH)):
        return None, None, 0.0

    if _ml_model is None:
        _ml_model = joblib.load(CHATBOT_MODEL_PATH)
        _ml_vectorizer = joblib.load(CHATBOT_VECTORIZER_PATH)

    X = _ml_vectorizer.transform([clean_text(message)])
    proba = _ml_model.predict_proba(X)[0]
    idx = proba.argmax()
    tag = _ml_model.classes_[idx]
    confidence = float(proba[idx])

    if confidence < ML_CONFIDENCE_THRESHOLD:
        return None, None, confidence

    intents = load_intents()
    intent = next((i for i in intents["intents"] if i["tag"] == tag), None)
    if not intent:
        return None, None, confidence

    import random
    return tag, random.choice(intent["responses"]), confidence


def get_reply(message: str) -> dict:
    tag, reply = _rule_based_match(message)
    if reply:
        return {"reply": reply, "matched_intent": tag, "source": "rule"}

    tag, reply, _ = _ml_fallback(message)
    if reply:
        return {"reply": reply, "matched_intent": tag, "source": "ml_fallback"}

    return {"reply": DEFAULT_REPLY, "matched_intent": None, "source": "default"}
