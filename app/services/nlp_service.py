"""
NLP service: text preprocessing + sentiment analysis.

Baseline model: TF-IDF + Logistic Regression (trained in
notebooks/03_sentiment_model_training.ipynb). Swap in a fine-tuned
DistilBERT later as a stretch goal without changing this module's
public interface (`analyze_sentiment`).
"""
import os
import re
import string
import joblib

from app.config import MODELS_DIR

SENTIMENT_MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

_model = None
_vectorizer = None
_stopwords: set[str] | None = None


def _load_stopwords() -> set[str]:
    """NLTK stopwords with a hardcoded fallback so this never hard-fails
    if the NLTK corpus wasn't downloaded."""
    global _stopwords
    if _stopwords is not None:
        return _stopwords
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            _stopwords = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            _stopwords = set(stopwords.words("english"))
    except Exception:
        _stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
            "to", "of", "in", "on", "for", "with", "this", "that", "it", "i",
        }
    return _stopwords


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits, remove stopwords. Lemmatization
    is intentionally left to spaCy in the training notebook (heavier
    dependency); this fast path is what the API uses at inference time."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    stops = _load_stopwords()
    tokens = [t for t in tokens if t not in stops]
    return " ".join(tokens)


def load_sentiment_model():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        if not (os.path.exists(SENTIMENT_MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
            raise FileNotFoundError(
                "Sentiment model/vectorizer not found. Run "
                "notebooks/03_sentiment_model_training.ipynb first."
            )
        _model = joblib.load(SENTIMENT_MODEL_PATH)
        _vectorizer = joblib.load(VECTORIZER_PATH)
    return _model, _vectorizer


def analyze_sentiment(text: str) -> dict:
    model, vectorizer = load_sentiment_model()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    label = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(max(proba))
    else:
        confidence = 1.0  # e.g. plain LinearSVC has no predict_proba

    return {"label": str(label), "confidence": round(confidence, 4)}
