"""
Basic endpoint tests. Model-dependent endpoints will 503 until you've run
the training notebooks and populated app/models/ — that's expected and the
tests account for it, so CI can run before models exist.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SMART_RETAIL_API_KEY", "test-key")

from app.main import app  # noqa: E402
from app.config import API_KEY  # noqa: E402

client = TestClient(app)
HEADERS = {"X-API-Key": API_KEY}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_docs_available():
    resp = client.get("/docs")
    assert resp.status_code == 200


def test_missing_api_key_rejected():
    resp = client.post("/chatbot", json={"message": "hi"})
    assert resp.status_code == 401


def test_chatbot_rule_based_greeting():
    resp = client.post("/chatbot", json={"message": "hello"}, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] in ("rule", "ml_fallback", "default")
    assert "reply" in body


def test_chatbot_known_intent():
    resp = client.post(
        "/chatbot", json={"message": "what are your store hours"}, headers=HEADERS
    )
    assert resp.status_code == 200
    assert resp.json()["matched_intent"] == "store_hours"


def test_sentiment_without_trained_model_returns_503():
    resp = client.post(
        "/analyze-sentiment", json={"text": "great product"}, headers=HEADERS
    )
    # Passes whether or not the model has been trained yet.
    assert resp.status_code in (200, 503)


def test_dashboard_stats_shape():
    resp = client.get("/dashboard/stats", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert "total_visits" in body
    assert "unique_customers" in body


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
