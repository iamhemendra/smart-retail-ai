<div align="center">

# 🛍️ Smart Retail & Customer Intelligence Platform

**AI-powered retail backend combining computer vision, NLP, and a hybrid chatbot — served behind one FastAPI gateway.**

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#-license)

</div>

---

## 📖 Overview

This project brings three AI capabilities into a single retail backend:

- 👤 **Returning-customer recognition** via face encodings
- 📦 **Product image classification** via transfer learning (MobileNetV2)
- 💬 **Sentiment analysis** on customer reviews/chat
- 🤖 **Hybrid FAQ chatbot** — rule-based first, ML fallback second

All served through one authenticated, containerized, testable **FastAPI** service, with model training fully decoupled into standalone Jupyter notebooks — retrain or swap any model without touching the API contract.

## 📑 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Reference](#-api-reference)
- [Training the Models](#-training-the-models)
- [Testing](#-testing)
- [Docker](#-docker)
- [CI/CD](#-cicd)
- [Ethical Considerations](#-ethical-considerations)
- [Roadmap](#-roadmap)
- [License](#-license)

## ✨ Features

| Module | Capability | Approach |
|---|---|---|
| Computer Vision | Returning-customer detection | 128-d face encodings + nearest-neighbour match |
| Computer Vision | Product image classification | MobileNetV2 transfer learning (5 categories) |
| NLP | Review/chat sentiment analysis | TF-IDF + Logistic Regression (SVM comparison included) |
| Chatbot | FAQ handling | Keyword-overlap rules with 20 curated intents + optional ML fallback |
| Platform | Auth | Shared `X-API-Key` dependency across all routes |
| Platform | Dashboard | `/dashboard/stats` aggregate endpoint, ready to wire to a frontend |

## 🏗 Architecture

```
Client (POS / kiosk / mobile app)
        │
        ▼
  FastAPI Gateway (app/main.py)
        │── X-API-Key auth dependency (app/config.py)
        │
   ┌────┼───────────┬─────────────────┐
   │                │                 │
vision.py         nlp.py          chatbot.py        (app/routers/)
   │                │                 │
cv_service.py   nlp_service.py   chatbot_service.py  (app/services/)
   │                │                 │
product_classifier.h5   sentiment_model.pkl    intents.json
face_db.pkl              vectorizer.pkl         chatbot_model.pkl (optional)
        (app/models/ — produced by notebooks/*.ipynb)
```

The service layer never changes when a model is retrained or upgraded — only the artifact in `app/models/` does.

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn, Pydantic v2 |
| Computer Vision | TensorFlow/Keras (MobileNetV2), OpenCV, `face_recognition` (dlib) |
| NLP | scikit-learn (TF-IDF + Logistic Regression), NLTK |
| Persistence | HDF5 (`.h5`) / Pickle |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Testing | pytest + FastAPI `TestClient` |

## 📂 Project Structure

```
smart-retail-ai/
├── app/
│   ├── main.py                # FastAPI entrypoint, lifespan model loading
│   ├── config.py               # settings + API key auth dependency
│   ├── schemas.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── vision.py            # /recognize-face, /classify-product
│   │   ├── nlp.py                # /analyze-sentiment
│   │   └── chatbot.py            # /chatbot
│   ├── models/                   # trained artifacts land here (.h5 / .pkl)
│   └── services/
│       ├── cv_service.py          # preprocessing, classifier, face matching
│       ├── nlp_service.py          # text cleaning, sentiment inference
│       └── chatbot_service.py       # rule engine + ML fallback
├── notebooks/                        # training notebooks
│   ├── 01_image_classifier_training.ipynb
│   ├── 02_face_recognition_setup.ipynb
│   └── 03_sentiment_model_training.ipynb
├── data/
│   ├── reviews.csv                    # sample sentiment training data
│   └── intents.json                    # 20 curated chatbot intents
├── tests/
│   └── test_endpoints.py
├── requirements.txt
├── Dockerfile
├── .github/workflows/deploy.yml
└── README.md
```

## 🚀 Getting Started

```bash
# 1. Clone and set up the environment
git clone https://github.com/<your-username>/smart-retail-ai.git
cd smart-retail-ai
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Train the models (run once — see "Training the Models" below)
jupyter notebook notebooks/01_image_classifier_training.ipynb
jupyter notebook notebooks/02_face_recognition_setup.ipynb
jupyter notebook notebooks/03_sentiment_model_training.ipynb

# 3. Run the API
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger docs.

> All endpoints except `/health` and `/docs` require an `X-API-Key` header. Set it via the `SMART_RETAIL_API_KEY` environment variable (defaults to a dev key — change this before deploying).

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/recognize-face` | Upload an image → matched customer ID or `unrecognized` |
| `POST` | `/classify-product` | Upload an image → predicted category + confidence |
| `POST` | `/analyze-sentiment` | Send text → `positive` / `negative` / `neutral` + confidence |
| `POST` | `/chatbot` | Send a message → bot reply + which strategy answered it |
| `GET` | `/dashboard/stats` | Aggregate visit/sentiment stats (JSON) |
| `GET` | `/health` | Liveness check (no auth required) |

<details>
<summary>Example: <code>POST /chatbot</code></summary>

```bash
curl -X POST http://127.0.0.1:8000/chatbot \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are your store hours"}'
```

```json
{
  "reply": "Our stores are open Monday to Saturday, 10 AM to 9 PM, and Sunday 11 AM to 6 PM.",
  "matched_intent": "store_hours",
  "source": "rule"
}
```

</details>

## 🧠 Training the Models

Each notebook is self-contained and saves its artifact straight into `app/models/`, the exact path the serving layer expects:

| Notebook | Produces | Expects |
|---|---|---|
| `01_image_classifier_training.ipynb` | `product_classifier.h5` | Images in `data/products/<category>/` |
| `02_face_recognition_setup.ipynb` | `face_db.pkl` | Enrollment photos in `data/faces/<customer_id>/` |
| `03_sentiment_model_training.ipynb` | `sentiment_model.pkl`, `vectorizer.pkl` | `data/reviews.csv` (sample included) |

The repo ships small illustrative datasets so every notebook runs end-to-end out of the box — swap in real data for production-grade accuracy.

## 🧪 Testing

```bash
pytest tests/ -v
```

The suite passes both before and after training — model-dependent endpoints gracefully return `503` if their artifact hasn't been generated yet, so CI stays green on a fresh checkout.

## 🐳 Docker

```bash
docker build -t smart-retail-ai .
docker run -p 8000:8000 -e SMART_RETAIL_API_KEY=your-secret-key smart-retail-ai
```

## ⚙️ CI/CD

`.github/workflows/deploy.yml` runs on every push/PR to `main`:

1. **Lint** with `flake8`
2. **Test** with `pytest`
3. **Build** the Docker image (on `main` only, after tests pass)

Add a registry push step (Docker Hub / GHCR / your PaaS deploy hook) once you've picked a hosting target.

## ⚖️ Ethical Considerations

Facial recognition for returning-customer detection carries real consent, privacy, and bias implications:

- **Consent** — production use needs opt-in enrollment, not silent capture.
- **Bias** — face recognition accuracy varies across demographic groups; evaluate per-group, not just in aggregate.
- **Data retention** — define how long face encodings are stored and how a customer can request deletion.
- **Security** — encrypt biometric data at rest in any real deployment (this scaffold uses a plain pickle file for simplicity only).

## 🗺 Roadmap

- [ ] Fine-tuned DistilBERT sentiment model as a drop-in upgrade
- [ ] Train the chatbot's ML fallback classifier for better paraphrase coverage
- [ ] Persist visit/sentiment logs to a real database for `/dashboard/stats`
- [ ] Rate limiting + audit logging on `/recognize-face`
- [ ] Fairness evaluation across demographic subgroups

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Built by <b>Hemendra Singh</b>
</div>
