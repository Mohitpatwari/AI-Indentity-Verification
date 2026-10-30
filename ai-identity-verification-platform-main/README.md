# AI Identity Verification & Document Intelligence Platform

A production-oriented identity verification pipeline combining OCR, document classification, information extraction, face verification, deep-learning-based forgery detection, and LLM-generated reporting — built with FastAPI, PyTorch, and modern pretrained vision models.



---

## Overview

This platform ingests identity documents (Aadhaar, PAN, Passport, etc.) alongside a selfie, and returns a structured verification result: document type, extracted fields, face match confidence, a forgery risk score from a custom-trained CNN, and an AI-generated summary report — all through a single REST API and a live dashboard.

## Features

- **Document classification** — identifies document type from OCR text
- **OCR** — PaddleOCR-based text extraction with per-line confidence scores
- **Information extraction** — regex-based structured field extraction (ID numbers, dates)
- **Face verification** — ArcFace embeddings via InsightFace, cosine-similarity matching between ID photo and selfie
- **Forgery detection** — a custom CNN trained from scratch in PyTorch on synthetically generated tampered/authentic image pairs, deployed via ONNX Runtime
- **AI-generated verification reports** — Gemini-powered natural language summaries, with automatic fallback to a template report if the LLM call fails
- **Verification history** — persisted to SQLite, queryable via a `/history` endpoint
- **Dashboard UI** — single-page frontend with live results visualization
- **Dockerized** — fully containerized for reproducible deployment

## Architecture

Browser (dashboard)
│
▼
FastAPI (versioned REST API)
│
├── OCR Service (PaddleOCR)
├── Classification Service (keyword-based, extensible to a trained classifier)
├── Extraction Service (regex-based structured field parsing)
├── Face Verification (InsightFace / ArcFace, ONNX Runtime)
├── Forgery Detection (custom PyTorch CNN, exported to ONNX for inference)
└── Report Generation (Gemini LLM, template fallback)
│
▼
SQLite (verification history)


**Design note:** training uses PyTorch (autograd, custom training loop, evaluation metrics). Deployed inference uses ONNX Runtime — this avoids a real dependency conflict between PaddlePaddle and PyTorch's native libraries in the same process, and is also standard practice for optimized, framework-agnostic production inference.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Pydantic, SQLAlchemy |
| OCR | PaddleOCR |
| Face Verification | InsightFace (ArcFace), ONNX Runtime |
| Forgery Detection | PyTorch (training), ONNX Runtime (inference) |
| LLM Reporting | Google Gemini API |
| Database | SQLite |
| Deployment | Docker, Hugging Face Spaces |

## Forgery Detection Model

Since no public labeled dataset of forged identity documents exists (for good reason — privacy/legal), training data was **synthetically generated**: base document images augmented with natural transformations (authentic class) versus localized recompression/blending edits simulating tampering (tampered class). This is a standard bootstrapping technique in forensic ML research.

**Architecture:** a compact 4-block CNN with BatchNorm and max-pooling (max pooling was deliberately chosen over average pooling — localized tampering signals get diluted by global average pooling, since the anomaly occupies only a small region of the image).

**Evaluation** (on held-out synthetic validation split, 1200 images):

| Metric | Score |
|---|---|
| Accuracy | 88.58% |
| Precision | 100% |
| Recall | 77.17% |
| F1 Score | 87.11% |

Precision of 100% means the model never falsely flagged an authentic image as tampered in this evaluation — a deliberately conservative decision boundary, prioritizing avoiding false accusations over catching every case. Full metrics in `docs/evaluation_report.txt`.

**Limitation:** evaluated on synthetic data; real-world document evaluation with genuine forged samples is a natural next step once labeled data becomes available.

## Project Structure

├── app/ # FastAPI application
│ ├── api/v1/ # Versioned API routes
│ ├── core/ # Config, logging
│ ├── db/ # SQLAlchemy models & session
│ ├── ml_models/ # Deployed ONNX model
│ ├── schemas/ # Pydantic request/response models
│ └── services/ # OCR, classification, extraction, face, forgery, report
├── ml/training/ # Model training pipeline (PyTorch)
│ ├── generate_synthetic_data.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ └── export_to_onnx.py
├── static/ # Dashboard frontend
├── docs/ # Evaluation reports
├── Dockerfile
└── requirements.txt


## Running Locally

```bash
git clone https://github.com/SouravStark009/ai-identity-verification-platform.git
cd ai-identity-verification-platform

conda create -n idverify python=3.11 -y
conda activate idverify

pip install -r requirements.txt

echo "GEMINI_API_KEY=your_key_here" >> .env

python -m uvicorn app.main:app
```

Visit `http://127.0.0.1:8000`.

## Running with Docker

```bash
docker build -t identity-verification-app .
docker run -p 8000:7860 identity-verification-app
```

## Retraining the Forgery Detection Model

```bash
cd ml/training
python generate_synthetic_data.py
python train.py
python evaluate.py
python export_to_onnx.py
```

## API Reference

Interactive API docs available at `/docs` (Swagger UI) once running.

- `POST /api/v1/verify` — full verification pipeline (multipart: `id_document`, `selfie`)
- `GET /api/v1/history` — recent verification records

## Roadmap

- [ ] Fine-tuned document classifier (replacing keyword matching)
- [ ] LayoutLMv3/Donut-based structured extraction
- [ ] Real-world forgery evaluation dataset
- [ ] React frontend with authentication
- [ ] Cloud object storage integration

## License

