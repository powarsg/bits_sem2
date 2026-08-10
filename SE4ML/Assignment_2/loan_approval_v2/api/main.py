"""FastAPI application — REST API for loan-approval inference.

Endpoints
---------
GET  /health          — liveness/readiness probe
GET  /metrics         — model evaluation metrics
POST /predict         — single-record inference
POST /predict/batch   — batch inference (list of records)
"""

import json
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

from src.inference import InferenceEngine  # noqa: E402
from src.logger import get_logger  # noqa: E402
from api.schemas import (  # noqa: E402
    LoanApplicationRequest,
    PredictionResponse,
    HealthResponse,
    MetricsResponse,
)

logger = get_logger("api")

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "model_artifacts"),
)
TEST_DATA_PATH = os.path.join(MODEL_DIR, "test_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Application-level state
_engine: InferenceEngine | None = None
_metrics: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup."""
    global _engine, _metrics
    logger.info("API startup — loading artifacts from '%s'", MODEL_DIR)
    try:
        _engine = InferenceEngine(model_dir=MODEL_DIR)
        _engine.load_artifacts()
        logger.info("Artifacts loaded successfully")
    except FileNotFoundError as exc:
        logger.warning("Artifacts not found at startup: %s — run train.py first", exc)
        _engine = None

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
        logger.info("Metrics loaded from '%s'", METRICS_PATH)
    yield
    logger.info("API shutdown")


app = FastAPI(
    title="Loan Approval Prediction API",
    description=(
        "REST API exposing the Loan Approval ML model inference. "
        "Group 216 — AIMLCZG546 SE4ML Assignment II."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and readiness probe",
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_engine is not None)


# ---------------------------------------------------------------------------
# Model metrics
# ---------------------------------------------------------------------------

@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Retrieve model evaluation metrics",
    tags=["Model"],
)
def get_metrics() -> MetricsResponse:
    if _metrics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metrics not available — model has not been trained yet",
        )
    return MetricsResponse(**_metrics)


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict loan approval for a single applicant",
    tags=["Inference"],
)
def predict(request: LoanApplicationRequest) -> PredictionResponse:
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — run train.py first",
        )
    try:
        record = request.model_dump()
        result = _engine.predict_single(record)
        logger.info(
            "POST /predict — decision=%s  p=%.4f",
            result["decision"],
            result["approval_probability"],
        )
        return PredictionResponse(**result)
    except Exception as exc:
        logger.error("Prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

@app.post(
    "/predict/batch",
    summary="Predict loan approval for a list of applicants",
    tags=["Inference"],
)
def predict_batch(requests: list[LoanApplicationRequest]) -> list[PredictionResponse]:
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded — run train.py first",
        )
    if not requests:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request list must not be empty",
        )
    try:
        records = [r.model_dump() for r in requests]
        df = pd.DataFrame(records)
        probs = _engine.predict_proba(df)
        labels = _engine.predict(df)
        results = []
        for label, prob_row in zip(labels, probs):
            results.append(
                PredictionResponse(
                    loan_status=int(label),
                    decision="Approved" if label == 1 else "Rejected",
                    approval_probability=round(float(prob_row[1]), 4),
                )
            )
        logger.info("POST /predict/batch — %d records processed", len(results))
        return results
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {exc}",
        ) from exc
