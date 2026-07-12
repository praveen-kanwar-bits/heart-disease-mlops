from __future__ import annotations

import json
from contextlib import asynccontextmanager
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from heart_disease_mlops.api.logging_utils import configure_logging, log_request
from heart_disease_mlops.api.schemas import PredictionRequest, PredictionResponse
from heart_disease_mlops.config import load_settings
from heart_disease_mlops.ml.artifacts import load_pipeline

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
REQUEST_ERROR_COUNT = Counter(
    "api_request_errors_total", "Total request errors", ["endpoint", "error_type"]
)
PREDICTION_COUNT = Counter(
    "model_predictions_total", "Total predictions made", ["prediction", "model_version"]
)

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    model_path = settings.path("paths", "model_joblib")
    metadata_path = settings.path("paths", "model_metadata")
    app.state.settings = settings
    app.state.model_ready = model_path.exists()
    app.state.model = load_pipeline(model_path) if model_path.exists() else None

    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
            app.state.threshold = float(
                meta.get("decision_threshold", settings.values["api"]["confidence_threshold"])
            )
    else:
        app.state.threshold = float(settings.values["api"]["confidence_threshold"])
    yield


app = FastAPI(title="Heart Disease Prediction API", version="1.0.0", lifespan=lifespan)
app.middleware("http")(log_request)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"ready": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not app.state.model_ready or app.state.model is None:
        REQUEST_ERROR_COUNT.labels(endpoint="/predict", error_type="model_not_loaded").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    started = perf_counter()
    try:
        frame = pd.DataFrame([payload.model_dump()])
        probability = float(app.state.model.predict_proba(frame)[0][1])
        threshold = app.state.threshold
        prediction = int(probability >= threshold)
        model_version = app.state.settings.values["api"]["model_version"]

        PREDICTION_COUNT.labels(prediction=str(prediction), model_version=model_version).inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(perf_counter() - started)
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

        return PredictionResponse(
            prediction=prediction,
            confidence=probability,
            decision_threshold=threshold,
            model_name=app.state.settings.values["api"]["model_name"],
            model_version=model_version,
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        REQUEST_ERROR_COUNT.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from e


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
