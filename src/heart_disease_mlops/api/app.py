from __future__ import annotations

from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from heart_disease_mlops.api.logging_utils import configure_logging, log_request
from heart_disease_mlops.api.schemas import PredictionRequest, PredictionResponse
from heart_disease_mlops.config import load_settings
from heart_disease_mlops.ml.artifacts import load_pipeline

REQUEST_COUNT = Counter("heart_predict_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("heart_predict_latency_seconds", "Prediction latency")

configure_logging()
app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.middleware("http")(log_request)


@app.on_event("startup")
def startup() -> None:
    settings = load_settings()
    model_path = settings.path("paths", "model_joblib")
    app.state.settings = settings
    app.state.model_ready = model_path.exists()
    app.state.model = load_pipeline(model_path) if model_path.exists() else None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, bool]:
    return {"ready": bool(app.state.model_ready)}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if not app.state.model_ready or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    started = perf_counter()
    REQUEST_COUNT.inc()

    frame = pd.DataFrame([payload.model_dump()])
    probability = float(app.state.model.predict_proba(frame)[0][1])
    threshold = app.state.settings.values["api"]["confidence_threshold"]
    prediction = int(probability >= threshold)

    REQUEST_LATENCY.observe(perf_counter() - started)
    return PredictionResponse(
        prediction=prediction,
        confidence=probability,
        model_name=app.state.settings.values["api"]["model_name"],
        model_version=app.state.settings.values["api"]["model_version"],
    )


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
