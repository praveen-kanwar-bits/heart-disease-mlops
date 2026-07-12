from __future__ import annotations

from fastapi.testclient import TestClient

from heart_disease_mlops.api.app import app
from heart_disease_mlops.ml.train import train


def _ensure_model_ready() -> None:
    train(smoke_test=True)


def test_health_and_ready_endpoints():
    _ensure_model_ready()
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["ready"] is True


def test_predict_endpoint_success():
    _ensure_model_ready()
    with TestClient(app) as client:
        payload = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert set(body).issuperset({"prediction", "confidence", "model_name", "model_version"})


def test_predict_endpoint_validation_error():
    _ensure_model_ready()
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": -1})
        assert response.status_code == 422
