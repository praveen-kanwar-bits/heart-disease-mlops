from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from heart_disease_mlops.api.app import app
from heart_disease_mlops.ml.train import train


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Ensure the model is trained exactly once for the entire test session."""
    train(smoke_test=True)


def test_health_and_ready_endpoints():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").json()["ready"] is True


def test_predict_endpoint_success():
    with TestClient(app) as client:
        payload = {
            "age": 63,
            "sex": 1,
            "cp": 4,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 7,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        
        assert "prediction" in body
        assert body["prediction"] in [0, 1]
        
        assert "confidence" in body
        assert 0.0 <= body["confidence"] <= 1.0
        
        assert "decision_threshold" in body
        assert 0.0 <= body["decision_threshold"] <= 1.0
        
        assert "model_name" in body
        assert isinstance(body["model_name"], str)
        assert len(body["model_name"]) > 0
        
        assert "model_version" in body
        assert isinstance(body["model_version"], str)


def test_predict_endpoint_validation_error():
    with TestClient(app) as client:
        response = client.post("/predict", json={"age": -1})
        assert response.status_code == 422


def test_valid_uci_category_codes():
    """Prove that valid UCI categories are accepted by the schema."""
    with TestClient(app) as client:
        payload = {
            "age": 63,
            "sex": 1,
            "cp": 4,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 3,
            "ca": 0,
            "thal": 7,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


def test_model_missing_returns_503():
    """Test that the API returns 503 if the model is not ready."""
    with TestClient(app) as client:
        app.state.model_ready = False
        app.state.model = None
        
        response = client.get("/ready")
        assert response.status_code == 503
        
        payload = {
            "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1, 
            "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, 
            "ca": 0, "thal": 7
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
