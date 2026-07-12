from __future__ import annotations

from heart_disease_mlops.config import load_settings
from heart_disease_mlops.ml.artifacts import load_pipeline
from heart_disease_mlops.ml.train import train


def test_train_smoke_produces_metrics_and_artifacts():
    metrics = train(smoke_test=True)
    assert "roc_auc" in metrics

    settings = load_settings()
    assert settings.path("paths", "model_joblib").exists()
    assert settings.path("paths", "model_metadata").exists()


def test_artifact_loader_returns_predictable_pipeline(trained_artifacts):
    settings = load_settings()
    model = load_pipeline(settings.path("paths", "model_joblib"))
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
