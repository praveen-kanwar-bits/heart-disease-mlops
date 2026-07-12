from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_disease_mlops.config import load_settings
from heart_disease_mlops.ml.artifacts import load_pipeline

if __name__ == "__main__":
    settings = load_settings()
    pipeline = load_pipeline(settings.path("paths", "model_joblib"))
    if not hasattr(pipeline, "predict_proba"):
        raise SystemExit("Loaded artifact is not a probability classifier pipeline")
    print("Model artifact verified")
