from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Pipeline:
    return joblib.load(path)


def save_metadata(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
