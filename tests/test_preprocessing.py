from __future__ import annotations

from heart_disease_mlops.config import load_settings
from heart_disease_mlops.ml.preprocessing import build_preprocessor


def test_preprocessor_builds_with_expected_transformers():
    settings = load_settings().values
    preprocessor = build_preprocessor(
        settings["data"]["numeric_features"], settings["data"]["categorical_features"]
    )
    assert preprocessor.transformers[0][0] == "num"
    assert preprocessor.transformers[1][0] == "cat"
