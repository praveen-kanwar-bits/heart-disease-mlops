from __future__ import annotations

from heart_disease_mlops.data.validation import validate_dataset


def test_dataset_validation_passes(trained_artifacts):
    frame = validate_dataset(trained_artifacts)
    assert not frame.empty
    assert set(frame["target"].unique()).issubset({0, 1})
