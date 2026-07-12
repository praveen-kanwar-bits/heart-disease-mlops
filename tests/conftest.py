from __future__ import annotations

from pathlib import Path

import pytest

from heart_disease_mlops.data.download_data import download_dataset
from heart_disease_mlops.ml.train import train


@pytest.fixture(scope="session")
def trained_artifacts() -> Path:
    train(smoke_test=True)
    return download_dataset()
