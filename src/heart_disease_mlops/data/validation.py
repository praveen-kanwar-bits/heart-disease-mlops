from __future__ import annotations

from pathlib import Path

import pandas as pd

from heart_disease_mlops.config import load_settings


class DataValidationError(ValueError):
    pass


def validate_dataset(path: Path) -> pd.DataFrame:
    settings = load_settings()
    expected_columns = settings.values["data"]["expected_columns"]
    frame = pd.read_csv(path)

    if list(frame.columns) != expected_columns:
        raise DataValidationError("Dataset columns do not match expected schema.")

    if frame.empty:
        raise DataValidationError("Dataset is empty.")

    if frame[settings.values["data"]["target_column"]].nunique() < 2:
        raise DataValidationError("Target must contain at least two classes.")

    duplicates = frame.duplicated().sum()
    if duplicates > 0:
        frame = frame.drop_duplicates().reset_index(drop=True)

    return frame
