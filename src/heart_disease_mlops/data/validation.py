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

    if frame.empty or len(frame) < 50:
        raise DataValidationError("Dataset is empty or has too few rows.")

    target_col = settings.values["data"]["target_column"]
    if not set(frame[target_col].dropna().unique()).issubset({0, 1}):
        raise DataValidationError("Target column must be strictly binary after conversion.")

    if frame[target_col].nunique() < 2:
        raise DataValidationError("Target must contain at least two classes.")

    if not set(frame["cp"].dropna().unique()).issubset({1, 2, 3, 4}):
        raise DataValidationError("Feature 'cp' contains invalid category codes.")
        
    if not set(frame["slope"].dropna().unique()).issubset({1, 2, 3}):
        raise DataValidationError("Feature 'slope' contains invalid category codes.")
        
    if not set(frame["thal"].dropna().unique()).issubset({3, 6, 7}):
        raise DataValidationError("Feature 'thal' contains invalid category codes.")

    duplicates = frame.duplicated().sum()
    if duplicates > 0:
        frame = frame.drop_duplicates().reset_index(drop=True)

    return frame
