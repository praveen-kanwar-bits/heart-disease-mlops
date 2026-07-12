from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import pandas as pd

from heart_disease_mlops.config import load_settings

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


def download_dataset(destination: Path | None = None) -> Path:
    settings = load_settings()
    output_path = destination or settings.path("paths", "raw_data_file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_url = settings.values["data"]["dataset_url"]

    try:
        dataframe = pd.read_csv(source_url, names=COLUMNS)
    except URLError:
        fallback_path = settings.path("data", "fallback_dataset_file")
        dataframe = pd.read_csv(fallback_path, names=COLUMNS)
    dataframe = dataframe.replace("?", pd.NA)
    dataframe = dataframe.apply(pd.to_numeric, errors="coerce")
    dataframe["target"] = dataframe["target"].astype(float).ge(1).astype(int)
    dataframe.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    print(download_dataset())
