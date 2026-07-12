from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart_disease_mlops.data.download_data import download_dataset
from heart_disease_mlops.data.validation import validate_dataset

if __name__ == "__main__":
    path = download_dataset()
    frame = validate_dataset(path)
    print(f"Saved validated dataset to {path} with shape {frame.shape}")
