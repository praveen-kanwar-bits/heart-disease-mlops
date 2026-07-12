from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    scores = (2 * precision * recall) / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(scores[:-1])]) if len(thresholds) else 0.5


def evaluate_and_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    output_dir: Path,
) -> tuple[dict[str, float], list[Path]]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    confusion_path = output_dir / "confusion_matrix.png"
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(confusion_path)
    plt.close()
    artifacts.append(confusion_path)

    roc_path = output_dir / "roc_curve.png"
    RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()
    artifacts.append(roc_path)

    pr_path = output_dir / "precision_recall_curve.png"
    PrecisionRecallDisplay.from_predictions(y_true, y_prob)
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    artifacts.append(pr_path)

    return metrics, artifacts
