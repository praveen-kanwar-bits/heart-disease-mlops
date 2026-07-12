from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(dataframe: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []

    missing = dataframe.isna().sum().sort_values(ascending=False)
    missing_path = output_dir / "missing_values.png"
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing.index, y=missing.values)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(missing_path)
    plt.close()
    artifacts.append(missing_path)

    histogram_path = output_dir / "histograms.png"
    dataframe.hist(figsize=(14, 10))
    plt.tight_layout()
    plt.savefig(histogram_path)
    plt.close()
    artifacts.append(histogram_path)

    class_path = output_dir / "class_distribution.png"
    plt.figure(figsize=(5, 4))
    sns.countplot(data=dataframe, x="target")
    plt.tight_layout()
    plt.savefig(class_path)
    plt.close()
    artifacts.append(class_path)

    heatmap_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    artifacts.append(heatmap_path)

    pairplot_path = output_dir / "feature_relationships.png"
    subset = dataframe[["age", "chol", "thalach", "oldpeak", "target"]].copy()
    sns.pairplot(subset, hue="target")
    plt.savefig(pairplot_path)
    plt.close()
    artifacts.append(pairplot_path)

    boxplot_path = output_dir / "boxplots.png"
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(["age", "trestbps", "chol", "thalach", "oldpeak"], 1):
        plt.subplot(2, 3, i)
        sns.boxplot(data=dataframe, x="target", y=feature)
    plt.tight_layout()
    plt.savefig(boxplot_path)
    plt.close()
    artifacts.append(boxplot_path)

    return artifacts
