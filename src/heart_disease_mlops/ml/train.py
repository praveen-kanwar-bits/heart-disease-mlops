from __future__ import annotations

import argparse
from datetime import UTC, datetime

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from heart_disease_mlops.config import load_settings
from heart_disease_mlops.data.download_data import download_dataset
from heart_disease_mlops.data.eda import run_eda
from heart_disease_mlops.data.validation import validate_dataset
from heart_disease_mlops.ml.artifacts import save_metadata, save_pipeline
from heart_disease_mlops.ml.evaluation import evaluate_and_plot, select_threshold
from heart_disease_mlops.ml.preprocessing import build_preprocessor


def train(smoke_test: bool = False) -> dict[str, float]:
    settings = load_settings()
    config = settings.values

    dataset_path = download_dataset()
    dataframe = validate_dataset(dataset_path)

    if smoke_test:
        dataframe = dataframe.sample(
            min(80, len(dataframe)), random_state=config["project"]["random_seed"]
        )

    features = config["data"]["numeric_features"] + config["data"]["categorical_features"]
    target = config["data"]["target_column"]

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe[features],
        dataframe[target],
        test_size=config["training"]["test_size"],
        random_state=config["project"]["random_seed"],
        stratify=dataframe[target],
    )

    preprocessor = build_preprocessor(
        config["data"]["numeric_features"], config["data"]["categorical_features"]
    )
    cv = StratifiedKFold(
        n_splits=config["training"]["cv_folds"],
        shuffle=True,
        random_state=config["project"]["random_seed"],
    )

    candidate_models: dict[str, tuple[object, dict[str, list[object]]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, random_state=config["project"]["random_seed"]),
            {
                "model__C": config["training"]["logistic_regression_grid"]["C"],
                "model__solver": config["training"]["logistic_regression_grid"]["solver"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config["project"]["random_seed"]),
            {
                "model__n_estimators": config["training"]["random_forest_grid"]["n_estimators"],
                "model__max_depth": config["training"]["random_forest_grid"]["max_depth"],
                "model__min_samples_split": config["training"]["random_forest_grid"][
                    "min_samples_split"
                ],
            },
        ),
    }

    mlflow.set_experiment("heart_disease_prediction")

    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -np.inf
    comparison_data = []

    with mlflow.start_run(run_name="training_pipeline") as parent_run:
        for name, (estimator, grid) in candidate_models.items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
                search = GridSearchCV(
                    pipeline,
                    param_grid=grid,
                    cv=cv,
                    scoring=config["training"]["scoring"],
                    n_jobs=-1,
                )
                search.fit(x_train, y_train)
                score = float(search.best_score_)
                mlflow.log_metric(f"{name}_cv_{config['training']['scoring']}", score)
                mlflow.log_params({f"{name}_{k}": v for k, v in search.best_params_.items()})

                if score > best_score:
                    best_name = name
                    best_score = score
                    best_pipeline = search.best_estimator_

                y_train_prob_temp = cross_val_predict(
                    search.best_estimator_, x_train, y_train, cv=cv, method="predict_proba"
                )[:, 1]
                temp_threshold = select_threshold(y_train.to_numpy(), y_train_prob_temp)
                y_test_prob_temp = search.best_estimator_.predict_proba(x_test)[:, 1]
                y_test_pred_temp = (y_test_prob_temp >= temp_threshold).astype(int)
                
                comparison_data.append({
                    "Model": name,
                    "CV ROC-AUC": score,
                    "Test Accuracy": accuracy_score(y_test, y_test_pred_temp),
                    "Precision": precision_score(y_test, y_test_pred_temp),
                    "Recall": recall_score(y_test, y_test_pred_temp),
                    "F1": f1_score(y_test, y_test_pred_temp),
                    "Test ROC-AUC": roc_auc_score(y_test, y_test_prob_temp),
                    "Best Parameters": str(search.best_params_),
                })

        assert best_pipeline is not None
        
        comp_df = pd.DataFrame(comparison_data)
        comp_path = settings.path("paths", "artifacts_dir") / "model_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        mlflow.log_artifact(str(comp_path))
        
        y_train_prob = cross_val_predict(
            best_pipeline, x_train, y_train, cv=cv, method="predict_proba"
        )[:, 1]
        threshold = select_threshold(y_train.to_numpy(), y_train_prob)
        
        y_prob = best_pipeline.predict_proba(x_test)[:, 1]
        metrics, metric_artifacts = evaluate_and_plot(
            y_test.to_numpy(), y_prob, threshold, settings.path("paths", "artifacts_dir") / "plots"
        )
        mlflow.log_metrics(metrics)
        mlflow.log_metric("decision_threshold", threshold)

        eda_artifacts = run_eda(dataframe, settings.path("paths", "artifacts_dir") / "eda")
        for artifact in metric_artifacts + eda_artifacts:
            mlflow.log_artifact(str(artifact))

        dataset_metadata = {
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "source": config["data"]["dataset_url"],
            "downloaded_utc": datetime.now(UTC).isoformat(),
            "selected_model": best_name,
            "decision_threshold": threshold,
            "mlflow_run_id": parent_run.info.run_id,
        }
        save_pipeline(best_pipeline, settings.path("paths", "model_joblib"))
        save_metadata(dataset_metadata, settings.path("paths", "model_metadata"))
        mlflow.log_artifact(str(settings.path("paths", "model_joblib")))
        mlflow.log_artifact(str(settings.path("paths", "model_metadata")))
        mlflow.sklearn.log_model(best_pipeline, artifact_path="model")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
