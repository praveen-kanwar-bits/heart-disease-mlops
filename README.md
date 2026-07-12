# Heart Disease MLOps (UCI Cleveland)

Production-ready end-to-end MLOps project for heart disease risk prediction using the official UCI Heart Disease Cleveland dataset.

## Architecture

- **Data pipeline**: reproducible dataset download + schema validation
- **EDA**: missing values, histograms, class balance, correlations, and feature relationships
- **ML pipeline**: `ColumnTransformer` + `Pipeline`, model comparison, hyperparameter tuning
- **Experiment tracking**: MLflow metrics, params, artifacts, model logging
- **Serving**: FastAPI with health/readiness/prediction/Prometheus metrics
- **Ops**: Docker, GitHub Actions CI, Kubernetes manifests, Prometheus + Grafana

## Project Structure

```text
src/heart_disease_mlops/
  api/
  data/
  ml/
config/config.yaml
k8s/
monitoring/
tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Training

```bash
python -m heart_disease_mlops.ml.train
```

To download and validate dataset only:

```bash
python scripts/download_dataset.py
```

Smoke test:

```bash
python -m heart_disease_mlops.ml.train --smoke-test
```

Outputs:
- `artifacts/model_pipeline.joblib`
- `artifacts/model_metadata.json`
- plots under `artifacts/eda` and `artifacts/plots`
- MLflow run in local `mlruns/`

If the UCI host is unreachable in restricted environments, the downloader uses
the versioned local fallback snapshot at `data/source/processed.cleveland.data`.

## MLflow

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Track model comparison, best parameters, evaluation metrics, plots, and serialized model artifacts.

## API

Run:

```bash
uvicorn heart_disease_mlops.api.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /ready`
- `POST /predict`
- `GET /metrics`
- Swagger: `/docs`

Sample request body: `sample_request.json`.

## Testing and Quality

```bash
ruff check src tests scripts
black --check src tests scripts
pytest
```

Coverage is configured via `pytest-cov` in `pyproject.toml`.

## Docker

```bash
docker build -t heart-disease-mlops:latest .
docker run -p 8000:8000 heart-disease-mlops:latest
```

- non-root runtime user
- built-in health check against `/health`

## Kubernetes

Apply:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Deployment includes 2 replicas, readiness/liveness probes, and resource requests/limits.

## Monitoring

- Prometheus config: `monitoring/prometheus.yml`
- Grafana dashboard definition: `monitoring/grafana_dashboard.json`

## CI Workflow

`.github/workflows/ci.yml` executes:
- dependency install
- ruff lint
- black format check
- pytest
- training smoke test
- model artifact verification
- Docker build and container startup
- `/health` and `/predict` endpoint checks
- log/artifact upload

## Limitations

- Trains on Cleveland subset only.
- Intended as educational baseline; clinical use is inappropriate.
- Threshold selection is data-driven and should be revisited for deployment context.

## Responsible Use

This project is for academic and engineering practice only. It is **not** a medical device and must not be used for diagnosis or treatment decisions. Predictions can reflect dataset bias and uncertainty.
