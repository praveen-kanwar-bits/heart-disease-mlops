.PHONY: install install-dev lint format-check test train smoke-api

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check src tests scripts

format-check:
	black --check src tests scripts

test:
	pytest

train:
	python -m heart_disease_mlops.ml.train

smoke-api:
	python -m heart_disease_mlops.ml.train --smoke-test
	uvicorn heart_disease_mlops.api.app:app --host 0.0.0.0 --port 8000 & \
	PID=$$!; \
	sleep 5; \
	curl -f http://localhost:8000/health; \
	kill $$PID
