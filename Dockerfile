FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
COPY sample_request.json ./sample_request.json

RUN mkdir -p /app/artifacts
COPY artifacts/model_pipeline.joblib /app/artifacts/model_pipeline.joblib
COPY artifacts/model_metadata.json /app/artifacts/model_metadata.json

ENV PYTHONPATH=/app/src

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "heart_disease_mlops.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
