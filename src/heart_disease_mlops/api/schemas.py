from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(ge=50, le=260)
    chol: float = Field(ge=100, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=50, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    decision_threshold: float
    model_name: str
    model_version: str
