from __future__ import annotations

from pydantic import BaseModel, Field


from typing import Literal

class PredictionRequest(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: float = Field(ge=50, le=260)
    chol: float = Field(ge=50, le=700)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: float = Field(ge=40, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=10)
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    decision_threshold: float
    model_name: str
    model_version: str
