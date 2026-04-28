from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    run_id: Optional[str] = Field(default=None, description="W&B run id (model source).")
    image_url: Optional[str] = Field(default=None, description="Image URL to fetch.")
    image_base64: Optional[str] = Field(default=None, description="Raw base64 or data URL.")
    top_k: int = Field(default=5, ge=1, le=50)


class ScoredLabel(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    run_id: Optional[str]
    artifact: Optional[str]
    predicted_label: str
    predicted_id: int
    labels: List[str]
    probs: List[float]
    top_k: List[ScoredLabel]


class BatchPredictRequest(BaseModel):
    run_id: Optional[str] = None
    items: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
