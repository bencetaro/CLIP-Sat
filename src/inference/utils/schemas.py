from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    run_id: Optional[str] = Field(default=None, description="W&B run id (model source).")
    image_url: Optional[str] = Field(default=None, description="Image URL to fetch.")
    image_base64: Optional[str] = Field(default=None, description="Raw base64 or data URL.")
    top_k: int = Field(default=5, ge=1, le=14)


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


class FeedbackRequest(BaseModel):
    user_rating: int = Field(ge=1, le=5, validation_alias="app_rating")
    user_feedback: Optional[str] = Field(default=None, max_length=2000, validation_alias="app_comment")


class LLMReviewRequest(BaseModel):
    labels: List[str] = Field(min_length=1)
    probs: List[float] = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=14)
    backend: str = Field(default="llama_cpp")
    use_gpu: bool = Field(default=False)
