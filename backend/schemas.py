from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class LegalMetadata(BaseModel):
    title: str = Field(min_length=2, max_length=500)
    document_number: str | None = Field(default=None, max_length=120)
    issuing_body: str | None = Field(default=None, max_length=300)
    issued_at: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    validity_status: Literal["unverified", "verified_active", "expired", "repealed"] = "unverified"
    source_url: str | None = Field(default=None, max_length=2000)


class URLImportRequest(LegalMetadata):
    url: HttpUrl


class ChatAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    session_id: str | None = None
    retrieval_mode: Literal["graphrag", "vector"] = "graphrag"
    top_k: int | None = Field(default=None, ge=1, le=20)


class FeedbackRequest(BaseModel):
    value: Literal[-1, 1]


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class EvaluationRequest(BaseModel):
    retrieval_mode: Literal["graphrag", "vector", "both"] = "both"
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievalSettingsRequest(BaseModel):
    default_top_k: int = Field(ge=1, le=20)
    chunk_size: int = Field(ge=500, le=4000)
    chunk_overlap: int = Field(ge=0, le=1000)


class EvaluationReviewRequest(BaseModel):
    answer_correctness_score: float | None = Field(default=None, ge=0, le=1)
    faithfulness_score: float | None = Field(default=None, ge=0, le=1)
    review_notes: str | None = Field(default=None, max_length=4000)


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    database: str
    llm: dict
    embeddings: dict
