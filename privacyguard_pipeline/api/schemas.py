"""Request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnonymizeRequest(BaseModel):
    """Request body for POST /v1/anonymize."""

    text: str = Field(..., description='Text that may contain PII.')
    system_prompt: str | None = Field(
        default=None,
        description='Optional system prompt for the LLM.',
    )


class AnonymizeResponse(BaseModel):
    """Response body for POST /v1/anonymize.

    Mirrors PrivacyGuardPipeline.process()'s return shape exactly.
    """

    anonymized_text: str
    llm_response: str
    stats: dict[str, Any]


class StatsResponse(BaseModel):
    """Response body for GET /v1/stats.

    Entity types/counts only — never raw PII values, matching the
    invariant AuditLogger already enforces for its own logs.
    """

    session_duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    entity_counts: dict[str, int]
    total_entities_detected: int


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = 'ok'


class PDFAnonymizeStats(BaseModel):
    """Non-PII statistics for a redacted PDF response.

    Sent as a response header (see routes.py) since the endpoint's
    body is the redacted PDF file itself.
    """

    pages_processed: int
    total_spans_redacted: int
    redacted_by_type: dict[str, int]
