"""Модели запроса/ответа для HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnonymizeRequest(BaseModel):
    """Тело запроса для POST /v1/anonymize."""

    text: str = Field(..., description='Текст, который может содержать PII.')
    system_prompt: str | None = Field(
        default=None,
        description='Опциональный системный промпт для LLM.',
    )


class AnonymizeResponse(BaseModel):
    """Тело ответа для POST /v1/anonymize.

    В точности повторяет форму возврата PrivacyGuardPipeline.process().
    """

    anonymized_text: str
    llm_response: str
    stats: dict[str, Any]


class StatsResponse(BaseModel):
    """Тело ответа для GET /v1/stats.

    Только типы/количества сущностей — никогда исходные значения PII,
    в соответствии с инвариантом, который AuditLogger уже применяет для
    собственных логов.
    """

    session_duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    entity_counts: dict[str, int]
    total_entities_detected: int


class HealthResponse(BaseModel):
    """Тело ответа для GET /health."""

    status: str = 'ok'


class PDFAnonymizeStats(BaseModel):
    """Статистика без PII для ответа с отредактированным PDF.

    Отправляется как заголовок ответа (см. routes.py), так как телом
    эндпоинта служит сам отредактированный файл PDF.
    """

    pages_processed: int
    total_spans_redacted: int
    redacted_by_type: dict[str, int]
