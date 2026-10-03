"""Прогон детектора по примерам бенчмарка."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.detection.common import resolve_min_confidence
from privacyguard_pipeline.evaluation.dataset import Example
from privacyguard_pipeline.evaluation.metrics import (
    Metrics,
    PredSpan,
    evaluate,
)


@dataclass
class RunResult:
    """Результат прогона детектора.

    Attributes:
        metrics: Счётчики по группам.
        min_confidence: Действующий порог уверенности прогона.
        natasha_available: Загрузилась ли Natasha.
        errors: Промахи для разбора: id примера, тип бенчмарка и оффсеты
            золотых сущностей, не покрытых ни одним спаном. Текстов нет.
    """

    metrics: Metrics
    min_confidence: float
    natasha_available: bool
    errors: list[dict[str, object]]


def run_detector(
    examples: Sequence[Example],
    min_confidence: float | None = None,
    detector: PIIDetector | None = None,
) -> RunResult:
    """Прогоняет один ``PIIDetector`` по всем примерам и считает метрики.

    Args:
        examples: Размеченные примеры.
        min_confidence: Порог уверенности; ``None`` — из настроек.
        detector: Готовый детектор (для тестов); по умолчанию создаётся
            один на весь прогон.

    Raises:
        InvalidConfidenceError: Если порог вне диапазона 0.0-1.0.
    """
    effective = resolve_min_confidence(min_confidence)
    detector = detector or PIIDetector()
    predictions: list[list[PredSpan]] = []
    errors: list[dict[str, object]] = []
    for example in examples:
        result = detector.detect(example.text, min_confidence=effective)
        spans = [PredSpan(s.start, s.end, s.entity_type) for s in result.spans]
        predictions.append(spans)
        for entity in example.entities:
            covered = any(
                min(entity.end, s.end) > max(entity.start, s.start)
                for s in spans
            )
            if not covered:
                errors.append(
                    {
                        'id': example.id,
                        'type': entity.type,
                        'start': entity.start,
                        'end': entity.end,
                    }
                )
    return RunResult(
        metrics=evaluate(examples, predictions),
        min_confidence=effective,
        natasha_available=detector.natasha_ner.is_available,
        errors=errors,
    )
