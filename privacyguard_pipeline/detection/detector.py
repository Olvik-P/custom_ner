"""Оркестратор трёхслойной детекции PII для PrivacyGuard Pipeline.

Слой 1: PatternMatcher — детекция на основе регулярных выражений.
Слой 2: NatashaNER — извлечение сущностей нейросетью через Natasha.
Слой 3: ContextualValidator — разрешение конфликтов и фильтрация по
whitelist.

Использование:
    detector = PIIDetector()
    result = detector.detect("some text with PII")
"""

from __future__ import annotations

import logging

from privacyguard_pipeline.detection.common import (
    DetectionResult,
    resolve_min_confidence,
)
from privacyguard_pipeline.detection.contextual_validator import (
    ContextualValidator,
)
from privacyguard_pipeline.detection.natasha_ner import NatashaNER
from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)


class PIIDetector:
    """Оркеструет все три слоя детекции.

    Использование:
        detector = PIIDetector()
        result = detector.detect("some text with PII")
    """

    def __init__(self) -> None:
        self.pattern_matcher = PatternMatcher()
        self.natasha_ner = NatashaNER()
        self.contextual_validator = ContextualValidator()

    @property
    def natasha_available(self) -> bool:
        """Проверяет, работоспособна ли Natasha NER."""
        return self.natasha_ner.is_available

    def detect(
        self,
        text: str,
        min_confidence: float | None = None,
    ) -> DetectionResult:
        """Прогоняет все три слоя детекции по входному тексту.

        Args:
            text: Входной текст для поиска PII.
            min_confidence: Порог уверенности только для этого вызова
                (0.0-1.0); ``None`` — порог из ``Settings``. Порог
                не хранится в состоянии детектора: он общий для
                параллельных запросов.

        Returns:
            DetectionResult с дедуплицированными спанами и статистикой
            по каждому слою.

        Raises:
            InvalidConfidenceError: Если порог вне диапазона 0.0-1.0.
        """
        threshold = resolve_min_confidence(min_confidence)
        result = DetectionResult()

        # Слой 1: PatternMatcher
        pattern_spans = self.pattern_matcher.detect(text, threshold)
        result.layer_stats['pattern'] = len(pattern_spans)
        logger.debug('PatternMatcher found %d spans', len(pattern_spans))

        # Слой 2: NatashaNER
        natasha_spans = self.natasha_ner.detect(text)
        result.layer_stats['natasha'] = len(natasha_spans)
        logger.debug('NatashaNER found %d spans', len(natasha_spans))

        # Слой 3: ContextualValidator
        final_spans = self.contextual_validator.validate(
            pattern_spans=pattern_spans,
            natasha_spans=natasha_spans,
            text=text,
        )
        final_spans = [s for s in final_spans if s.confidence >= threshold]
        result.layer_stats['context'] = len(final_spans)
        logger.debug(
            'ContextualValidator produced %d final spans',
            len(final_spans),
        )

        result.spans = final_spans
        return result
