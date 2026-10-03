"""Оркестратор трёхслойной детекции PII для PrivacyGuard Pipeline.

Слой 1: PatternMatcher — детекция на основе регулярных выражений.
Слой 2: NatashaNER — извлечение сущностей нейросетью через Natasha.
Слой 3: ContextualValidator — разрешение конфликтов, фильтрация по
whitelist и морфологические фильтры PER.

Поверх слоёв — списки оператора: allow (спан не маскируется) и deny
(строка маскируется всегда, как тип CUSTOM).

Использование:
    detector = PIIDetector()
    result = detector.detect("some text with PII")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.detection.common import (
    DetectionResult,
    PIISpan,
    merge_overlapping_spans,
    resolve_min_confidence,
)
from privacyguard_pipeline.detection.contextual_validator import (
    ContextualValidator,
)
from privacyguard_pipeline.detection.lists import compile_lists
from privacyguard_pipeline.detection.natasha_ner import NatashaNER
from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)


def _prefer_existing(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Тай-брейк слияния с deny: тип и источник остаются у не-deny спана."""
    if kept.source == 'deny' and incoming.source != 'deny':
        return incoming
    return kept


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
        allow_list: Sequence[str] | None = None,
        deny_list: Sequence[str] | None = None,
    ) -> DetectionResult:
        """Прогоняет все три слоя детекции по входному тексту.

        Args:
            text: Входной текст для поиска PII.
            min_confidence: Порог уверенности только для этого вызова
                (0.0-1.0); ``None`` — порог из ``Settings``. Порог
                не хранится в состоянии детектора: он общий для
                параллельных запросов.
            allow_list: Строки, которые не маскируются, только для этого
                вызова; дополняют ``ALLOW_LIST`` из настроек.
            deny_list: Строки, которые маскируются всегда (тип CUSTOM),
                только для этого вызова; дополняют ``DENY_LIST`` из
                настроек. При конфликте deny сильнее allow. Списки не
                хранятся в состоянии детектора.

        Returns:
            DetectionResult с дедуплицированными спанами и статистикой
            по каждому слою.

        Raises:
            InvalidConfidenceError: Если порог вне диапазона 0.0-1.0.
            InvalidListEntryError: Если запись списка невалидна.
        """
        threshold = resolve_min_confidence(min_confidence)
        allow, deny = compile_lists(
            settings.allow_list,
            allow_list,
            settings.deny_list,
            deny_list,
        )
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

        # Списки оператора. Allow снимает спаны слоёв до добавления deny —
        # поэтому deny сильнее allow и не зависит от порога и whitelist.
        if len(allow):
            final_spans = [s for s in final_spans if not allow.matches(s.text)]
        deny_spans = deny.find_spans(text)
        if deny_spans:
            final_spans = merge_overlapping_spans(
                final_spans + deny_spans,
                text,
                prefer=_prefer_existing,
            )
        logger.debug(
            'Lists applied: %d allow entries, %d deny entries, '
            '%d deny matches',
            len(allow),
            len(deny),
            len(deny_spans),
        )

        result.spans = final_spans
        return result
