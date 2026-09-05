"""PatternMatcher — слой 1: детекция PII на основе регулярных выражений.

Покрывает телефоны, email, российские документы (паспорт, ИНН, СНИЛС,
ОГРН), банковские карты (с проверкой по алгоритму Луна), IP-адреса,
URL и координаты.
"""

from __future__ import annotations

import logging

from privacyguard_pipeline.constants import PATTERN_CONFIDENCE
from privacyguard_pipeline.detection.common import (
    PIISpan,
    merge_overlapping_spans,
    prefer_greater_end,
)
from privacyguard_pipeline.detection.patterns import PATTERN_REGISTRY
from privacyguard_pipeline.detection.validators import VALIDATOR_REGISTRY

logger = logging.getLogger(__name__)

# Паттерны PASSPORT и INN могут совпасть на одном и том же голом
# 10-значном числе. validate_inn() теперь требует прохождения
# контрольной суммы ФНС, поэтому совпадение по одному и тому же
# диапазону означает, что кандидат INN — *настоящий* ИНН; отдаём ему
# предпочтение перед формато-независимым совпадением PASSPORT, вместо
# того чтобы полагаться на порядок регистрации в PATTERN_REGISTRY.
_PASSPORT_INN_TIE: frozenset[str] = frozenset({'PASSPORT', 'INN'})


def _prefer_pattern_span(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Тай-брейк для спанов одного уровня (pattern).

    По умолчанию делегирует :func:`prefer_greater_end`, кроме случая
    точного совпадения диапазона PASSPORT/INN, где побеждает ИНН,
    прошедший проверку контрольной суммы.
    """
    if (
        kept.start == incoming.start
        and kept.end == incoming.end
        and {kept.entity_type, incoming.entity_type} == _PASSPORT_INN_TIE
    ):
        return kept if kept.entity_type == 'INN' else incoming
    return prefer_greater_end(kept, incoming)


class PatternMatcher:
    """Слой 1: детекция PII на основе регулярных выражений.

    Покрывает телефоны, email, российские документы (паспорт, ИНН,
    СНИЛС, ОГРН), банковские карты (с проверкой по алгоритму Луна),
    IP-адреса, URL и координаты.
    """

    def __init__(self) -> None:
        self._patterns = PATTERN_REGISTRY

    # ------------------------------------------------------------------
    # Слияние спанов
    # ------------------------------------------------------------------

    @staticmethod
    def merge_overlapping(
        spans: list[PIISpan],
        text: str,
    ) -> list[PIISpan]:
        """Сливает пересекающиеся спаны в их объединение.

        Args:
            spans: Список спанов, возможно пересекающихся.
            text: Полный исходный текст, в котором были обнаружены
                спаны (используется для пересчёта ``.text`` слитого
                спана под его расширенные границы).

        Returns:
            Дедуплицированный список спанов, каждый из которых
            покрывает полное объединение всех входных диапазонов,
            пересекавшихся с ним — ни один PII-символ, покрытый
            входным спаном, не теряется в результате.
        """
        return merge_overlapping_spans(
            spans,
            text,
            prefer=_prefer_pattern_span,
        )

    # ------------------------------------------------------------------
    # Основная детекция
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[PIISpan]:
        """Прогоняет все regex-паттерны по тексту.

        Args:
            text: Входной текст для сканирования.

        Returns:
            Список обнаруженных PII-спанов.
        """
        spans: list[PIISpan] = []

        for entity_type, pattern in self._patterns:
            for match in pattern.finditer(text):
                raw = match.group(0)
                start, end = match.start(), match.end()

                if not self._validate_match(
                    entity_type,
                    raw,
                    text,
                    start,
                    end,
                ):
                    continue

                spans.append(
                    PIISpan(
                        start=start,
                        end=end,
                        text=raw,
                        entity_type=entity_type,
                        source='pattern',
                        confidence=PATTERN_CONFIDENCE,
                    ),
                )

        return self.merge_overlapping(spans, text)

    def _validate_match(
        self,
        entity_type: str,
        raw: str,
        text: str,
        start: int,
        end: int,
    ) -> bool:
        """Прогоняет валидацию, специфичную для типа сущности.

        Args:
            entity_type: Тип совпавшей сущности.
            raw: Совпавший исходный текст.
            text: Полный входной текст.
            start: Индекс начала совпадения.
            end: Индекс конца совпадения.

        Returns:
            True, если совпадение валидно, False — чтобы его
            пропустить.
        """
        validator = VALIDATOR_REGISTRY.get(entity_type)
        if validator is not None:
            return validator(raw, text, start, end)
        return True
