"""PatternMatcher — слой 1: детекция PII на основе регулярных выражений.

Покрывает телефоны, email, российские документы (паспорт, ИНН, СНИЛС,
ОГРН), банковские карты (с проверкой по алгоритму Луна), IP-адреса,
URL и координаты.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from privacyguard_pipeline.constants import OMS_PREFER_CONFIDENCE
from privacyguard_pipeline.detection.common import (
    PIISpan,
    merge_overlapping_spans,
    prefer_greater_end,
    resolve_min_confidence,
)
from privacyguard_pipeline.detection.patterns import PATTERN_REGISTRY
from privacyguard_pipeline.detection.scoring import score_match
from privacyguard_pipeline.detection.validators import VALIDATOR_REGISTRY

logger = logging.getLogger(__name__)

TieRule = Callable[[PIISpan, PIISpan], PIISpan]


def _winner_of_type(
    kept: PIISpan,
    incoming: PIISpan,
    entity_type: str,
) -> PIISpan:
    """Возвращает тот из двух спанов, у которого заданный тип."""
    return kept if kept.entity_type == entity_type else incoming


def _inn_wins(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    return _winner_of_type(kept, incoming, 'INN')


def _by_confidence_passport_on_tie(
    kept: PIISpan,
    incoming: PIISpan,
) -> PIISpan:
    """PASSPORT/DRIVER_LICENSE: побеждает большая уверенность.

    Серия и номер у паспорта и водительского удостоверения одного
    формата; тип решает ключевое слово рядом. При равенстве оценок
    остаётся прежнее поведение - ``PASSPORT``.
    """
    if kept.confidence == incoming.confidence:
        return _winner_of_type(kept, incoming, 'PASSPORT')
    return kept if kept.confidence > incoming.confidence else incoming


def _oms_if_confident(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """CARD/OMS: побеждает ``OMS`` только при ключевом слове полиса.

    16-значный номер полиса ОМС и номер банковской карты проходят одну
    и ту же проверку; уверенность ``OMS`` достигает
    ``OMS_PREFER_CONFIDENCE`` лишь при ключевом слове полиса вместе с
    верной контрольной цифрой. Иначе остаётся ``CARD``.
    """
    oms = _winner_of_type(kept, incoming, 'OMS')
    if oms.confidence >= OMS_PREFER_CONFIDENCE:
        return oms
    return _winner_of_type(kept, incoming, 'CARD')


def _telegram_wins(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    return _winner_of_type(kept, incoming, 'TELEGRAM')


# Правила для спанов с ТОЧНО совпадающим диапазоном: ключ - пара типов.
# Вместо порядка регистрации в PATTERN_REGISTRY побеждает тип с более
# сильными доказательствами.
# - PASSPORT и PHONE могут оба совпасть на одном и том же голом
#   10-значном числе, что и INN (PASSPORT - формато-независимо; PHONE -
#   потому что phonenumbers принимает голую последовательность формата
#   9XXXXXXXXX как правдоподобный российский номер). validate_inn()
#   требует прохождения контрольной суммы ФНС, поэтому совпадение по
#   одному и тому же диапазону означает, что кандидат INN - *настоящий*
#   ИНН, и он побеждает.
# - URL и TELEGRAM: ссылка https://t.me/name - это и URL, и ник; тип
#   TELEGRAM точнее (и редактируется в PDF, в отличие от URL).
_EXACT_SPAN_RULES: dict[frozenset[str], TieRule] = {
    frozenset({'PASSPORT', 'INN'}): _inn_wins,
    frozenset({'PHONE', 'INN'}): _inn_wins,
    frozenset({'PASSPORT', 'DRIVER_LICENSE'}): _by_confidence_passport_on_tie,
    frozenset({'CARD', 'OMS'}): _oms_if_confident,
    frozenset({'URL', 'TELEGRAM'}): _telegram_wins,
}


def _prefer_pattern_span(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Тай-брейк для спанов одного уровня (pattern).

    По умолчанию делегирует :func:`prefer_greater_end`, кроме случая
    точного совпадения диапазона у пары типов из ``_EXACT_SPAN_RULES``.
    """
    if kept.start == incoming.start and kept.end == incoming.end:
        rule = _EXACT_SPAN_RULES.get(
            frozenset({kept.entity_type, incoming.entity_type}),
        )
        if rule is not None:
            return rule(kept, incoming)
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

    def detect(
        self,
        text: str,
        min_confidence: float | None = None,
    ) -> list[PIISpan]:
        """Прогоняет все regex-паттерны по тексту.

        Args:
            text: Входной текст для сканирования.
            min_confidence: Порог уверенности для этого вызова; ``None``
                — порог из настроек. Совпадения ниже порога
                отбрасываются до слияния спанов.

        Returns:
            Список обнаруженных PII-спанов.

        Raises:
            InvalidConfidenceError: Если порог вне диапазона 0.0-1.0.
        """
        threshold = resolve_min_confidence(min_confidence)
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

                confidence = score_match(entity_type, raw, text, start, end)
                if confidence < threshold:
                    continue

                spans.append(
                    PIISpan(
                        start=start,
                        end=end,
                        text=raw,
                        entity_type=entity_type,
                        source='pattern',
                        confidence=confidence,
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
