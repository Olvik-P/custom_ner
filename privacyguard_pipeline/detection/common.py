"""Common data structures for PII detection in PrivacyGuard Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PIISpan:
    """A detected PII span in the text.

    Attributes:
        start: Start character index.
        end: End character index.
        text: The original text of the span.
        entity_type: Type of PII entity (PHONE, EMAIL, PASSPORT, etc.).
        source: Which layer detected it ('pattern', 'natasha', 'context').
        confidence: Confidence score (0.0 to 1.0).
    """

    start: int
    end: int
    text: str
    entity_type: str
    source: str = 'pattern'
    confidence: float = 1.0


@dataclass
class DetectionResult:
    """Result of running all three detection layers.

    Attributes:
        spans: All detected PII spans, deduplicated and validated.
        layer_stats: Per-layer detection counts.
    """

    spans: list[PIISpan] = field(default_factory=list)
    layer_stats: dict[str, int] = field(
        default_factory=lambda: {
            'pattern': 0,
            'natasha': 0,
            'context': 0,
        }
    )


# Whitelist for common words that might be mistaken for names
WHITELIST: set[str] = {
    # Days of week
    'понедельник',
    'вторник',
    'среда',
    'четверг',
    'пятница',
    'суббота',
    'воскресенье',
    # Months
    'январь',
    'февраль',
    'март',
    'апрель',
    'май',
    'июнь',
    'июль',
    'август',
    'сентябрь',
    'октябрь',
    'ноябрь',
    'декабрь',
    # Common words that look like names
    'роза',
    'лилия',
    'ромашка',
    'гвоздика',
    'фиалка',
    'ландыш',
    'вишня',
    'груша',
    'слива',
    'яблоко',
    'надежда',
    'вера',
    'любовь',
    'софия',
    'рай',
    'ад',
    'мир',
    'воля',
    'слава',
    'искра',
    'заря',
    'утро',
    'вечер',
    'ночь',
    'камень',
    'река',
    'озеро',
    'море',
    'поле',
    'лес',
    'зима',
    'весна',
    'лето',
    'осень',
    'север',
    'юг',
    'запад',
    'восток',
    'белый',
    'черный',
    'красный',
    'синий',
    'зеленый',
    'белое',
    'черное',
    'красное',
    'синее',
    'зеленое',
    'белая',
    'черная',
    'красная',
    'синяя',
    'зеленая',
}
