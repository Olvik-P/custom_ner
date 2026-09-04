"""Common data structures for PII detection in PrivacyGuard Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


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


PreferSpan = Callable[[PIISpan, PIISpan], PIISpan]


def prefer_greater_end(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Default tie-break: keep whichever span currently reaches further.

    Matches the historical ``PatternMatcher.merge_overlapping`` rule —
    on an exact tie (equal ``end``) the already-kept span wins.
    """
    return incoming if incoming.end > kept.end else kept


def prefer_first(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Tie-break that always keeps the already-kept (earlier) span."""
    return kept


def merge_overlapping_spans(
    spans: list[PIISpan],
    text: str,
    prefer: PreferSpan = prefer_greater_end,
) -> list[PIISpan]:
    """Merge overlapping spans into their union, without losing coverage.

    Sorts by ``(start, -end)`` and, whenever the next span starts before
    the currently-kept span ends, extends the kept span's boundaries to
    the union of both ranges — unlike a plain "keep the winner" merge,
    this never drops the non-overlapping region of the losing span.
    ``text`` is used to recompute the merged span's ``.text`` substring
    for its (possibly extended) boundaries. ``prefer(kept, incoming)``
    decides whose ``entity_type``/``source``/``confidence`` the merged
    span carries when the two differ.

    Args:
        spans: Spans to merge, possibly overlapping.
        text: Full source text the spans were detected in.
        prefer: Tie-break callable; defaults to keeping whichever span
            currently reaches further (matching the historical
            pattern-layer merge rule).

    Returns:
        Deduplicated list of spans covering the full union of input
        ranges, sorted by start position.
    """
    if not spans:
        return []

    ordered = sorted(spans, key=lambda s: (s.start, -s.end))
    merged: list[PIISpan] = [ordered[0]]
    for span in ordered[1:]:
        last = merged[-1]
        if span.start >= last.end:
            merged.append(span)
            continue

        winner = prefer(last, span)
        new_start = min(last.start, span.start)
        new_end = max(last.end, span.end)
        merged[-1] = PIISpan(
            start=new_start,
            end=new_end,
            text=text[new_start:new_end],
            entity_type=winner.entity_type,
            source=winner.source,
            confidence=winner.confidence,
        )
    return merged


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
