"""Contextual validation and conflict resolution for PII detection.

Layer 3: Conflict resolution and context-based validation.

Rules:
1. If a span is found by both pattern and NER — pattern takes priority.
2. If NER found PER but the word is in the whitelist — skip masking.
3. Ambiguous location/person resolution via surrounding token heuristics.
"""

from __future__ import annotations

from typing import ClassVar

from privacyguard_pipeline.constants import (
    CONTEXT_LOOKBACK_WORDS,
    CONTEXT_RESOLVED_CONFIDENCE,
)
from privacyguard_pipeline.detection.common import (
    WHITELIST,
    PIISpan,
    merge_overlapping_spans,
    prefer_first,
)


def _subtract_covered_ranges(
    span: PIISpan,
    covering: list[tuple[int, int]],
    text: str,
) -> list[PIISpan]:
    """Split `span` into remainder spans excluding `covering` ranges.

    `covering` must be sorted by start and already filtered to ranges
    that actually overlap `span`. Used so a pattern span only claims the
    characters it overlaps, instead of an overlapping NER span being
    dropped in full — a longer NER span may keep a leading and/or
    trailing remainder around a shorter pattern match in its middle.
    """
    cursor = span.start
    remainders: list[PIISpan] = []
    for c_start, c_end in covering:
        gap_end = min(c_start, span.end)
        if gap_end > cursor:
            remainders.append(
                PIISpan(
                    start=cursor,
                    end=gap_end,
                    text=text[cursor:gap_end],
                    entity_type=span.entity_type,
                    source=span.source,
                    confidence=span.confidence,
                ),
            )
        cursor = max(cursor, c_end)
    if cursor < span.end:
        remainders.append(
            PIISpan(
                start=cursor,
                end=span.end,
                text=text[cursor : span.end],
                entity_type=span.entity_type,
                source=span.source,
                confidence=span.confidence,
            ),
        )
    return remainders


class ContextualValidator:
    """Layer 3: Conflict resolution and context-based validation.

    Rules:
    1. If a span is found by both pattern and NER — pattern takes priority.
    2. If NER found PER but the word is in the whitelist — skip masking.
    3. Ambiguous location/person resolution via surrounding token heuristics.
    """

    # Prepositions that indicate a location follows
    _LOC_PREPOSITIONS: ClassVar[set[str]] = {
        'в',
        'во',
        'на',
        'из',
        'со',
        'от',
        'до',
        'у',
        'за',
        'под',
        'над',
        'перед',
        'между',
        'через',
        'около',
        'возле',
        'мимо',
        'против',
        'среди',
        'внутри',
        'снаружи',
        'вдоль',
        'поперёк',
        'к',
        'ко',
        'по',
        'через',
    }

    # Verbs that indicate movement/direction (location context)
    _LOC_VERBS: ClassVar[set[str]] = {
        'поехать',
        'ехать',
        'приехать',
        'уехать',
        'пойти',
        'идти',
        'прийти',
        'уйти',
        'полететь',
        'лететь',
        'прилететь',
        'отправиться',
        'направиться',
        'прибыть',
        'прибывать',
        'находиться',
        'расположен',
        'проживать',
        'живёт',
        'живут',
        'жить',
        'находится',
        'находятся',
    }

    def __init__(self) -> None:
        self._whitelist = WHITELIST

    def validate(
        self,
        pattern_spans: list[PIISpan],
        natasha_spans: list[PIISpan],
        text: str,
    ) -> list[PIISpan]:
        """Resolve conflicts between pattern and NER detections.

        Args:
            pattern_spans: Spans from PatternMatcher.
            natasha_spans: Spans from NatashaNER.
            text: Original text for context analysis.

        Returns:
            Final deduplicated and validated list of PII spans.
        """
        pattern_intervals = sorted((s.start, s.end) for s in pattern_spans)

        # Filter natasha spans:
        # - Pattern wins on overlap, but only for the characters it
        #   actually matched — a partially-overlapping NER span keeps
        #   its non-overlapping remainder(s) rather than being dropped
        #   whole.
        # - Remove if PER and whitelisted
        # - Resolve ambiguous PER/LOC
        filtered_natasha: list[PIISpan] = []
        for span in natasha_spans:
            overlapping = [
                (c_start, c_end)
                for c_start, c_end in pattern_intervals
                if c_start < span.end and c_end > span.start
            ]
            remainders = (
                _subtract_covered_ranges(span, overlapping, text)
                if overlapping
                else [span]
            )

            for remainder in remainders:
                # Whitelist check for PER
                if remainder.entity_type == 'PER':
                    word_lower = remainder.text.lower().strip()
                    if word_lower in self._whitelist:
                        continue

                # Ambiguous resolution: NER says PER but context
                # suggests LOC
                if remainder.entity_type == 'PER':
                    resolved_type = self._resolve_ambiguous(
                        remainder,
                        text,
                    )
                    if resolved_type != 'PER':
                        remainder = PIISpan(
                            start=remainder.start,
                            end=remainder.end,
                            text=remainder.text,
                            entity_type=resolved_type,
                            source='context',
                            confidence=CONTEXT_RESOLVED_CONFIDENCE,
                        )

                filtered_natasha.append(remainder)

        # Merge: pattern spans and filtered natasha spans, extending
        # overlapping ranges to their union rather than dropping either
        # side's non-overlapping coverage. Pattern spans were already
        # merged by PatternMatcher, so ties here favor the earlier
        # (pattern) span, matching the historical precedence rule.
        all_spans = list(pattern_spans) + filtered_natasha
        return merge_overlapping_spans(all_spans, text, prefer=prefer_first)

    def _resolve_ambiguous(
        self,
        span: PIISpan,
        text: str,
    ) -> str:
        """Resolve whether a PER-tagged span is actually a location.

        Uses heuristics based on surrounding tokens:
        - If preceded by location prepositions → LOC
        - If preceded by movement verbs → LOC
        - Otherwise → PER

        Args:
            span: The detected span.
            text: Full text for context.

        Returns:
            Resolved entity type ('PER' or 'LOC').
        """
        # Look at tokens before the span
        before = text[: span.start].strip().lower()
        if not before:
            return 'PER'

        # Get the last few words before the span
        tokens_before = before.split()
        context_words = (
            tokens_before[-CONTEXT_LOOKBACK_WORDS:]
            if len(tokens_before) >= CONTEXT_LOOKBACK_WORDS
            else tokens_before
        )

        for word in context_words:
            word_clean = word.strip('«»"(),.!?;:')
            if word_clean in self._LOC_PREPOSITIONS:
                return 'LOC'
            if word_clean in self._LOC_VERBS:
                return 'LOC'

        return 'PER'
