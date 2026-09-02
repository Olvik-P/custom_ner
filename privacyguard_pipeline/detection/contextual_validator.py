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
from privacyguard_pipeline.detection.common import WHITELIST, PIISpan


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
        # Build a set of character indices covered by pattern spans
        pattern_covered: set[int] = set()
        for span in pattern_spans:
            for i in range(span.start, span.end):
                pattern_covered.add(i)

        # Filter natasha spans:
        # - Remove if overlapping with pattern (pattern wins)
        # - Remove if PER and whitelisted
        # - Resolve ambiguous PER/LOC
        filtered_natasha: list[PIISpan] = []
        for span in natasha_spans:
            # Check overlap with pattern spans
            overlap = any(
                i in pattern_covered for i in range(span.start, span.end)
            )
            if overlap:
                continue

            # Whitelist check for PER
            if span.entity_type == 'PER':
                word_lower = span.text.lower().strip()
                if word_lower in self._whitelist:
                    continue

            # Ambiguous resolution: if NER says PER but context suggests LOC
            if span.entity_type == 'PER':
                resolved_type = self._resolve_ambiguous(
                    span,
                    text,
                )
                if resolved_type != 'PER':
                    span = PIISpan(
                        start=span.start,
                        end=span.end,
                        text=span.text,
                        entity_type=resolved_type,
                        source='context',
                        confidence=CONTEXT_RESOLVED_CONFIDENCE,
                    )

            filtered_natasha.append(span)

        # Merge: pattern spans first, then filtered natasha spans
        all_spans = list(pattern_spans) + filtered_natasha
        all_spans.sort(key=lambda s: (s.start, -s.end))

        # Final deduplication
        final: list[PIISpan] = []
        for span in all_spans:
            if final and span.start < final[-1].end:
                continue
            final.append(span)

        return final

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
