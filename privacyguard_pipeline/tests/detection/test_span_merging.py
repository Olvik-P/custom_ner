"""Overlap-resolution must preserve coverage, never drop it.

Regression tests for the three merge sites that used to discard a
span's non-overlapping region on partial overlap: the shared
merge_overlapping_spans() helper, PatternMatcher's same-tier merge, and
ContextualValidator's pattern-vs-NER conflict resolution.
"""

from __future__ import annotations

from privacyguard_pipeline.detection.common import (
    PIISpan,
    merge_overlapping_spans,
    prefer_first,
)
from privacyguard_pipeline.detection.contextual_validator import (
    ContextualValidator,
)


def _span(start: int, end: int, entity_type: str, text: str) -> PIISpan:
    return PIISpan(start=start, end=end, text=text, entity_type=entity_type)


class TestMergeOverlappingSpans:
    def test_partially_overlapping_spans_merge_to_full_union(
        self,
    ) -> None:
        text = '0123456789012345678901234567890123456789'
        spans = [
            _span(0, 10, 'PASSPORT', text[0:10]),
            _span(5, 20, 'CARD', text[5:20]),
        ]

        merged = merge_overlapping_spans(spans, text)

        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end == 20
        assert merged[0].text == text[0:20]

    def test_non_overlapping_spans_are_kept_separate(self) -> None:
        text = 'A' * 40
        spans = [
            _span(0, 5, 'PHONE', text[0:5]),
            _span(10, 15, 'EMAIL', text[10:15]),
        ]

        merged = merge_overlapping_spans(spans, text)

        assert [(s.start, s.end) for s in merged] == [(0, 5), (10, 15)]

    def test_prefer_first_keeps_earlier_spans_type_but_extends_range(
        self,
    ) -> None:
        text = 'X' * 30
        spans = [
            _span(0, 10, 'PER', text[0:10]),
            _span(5, 20, 'LOC', text[5:20]),
        ]

        merged = merge_overlapping_spans(spans, text, prefer=prefer_first)

        assert len(merged) == 1
        assert merged[0].entity_type == 'PER'
        assert (merged[0].start, merged[0].end) == (0, 20)


class TestPatternVsNerConflictResolution:
    def test_pattern_overlapping_middle_of_ner_span_leaves_both_remainders(
        self,
    ) -> None:
        # NER tags the whole thing as one PER span; a pattern span
        # (e.g. a phone-shaped substring) matches only the middle.
        text = 'Иван 1234567890 Петров'
        pattern_spans = [_span(5, 15, 'PHONE', text[5:15])]
        natasha_spans = [_span(0, 23, 'PER', text[0:23])]

        result = ContextualValidator().validate(
            pattern_spans=pattern_spans,
            natasha_spans=natasha_spans,
            text=text,
        )

        types_and_ranges = sorted(
            (s.entity_type, s.start, s.end) for s in result
        )
        assert ('PHONE', 5, 15) in types_and_ranges
        # Leading remainder "Иван " and trailing remainder " Петров"
        # both survive as PER spans instead of the whole PER span
        # being dropped.
        per_ranges = [
            (s.start, s.end) for s in result if s.entity_type == 'PER'
        ]
        assert (0, 5) in per_ranges
        assert (15, 23) in per_ranges

    def test_pattern_overlapping_one_end_leaves_single_remainder(
        self,
    ) -> None:
        text = 'Смирнова 1234567890'
        pattern_spans = [_span(9, 19, 'PHONE', text[9:19])]
        natasha_spans = [_span(0, 19, 'PER', text[0:19])]

        result = ContextualValidator().validate(
            pattern_spans=pattern_spans,
            natasha_spans=natasha_spans,
            text=text,
        )

        per_spans = [s for s in result if s.entity_type == 'PER']
        assert len(per_spans) == 1
        assert (per_spans[0].start, per_spans[0].end) == (0, 9)
        assert per_spans[0].text == 'Смирнова '
