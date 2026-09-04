"""PatternMatcher — Layer 1: Regex-based PII detection.

Covers phones, emails, Russian documents (passport, INN, SNILS, OGRN),
bank cards (with Luhn check), IP addresses, URLs, and coordinates.
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

# Both PASSPORT and INN patterns can match an identical bare 10-digit
# run. validate_inn() now requires a passing ФНС control-digit checksum,
# so a same-range tie means the INN candidate is a *real* INN — prefer
# it over the format-agnostic PASSPORT match rather than relying on
# PATTERN_REGISTRY's registration order.
_PASSPORT_INN_TIE: frozenset[str] = frozenset({'PASSPORT', 'INN'})


def _prefer_pattern_span(kept: PIISpan, incoming: PIISpan) -> PIISpan:
    """Tie-break for same-tier pattern spans.

    Falls back to :func:`prefer_greater_end` except for an exact-range
    PASSPORT/INN tie, where the checksum-validated INN wins.
    """
    if (
        kept.start == incoming.start
        and kept.end == incoming.end
        and {kept.entity_type, incoming.entity_type} == _PASSPORT_INN_TIE
    ):
        return kept if kept.entity_type == 'INN' else incoming
    return prefer_greater_end(kept, incoming)


class PatternMatcher:
    """Layer 1: Regex-based PII detection.

    Covers phones, emails, Russian documents (passport, INN, SNILS, OGRN),
    bank cards (with Luhn check), IP addresses, URLs, and coordinates.
    """

    def __init__(self) -> None:
        self._patterns = PATTERN_REGISTRY

    # ------------------------------------------------------------------
    # Span merging
    # ------------------------------------------------------------------

    @staticmethod
    def merge_overlapping(
        spans: list[PIISpan],
        text: str,
    ) -> list[PIISpan]:
        """Merge overlapping spans into their union.

        Args:
            spans: List of spans, possibly overlapping.
            text: Full source text the spans were detected in (used to
                recompute a merged span's ``.text`` for its extended
                boundaries).

        Returns:
            Deduplicated list of spans, each covering the full union of
            whatever input ranges overlapped it — no PII characters
            covered by an input span are left out of the result.
        """
        return merge_overlapping_spans(
            spans,
            text,
            prefer=_prefer_pattern_span,
        )

    # ------------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[PIISpan]:
        """Run all regex patterns against the text.

        Args:
            text: Input text to scan.

        Returns:
            List of detected PII spans.
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
        """Run type-specific validation on a regex match.

        Args:
            entity_type: Type of the matched entity.
            raw: Raw matched text.
            text: Full input text.
            start: Match start index.
            end: Match end index.

        Returns:
            True if the match is valid, False to skip it.
        """
        validator = VALIDATOR_REGISTRY.get(entity_type)
        if validator is not None:
            return validator(raw, text, start, end)
        return True
