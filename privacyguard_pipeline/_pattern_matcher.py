"""PatternMatcher — Layer 1: Regex-based PII detection.

Covers phones, emails, Russian documents (passport, INN, SNILS, OGRN),
bank cards (with Luhn check), IP addresses, URLs, and coordinates.
"""

from __future__ import annotations

import logging

from privacyguard_pipeline._common import PIISpan
from privacyguard_pipeline._patterns import PATTERN_REGISTRY
from privacyguard_pipeline._validators import VALIDATOR_REGISTRY

logger = logging.getLogger(__name__)


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
    def merge_overlapping(spans: list[PIISpan]) -> list[PIISpan]:
        """Merge overlapping spans, keeping the longer one.

        Args:
            spans: List of spans, possibly overlapping.

        Returns:
            Deduplicated list of spans.
        """
        if not spans:
            return []

        spans.sort(key=lambda s: (s.start, -s.end))
        merged: list[PIISpan] = []
        for span in spans:
            if merged and span.start < merged[-1].end:
                if span.end > merged[-1].end:
                    merged[-1] = span
            else:
                merged.append(span)
        return merged

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
                    entity_type, raw, text, start, end,
                ):
                    continue

                spans.append(
                    PIISpan(
                        start=start,
                        end=end,
                        text=raw,
                        entity_type=entity_type,
                        source="pattern",
                        confidence=0.95,
                    ),
                )

        return self.merge_overlapping(spans)

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
