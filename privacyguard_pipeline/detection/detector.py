"""Three-layer PII detection orchestrator for PrivacyGuard Pipeline.

Layer 1: PatternMatcher — regex-based detection.
Layer 2: NatashaNER — neural network entity extraction via Natasha.
Layer 3: ContextualValidator — conflict resolution and whitelist filtering.

Usage:
    detector = PIIDetector()
    result = detector.detect("some text with PII")
"""

from __future__ import annotations

import logging

from privacyguard_pipeline.detection.common import DetectionResult
from privacyguard_pipeline.detection.contextual_validator import (
    ContextualValidator,
)
from privacyguard_pipeline.detection.natasha_ner import NatashaNER
from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher

logger = logging.getLogger(__name__)


class PIIDetector:
    """Orchestrates all three detection layers.

    Usage:
        detector = PIIDetector()
        result = detector.detect("some text with PII")
    """

    def __init__(self) -> None:
        self.pattern_matcher = PatternMatcher()
        self.natasha_ner = NatashaNER()
        self.contextual_validator = ContextualValidator()

    @property
    def natasha_available(self) -> bool:
        """Check if Natasha NER is operational."""
        return self.natasha_ner.is_available

    def detect(self, text: str) -> DetectionResult:
        """Run all three detection layers on the input text.

        Args:
            text: Input text to scan for PII.

        Returns:
            DetectionResult with deduplicated spans and per-layer stats.
        """
        result = DetectionResult()

        # Layer 1: PatternMatcher
        pattern_spans = self.pattern_matcher.detect(text)
        result.layer_stats['pattern'] = len(pattern_spans)
        logger.debug('PatternMatcher found %d spans', len(pattern_spans))

        # Layer 2: NatashaNER
        natasha_spans = self.natasha_ner.detect(text)
        result.layer_stats['natasha'] = len(natasha_spans)
        logger.debug('NatashaNER found %d spans', len(natasha_spans))

        # Layer 3: ContextualValidator
        final_spans = self.contextual_validator.validate(
            pattern_spans=pattern_spans,
            natasha_spans=natasha_spans,
            text=text,
        )
        result.layer_stats['context'] = len(final_spans)
        logger.debug(
            'ContextualValidator produced %d final spans',
            len(final_spans),
        )

        result.spans = final_spans
        return result
