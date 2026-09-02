"""PII detection layer for PrivacyGuard Pipeline.

Three-layer detection: PatternMatcher (regex) -> NatashaNER (neural NER)
-> ContextualValidator (conflict resolution), orchestrated by PIIDetector.
"""

from __future__ import annotations

from privacyguard_pipeline.detection.common import DetectionResult, PIISpan
from privacyguard_pipeline.detection.detector import PIIDetector

__all__ = ['DetectionResult', 'PIIDetector', 'PIISpan']
