"""Слой детекции PII для PrivacyGuard Pipeline.

Трёхслойная детекция: PatternMatcher (регулярные выражения) ->
NatashaNER (нейросетевой NER) -> ContextualValidator (разрешение
конфликтов), оркеструется классом PIIDetector.
"""

from __future__ import annotations

from privacyguard_pipeline.detection.common import DetectionResult, PIISpan
from privacyguard_pipeline.detection.detector import PIIDetector

__all__ = ['DetectionResult', 'PIIDetector', 'PIISpan']
