"""Natasha NER integration for PrivacyGuard Pipeline.

Layer 2: Neural network entity extraction using Natasha library.
Detects PER (persons), LOC (locations), ORG (organizations).
Gracefully degrades if models are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from privacyguard_pipeline.constants import (
    NATASHA_ADDR_CONFIDENCE,
    NATASHA_NER_CONFIDENCE,
)
from privacyguard_pipeline.detection.common import PIISpan

logger = logging.getLogger(__name__)


class NatashaNER:
    """Layer 2: Neural network entity extraction using Natasha library.

    Detects PER (persons), LOC (locations), ORG (organizations).
    Gracefully degrades if models are unavailable.
    """

    def __init__(self) -> None:
        self._available = False
        self._segmenter: Any = None
        self._ner_tagger: Any = None  # NewsNERTagger
        self._addr_tagger: Any = None  # AddrExtractor
        # MorphVocab (только для AddrExtractor)
        self._morph_vocab: Any = None
        self._emb: Any = None  # NewsEmbedding (для NewsNERTagger)
        self._load_models()

    def _load_models(self) -> None:
        """Load Natasha models. Gracefully degrades on failure."""
        try:
            # Правильные импорты для Natasha 1.6.0
            from natasha import (
                AddrExtractor,
                MorphVocab,
                NewsEmbedding,
                NewsNERTagger,
                Segmenter,
            )

            self._morph_vocab = MorphVocab()
            self._segmenter = Segmenter()

            # NewsEmbedding — скачает модель при первом запуске (~100 МБ)
            self._emb = NewsEmbedding()

            # NewsNERTagger принимает NewsEmbedding, НЕ MorphVocab
            self._ner_tagger = NewsNERTagger(self._emb)

            # AddrExtractor принимает MorphVocab
            self._addr_tagger = AddrExtractor(self._morph_vocab)

            self._available = True
            logger.info('Natasha models loaded successfully')
        except Exception as exc:
            logger.warning(
                'Failed to load Natasha models: %s. '
                'Continuing with PatternMatcher only.',
                exc,
            )
            self._available = False

    @property
    def is_available(self) -> bool:
        """Check if Natasha models are loaded and ready."""
        return self._available

    def detect(self, text: str) -> list[PIISpan]:
        """Extract named entities using Natasha NER.

        Args:
            text: Input text.

        Returns:
            List of detected PII spans from Natasha.
        """
        if not self._available:
            return []

        spans: list[PIISpan] = []

        try:
            from natasha import Doc

            doc = Doc(text)
            doc.segment(self._segmenter)

            # NER — имена, организации, локации
            doc.tag_ner(self._ner_tagger)

            for span in doc.spans:
                entity_type = span.type  # PER, LOC, ORG
                if entity_type not in ('PER', 'LOC', 'ORG'):
                    continue

                spans.append(
                    PIISpan(
                        start=span.start,
                        end=span.stop,
                        text=span.text,
                        entity_type=entity_type,
                        source='natasha',
                        confidence=NATASHA_NER_CONFIDENCE,
                    ),
                )

            # Address extraction
            try:
                addr_matches = self._addr_tagger.find(text)
                for match in addr_matches:
                    # Проверяем, не пересекается ли с уже найденным
                    if any(
                        s.start <= match.span.start
                        and s.end >= match.span.stop
                        for s in spans
                    ):
                        continue

                    spans.append(
                        PIISpan(
                            start=match.span.start,
                            end=match.span.stop,
                            text=match.text,
                            entity_type='LOC',
                            source='natasha',
                            confidence=NATASHA_ADDR_CONFIDENCE,
                        ),
                    )
            except Exception as exc:
                logger.debug('Address extraction skipped: %s', exc)

        except Exception as exc:
            logger.warning('Natasha NER error: %s', exc)

        return spans
