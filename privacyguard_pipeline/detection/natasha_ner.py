"""Natasha NER integration for PrivacyGuard Pipeline.

Layer 2: Neural network entity extraction using Natasha library.
Detects PER (persons), LOC (locations), ORG (organizations).
Gracefully degrades if models are unavailable.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from privacyguard_pipeline.constants import (
    NATASHA_ADDR_CONFIDENCE,
    NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE,
    NATASHA_NER_CONFIDENCE,
)
from privacyguard_pipeline.detection.common import PIISpan

logger = logging.getLogger(__name__)

# AddrExtractor's own "дом" grammar rule only fires with an explicit
# marker written as "д." (with a period) or the full word "дом" - a bare
# number ("ул. Малышева, 101") or the common abbreviated marker without a
# period ("ул Бутырский Вал, д 68/70") isn't recognized as a house number
# at all. This regex catches both, anchored immediately after a matched
# street ("улица") component, so it can't fire on unrelated numbers
# elsewhere in the text.
_HOUSE_NUMBER_RE = re.compile(
    r'[,\s]+((?:д\.?|дом)?\s*\d+(?:/\d+)?[а-яёА-ЯЁa-zA-Z]?)\b',
)

# AddrExtractor has no "помещение" (room/premises) part type at all -
# unlike "офис", which it does recognize, "помещ./помещение" plus a
# number is never matched, in any spelling. Anchored immediately after a
# house number (AddrExtractor's own "дом" match, or the heuristic one
# above), same rationale as _HOUSE_NUMBER_RE.
_ROOM_NUMBER_RE = re.compile(
    r'[,\s]+(помещ(?:ение)?\.?\s*\d+(?:/\d+)?[а-яёА-ЯЁa-zA-Z]?)\b',
    re.IGNORECASE,
)


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

            # Address extraction. The extractor is called directly (not
            # via .find(), which collapses every match in the text into
            # one pre-merged span) so each address component — index,
            # city, street, house, building, office, etc. — comes back
            # as its own match with its own start/stop.
            try:
                for match in self._addr_tagger(text):
                    # Проверяем, не пересекается ли с уже найденным.
                    # Only guards whether we add *this* match's own
                    # span - house/room continuation checks below still
                    # run even when it's already covered, since a house
                    # component logically exists here either way.
                    already_covered = any(
                        s.start <= match.start and s.end >= match.stop
                        for s in spans
                    )
                    if not already_covered:
                        spans.append(
                            PIISpan(
                                start=match.start,
                                end=match.stop,
                                text=text[match.start : match.stop],
                                entity_type='LOC',
                                source='natasha',
                                confidence=NATASHA_ADDR_CONFIDENCE,
                            ),
                        )

                    house_end = match.stop
                    if match.fact.type == 'улица':
                        house_match = _HOUSE_NUMBER_RE.match(
                            text,
                            match.stop,
                        )
                        if house_match is None:
                            continue

                        h_start, h_end = house_match.span(1)
                        if not any(
                            s.start <= h_start and s.end >= h_end
                            for s in spans
                        ):
                            spans.append(
                                PIISpan(
                                    start=h_start,
                                    end=h_end,
                                    text=text[h_start:h_end],
                                    entity_type='LOC',
                                    source='natasha',
                                    confidence=(
                                        NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE
                                    ),
                                ),
                            )
                        house_end = h_end
                    elif match.fact.type != 'дом':
                        continue

                    room_match = _ROOM_NUMBER_RE.match(text, house_end)
                    if room_match is None:
                        continue

                    r_start, r_end = room_match.span(1)
                    if any(
                        s.start <= r_start and s.end >= r_end for s in spans
                    ):
                        continue

                    spans.append(
                        PIISpan(
                            start=r_start,
                            end=r_end,
                            text=text[r_start:r_end],
                            entity_type='LOC',
                            source='natasha',
                            confidence=(
                                NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE
                            ),
                        ),
                    )
            except Exception as exc:
                logger.debug('Address extraction skipped: %s', exc)

        except Exception as exc:
            logger.warning('Natasha NER error: %s', exc)

        return spans
