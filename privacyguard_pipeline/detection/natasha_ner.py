"""Интеграция Natasha NER для PrivacyGuard Pipeline.

Слой 2: извлечение сущностей нейросетью с помощью библиотеки Natasha.
Обнаруживает PER (персоны), LOC (локации), ORG (организации).
Корректно деградирует, если модели недоступны.
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

# Собственное грамматическое правило "дом" у AddrExtractor срабатывает
# только при явном маркере "д." (с точкой) или полном слове "дом" — голое
# число ("ул. Малышева, 101") или распространённый сокращённый маркер без
# точки ("ул Бутырский Вал, д 68/70") номером дома вообще не
# распознаётся. Этот regex ловит оба случая, привязан сразу после
# совпавшего компонента улицы ("улица"), поэтому не может сработать на
# посторонних числах в другом месте текста.
_HOUSE_NUMBER_RE = re.compile(
    r'[,\s]+((?:д\.?|дом)?\s*\d+(?:/\d+)?[а-яёА-ЯЁa-zA-Z]?)\b',
)

# У AddrExtractor вообще нет типа части "помещение" - в отличие от
# "офис", который он распознаёт, "помещ./помещение" плюс число не
# совпадает никогда, в любом написании. Привязан сразу после номера
# дома (собственного совпадения "дом" у AddrExtractor или эвристического
# выше), та же логика, что и у _HOUSE_NUMBER_RE.
_ROOM_NUMBER_RE = re.compile(
    r'[,\s]+(помещ(?:ение)?\.?\s*\d+(?:/\d+)?[а-яёА-ЯЁa-zA-Z]?)\b',
    re.IGNORECASE,
)


class NatashaNER:
    """Слой 2: извлечение сущностей нейросетью с помощью Natasha.

    Обнаруживает PER (персоны), LOC (локации), ORG (организации).
    Корректно деградирует, если модели недоступны.
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
        """Загружает модели Natasha. Корректно деградирует при сбое."""
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
        """Проверяет, загружены ли модели Natasha и готовы ли к работе."""
        return self._available

    def detect(self, text: str) -> list[PIISpan]:
        """Извлекает именованные сущности с помощью Natasha NER.

        Args:
            text: Входной текст.

        Returns:
            Список PII-спанов, обнаруженных Natasha.
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

            # Извлечение адресов. Экстрактор вызывается напрямую (не
            # через .find(), который схлопывает все совпадения в тексте
            # в один заранее слитый спан), поэтому каждый компонент
            # адреса — индекс, город, улица, дом, корпус, офис и т.д. —
            # возвращается отдельным совпадением со своим start/stop.
            try:
                for match in self._addr_tagger(text):
                    # Проверяем, не пересекается ли с уже найденным.
                    # Защищает только добавление спана *этого* конкретного
                    # совпадения - проверки продолжения дома/помещения
                    # ниже всё равно выполняются, даже если оно уже
                    # покрыто, так как компонент дома логически
                    # существует здесь в любом случае.
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
