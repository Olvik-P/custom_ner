"""Контекстная валидация и разрешение конфликтов для детекции PII.

Слой 3: разрешение конфликтов и валидация на основе контекста.

Правила:
1. Если спан найден и pattern, и NER — приоритет у pattern.
2. Если NER нашёл PER, но слово есть в whitelist (в том числе любая
   лемма одиночного слова) — маскирование пропускается.
3. Одиночный PER, все разборы которого — обычные словарные слова без
   чтений имени/фамилии/отчества, отбрасывается, если перед ним нет
   маркера персоны.
4. Разрешение неоднозначности локация/персона по грамматическим
   признакам, а без них — через эвристику по окружающим токенам.

Правила 2 (по лемме), 3 и грамматическая часть 4 требуют морфологии
(pymorphy3); без неё они пропускаются, действует прежнее поведение.
"""

from __future__ import annotations

import re
from typing import ClassVar

from privacyguard_pipeline.constants import (
    CONTEXT_LOOKBACK_WORDS,
    CONTEXT_RESOLVED_CONFIDENCE,
    PERSON_MARKERS,
)
from privacyguard_pipeline.detection.common import (
    WHITELIST,
    PIISpan,
    merge_overlapping_spans,
    prefer_first,
)
from privacyguard_pipeline.detection.morphology import (
    GEO_READING,
    PERSON_READINGS,
    Morphology,
    get_morphology,
)

_WORD_RE = re.compile(r'[^\W\d_]+')
_SURROUNDING_PUNCT = '«»"\'(),.!?;:'


def _subtract_covered_ranges(
    span: PIISpan,
    covering: list[tuple[int, int]],
    text: str,
) -> list[PIISpan]:
    """Разбивает `span` на остаточные спаны, исключая диапазоны `covering`.

    `covering` должен быть отсортирован по началу и уже отфильтрован
    до диапазонов, реально пересекающихся с `span`. Используется, чтобы
    pattern-спан забирал только те символы, которые он пересекает,
    вместо полного отбрасывания пересекающегося NER-спана — более
    длинный NER-спан может сохранить остаток слева и/или справа вокруг
    более короткого совпадения pattern посередине.
    """
    cursor = span.start
    remainders: list[PIISpan] = []
    for c_start, c_end in covering:
        gap_end = min(c_start, span.end)
        if gap_end > cursor:
            remainders.append(
                PIISpan(
                    start=cursor,
                    end=gap_end,
                    text=text[cursor:gap_end],
                    entity_type=span.entity_type,
                    source=span.source,
                    confidence=span.confidence,
                ),
            )
        cursor = max(cursor, c_end)
    if cursor < span.end:
        remainders.append(
            PIISpan(
                start=cursor,
                end=span.end,
                text=text[cursor : span.end],
                entity_type=span.entity_type,
                source=span.source,
                confidence=span.confidence,
            ),
        )
    return remainders


class ContextualValidator:
    """Слой 3: разрешение конфликтов и валидация на основе контекста.

    Правила:
    1. Если спан найден и pattern, и NER — приоритет у pattern.
    2. Если NER нашёл PER, но слово есть в whitelist — маскирование
       пропускается.
    3. Разрешение неоднозначности локация/персона через эвристику по
       окружающим токенам.
    """

    # Предлоги, за которыми следует локация
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

    # Глаголы, указывающие на движение/направление (контекст локации)
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

    def __init__(self, morphology: Morphology | None = None) -> None:
        self._whitelist = WHITELIST
        self._morph = morphology or get_morphology()

    def validate(
        self,
        pattern_spans: list[PIISpan],
        natasha_spans: list[PIISpan],
        text: str,
    ) -> list[PIISpan]:
        """Разрешает конфликты между детекциями pattern и NER.

        Args:
            pattern_spans: Спаны от PatternMatcher.
            natasha_spans: Спаны от NatashaNER.
            text: Исходный текст для анализа контекста.

        Returns:
            Итоговый дедуплицированный и провалидированный список
            PII-спанов.
        """
        pattern_intervals = sorted((s.start, s.end) for s in pattern_spans)

        # Фильтрация спанов natasha:
        # - Pattern побеждает при пересечении, но только на тех
        #   символах, которые он реально захватил — частично
        #   пересекающийся NER-спан сохраняет свой(и)
        #   непересекающийся(еся) остаток(и), а не отбрасывается целиком.
        # - Удаляется, если PER и есть в whitelist
        # - Разрешение неоднозначного PER/LOC
        filtered_natasha: list[PIISpan] = []
        for span in natasha_spans:
            overlapping = [
                (c_start, c_end)
                for c_start, c_end in pattern_intervals
                if c_start < span.end and c_end > span.start
            ]
            remainders = (
                _subtract_covered_ranges(span, overlapping, text)
                if overlapping
                else [span]
            )

            for remainder in remainders:
                # Проверка whitelist и морфологических фильтров для PER
                if remainder.entity_type == 'PER':
                    word_lower = remainder.text.lower().strip()
                    if word_lower in self._whitelist:
                        continue
                    if self._is_false_single_word_person(remainder, text):
                        continue

                # Разрешение неоднозначности: NER говорит PER, но
                # контекст указывает на LOC
                if remainder.entity_type == 'PER':
                    resolved_type = self._resolve_ambiguous(
                        remainder,
                        text,
                    )
                    if resolved_type != 'PER':
                        remainder = PIISpan(
                            start=remainder.start,
                            end=remainder.end,
                            text=remainder.text,
                            entity_type=resolved_type,
                            source='context',
                            confidence=CONTEXT_RESOLVED_CONFIDENCE,
                        )

                filtered_natasha.append(remainder)

        # Слияние: pattern-спаны и отфильтрованные natasha-спаны,
        # с расширением пересекающихся диапазонов до их объединения,
        # а не отбрасыванием непересекающегося покрытия любой стороны.
        # Pattern-спаны уже были слиты PatternMatcher, поэтому при
        # ничьей здесь побеждает более ранний (pattern) спан — согласно
        # историческому правилу приоритета.
        all_spans = list(pattern_spans) + filtered_natasha
        return merge_overlapping_spans(all_spans, text, prefer=prefer_first)

    @staticmethod
    def _single_word(span_text: str) -> str | None:
        """Единственное слово спана или ``None``, если слов не одно.

        Кавычки и окружающая пунктуация снимаются; «слово» — токен из
        одних букв (без цифр и дефисов).
        """
        cleaned = span_text.strip().strip(_SURROUNDING_PUNCT).strip()
        return cleaned if _WORD_RE.fullmatch(cleaned) else None

    @staticmethod
    def _context_words(text: str, start: int) -> list[str]:
        """Последние слова перед позицией ``start`` в нижнем регистре."""
        before = text[:start].strip().lower()
        tokens_before = before.split()
        return tokens_before[-CONTEXT_LOOKBACK_WORDS:]

    def _is_false_single_word_person(self, span: PIISpan, text: str) -> bool:
        """Одиночный PER, который на самом деле не персона.

        Отбрасывается (``True``), если слово: (1) имеет лемму из
        whitelist; либо (2) целиком словарное без чтений
        имени/фамилии/отчества и без чтения «топоним» (топоним — не
        обычное слово, его тип решает PER/LOC), и перед ним нет маркера
        персоны. Слова вне словаря сохраняются — это может быть редкая
        фамилия.
        Многословные спаны не затрагиваются. Без морфологии — ``False``.
        """
        if not self._morph.is_available:
            return False
        word = self._single_word(span.text)
        if word is None:
            return False
        if self._morph.lemmas(word) & self._whitelist:
            return True
        if not self._morph.is_known(word):
            return False
        if self._morph.readings(word) & (PERSON_READINGS | {GEO_READING}):
            return False
        markers = {
            w.strip(_SURROUNDING_PUNCT)
            for w in self._context_words(text, span.start)
        }
        return not markers & PERSON_MARKERS

    def _resolve_ambiguous(
        self,
        span: PIISpan,
        text: str,
    ) -> str:
        """Определяет, является ли спан с тегом PER на самом деле локацией.

        Сначала по грамматическим признакам (при наличии морфологии):
        - есть чтение имени/фамилии/отчества → PER, какой бы предлог ни
          стоял перед спаном (даже если у слова есть и чтение «топоним»);
        - есть только чтение «топоним» → LOC при контексте локации,
          иначе PER.
        Если словарь не даёт ни того, ни другого, решает прежняя
        эвристика по окружающим токенам:
        - Если перед спаном предлог локации → LOC
        - Если перед спаном глагол движения → LOC
        - Иначе → PER

        Args:
            span: Обнаруженный спан.
            text: Полный текст для анализа контекста.

        Returns:
            Разрешённый тип сущности ('PER' или 'LOC').
        """
        if self._morph.is_available:
            readings: set[str] = set()
            for word in _WORD_RE.findall(span.text):
                readings |= self._morph.readings(word)
            if readings & PERSON_READINGS:
                return 'PER'
            if GEO_READING in readings:
                return (
                    'LOC' if self._has_location_context(span, text) else 'PER'
                )
        return 'LOC' if self._has_location_context(span, text) else 'PER'

    def _has_location_context(self, span: PIISpan, text: str) -> bool:
        """Есть ли перед спаном предлог локации или глагол движения."""
        for word in self._context_words(text, span.start):
            word_clean = word.strip(_SURROUNDING_PUNCT)
            if word_clean in self._LOC_PREPOSITIONS:
                return True
            if word_clean in self._LOC_VERBS:
                return True
        return False
