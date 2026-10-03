"""Морфологический слой на pymorphy3: леммы, чтения слова, словарность.

Узкая обёртка над ``pymorphy3.MorphAnalyzer`` для детекции PII. Как и
``NatashaNER``, никогда не выбрасывает исключение при сбое загрузки:
выставляет ``is_available = False``, а потребители (контекстный
валидатор, списки allow/deny) в этом случае становятся no-op или
сравнивают буквальный текст.

Использование:
    morph = get_morphology()
    if morph.is_available:
        morph.lemmas('Надежды')   # frozenset({'надежда'})
        morph.readings('Иванов')  # frozenset({'Surn', 'Name'})
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Грамматические признаки, которые нужны детектору: личные чтения
# (имя, фамилия, отчество) и географическое.
PERSON_READINGS = frozenset({'Name', 'Surn', 'Patr'})
GEO_READING = 'Geox'
_TRACKED_READINGS = PERSON_READINGS | {GEO_READING}

_ANALYZER_CACHE_SIZE = 4096


def normalize_word(word: str) -> str:
    """Нижний регистр и ``ё`` -> ``е``: единая форма для сравнений."""
    return word.lower().replace('ё', 'е')


def _default_factory() -> Any:
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


class Morphology:
    """Леммы и грамматические признаки русских слов.

    Attributes:
        is_available: ``False``, если pymorphy3 или его словарь не
            загрузились; тогда методы возвращают нейтральные значения.
    """

    def __init__(
        self,
        analyzer_factory: Callable[[], Any] = _default_factory,
    ) -> None:
        self._analyzer: Any = None
        self.is_available = False
        try:
            self._analyzer = analyzer_factory()
            self.is_available = True
        except Exception as exc:  # noqa: BLE001 - любой сбой = деградация
            logger.warning(
                'Morphology unavailable (%s): lemma whitelist, noun filter '
                'and grammatical PER/LOC rule are skipped',
                type(exc).__name__,
            )
        self._parse = lru_cache(maxsize=_ANALYZER_CACHE_SIZE)(self._parse_raw)

    def _parse_raw(self, word: str) -> tuple[Any, ...]:
        return tuple(self._analyzer.parse(word))

    def lemmas(self, word: str) -> frozenset[str]:
        """Все леммы слова по всем разборам (в нижнем регистре, ё->е).

        Без морфологии — само слово в нормализованной форме.
        """
        if not self.is_available:
            return frozenset({normalize_word(word)})
        return frozenset(
            normalize_word(p.normal_form) for p in self._parse(word)
        )

    def first_lemma(self, word: str) -> str:
        """Лемма самого вероятного разбора (ё->е); без морфологии — слово."""
        if not self.is_available:
            return normalize_word(word)
        parses = self._parse(word)
        return normalize_word(parses[0].normal_form if parses else word)

    def readings(self, word: str) -> frozenset[str]:
        """Подмножество ``{Name, Surn, Patr, Geox}`` по всем разборам слова.

        Без морфологии — пустое множество.
        """
        if not self.is_available:
            return frozenset()
        found: set[str] = set()
        for parse in self._parse(word):
            found.update(r for r in _TRACKED_READINGS if r in parse.tag)
        return frozenset(found)

    def is_known(self, word: str) -> bool:
        """Все разборы слова взяты из словаря (а не угаданы по суффиксу).

        Без морфологии — ``False``: слово не считается словарным.
        """
        if not self.is_available:
            return False
        parses = self._parse(word)
        return bool(parses) and all(p.is_known for p in parses)


@lru_cache(maxsize=1)
def get_morphology() -> Morphology:
    """Возвращает общий лениво созданный экземпляр ``Morphology``."""
    return Morphology()
