"""Тесты морфологического слоя (pymorphy3)."""

from __future__ import annotations

import logging

import pytest

from privacyguard_pipeline.detection.morphology import (
    Morphology,
    get_morphology,
    normalize_word,
)


@pytest.fixture(scope='module')
def morph() -> Morphology:
    m = get_morphology()
    assert m.is_available
    return m


class TestLemmas:
    def test_case_forms_reduce_to_nominative(self, morph: Morphology) -> None:
        assert 'надежда' in morph.lemmas('Надежды')
        assert 'ромашка' in morph.lemmas('Ромашки')

    def test_yo_is_normalized(self, morph: Morphology) -> None:
        assert 'черный' in morph.lemmas('Чёрный')

    def test_first_lemma(self, morph: Morphology) -> None:
        assert morph.first_lemma('Ромашки') == 'ромашка'
        assert morph.first_lemma('проекта') == 'проект'


class TestReadings:
    def test_surname_reading(self, morph: Morphology) -> None:
        assert 'Surn' in morph.readings('Иванов')

    def test_only_geographic_reading(self, morph: Morphology) -> None:
        assert morph.readings('Москве') == {'Geox'}

    def test_both_surname_and_geographic(self, morph: Morphology) -> None:
        assert {'Surn', 'Geox'} <= morph.readings('Иванова')

    def test_common_noun_has_no_tracked_reading(
        self,
        morph: Morphology,
    ) -> None:
        assert morph.readings('Рыба') == frozenset()


class TestIsKnown:
    def test_dictionary_word(self, morph: Morphology) -> None:
        assert morph.is_known('Рыба')

    def test_out_of_vocabulary_word(self, morph: Morphology) -> None:
        assert not morph.is_known('Цзинь')


class TestUnavailable:
    def _broken(self) -> Morphology:
        def boom() -> object:
            raise RuntimeError('no dictionary')

        return Morphology(analyzer_factory=boom)

    def test_no_exception_and_flag_false(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            m = self._broken()
        assert not m.is_available
        assert any(
            'Morphology unavailable' in r.message for r in caplog.records
        )

    def test_neutral_values(self) -> None:
        m = self._broken()
        assert m.lemmas('Надежды') == {'надежды'}
        assert m.first_lemma('Ё') == 'е'
        assert m.readings('Иванов') == frozenset()
        assert not m.is_known('Рыба')


def test_normalize_word() -> None:
    assert normalize_word('ПЁТР') == 'петр'
