"""Морфологические фильтры ContextualValidator по сценариям спеки."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.detection.common import PIISpan
from privacyguard_pipeline.detection.contextual_validator import (
    ContextualValidator,
)
from privacyguard_pipeline.detection.morphology import Morphology


def _unavailable() -> Morphology:
    def boom() -> object:
        raise RuntimeError('no dictionary')

    return Morphology(analyzer_factory=boom)


def _validate(
    text: str,
    fragment: str,
    entity_type: str = 'PER',
    validator: ContextualValidator | None = None,
) -> list[PIISpan]:
    start = text.index(fragment)
    span = PIISpan(
        start=start,
        end=start + len(fragment),
        text=fragment,
        entity_type=entity_type,
        source='natasha',
        confidence=0.85,
    )
    validator = validator or ContextualValidator()
    return validator.validate([], [span], text)


class TestLemmaWhitelist:
    @pytest.mark.parametrize('word', ['Надежды', 'Веры', 'Ромашки'])
    def test_inflected_whitelisted_word_dropped(self, word: str) -> None:
        assert _validate(f'пришло письмо {word} вчера', word) == []

    @pytest.mark.parametrize('word', ['Надежда', 'Ромашка'])
    def test_nominative_still_dropped(self, word: str) -> None:
        assert _validate(f'пришло письмо {word} вчера', word) == []

    def test_whitelisted_word_inside_full_name_kept(self) -> None:
        spans = _validate('звонила Вера Петрова утром', 'Вера Петрова')
        assert [s.entity_type for s in spans] == ['PER']


class TestCommonNounFilter:
    @pytest.mark.parametrize('word', ['Рыба', 'Корень'])
    def test_common_noun_dropped(self, word: str) -> None:
        assert _validate(f'на ужин была {word} с овощами', word) == []

    def test_noun_with_person_marker_kept(self) -> None:
        spans = _validate('гражданин Гусь явился в суд', 'Гусь')
        assert [s.entity_type for s in spans] == ['PER']

    @pytest.mark.parametrize('marker', ['пациент', 'ФИО:', 'г-н', 'клиент'])
    def test_other_markers_keep_span(self, marker: str) -> None:
        spans = _validate(f'{marker} Корень пришёл', 'Корень')
        assert [s.entity_type for s in spans] == ['PER']

    @pytest.mark.parametrize('word', ['Иванов', 'Белов', 'Ленина'])
    def test_real_surname_kept(self, word: str) -> None:
        spans = _validate(f'это сказал {word} вчера', word)
        assert [s.entity_type for s in spans] == ['PER']

    def test_out_of_vocabulary_word_kept(self) -> None:
        spans = _validate('это сказал Цзинь вчера', 'Цзинь')
        assert [s.entity_type for s in spans] == ['PER']

    def test_multiword_span_not_filtered(self) -> None:
        spans = _validate('сказали Рыба Корень вчера', 'Рыба Корень')
        assert len(spans) == 1


class TestPersonVersusLocation:
    @pytest.mark.parametrize(
        ('text', 'word'),
        [
            ('письмо от Иванова пришло', 'Иванова'),
            ('обратился к Петрову с вопросом', 'Петрову'),
            ('получил у Сидорова деньги', 'Сидорова'),
        ],
    )
    def test_preposition_before_surname_stays_per(
        self,
        text: str,
        word: str,
    ) -> None:
        spans = _validate(text, word)
        assert [s.entity_type for s in spans] == ['PER']

    @pytest.mark.parametrize(
        ('text', 'word'),
        [
            ('приехал в Москве вчера', 'Москве'),
            ('живёт в Казани давно', 'Казани'),
        ],
    )
    def test_place_name_after_preposition_becomes_loc(
        self,
        text: str,
        word: str,
    ) -> None:
        spans = _validate(text, word)
        assert [s.entity_type for s in spans] == ['LOC']

    def test_surname_that_is_also_a_place_stays_per(self) -> None:
        spans = _validate('живёт в Иванове давно', 'Иванове')
        assert [s.entity_type for s in spans] == ['PER']

    def test_full_name_after_preposition_stays_per(self) -> None:
        spans = _validate('письмо от Ивана Петрова пришло', 'Ивана Петрова')
        assert [s.entity_type for s in spans] == ['PER']

    def test_geo_reading_without_location_context_stays_per(self) -> None:
        spans = _validate('сказала Москве всё', 'Москве')
        assert [s.entity_type for s in spans] == ['PER']


class TestWithoutMorphology:
    def _validator(self) -> ContextualValidator:
        return ContextualValidator(morphology=_unavailable())

    def test_inflected_whitelisted_word_not_filtered(self) -> None:
        spans = _validate(
            'пришло письмо Надежды вчера',
            'Надежды',
            validator=self._validator(),
        )
        assert len(spans) == 1

    def test_exact_whitelist_still_works(self) -> None:
        spans = _validate(
            'пришло письмо Ромашка вчера',
            'Ромашка',
            validator=self._validator(),
        )
        assert spans == []

    def test_common_noun_kept(self) -> None:
        spans = _validate(
            'сегодня была Рыба с овощами',
            'Рыба',
            validator=self._validator(),
        )
        assert [s.entity_type for s in spans] == ['PER']

    def test_previous_preposition_heuristic_applies(self) -> None:
        spans = _validate(
            'письмо от Иванова пришло',
            'Иванова',
            validator=self._validator(),
        )
        assert [s.entity_type for s in spans] == ['LOC']
