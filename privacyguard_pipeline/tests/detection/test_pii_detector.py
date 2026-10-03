"""Сквозные тесты PIIDetector + Masker для структурированных адресных спанов.

Подтверждают, что спаны из фикса AddrExtractor переживают проход
слияния/дедупликации ContextualValidator, и что каждый компонент
адреса в итоге маскируется собственным токеном, а не проглатывается и
не остаётся открытым текстом.
"""

from __future__ import annotations

import pytest

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.exceptions import InvalidConfidenceError
from privacyguard_pipeline.masker import Masker


class TestAddressComponentsSurviveDetectionAndMasking:
    def test_each_address_component_becomes_its_own_masked_token(
        self,
        detector: PIIDetector,
    ) -> None:
        text = (
            '620004, г. Екатеринбург, ул. Малышева, 101, '
            'тел. (343) 312-00-32, эл. почта: nadzor@egov66.ru'
        )
        result = detector.detect(text)
        masker = Masker()
        masked = masker.mask(text, result.spans)

        for original in (
            '620004',
            'Екатеринбург',
            'Малышева',
            '312-00-32',
            'nadzor@egov66.ru',
        ):
            assert original not in masked

        loc_tokens = [
            entry
            for entry in masker.mapping.values()
            if entry.entity_type == 'LOC'
        ]
        # индекс, город, улица, дом - каждый своим токеном, а не один
        # слитый адресный спан.
        assert len(loc_tokens) >= 4

        demasked = masker.demask(masked)
        assert demasked == text


class TestSpacedPassportSeriesIsFullyMasked:
    def test_no_series_or_number_digits_survive_masking(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'паспорт 45 10 123456, тел. +7(916)123-45-67'
        result = detector.detect(text)
        masker = Masker()
        masked = masker.mask(text, result.spans)

        assert '45 10' not in masked
        assert '123456' not in masked
        assert '123-45-67' not in masked

        assert masker.demask(masked) == text


class TestMinConfidenceThreshold:
    _TEXT = 'номер 4510123456'

    def test_default_threshold_filters_bare_passport(
        self,
        detector: PIIDetector,
    ) -> None:
        result = detector.detect(self._TEXT)

        assert [s for s in result.spans if s.entity_type == 'PASSPORT'] == []
        assert self._TEXT in Masker().mask(self._TEXT, result.spans)

    def test_per_call_zero_restores_candidate(
        self,
        detector: PIIDetector,
    ) -> None:
        result = detector.detect(self._TEXT, min_confidence=0.0)

        assert [s.entity_type for s in result.spans] == ['PASSPORT']

    def test_override_does_not_leak_into_next_call(
        self,
        detector: PIIDetector,
    ) -> None:
        detector.detect(self._TEXT, min_confidence=0.0)
        later = detector.detect(self._TEXT)

        assert later.spans == []

    def test_high_threshold_also_filters_ner_layer(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Иван Петров живёт в Москве'
        everything = detector.detect(text, min_confidence=0.0)
        strict = detector.detect(text, min_confidence=1.0)

        assert everything.spans != []
        assert strict.spans == []

    @pytest.mark.parametrize('value', [-0.01, 1.01])
    def test_out_of_range_threshold_rejected(
        self,
        detector: PIIDetector,
        value: float,
    ) -> None:
        with pytest.raises(InvalidConfidenceError):
            detector.detect(self._TEXT, min_confidence=value)

    def test_inn_still_wins_over_passport_at_threshold_zero(
        self,
        detector: PIIDetector,
    ) -> None:
        result = detector.detect('ИНН: 7707083893', min_confidence=0.0)

        assert [s.entity_type for s in result.spans] == ['INN']
