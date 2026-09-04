"""Tests for NatashaNER's structured postal address extraction.

Covers the AddrExtractor integration fix (calling the extractor as a
match generator instead of the single pre-merged `.find()` result) and
the regex fallback for house numbers AddrExtractor's own "дом" grammar
misses (bare numbers and the unpunctuated "д" marker).
"""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from privacyguard_pipeline.detection.common import PIISpan
from privacyguard_pipeline.detection.natasha_ner import NatashaNER


@pytest.fixture(scope='module')
def natasha() -> NatashaNER:
    instance = NatashaNER()
    if not instance.is_available:
        pytest.skip('Natasha models unavailable in this environment')
    return instance


def _texts(spans: list[PIISpan]) -> set[str]:
    return {span.text for span in spans}


class TestAddressComponentDetection:
    def test_index_city_street_and_house_all_detected(
        self,
        natasha: NatashaNER,
    ) -> None:
        text = (
            '620004, г. Екатеринбург, ул. Малышева, 101, тел. (343) 312-00-32'
        )
        spans = natasha.detect(text)
        assert {
            '620004',
            'г. Екатеринбург',
            'ул. Малышева',
            '101',
        } <= _texts(spans)

    def test_building_and_office_qualifiers_each_get_own_span(
        self,
        natasha: NatashaNER,
    ) -> None:
        text = 'ул Бутырский Вал, д 68/70 стр 1, офис 54'
        spans = natasha.detect(text)
        assert {
            'ул Бутырский Вал',
            'д 68/70',
            'стр 1',
            'офис 54',
        } <= _texts(spans)

    def test_two_addresses_in_one_block_are_not_merged(
        self,
        natasha: NatashaNER,
    ) -> None:
        text = (
            'адрес: 127055, г Москва, ул Бутырский Вал, д 68/70, '
            'Технический заказчик: ОБЩЕСТВО, '
            'адрес: 125315, г. Москва, ул. Балтийская, д. 14.'
        )
        spans = natasha.detect(text)

        assert {'127055', '125315'} <= _texts(spans)
        # The old .find()-based bug merged every match in the text into
        # one span from the first match's start to the last match's
        # stop - guard against that regressing.
        assert not any(
            '127055' in span.text and '125315' in span.text for span in spans
        )
        assert not any('Технический заказчик' in span.text for span in spans)


class TestHouseNumberHeuristic:
    def test_bare_number_with_no_marker(self, natasha: NatashaNER) -> None:
        text = 'ул. Малышева, 101, тел. (343) 312-00-32'
        spans = natasha.detect(text)
        assert '101' in _texts(spans)

    def test_unpunctuated_abbreviated_marker(
        self,
        natasha: NatashaNER,
    ) -> None:
        text = 'ул Бутырский Вал, д 68/70 стр 1'
        spans = natasha.detect(text)
        assert 'д 68/70' in _texts(spans)

    def test_no_spurious_match_without_a_trailing_number(
        self,
        natasha: NatashaNER,
    ) -> None:
        text = 'ул. Малышева расположена в центре города Екатеринбурга.'
        spans = natasha.detect(text)
        assert not any(span.text.strip(',. ').isdigit() for span in spans)

    def test_room_qualifier_after_grammar_matched_house_number(
        self,
        natasha: NatashaNER,
    ) -> None:
        # AddrExtractor has no 'помещение' part type at all - this must
        # come from the room-number heuristic, anchored on its own
        # 'дом' match ('д. 14').
        text = 'ул. Балтийская, д. 14, помещ. 1/1.'
        spans = natasha.detect(text)
        assert 'помещ. 1/1' in _texts(spans)

    def test_room_qualifier_after_heuristic_matched_house_number(
        self,
        natasha: NatashaNER,
    ) -> None:
        # Here the house number itself only comes from the 1.5
        # heuristic (unpunctuated "д" marker) - the room check must
        # still anchor on it correctly.
        text = 'ул Бутырский Вал, д 68/70, помещение 3'
        spans = natasha.detect(text)
        assert 'д 68/70' in _texts(spans)
        assert 'помещение 3' in _texts(spans)


class TestAddressExtractionGracefulDegradation:
    def test_exception_during_extraction_does_not_propagate(
        self,
        natasha: NatashaNER,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def _raise(_text: str) -> Iterator[Any]:
            raise RuntimeError('boom')
            yield  # pragma: no cover - unreachable, keeps this a generator

        monkeypatch.setattr(natasha, '_addr_tagger', _raise)
        text = 'Директор Иванов Пётр Сергеевич подписал документ.'

        spans = natasha.detect(text)  # must not raise

        assert any(span.entity_type == 'PER' for span in spans)
