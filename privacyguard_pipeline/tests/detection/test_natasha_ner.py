"""Тесты извлечения структурированных почтовых адресов в NatashaNER.

Покрывают фикс интеграции AddrExtractor (вызов экстрактора как
генератора совпадений вместо единого заранее слитого результата
`.find()`) и regex-фолбэк для номеров домов, которые собственная
грамматика "дом" AddrExtractor пропускает (голые числа и
непунктуированный маркер "д").
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
        # Старый баг на основе .find() сливал каждое совпадение в
        # тексте в один спан от начала первого совпадения до конца
        # последнего - защита от регрессии этого поведения.
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
        # У AddrExtractor вообще нет типа части 'помещение' - это должно
        # приходить из эвристики номера помещения, привязанной к его
        # собственному совпадению 'дом' ('д. 14').
        text = 'ул. Балтийская, д. 14, помещ. 1/1.'
        spans = natasha.detect(text)
        assert 'помещ. 1/1' in _texts(spans)

    def test_room_qualifier_after_heuristic_matched_house_number(
        self,
        natasha: NatashaNER,
    ) -> None:
        # Здесь сам номер дома приходит только из эвристики 1.5
        # (непунктуированный маркер "д") - проверка помещения всё равно
        # должна корректно к нему привязаться.
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
            yield  # pragma: no cover - недостижимо, оставляет это генератором

        monkeypatch.setattr(natasha, '_addr_tagger', _raise)
        text = 'Директор Иванов Пётр Сергеевич подписал документ.'

        spans = natasha.detect(text)  # must not raise

        assert any(span.entity_type == 'PER' for span in spans)
