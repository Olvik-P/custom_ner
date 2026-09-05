"""Разрешение неоднозначности PASSPORT/INN и PHONE/INN: ИНН с валидной
контрольной суммой не должен ошибочно классифицироваться как PASSPORT
или PHONE только потому, что несколько паттернов совпадают на одном и
том же голом 10-значном числе.
"""

from __future__ import annotations

from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher
from privacyguard_pipeline.detection.validators import validate_inn


class TestInnChecksum:
    def test_real_organization_inn_passes_checksum(self) -> None:
        assert validate_inn('7707083893') is True

    def test_real_individual_inn_passes_checksum(self) -> None:
        assert validate_inn('500100732259') is True

    def test_passport_shaped_number_failing_checksum_is_rejected(
        self,
    ) -> None:
        assert validate_inn('1234567890') is False


class TestPassportInnTieBreak:
    def test_checksum_valid_inn_wins_over_passport(self) -> None:
        text = 'ИНН: 7707083893'
        spans = PatternMatcher().detect(text)

        assert len(spans) == 1
        assert spans[0].entity_type == 'INN'

    def test_checksum_invalid_number_stays_passport(self) -> None:
        text = 'Паспорт серия и номер 1234567890'
        spans = PatternMatcher().detect(text)

        assert len(spans) == 1
        assert spans[0].entity_type == 'PASSPORT'


class TestPhoneInnTieBreak:
    def test_checksum_valid_inn_wins_over_phone(self) -> None:
        # Реальный кейс, найденный на практике (решение о рейдовом
        # осмотре): без исправления PHONE_RE захватывал пробел перед
        # цифрами, из-за чего этот ИНН детектировался как PHONE.
        text = 'ИНН 9702010905, адрес: 127055, г Москва'
        spans = PatternMatcher().detect(text)

        inn_spans = [s for s in spans if s.text.strip() == '9702010905']
        assert len(inn_spans) == 1
        assert inn_spans[0].entity_type == 'INN'
        assert inn_spans[0].text == '9702010905'

    def test_real_phone_with_prefix_is_unaffected(self) -> None:
        text = 'мой телефон +7 999 123 45 67.'
        spans = PatternMatcher().detect(text)

        phone_spans = [s for s in spans if s.entity_type == 'PHONE']
        assert len(phone_spans) == 1
        assert phone_spans[0].text == '+7 999 123 45 67'

    def test_real_phone_without_prefix_is_unaffected(self) -> None:
        text = 'звоните на 8 (916) 123-45-67 в любое время'
        spans = PatternMatcher().detect(text)

        phone_spans = [s for s in spans if s.entity_type == 'PHONE']
        assert len(phone_spans) == 1
        assert phone_spans[0].text == '8 (916) 123-45-67'

    def test_bare_number_failing_inn_checksum_stays_phone(self) -> None:
        # 10 цифр, похожих на телефон, но не проходящих контрольную
        # сумму ИНН — тай-брейк не должен подавлять валидный PHONE.
        assert validate_inn('9161234568') is False
        text = 'звонили с номера 9161234568 вчера'
        spans = PatternMatcher().detect(text)

        matching = [s for s in spans if '9161234568' in s.text]
        assert len(matching) == 1
        assert matching[0].entity_type == 'PHONE'
