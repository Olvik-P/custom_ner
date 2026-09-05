"""Разрешение неоднозначности PASSPORT/INN: ИНН с валидной контрольной
суммой не должен ошибочно классифицироваться как PASSPORT только
потому, что оба паттерна совпадают на одном и том же голом 10-значном
числе.
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
