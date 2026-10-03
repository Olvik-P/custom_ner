"""Контрольные суммы СНИЛС и ОГРН/ОГРНИП."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.detection.validators import (
    validate_ogrn_checksum,
    validate_snils_checksum,
)


class TestSnilsChecksum:
    @pytest.mark.parametrize(
        'raw',
        ['112-233-445 95', '11223344595', '112 233 445 95'],
    )
    def test_valid_snils(self, raw: str) -> None:
        assert validate_snils_checksum(raw) is True

    def test_wrong_control_number_fails(self) -> None:
        assert validate_snils_checksum('112-233-445 96') is False

    def test_wrong_length_fails(self) -> None:
        assert validate_snils_checksum('1122334459') is False

    @pytest.mark.parametrize('raw', ['002-008-999 00', '002-009-899 00'])
    def test_sum_100_or_101_gives_control_00(self, raw: str) -> None:
        assert validate_snils_checksum(raw) is True

    def test_number_at_or_below_threshold_is_not_checked(self) -> None:
        # Для 001-001-998 и ниже контрольное число не определено.
        assert validate_snils_checksum('001-001-998 00') is False
        assert validate_snils_checksum('000-000-001 00') is False

    def test_number_above_threshold_is_checked(self) -> None:
        # 001-001-999: сумма = 0+0+1*7+0+0+1*4+9*3+9*2+9*1 = 65
        assert validate_snils_checksum('001-001-999 65') is True


class TestOgrnChecksum:
    def test_valid_ogrn_13(self) -> None:
        assert validate_ogrn_checksum('1027700132195') is True

    def test_valid_ogrnip_15(self) -> None:
        assert validate_ogrn_checksum('304500116000157') is True

    def test_wrong_control_digit_fails(self) -> None:
        assert validate_ogrn_checksum('1027700132196') is False
        assert validate_ogrn_checksum('304500116000158') is False

    @pytest.mark.parametrize('raw', ['102770013219', '10277001321950'])
    def test_wrong_length_fails(self, raw: str) -> None:
        assert validate_ogrn_checksum(raw) is False
