"""Скоринг pattern-совпадений: признаки, контекст, порог."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.constants import PATTERN_CONFIDENCE
from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher
from privacyguard_pipeline.detection.scoring import score_match


def _score(entity_type: str, text: str, raw: str) -> float:
    start = text.index(raw)
    return score_match(entity_type, raw, text, start, start + len(raw))


def _types(text: str, min_confidence: float | None = None) -> list[str]:
    spans = PatternMatcher().detect(text, min_confidence=min_confidence)
    return [s.entity_type for s in spans]


class TestPassportScoring:
    def test_keyword_raises_score(self) -> None:
        with_kw = _score('PASSPORT', 'паспорт 4510123456', '4510123456')
        bare = _score('PASSPORT', 'номер 4510123456', '4510123456')
        assert with_kw > bare

    def test_order_marker_lowers_score(self) -> None:
        order = _score('PASSPORT', 'заказ № 45 10 123456', '45 10 123456')
        bare = _score('PASSPORT', 'номер 45 10 123456', '45 10 123456')
        assert order < bare

    def test_bare_ten_digits_below_default_threshold(self) -> None:
        assert _types('номер 4510123456') == []

    def test_bare_ten_digits_kept_at_threshold_zero(self) -> None:
        assert _types('номер 4510123456', 0.0) == ['PASSPORT']

    def test_paired_series_kept_without_keyword(self) -> None:
        assert _types('номер 45 10 123456') == ['PASSPORT']

    @pytest.mark.parametrize(
        'passport', ['4510 123456', '45-10-123456', '4510123456']
    )
    def test_keyword_makes_any_spelling_pass(self, passport: str) -> None:
        assert _types(f'паспорт {passport}') == ['PASSPORT']

    def test_order_marker_drops_paired_series(self) -> None:
        assert _types('заказ № 45 10 123456') == []

    def test_word_zakazchik_is_not_anti_context(self) -> None:
        text = 'Заказчик: паспорт 45 10 123456'
        assert _types(text) == ['PASSPORT']

    def test_keyword_on_previous_line_gives_no_bonus(self) -> None:
        text = 'паспорт\n4510123456'
        assert _types(text) == []


class TestSnilsScoring:
    def test_valid_checksum_detected_without_keyword(self) -> None:
        assert _types('номер 112-233-445 95') == ['SNILS']

    def test_checksum_raises_score(self) -> None:
        valid = _score('SNILS', 'x 11223344595', '11223344595')
        invalid = _score('SNILS', 'x 11223344596', '11223344596')
        assert valid > invalid

    def test_failed_checksum_without_keyword_not_detected(self) -> None:
        assert _types('номер 11223344596') == []

    def test_failed_checksum_with_keyword_detected(self) -> None:
        assert _types('СНИЛС 11223344596') == ['SNILS']


class TestOgrnScoring:
    def test_valid_ogrn_detected_without_keyword(self) -> None:
        assert _types('рег. 1027700132195') == ['OGRN']

    def test_valid_ogrnip_detected_without_keyword(self) -> None:
        assert _types('рег. 304500116000157') == ['OGRNIP']

    def test_failed_checksum_without_keyword_not_detected(self) -> None:
        assert _types('рег. 1027700132196') == []

    def test_failed_checksum_with_keyword_detected(self) -> None:
        assert _types('ОГРН 1027700132196') == ['OGRN']


class TestCoordsScoring:
    def test_price_pair_is_not_coordinates(self) -> None:
        assert _types('товары по 12.5, 3.7 за штуку') == []

    def test_precise_pair_without_keyword_detected(self) -> None:
        assert _types('точка 55.751244, 37.618423') == ['COORDS']

    def test_keyword_makes_short_decimals_pass(self) -> None:
        assert _types('координаты 55.75, 37.61') == ['COORDS']

    def test_dms_notation_detected(self) -> None:
        text = '55°45\'07" N 37°37\'04" E'
        assert _types(text) == ['COORDS']


class TestIpScoring:
    def test_version_word_is_not_ip(self) -> None:
        assert _types('версия 1.2.3.4') == []

    def test_v_prefix_is_not_ip(self) -> None:
        assert _types('релиз v10.0.19.42') == []

    def test_keyword_makes_ip_pass(self) -> None:
        assert _types('IP-адрес 192.168.1.10') == ['IP']

    def test_version_marker_beats_keyword(self) -> None:
        assert _types('адрес, версия 192.168.1.10') == []

    def test_bare_ip_below_default_threshold(self) -> None:
        assert _types('клиент 84.201.12.5') == []

    def test_bare_ip_kept_at_lower_threshold(self) -> None:
        assert _types('клиент 84.201.12.5', 0.4) == ['IP']


class TestStrictlyValidatedTypesKeepFixedScore:
    def test_email_next_to_anti_context_keeps_score(self) -> None:
        text = 'заказ № a@b.ru'
        assert _score('EMAIL', text, 'a@b.ru') == PATTERN_CONFIDENCE
        assert _types(text) == ['EMAIL']

    def test_luhn_valid_card_next_to_order_marker(self) -> None:
        text = 'заказ № 4111 1111 1111 1111'
        assert _types(text) == ['CARD']


class TestThresholdValidation:
    @pytest.mark.parametrize('value', [-0.1, 1.1])
    def test_out_of_range_rejected(self, value: float) -> None:
        with pytest.raises(ValueError):
            PatternMatcher().detect('x', min_confidence=value)
