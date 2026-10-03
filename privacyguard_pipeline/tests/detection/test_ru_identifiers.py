"""Новые российские идентификаторы по сценариям спеки pii-detection."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.detection import PIIDetector
from privacyguard_pipeline.detection.validators import validate_kpp


def _found(
    detector: PIIDetector,
    text: str,
    min_confidence: float | None = None,
) -> list[tuple[str, str]]:
    result = detector.detect(text, min_confidence=min_confidence)
    return [(s.text, s.entity_type) for s in result.spans]


class TestOgrnAndOgrnip:
    def test_thirteen_digit_ogrn(self, detector: PIIDetector) -> None:
        got = _found(detector, 'рег. 1027700132195')
        assert got == [('1027700132195', 'OGRN')]

    def test_fifteen_digit_ogrnip_is_its_own_type(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'рег. 304500116000157')
        assert got == [('304500116000157', 'OGRNIP')]

    def test_ogrnip_keyword_with_failed_checksum(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'ОГРНИП 304500116000158')
        assert got == [('304500116000158', 'OGRNIP')]

    def test_failed_checksum_without_keyword_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'рег. 304500116000158') == []

    def test_masking_uses_the_new_token_type(
        self,
        detector: PIIDetector,
    ) -> None:
        from privacyguard_pipeline.masker import Masker

        text = 'ОГРНИП 304500116000157'
        masked = Masker().mask(text, detector.detect(text).spans)
        assert '<OGRNIP_' in masked
        assert '<OGRN_' not in masked


class TestKpp:
    @pytest.mark.parametrize(
        'text',
        [
            'КПП 771234234 требуется для оформления',
            'КПП организации 501401567, необходимо получить выписку',
            'Прошу проверить реквизиты контрагента. КПП 222005584.',
        ],
    )
    def test_with_keyword(self, detector: PIIDetector, text: str) -> None:
        got = _found(detector, text)
        assert [t for _, t in got] == ['KPP']

    def test_bare_nine_digits_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'номер 771234234 в системе') == []

    def test_bare_nine_digits_detected_at_zero_threshold(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'номер 771234234 в системе', 0.0)
        assert ('771234234', 'KPP') in got

    def test_invalid_reason_code_not_detected_even_with_keyword(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'КПП 771200234') == []

    def test_unknown_region_code_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'КПП 801234234') == []

    def test_other_label_does_not_borrow_the_kpp_keyword(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'КПП 771234234, БИК 044525225')
        assert got == [('771234234', 'KPP')]

    @pytest.mark.parametrize(
        ('raw', 'ok'),
        [
            ('771234234', True),
            ('7712AB234', True),
            ('771200234', False),
            ('801234234', False),
            ('77123423', False),
            ('7712345A4', False),
        ],
    )
    def test_validate_kpp(self, raw: str, ok: bool) -> None:
        assert validate_kpp(raw) is ok


class TestBankAccount:
    ACCOUNT = '40702810200000012345'
    CORR = '30101810400000000225'

    def test_account_verified_by_bik_in_the_next_line(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, f'БИК 044525225\n{self.ACCOUNT}')
        assert got == [(self.ACCOUNT, 'BANK_ACCOUNT')]

    def test_correspondent_account_verified_by_bik(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, f'БИК 044525225 {self.CORR}')
        assert (self.CORR, 'BANK_ACCOUNT') in got

    def test_wrong_key_for_the_bik_does_not_raise_confidence(self) -> None:
        from privacyguard_pipeline.detection.validators import (
            validate_account_key,
        )

        assert validate_account_key(self.ACCOUNT, '044525225')
        wrong = self.ACCOUNT[:8] + '5' + self.ACCOUNT[9:]
        assert not validate_account_key(wrong, '044525225')
        assert validate_account_key(self.CORR, '044525225')

    def test_keyword_without_bik(self, detector: PIIDetector) -> None:
        got = _found(detector, f'расчётный счёт {self.ACCOUNT} открыт')
        assert got == [(self.ACCOUNT, 'BANK_ACCOUNT')]

    def test_bare_account_recognised_by_structure(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, f'перевод на {self.ACCOUNT}')
        assert got == [(self.ACCOUNT, 'BANK_ACCOUNT')]

    def test_grouped_account(self, detector: PIIDetector) -> None:
        grouped = '40702 810 2 0000 0012345'
        got = _found(detector, f'реквизиты: {grouped}')
        assert got == [(grouped, 'BANK_ACCOUNT')]

    def test_bare_number_without_structure_is_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'отправление 12345678901234567890 передано'
        assert _found(detector, text) == []
        assert ('12345678901234567890', 'BANK_ACCOUNT') in _found(
            detector,
            text,
            0.0,
        )

    def test_keyword_without_valid_prefix_is_not_enough(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'счёт 12345678901234567890') == []

    def test_structurally_valid_tracking_number_is_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, f'трек-номер {self.ACCOUNT}') == []

    def test_account_is_not_reduced_to_a_phone_fragment(
        self,
        detector: PIIDetector,
    ) -> None:
        # В номере есть фрагмент 9910004312, похожий на телефон.
        account = '40817810099910004312'
        got = _found(detector, f'р/с {account} в банке')
        assert got == [(account, 'BANK_ACCOUNT')]

    def test_random_twenty_digit_numbers_rarely_look_like_accounts(
        self,
    ) -> None:
        import random

        from privacyguard_pipeline.detection.pattern_matcher import (
            PatternMatcher,
        )

        matcher = PatternMatcher()
        rng = random.Random(20261003)
        total = 20000
        hits = 0
        for _ in range(total):
            number = ''.join(rng.choice('0123456789') for _ in range(20))
            spans = matcher.detect(f'номер {number} в системе')
            hits += any(s.entity_type == 'BANK_ACCOUNT' for s in spans)
        # Ожидание ~1.5e-4: префикс из суженного набора (1.5%) и код
        # валюты из десяти распространённых (1.0%) одновременно.
        assert hits / total < 0.001, hits


class TestCarPlate:
    @pytest.mark.parametrize('plate', ['А123ВС77', 'A123BC77', 'а123вс77'])
    def test_cyrillic_latin_and_lowercase(
        self,
        detector: PIIDetector,
        plate: str,
    ) -> None:
        assert _found(detector, f'машина {plate}') == [(plate, 'CAR_PLATE')]

    def test_three_digit_region_with_space(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'К777ХН 799 стоит во дворе')
        assert got == [('К777ХН 799', 'CAR_PLATE')]

    @pytest.mark.parametrize('plate', ['А123ВС00', 'Д123ЖЗ77', 'А123ВС999'])
    def test_invalid_region_or_letter_not_detected(
        self,
        detector: PIIDetector,
        plate: str,
    ) -> None:
        got = _found(detector, f'машина {plate}')
        assert all(t != 'CAR_PLATE' for _, t in got)

    def test_valid_three_digit_region_other_than_moscow(
        self,
        detector: PIIDetector,
    ) -> None:
        # 778 - код Санкт-Петербурга.
        got = _found(detector, 'машина А123ВС778')
        assert ('А123ВС778', 'CAR_PLATE') in got

    def test_keyword_only_raises_confidence(
        self,
        detector: PIIDetector,
    ) -> None:
        plain = detector.detect('машина А123ВС77').spans[0].confidence
        keyed = detector.detect('гос номер А123ВС77').spans[0].confidence
        assert keyed > plain

    def test_anti_context_drops_a_product_code(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'артикул А123ВС77') == []

    def test_not_taken_from_the_middle_of_a_word(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'ХА123ВС77') == []


class TestDriverLicenceVersusPassport:
    def test_licence_keyword(self, detector: PIIDetector) -> None:
        got = _found(detector, 'водительское удостоверение 99 01 123456')
        assert got == [('99 01 123456', 'DRIVER_LICENSE')]

    def test_short_keyword_and_cyrillic_series(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'в/у 77 ТТ 123456')
        assert got == [('77 ТТ 123456', 'DRIVER_LICENSE')]

    def test_passport_keyword(self, detector: PIIDetector) -> None:
        got = _found(detector, 'паспорт 45 10 123456')
        assert got == [('45 10 123456', 'PASSPORT')]

    def test_neither_keyword_is_exactly_the_passport_result(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'номер 45 10 123456') == [
            ('45 10 123456', 'PASSPORT'),
        ]

    def test_licence_candidate_without_keyword_never_wins_at_zero(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'номер 45 10 123456', 0.0)
        assert ('45 10 123456', 'PASSPORT') in got
        assert all(t != 'DRIVER_LICENSE' for _, t in got)


class TestOms:
    POLICY = '1234567890123452'  # Mod10 верен

    def test_policy_with_keyword_is_not_a_card(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, f'полис ОМС {self.POLICY}')
        assert (self.POLICY, 'OMS') in got
        assert (self.POLICY, 'CARD') not in got

    def test_same_digits_without_keyword_stay_a_card(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, f'номер {self.POLICY}') == [
            (self.POLICY, 'CARD'),
        ]

    def test_keyword_with_luhn_valid_real_card_number(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'полис 4111 1111 1111 1111')
        assert got == [('4111 1111 1111 1111', 'OMS')]

    def test_bad_checksum_without_keyword_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'номер 1234567890123456') == []

    def test_card_next_to_unrelated_words_unchanged(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'карта 4111 1111 1111 1111')
        assert got == [('4111 1111 1111 1111', 'CARD')]


class TestBirthdate:
    @pytest.mark.parametrize(
        ('text', 'date_text'),
        [
            ('дата рождения 05.03.1990', '05.03.1990'),
            ('родился 5 мая 1990 г.', '5 мая 1990'),
            ('д.р. 5/3/1990', '5/3/1990'),
            ('05.03.1990 г.р.', '05.03.1990'),
            ('Дата рождения: 05-03-1990', '05-03-1990'),
        ],
    )
    def test_date_next_to_the_keyword(
        self,
        detector: PIIDetector,
        text: str,
        date_text: str,
    ) -> None:
        assert _found(detector, text) == [(date_text, 'BIRTHDATE')]

    def test_keyword_is_left_unmasked(self, detector: PIIDetector) -> None:
        text = 'дата рождения 05.03.1990'
        spans = detector.detect(text).spans
        assert [s.text for s in spans] == ['05.03.1990']

    @pytest.mark.parametrize(
        'text',
        ['договор от 05.03.2024', 'встреча 12.10.2025', 'отчёт за 01.02.2023'],
    )
    def test_date_without_keyword(
        self,
        detector: PIIDetector,
        text: str,
    ) -> None:
        got = _found(detector, text)
        assert all(t != 'BIRTHDATE' for _, t in got)

    @pytest.mark.parametrize(
        'text',
        [
            'дата рождения 31.02.1990',
            'дата рождения 05.03.2999',
            'дата рождения 05.03.1850',
        ],
    )
    def test_impossible_or_implausible_date(
        self,
        detector: PIIDetector,
        text: str,
    ) -> None:
        got = _found(detector, text)
        assert all(t != 'BIRTHDATE' for _, t in got)

    def test_leap_day_is_accepted_only_in_a_leap_year(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'родился 29.02.2000') == [
            ('29.02.2000', 'BIRTHDATE'),
        ]
        got = _found(detector, 'родился 29.02.1999')
        assert all(t != 'BIRTHDATE' for _, t in got)

    def test_other_date_in_the_same_sentence_is_not_a_birthdate(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'дата рождения: 05.03.1990, дата выдачи 12.10.2015'
        assert _found(detector, text) == [('05.03.1990', 'BIRTHDATE')]

    def test_issue_date_before_the_birthdate_does_not_hide_it(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Паспорт выдан 01.01.2010, дата рождения 05.03.1990'
        assert _found(detector, text) == [('05.03.1990', 'BIRTHDATE')]

    def test_keyword_does_not_extend_to_a_second_date(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'родился 05.03.1990, а поступил 01.09.2008'
        assert _found(detector, text) == [('05.03.1990', 'BIRTHDATE')]

    def test_validate_birthdate_boundaries(self) -> None:
        from datetime import date

        from privacyguard_pipeline.detection.validators import (
            validate_birthdate,
        )

        today = date(2026, 10, 3)
        assert validate_birthdate('03.10.2026', today)
        assert not validate_birthdate('04.10.2026', today)
        assert validate_birthdate('03.10.1906', today)
        assert not validate_birthdate('02.10.1906', today)


class TestTelegram:
    @pytest.mark.parametrize(
        'text',
        [
            'мой телеграм @ivan_petrov',
            'tg: @ivan_petrov',
            'в Telegram @ivan_petrov',
        ],
    )
    def test_handle_with_keyword(
        self,
        detector: PIIDetector,
        text: str,
    ) -> None:
        # Natasha может отдельно пометить слово (Telegram как ORG).
        assert ('@ivan_petrov', 'TELEGRAM') in _found(detector, text)

    def test_handle_without_keyword_not_detected(
        self,
        detector: PIIDetector,
    ) -> None:
        assert _found(detector, 'пишите @ivan_petrov без слов') == []

    def test_handle_without_keyword_detected_at_zero(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'пишите @ivan_petrov', 0.0)
        assert ('@ivan_petrov', 'TELEGRAM') in got

    @pytest.mark.parametrize(
        'link',
        ['t.me/ivan_petrov', 'https://t.me/ivan_petrov'],
    )
    def test_link_needs_no_keyword(
        self,
        detector: PIIDetector,
        link: str,
    ) -> None:
        assert _found(detector, f'вот {link}') == [(link, 'TELEGRAM')]

    def test_email_stays_an_email(self, detector: PIIDetector) -> None:
        assert _found(detector, 'ivan_petrov@example.com') == [
            ('ivan_petrov@example.com', 'EMAIL'),
        ]

    def test_too_short_name_is_not_a_handle(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _found(detector, 'телеграм @abc')
        assert all(t != 'TELEGRAM' for _, t in got)

    def test_masking_round_trip(self, detector: PIIDetector) -> None:
        from privacyguard_pipeline.masker import Masker

        text = 'мой телеграм @ivan_petrov, пишите'
        masker = Masker()
        masked = masker.mask(text, detector.detect(text).spans)
        assert '@ivan_petrov' not in masked
        assert '<TELEGRAM_' in masked
        assert masker.demask(masked) == text
