"""Валидаторы для паттернов детекции PII.

Каждая функция-валидатор принимает результат совпадения и возвращает
True/False.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Callable

from privacyguard_pipeline.constants import (
    ACCOUNT_BIK_TAIL_DIGITS,
    ACCOUNT_CORR_BIK_SLICE,
    ACCOUNT_KEY_MODULO,
    ACCOUNT_KEY_WEIGHTS,
    ACCOUNT_LENGTH,
    BIK_LENGTH,
    BIK_PREFIX,
    BIRTHDATE_MAX_AGE_YEARS,
    FNS_SUBJECT_CODES,
    INN_10_CHECKSUM_WEIGHTS,
    INN_12_CHECKSUM_WEIGHTS_1,
    INN_12_CHECKSUM_WEIGHTS_2,
    INN_CHECKSUM_DIGIT_MODULO,
    INN_CHECKSUM_MODULO,
    INN_VALID_LENGTHS,
    IP_OCTET_MAX,
    IP_OCTET_MIN,
    KPP_INVALID_REASON_CODE,
    KPP_LENGTH,
    LUHN_DOUBLE_SUBTRACT,
    LUHN_MODULO,
    OGRN_13_LENGTH,
    OGRN_13_MODULO,
    OGRN_15_LENGTH,
    OGRN_15_MODULO,
    OGRN_CHECKSUM_DIGIT_MODULO,
    PASSPORT_DIGIT_COUNT,
    PLATE_REGION_CODES,
    SNILS_CHECKSUM_MIN_NUMBER,
    SNILS_CHECKSUM_MODULO,
    SNILS_DIGIT_COUNT,
    TELEGRAM_NAME_MAX_LENGTH,
    TELEGRAM_NAME_MIN_LENGTH,
)
from privacyguard_pipeline.detection.patterns import IP_RE, MONTHS_GENITIVE

# ---------------------------------------------------------------------------
# Алгоритм Луна
# ---------------------------------------------------------------------------


def luhn_check(digits: str) -> bool:
    """Проверяет число по алгоритму Луна."""
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        if not ch.isdigit():
            return False
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= LUHN_DOUBLE_SUBTRACT
        total += n
    return total % LUHN_MODULO == 0


# ---------------------------------------------------------------------------
# Валидаторы, специфичные для типа сущности
# ---------------------------------------------------------------------------


def validate_ip(octets: tuple[str, ...]) -> bool:
    """Проверяет октеты IPv4-адреса."""
    if len(octets) != 4:
        return False
    for octet in octets:
        val = int(octet)
        if val < IP_OCTET_MIN or val > IP_OCTET_MAX:
            return False
    return True


def validate_card(raw: str) -> bool:
    """Проверяет номер банковской карты по алгоритму Луна."""
    clean = raw.replace('-', '').replace(' ', '')
    return bool(clean) and luhn_check(clean)


def _inn_control_digit(digits: str, weights: tuple[int, ...]) -> int:
    """Вычисляет одну контрольную цифру ИНН по алгоритму ФНС."""
    total = sum(int(d) * w for d, w in zip(digits, weights))
    return (total % INN_CHECKSUM_MODULO) % INN_CHECKSUM_DIGIT_MODULO


def validate_inn_checksum(digits: str) -> bool:
    """Проверяет контрольную(ые) цифру(ы) ИНН по алгоритму ФНС.

    Предполагает, что ``digits`` уже состоит ровно из 10 или 12
    цифровых символов.
    """
    if len(digits) == INN_VALID_LENGTHS[0]:  # 10 цифр
        return _inn_control_digit(digits[:9], INN_10_CHECKSUM_WEIGHTS) == int(
            digits[9]
        )

    n11 = _inn_control_digit(digits[:10], INN_12_CHECKSUM_WEIGHTS_1)
    n12 = _inn_control_digit(digits[:11], INN_12_CHECKSUM_WEIGHTS_2)
    return n11 == int(digits[10]) and n12 == int(digits[11])


def validate_inn(raw: str) -> bool:
    """Проверяет ИНН: 10 или 12 цифр с валидной контрольной суммой."""
    clean = raw.strip()
    if len(clean) not in INN_VALID_LENGTHS or not clean.isdigit():
        return False
    return validate_inn_checksum(clean)


def validate_snils_checksum(raw: str) -> bool:
    """Проверяет контрольное число СНИЛС (последние две цифры).

    Для номеров не выше 001-001-998 контрольное число не определено,
    поэтому такие номера проверку не проходят.
    """
    digits = re.sub(r'\D', '', raw)
    if len(digits) != SNILS_DIGIT_COUNT:
        return False
    number, control = int(digits[:9]), int(digits[9:])
    if number <= SNILS_CHECKSUM_MIN_NUMBER:
        return False
    total = sum(int(d) * w for d, w in zip(digits[:9], range(9, 0, -1)))
    expected = total % SNILS_CHECKSUM_MODULO
    if expected == SNILS_CHECKSUM_MODULO - 1:
        expected = 0
    return expected == control


def validate_ogrn_checksum(raw: str) -> bool:
    """Проверяет контрольную цифру ОГРН (13 цифр) или ОГРНИП (15)."""
    digits = re.sub(r'\D', '', raw)
    if len(digits) == OGRN_13_LENGTH:
        modulo = OGRN_13_MODULO
    elif len(digits) == OGRN_15_LENGTH:
        modulo = OGRN_15_MODULO
    else:
        return False
    expected = (int(digits[:-1]) % modulo) % OGRN_CHECKSUM_DIGIT_MODULO
    return expected == int(digits[-1])


def validate_passport(raw: str) -> bool:
    """Проверяет паспорт: серия (4) + номер (6) = 10 цифр.

    Считает только цифры, поэтому любой разделитель (включая
    неразрывный пробел из текстового слоя PDF) не влияет на результат.
    """
    digits = re.sub(r'\D', '', raw)
    return len(digits) == PASSPORT_DIGIT_COUNT


_BIK_RE = re.compile(
    rf'бик\W{{0,3}}({BIK_PREFIX}\d{{{BIK_LENGTH - len(BIK_PREFIX)}}})(?!\d)',
    re.IGNORECASE,
)
# Корреспондентские счета кредитных организаций (301xx) проверяются по
# другой схеме, чем расчётные.
_CORRESPONDENT_PREFIX = '301'


def find_biks(text: str) -> list[str]:
    """Находит БИК (девять цифр с префиксом «04» после метки «БИК»)."""
    return _BIK_RE.findall(text)


def validate_account_key(account: str, bik: str) -> bool:
    """Проверяет ключ счёта по БИК (Положение Банка России об ЭБП).

    Для расчётного счёта берутся три последние цифры БИК и 20 цифр
    счёта, для корреспондентского (301xx) - ноль, 5-6-я цифры БИК и 20
    цифр счёта. Сумма ``(цифра * вес) mod 10`` по 23 цифрам с весами
    7, 1, 3 по кругу должна дать 0 по модулю 10.
    """
    if len(account) != ACCOUNT_LENGTH or len(bik) != BIK_LENGTH:
        return False
    if not (account.isdigit() and bik.isdigit()):
        return False
    if account.startswith(_CORRESPONDENT_PREFIX):
        lo, hi = ACCOUNT_CORR_BIK_SLICE
        digits = '0' + bik[lo:hi] + account
    else:
        digits = bik[-ACCOUNT_BIK_TAIL_DIGITS:] + account
    total = sum(
        (int(d) * w) % ACCOUNT_KEY_MODULO
        for d, w in zip(digits, ACCOUNT_KEY_WEIGHTS)
    )
    return total % ACCOUNT_KEY_MODULO == 0


def validate_kpp(raw: str) -> bool:
    """Проверяет структуру КПП: 9 знаков, регион и причина постановки.

    Первые две цифры кода налогового органа должны входить в справочник
    кодов субъектов ФНС, а код причины постановки (5-6-й знаки) не
    может быть ``00``. Контрольной суммы у КПП нет.
    """
    if (
        len(raw) != KPP_LENGTH
        or not raw[:4].isdigit()
        or not raw[6:].isdigit()
    ):
        return False
    if raw[:2] not in FNS_SUBJECT_CODES:
        return False
    return raw[4:6] != KPP_INVALID_REASON_CODE


# Часть знака до региона: буква, три цифры, две буквы.
_PLATE_HEAD_LENGTH = 6


def validate_car_plate(raw: str) -> bool:
    """Проверяет код региона госномера по справочнику регионов на знаках."""
    compact = re.sub(r'\s', '', raw)
    return compact[_PLATE_HEAD_LENGTH:] in PLATE_REGION_CODES


def _parse_birthdate(raw: str) -> date | None:
    """Разбирает дату в числовой или словесной форме; ``None`` - не дата."""
    numeric = re.fullmatch(r'(\d{1,2})[./-](\d{1,2})[./-](\d{4})', raw)
    if numeric:
        day, month, year = (int(g) for g in numeric.groups())
    else:
        words = re.fullmatch(r'(\d{1,2})\s+(\S+)\s+(\d{4})', raw)
        if not words or words.group(2).lower() not in MONTHS_GENITIVE:
            return None
        day = int(words.group(1))
        month = MONTHS_GENITIVE.index(words.group(2).lower()) + 1
        year = int(words.group(3))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def validate_birthdate(raw: str, today: date | None = None) -> bool:
    """Проверяет реальную календарную дату с правдоподобным возрастом.

    Невозможные даты (31.02), будущие даты и возраст старше
    ``BIRTHDATE_MAX_AGE_YEARS`` лет отклоняются.
    """
    parsed = _parse_birthdate(raw)
    if parsed is None:
        return False
    today = today or date.today()
    if parsed > today:
        return False
    try:
        oldest = today.replace(year=today.year - BIRTHDATE_MAX_AGE_YEARS)
    except ValueError:  # 29 февраля
        oldest = today.replace(
            year=today.year - BIRTHDATE_MAX_AGE_YEARS,
            day=28,
        )
    return parsed >= oldest


def validate_telegram(raw: str) -> bool:
    """Проверяет ник Telegram: 5-32 знака, начинается с латинской буквы."""
    name = raw.rsplit('/', 1)[-1].lstrip('@')
    return (
        TELEGRAM_NAME_MIN_LENGTH <= len(name) <= TELEGRAM_NAME_MAX_LENGTH
        and name[:1].isascii()
        and name[:1].isalpha()
    )


def validate_phone(text: str, start: int, end: int) -> bool:
    """Проверяет номер телефона с помощью библиотеки phonenumbers."""
    try:
        from phonenumbers import PhoneNumberMatcher

        matches = list(
            PhoneNumberMatcher(text[start:end], 'RU'),
        )
        if not matches:
            matches = list(
                PhoneNumberMatcher(text[start:end], None),
            )
        return bool(matches)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Реестр валидаторов
# ---------------------------------------------------------------------------

ValidatorFunc = Callable[[str, str, int, int], bool]

VALIDATOR_REGISTRY: dict[str, ValidatorFunc] = {
    'CARD': lambda raw, text, start, end: validate_card(raw),
    'IP': lambda raw, text, start, end: validate_ip(
        re.match(IP_RE, raw).groups() if re.match(IP_RE, raw) else (),  # type: ignore[union-attr]
    ),
    'INN': lambda raw, text, start, end: validate_inn(raw),
    'KPP': lambda raw, text, start, end: validate_kpp(raw),
    'CAR_PLATE': lambda raw, text, start, end: validate_car_plate(raw),
    'BIRTHDATE': lambda raw, text, start, end: validate_birthdate(raw),
    'TELEGRAM': lambda raw, text, start, end: validate_telegram(raw),
    'PASSPORT': lambda raw, text, start, end: validate_passport(raw),
    'PHONE': lambda raw, text, start, end: validate_phone(text, start, end),
}
