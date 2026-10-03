"""Скоринг pattern-совпадений: оценка уверенности по контексту и признакам.

Для типов с неоднозначным форматом (паспорт, СНИЛС, ОГРН, координаты,
IP) уверенность считается аддитивно: базовая оценка типа + бонусы
(контрольная сумма, ключевое слово рядом, признак формата) - штраф за
анти-контекст. Строго валидированные типы (EMAIL, URL, CARD, INN, PHONE)
в реестре отсутствуют и получают фиксированную ``PATTERN_CONFIDENCE``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from privacyguard_pipeline.constants import (
    ACCOUNT_BALANCE_PREFIXES,
    ACCOUNT_STRUCTURE_CURRENCY_CODES,
    ACCOUNT_STRUCTURE_PREFIXES,
    ACCOUNT_WINDOW_AFTER,
    ACCOUNT_WINDOW_BEFORE,
    COORDS_LATITUDE_MAX,
    COORDS_LONGITUDE_MAX,
    COORDS_PRECISE_MIN_DECIMALS,
    PASSPORT_YEAR_MAX,
    PASSPORT_YEAR_MIN_LEGACY,
    PATTERN_CONFIDENCE,
    SCORE_ACCOUNT_BASE,
    SCORE_ACCOUNT_BIK_KEY,
    SCORE_ACCOUNT_KEYWORD,
    SCORE_ACCOUNT_PREFIX,
    SCORE_ACCOUNT_STRUCTURE,
    SCORE_ANTI_CONTEXT_PENALTY,
    SCORE_BIRTHDATE_BASE,
    SCORE_BIRTHDATE_KEYWORD,
    SCORE_COORDS_DECIMAL_BASE,
    SCORE_COORDS_DMS_BASE,
    SCORE_COORDS_IN_RANGE,
    SCORE_COORDS_KEYWORD,
    SCORE_COORDS_PRECISE,
    SCORE_DRIVER_LICENSE_BASE,
    SCORE_DRIVER_LICENSE_KEYWORD,
    SCORE_DRIVER_LICENSE_PAIRED_SERIES,
    SCORE_IP_BASE,
    SCORE_IP_KEYWORD,
    SCORE_KPP_BASE,
    SCORE_KPP_KEYWORD,
    SCORE_OGRN_BASE,
    SCORE_OGRN_CHECKSUM,
    SCORE_OGRN_KEYWORD,
    SCORE_OMS_BASE,
    SCORE_OMS_CHECKSUM,
    SCORE_OMS_KEYWORD,
    SCORE_PASSPORT_BASE,
    SCORE_PASSPORT_KEYWORD,
    SCORE_PASSPORT_PAIRED_SERIES,
    SCORE_PASSPORT_PLAUSIBLE_YEAR,
    SCORE_PLATE_BASE,
    SCORE_PLATE_KEYWORD,
    SCORE_SNILS_BASE,
    SCORE_SNILS_CHECKSUM,
    SCORE_SNILS_FORMATTED,
    SCORE_SNILS_KEYWORD,
    SCORE_TELEGRAM_HANDLE_BASE,
    SCORE_TELEGRAM_KEYWORD,
    SCORE_TELEGRAM_LINK,
    SCORING_WINDOW_AFTER,
    SCORING_WINDOW_BEFORE,
)
from privacyguard_pipeline.detection.patterns import BIRTHDATE_RE
from privacyguard_pipeline.detection.validators import (
    find_biks,
    luhn_check,
    validate_account_key,
    validate_ogrn_checksum,
    validate_snils_checksum,
)

_FLAGS = re.IGNORECASE

# Ключевые слова типа: только собственные слова каждого типа, чтобы
# «ИНН» рядом не давал бонус паспорту и наоборот.
_PASSPORT_KEYWORD_RE = re.compile(
    r'паспорт\w*|серия|серии|выдан\w*|\bсер\.|passport',
    _FLAGS,
)
_SNILS_KEYWORD_RE = re.compile(
    r'снилс|страхов\w*\s+свидетельств\w*|пенсионн\w*|snils',
    _FLAGS,
)
_OGRN_KEYWORD_RE = re.compile(r'огрн\w*|ogrn', _FLAGS)
_KPP_KEYWORD_RE = re.compile(r'кпп|\bkpp\b', _FLAGS)
_PLATE_KEYWORD_RE = re.compile(
    r'гос\.?\s?номер\w*|госзнак\w*|номер\w*\s+(?:авто\w*|машин\w*|т/?с)'
    r'|регистрационн\w*\s+знак\w*|автомобил\w*|license\s+plate',
    _FLAGS,
)
_DRIVER_LICENSE_KEYWORD_RE = re.compile(
    r'водител\w*|\bв/у\b|\bв\.у\.|\bву\b|driver',
    _FLAGS,
)
_OMS_KEYWORD_RE = re.compile(
    r'полис\w*|\bомс\b|\bенп\b|медицинск\w*\s+страхов\w*'
    r'|обязательн\w*\s+медицинск\w*',
    _FLAGS,
)
_BIRTH_KEYWORD_RE = re.compile(
    r'рожд\w*|родил\w*|date\s+of\s+birth|\bdob\b|\bд\.\s?р\.|\bб-?day\b',
    _FLAGS,
)
# После даты: «05.03.1990 г.р.», «1990 года рождения».
_BIRTH_AFTER_RE = re.compile(
    r'^\s*(?:г\.\s?р\.|года\s+рожд\w*|г\.\s?рожд\w*)',
    _FLAGS,
)
# Подпись непосредственно перед датой, показывающая, что это другая
# дата (выдача, срок действия, договор), а не рождение.
_BIRTH_OTHER_LABEL_BEFORE_RE = re.compile(
    r'(?:выдан\w*|выдач\w*|действител\w*|договор\w*|срок\w*|заключ\w*'
    r'|оплат\w*|заказ\w*)\W{0,3}(?:от|до|по|с)?\W{0,3}$',
    _FLAGS,
)
_TELEGRAM_KEYWORD_RE = re.compile(
    r'телеграм\w*|telegram|\btg\b|\bтг\b|\bтелега\b',
    _FLAGS,
)
_ACCOUNT_KEYWORD_RE = re.compile(
    r'р/с|р\.\s?с\b|к/с|сч[её]т\w*|\baccount\b|\biban\b',
    _FLAGS,
)
# Число, которое подписано собственной меткой другого реквизита
# («БИК 044525225», «ИНН ...»), не получает бонус от ключевого слова КПП,
# стоящего дальше в той же строке: «КПП 771234234, БИК 044525225».
_KPP_OTHER_LABEL_BEFORE_RE = re.compile(
    r'(?:бик|инн|огрн\w*|р/с|к/с|сч[её]т|тел\.?|№)\W{0,3}$',
    _FLAGS,
)
_COORDS_KEYWORD_RE = re.compile(
    r'координат\w*|широт\w*|долгот\w*|\blat\b|\blon\b|\blng\b|\bgps\b'
    r'|coordinates?',
    _FLAGS,
)
_IP_KEYWORD_RE = re.compile(
    r'\bip\b|ip-?адрес\w*|адрес\w*|сервер\w*|хост\w*|\bhost\b|шлюз\w*'
    r'|address',
    _FLAGS,
)

# Анти-контекст: маркеры «это не PII, а номер заказа/цена/версия».
# Метки, подписывающие само число, учитываются только перед ним: слово
# после числа («..., тел. +7...») подписывает уже следующую сущность.
_ANTI_BEFORE_RE = re.compile(
    r'\bзаказ\w{0,2}\b|артикул\w*|\bарт\.|накладн\w*|\bтрек\w*|\bsku\b'
    r'|\bтел\.|\bцен[аыеу]\b|стоимост\w*|[$€₽]\s*$',
    _FLAGS,
)
# Единицы измерения идут сразу после числа: «1234 руб.», «50 $».
_ANTI_AFTER_RE = re.compile(r'^\s*(?:руб\w*|р\.|₽|\$|€)', _FLAGS)
_ANTI_VERSION_RE = re.compile(r'верси\w*|\bversion\b|\bver\.', _FLAGS)
# Приставка «v» вплотную перед числом: v10.0.19.42
_VERSION_PREFIX_RE = re.compile(r'(?:^|[\s(])v\.?\s*$', _FLAGS)

_PAIRED_SERIES_RE = re.compile(r'^\d{2}[\s-]\d{2}')
_DECIMAL_RE = re.compile(r'-?\d{1,3}\.(\d+)')


@dataclass(frozen=True)
class ScoringWindow:
    """Окно контекста вокруг совпадения для оценщика типа.

    Attributes:
        before: Сколько символов брать до совпадения.
        after: Сколько символов брать после совпадения.
        cross_lines: Пересекать ли границы строк (по умолчанию окно
            обрезается по строке, чтобы подпись соседней строки не
            влияла на оценку).
    """

    before: int
    after: int
    cross_lines: bool = False


_DEFAULT_WINDOW = ScoringWindow(SCORING_WINDOW_BEFORE, SCORING_WINDOW_AFTER)


def _context_window(
    text: str,
    start: int,
    end: int,
    window: ScoringWindow = _DEFAULT_WINDOW,
) -> tuple[str, str]:
    """Возвращает (до, после) вокруг совпадения.

    По умолчанию окно ограничено одной строкой; ``window.cross_lines``
    разрешает захватывать соседние строки.
    """
    before = text[max(0, start - window.before) : start]
    after = text[end : end + window.after]
    if not window.cross_lines:
        before = before.rsplit('\n', 1)[-1]
        after = after.split('\n', 1)[0]
    return before, after


def _has(pattern: re.Pattern[str], before: str, after: str) -> bool:
    return bool(pattern.search(before) or pattern.search(after))


def _has_anti_context(before: str, after: str) -> bool:
    return bool(_ANTI_BEFORE_RE.search(before) or _ANTI_AFTER_RE.search(after))


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, round(score, 4)))


def _score_passport(raw: str, before: str, after: str) -> float:
    digits = re.sub(r'\D', '', raw)
    score = SCORE_PASSPORT_BASE
    if _has(_PASSPORT_KEYWORD_RE, before, after):
        score += SCORE_PASSPORT_KEYWORD
    if _PAIRED_SERIES_RE.match(raw):
        score += SCORE_PASSPORT_PAIRED_SERIES
    year = int(digits[2:4])
    if year <= PASSPORT_YEAR_MAX or year >= PASSPORT_YEAR_MIN_LEGACY:
        score += SCORE_PASSPORT_PLAUSIBLE_YEAR
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_snils(raw: str, before: str, after: str) -> float:
    score = SCORE_SNILS_BASE
    if validate_snils_checksum(raw):
        score += SCORE_SNILS_CHECKSUM
    if _has(_SNILS_KEYWORD_RE, before, after):
        score += SCORE_SNILS_KEYWORD
    if re.search(r'[-\s]', raw):
        score += SCORE_SNILS_FORMATTED
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_ogrn(raw: str, before: str, after: str) -> float:
    score = SCORE_OGRN_BASE
    if validate_ogrn_checksum(raw):
        score += SCORE_OGRN_CHECKSUM
    if _has(_OGRN_KEYWORD_RE, before, after):
        score += SCORE_OGRN_KEYWORD
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_kpp(raw: str, before: str, after: str) -> float:
    score = SCORE_KPP_BASE
    labelled_by_other = _KPP_OTHER_LABEL_BEFORE_RE.search(before)
    if not labelled_by_other and _has(_KPP_KEYWORD_RE, before, after):
        score += SCORE_KPP_KEYWORD
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_car_plate(raw: str, before: str, after: str) -> float:
    score = SCORE_PLATE_BASE
    if _has(_PLATE_KEYWORD_RE, before, after):
        score += SCORE_PLATE_KEYWORD
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_driver_license(raw: str, before: str, after: str) -> float:
    score = SCORE_DRIVER_LICENSE_BASE
    if _has(_DRIVER_LICENSE_KEYWORD_RE, before, after):
        score += SCORE_DRIVER_LICENSE_KEYWORD
    if _PAIRED_SERIES_RE.match(raw) or re.match(
        r'^\d{2}[\s-][А-Яа-я]{2}', raw
    ):
        score += SCORE_DRIVER_LICENSE_PAIRED_SERIES
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_oms(raw: str, before: str, after: str) -> float:
    digits = re.sub(r'\D', '', raw)
    score = SCORE_OMS_BASE
    if luhn_check(digits):
        score += SCORE_OMS_CHECKSUM
    if _has(_OMS_KEYWORD_RE, before, after):
        score += SCORE_OMS_KEYWORD
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_birthdate(raw: str, before: str, after: str) -> float:
    score = SCORE_BIRTHDATE_BASE
    # Ключевое слово «рождения» относится к ближайшей дате: если между
    # словом и этой датой стоит другая дата, бонуса нет.
    keyword_before = None
    for keyword in _BIRTH_KEYWORD_RE.finditer(before):
        keyword_before = keyword
    related = keyword_before is not None and not BIRTHDATE_RE.search(
        before[keyword_before.end() :],
    )
    if related or _BIRTH_AFTER_RE.search(after):
        score += SCORE_BIRTHDATE_KEYWORD
    if _BIRTH_OTHER_LABEL_BEFORE_RE.search(before):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_telegram(raw: str, before: str, after: str) -> float:
    if 't.me/' in raw.lower():
        return _clamp(SCORE_TELEGRAM_LINK)
    score = SCORE_TELEGRAM_HANDLE_BASE
    if _has(_TELEGRAM_KEYWORD_RE, before, after):
        score += SCORE_TELEGRAM_KEYWORD
    return _clamp(score)


def _same_line_tail(before: str, after: str) -> tuple[str, str]:
    """Узкое окно (как по умолчанию) из широкого окна счёта."""
    near_before = before[-SCORING_WINDOW_BEFORE:].rsplit('\n', 1)[-1]
    near_after = after[:SCORING_WINDOW_AFTER].split('\n', 1)[0]
    return near_before, near_after


def _score_bank_account(raw: str, before: str, after: str) -> float:
    digits = re.sub(r'\D', '', raw)
    near_before, near_after = _same_line_tail(before, after)
    score = SCORE_ACCOUNT_BASE
    bik_context = f'{before}\n{after}'
    if any(validate_account_key(digits, b) for b in find_biks(bik_context)):
        score += SCORE_ACCOUNT_BIK_KEY
    prefix = digits[:3]
    if prefix in ACCOUNT_BALANCE_PREFIXES:
        score += SCORE_ACCOUNT_PREFIX
        if _has(_ACCOUNT_KEYWORD_RE, near_before, near_after):
            score += SCORE_ACCOUNT_KEYWORD
    if (
        prefix in ACCOUNT_STRUCTURE_PREFIXES
        and digits[5:8] in ACCOUNT_STRUCTURE_CURRENCY_CODES
    ):
        score += SCORE_ACCOUNT_STRUCTURE
    if _has_anti_context(near_before, near_after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_coords(raw: str, before: str, after: str) -> float:
    # Градусы/минуты/секунды с полусферами: явная запись координат.
    if '°' in raw:
        return _clamp(SCORE_COORDS_DMS_BASE)

    score = SCORE_COORDS_DECIMAL_BASE
    if _has(_COORDS_KEYWORD_RE, before, after):
        score += SCORE_COORDS_KEYWORD
    values = [float(m.group(0)) for m in _DECIMAL_RE.finditer(raw)]
    decimals = [len(m.group(1)) for m in _DECIMAL_RE.finditer(raw)]
    if len(values) == 2:
        lat, lon = values
        if (
            abs(lat) <= COORDS_LATITUDE_MAX
            and abs(lon) <= COORDS_LONGITUDE_MAX
        ):
            score += SCORE_COORDS_IN_RANGE
        if min(decimals) >= COORDS_PRECISE_MIN_DECIMALS:
            score += SCORE_COORDS_PRECISE
    if _has_anti_context(before, after):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


def _score_ip(raw: str, before: str, after: str) -> float:
    score = SCORE_IP_BASE
    if _has(_IP_KEYWORD_RE, before, after):
        score += SCORE_IP_KEYWORD
    if _ANTI_VERSION_RE.search(before) or _VERSION_PREFIX_RE.search(before):
        score -= SCORE_ANTI_CONTEXT_PENALTY
    return _clamp(score)


ScorerFunc = Callable[[str, str, str], float]

SCORING_REGISTRY: dict[str, ScorerFunc] = {
    'PASSPORT': _score_passport,
    'SNILS': _score_snils,
    'OGRN': _score_ogrn,
    'OGRNIP': _score_ogrn,
    'KPP': _score_kpp,
    'BANK_ACCOUNT': _score_bank_account,
    'CAR_PLATE': _score_car_plate,
    'DRIVER_LICENSE': _score_driver_license,
    'OMS': _score_oms,
    'BIRTHDATE': _score_birthdate,
    'TELEGRAM': _score_telegram,
    'COORDS': _score_coords,
    'IP': _score_ip,
}

# Типы с нестандартным окном контекста; остальные используют окно по
# умолчанию (40 символов до, 15 после, в пределах строки).
SCORING_WINDOWS: dict[str, ScoringWindow] = {
    # БИК стоит в соседней строке реквизитов, поэтому окно шире и
    # пересекает строки; слово-подпись и анти-контекст оценщик ищет
    # только в узкой части.
    'BANK_ACCOUNT': ScoringWindow(
        ACCOUNT_WINDOW_BEFORE,
        ACCOUNT_WINDOW_AFTER,
        cross_lines=True,
    ),
}


def score_match(
    entity_type: str,
    raw: str,
    text: str,
    start: int,
    end: int,
) -> float:
    """Считает уверенность pattern-совпадения.

    Типы без записи в реестре строго валидируются жёсткими
    валидаторами и получают фиксированную ``PATTERN_CONFIDENCE``.
    """
    scorer = SCORING_REGISTRY.get(entity_type)
    if scorer is None:
        return PATTERN_CONFIDENCE
    window = SCORING_WINDOWS.get(entity_type, _DEFAULT_WINDOW)
    before, after = _context_window(text, start, end, window)
    return scorer(raw, before, after)
