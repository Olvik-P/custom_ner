"""Скоринг pattern-совпадений: оценка уверенности по контексту и признакам.

Для типов с неоднозначным форматом (паспорт, СНИЛС, ОГРН, координаты,
IP) уверенность считается аддитивно: базовая оценка типа + бонусы
(контрольная сумма, ключевое слово рядом, признак формата) - штраф за
анти-контекст. Строго валидированные типы (EMAIL, URL, CARD, INN, PHONE)
в реестре отсутствуют и получают фиксированную ``PATTERN_CONFIDENCE``.
"""

from __future__ import annotations

import re
from typing import Callable

from privacyguard_pipeline.constants import (
    COORDS_LATITUDE_MAX,
    COORDS_LONGITUDE_MAX,
    COORDS_PRECISE_MIN_DECIMALS,
    PASSPORT_YEAR_MAX,
    PASSPORT_YEAR_MIN_LEGACY,
    PATTERN_CONFIDENCE,
    SCORE_ANTI_CONTEXT_PENALTY,
    SCORE_COORDS_DECIMAL_BASE,
    SCORE_COORDS_DMS_BASE,
    SCORE_COORDS_IN_RANGE,
    SCORE_COORDS_KEYWORD,
    SCORE_COORDS_PRECISE,
    SCORE_IP_BASE,
    SCORE_IP_KEYWORD,
    SCORE_OGRN_BASE,
    SCORE_OGRN_CHECKSUM,
    SCORE_OGRN_KEYWORD,
    SCORE_PASSPORT_BASE,
    SCORE_PASSPORT_KEYWORD,
    SCORE_PASSPORT_PAIRED_SERIES,
    SCORE_PASSPORT_PLAUSIBLE_YEAR,
    SCORE_SNILS_BASE,
    SCORE_SNILS_CHECKSUM,
    SCORE_SNILS_FORMATTED,
    SCORE_SNILS_KEYWORD,
    SCORING_WINDOW_AFTER,
    SCORING_WINDOW_BEFORE,
)
from privacyguard_pipeline.detection.validators import (
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


def _context_window(text: str, start: int, end: int) -> tuple[str, str]:
    """Возвращает (до, после) вокруг совпадения в пределах одной строки."""
    before = text[max(0, start - SCORING_WINDOW_BEFORE) : start]
    after = text[end : end + SCORING_WINDOW_AFTER]
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
    'COORDS': _score_coords,
    'IP': _score_ip,
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
    before, after = _context_window(text, start, end)
    return scorer(raw, before, after)
