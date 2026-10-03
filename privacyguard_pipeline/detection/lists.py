"""Списки оператора: allow (никогда не маскировать) и deny (всегда).

Записи — литеральные строки, не regex и не glob. Нормализация единая для
allow и deny: регистр, ``ё`` -> ``е``, кавычки и окружающая пунктуация,
пробелы. Дальше два режима сравнения:

* allow — спан целиком равен записи целиком (после снятия ведущей
  юридической формы и замены слов их леммой; строка из семи и более
  цифр сравнивается только по цифрам). Подстрока не подавляет спан, а
  лемма берётся строго по самому вероятному разбору — ошибка здесь
  в сторону «не подавлять».
* deny — последовательность токенов текста совпадает с токенами записи
  с точностью до формы слова (пересечение множеств лемм — ошибка здесь
  в сторону «замаскировать»); границы слов соблюдаются, потому что
  сравниваются токены, а не подстроки.

Содержимое списков — чувствительные данные: оно не попадает в логи,
аудит и тексты ошибок (в ошибке — индекс записи и причина).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.constants import (
    CUSTOM_ENTITY_TYPE,
    CUSTOM_SOURCE,
    DENY_CONFIDENCE,
    LEGAL_FORMS,
    LIST_DIGITS_ONLY_THRESHOLD,
    MAX_LIST_ENTRIES,
    MAX_LIST_ENTRY_CHARS,
)
from privacyguard_pipeline.detection.common import PIISpan
from privacyguard_pipeline.detection.morphology import (
    Morphology,
    get_morphology,
    normalize_word,
)
from privacyguard_pipeline.exceptions import InvalidListEntryError

# Токен: слова из букв/цифр, соединённые внутри дефисом, точкой, «@», «+»
# или «/» (e-mail, коды, составные слова) — как одно целое.
_TOKEN_RE = re.compile(r'\w+(?:[-.@+/]\w+)*')
_ALPHA_RE = re.compile(r'[^\W\d_]+')
# Между токенами многословной записи допустимы только пробелы, кавычки,
# скобки, запятые и тире.
_GAP_RE = re.compile(r'[\s«»"\'“”„()\[\],–—-]*')


@dataclass(frozen=True)
class _Token:
    """Токен текста: границы и множество допустимых лемм."""

    start: int
    end: int
    forms: frozenset[str]


def _is_alpha_compound(token: str) -> bool:
    """Слово из букв, возможно составное через дефис."""
    return all(_ALPHA_RE.fullmatch(part) for part in token.split('-'))


def _first_form(token: str, morph: Morphology) -> str:
    """Единственная нормализованная форма токена: лемма первого разбора."""
    low = normalize_word(token)
    if not _is_alpha_compound(low):
        return low
    return '-'.join(morph.first_lemma(p) for p in low.split('-'))


def _entry_forms(token: str, morph: Morphology) -> frozenset[str]:
    """Формы токена записи deny: каноничная лемма и написание как есть."""
    return frozenset({_first_form(token, morph), normalize_word(token)})


def _text_forms(token: str, morph: Morphology) -> frozenset[str]:
    """Формы токена текста: все его леммы и написание как есть.

    Запись сопоставляется по каноничной лемме, а токен текста
    предъявляет все возможные — так любая форма слова находит запись, но
    слово с омонимичной леммой («Ивана» при записи «Иванов») — нет.
    """
    low = normalize_word(token)
    if not _is_alpha_compound(low):
        return frozenset({low})
    if '-' in low:
        return frozenset({low, _first_form(low, morph)})
    return morph.lemmas(low) | {low}


def _digits_key(text: str) -> str | None:
    """``'#цифры'``, если в строке не меньше порога цифр, иначе ``None``."""
    digits = re.sub(r'\D', '', text)
    if len(digits) >= LIST_DIGITS_ONLY_THRESHOLD:
        return '#' + digits
    return None


def normalize_for_allow(text: str, morph: Morphology) -> tuple[str, ...]:
    """Нормализованная форма строки для сравнения в allow-списке."""
    digits_key = _digits_key(text)
    if digits_key is not None:
        return (digits_key,)
    forms = [_first_form(m.group(), morph) for m in _TOKEN_RE.finditer(text)]
    if len(forms) > 1 and forms[0] in LEGAL_FORMS:
        forms = forms[1:]
    return tuple(forms)


def validate_entries(entries: Sequence[str], label: str) -> None:
    """Проверяет записи списка; в ошибке — индекс и причина, не текст.

    Raises:
        InvalidListEntryError: Если записей больше лимита, запись не
            строка, длиннее лимита или пуста после нормализации.
    """
    if len(entries) > MAX_LIST_ENTRIES:
        msg = f'{label}: more than {MAX_LIST_ENTRIES} entries'
        raise InvalidListEntryError(msg)
    for index, entry in enumerate(entries):
        if not isinstance(entry, str):
            msg = f'{label}[{index}]: entry is not a string'
            raise InvalidListEntryError(msg)
        if len(entry) > MAX_LIST_ENTRY_CHARS:
            msg = (
                f'{label}[{index}]: entry is longer than '
                f'{MAX_LIST_ENTRY_CHARS} characters'
            )
            raise InvalidListEntryError(msg)
        if not _TOKEN_RE.search(entry):
            msg = f'{label}[{index}]: entry is empty after normalization'
            raise InvalidListEntryError(msg)


class AllowList:
    """Allow-список: спан целиком равен записи целиком после нормализации."""

    def __init__(
        self,
        entries: Sequence[str],
        morphology: Morphology | None = None,
    ) -> None:
        self._morph = morphology or get_morphology()
        self._keys = frozenset(
            normalize_for_allow(e, self._morph) for e in entries
        )

    def __len__(self) -> int:
        return len(self._keys)

    def matches(self, span_text: str) -> bool:
        """Подавляется ли спан с таким текстом."""
        if not self._keys:
            return False
        key = normalize_for_allow(span_text, self._morph)
        return bool(key) and key in self._keys


class DenyList:
    """Deny-список: поиск записей в тексте как спанов типа ``CUSTOM``.

    Сами строки записей не хранятся: только их нормализованные формы.
    """

    def __init__(
        self,
        entries: Sequence[str],
        morphology: Morphology | None = None,
    ) -> None:
        morph = morphology or get_morphology()
        self._morph = morph
        index: dict[str, list[tuple[frozenset[str], ...]]] = defaultdict(list)
        seen: set[tuple[frozenset[str], ...]] = set()
        for entry in entries:
            tokens = tuple(
                _entry_forms(m.group(), morph)
                for m in _TOKEN_RE.finditer(entry)
            )
            if not tokens or tokens in seen:
                continue
            seen.add(tokens)
            for form in tokens[0]:
                index[form].append(tokens)
        self._index = dict(index)
        self._count = len(seen)

    def __len__(self) -> int:
        return self._count

    def find_spans(self, text: str) -> list[PIISpan]:
        """Все вхождения записей deny-списка в тексте."""
        if not self._index:
            return []
        tokens = [
            _Token(m.start(), m.end(), _text_forms(m.group(), self._morph))
            for m in _TOKEN_RE.finditer(text)
        ]
        spans: list[PIISpan] = []
        for i, token in enumerate(tokens):
            candidates = {
                id(entry): entry
                for form in token.forms
                for entry in self._index.get(form, ())
            }
            best_end = -1
            for entry in candidates.values():
                end = self._match_at(tokens, i, entry, text)
                best_end = max(best_end, end)
            if best_end > 0:
                spans.append(
                    PIISpan(
                        start=token.start,
                        end=best_end,
                        text=text[token.start : best_end],
                        entity_type=CUSTOM_ENTITY_TYPE,
                        source=CUSTOM_SOURCE,
                        confidence=DENY_CONFIDENCE,
                    ),
                )
        return spans

    @staticmethod
    def _match_at(
        tokens: list[_Token],
        i: int,
        entry: tuple[frozenset[str], ...],
        text: str,
    ) -> int:
        """Конец совпадения записи с позиции ``i`` или ``-1``."""
        if i + len(entry) > len(tokens):
            return -1
        for offset, forms in enumerate(entry):
            token = tokens[i + offset]
            if not token.forms & forms:
                return -1
            if offset and not _GAP_RE.fullmatch(
                text[tokens[i + offset - 1].end : token.start],
            ):
                return -1
        return tokens[i + len(entry) - 1].end


def check_lists(
    allow_list: Sequence[str] | None,
    deny_list: Sequence[str] | None,
) -> None:
    """Проверяет списки настроек и вызова, не собирая их.

    Нужна вызывающим, которые обязаны отклонить невалидные записи до
    начала работы (создания файлов/handle), а не посреди обработки.

    Raises:
        InvalidListEntryError: Если любая запись невалидна.
    """
    validate_entries(settings.allow_list, 'allow_list(settings)')
    validate_entries(settings.deny_list, 'deny_list(settings)')
    validate_entries(allow_list or (), 'allow_list')
    validate_entries(deny_list or (), 'deny_list')


def compile_lists(
    settings_allow: Sequence[str],
    call_allow: Sequence[str] | None,
    settings_deny: Sequence[str],
    call_deny: Sequence[str] | None,
    morphology: Morphology | None = None,
) -> tuple[AllowList, DenyList]:
    """Валидирует и собирает списки: настройки плюс список вызова.

    Список вызова только дополняет настроечный и не может ничего из него
    убрать.

    Raises:
        InvalidListEntryError: Если любая запись невалидна.
    """
    validate_entries(settings_allow, 'allow_list(settings)')
    validate_entries(settings_deny, 'deny_list(settings)')
    validate_entries(call_allow or (), 'allow_list')
    validate_entries(call_deny or (), 'deny_list')
    morph = morphology or get_morphology()
    allow = AllowList([*settings_allow, *(call_allow or ())], morph)
    deny = DenyList([*settings_deny, *(call_deny or ())], morph)
    return allow, deny
