"""Allow/deny-списки: нормализация, сравнение, интеграция в PIIDetector."""

from __future__ import annotations

import logging

import pytest

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.detection import PIIDetector, lists
from privacyguard_pipeline.detection.lists import (
    AllowList,
    DenyList,
    compile_lists,
    normalize_for_allow,
)
from privacyguard_pipeline.detection.morphology import (
    Morphology,
    get_morphology,
)
from privacyguard_pipeline.exceptions import InvalidListEntryError
from privacyguard_pipeline.masker import Masker


def _unavailable() -> Morphology:
    def boom() -> object:
        raise RuntimeError('no dictionary')

    return Morphology(analyzer_factory=boom)


def _spans(
    detector: PIIDetector,
    text: str,
    **kwargs: object,
) -> list[tuple[str, str]]:
    result = detector.detect(text, **kwargs)  # type: ignore[arg-type]
    return [(s.text, s.entity_type) for s in result.spans]


class TestAllowNormalization:
    def test_case_yo_quotes_and_legal_form_ignored(self) -> None:
        morph = get_morphology()
        a = normalize_for_allow('ООО «Ромашка»', morph)
        b = normalize_for_allow('ромашка', morph)
        assert a == b == ('ромашка',)

    def test_grammatical_form_matches(self) -> None:
        morph = get_morphology()
        assert normalize_for_allow('ООО Ромашки', morph) == (
            normalize_for_allow('Ромашка', morph)
        )

    def test_digits_only_for_long_numbers(self) -> None:
        morph = get_morphology()
        assert normalize_for_allow('+7 (916) 123-45-67', morph) == (
            normalize_for_allow('79161234567', morph)
        )

    def test_short_numbers_stay_literal(self) -> None:
        morph = get_morphology()
        assert normalize_for_allow('12-34', morph) != normalize_for_allow(
            '1234',
            morph,
        )

    def test_lone_legal_form_is_kept(self) -> None:
        assert normalize_for_allow('ООО', get_morphology()) == ('ооо',)


class TestAllowViaDetector:
    def test_company_name_in_other_form_with_legal_form(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Договор заключён с ООО «Ромашка» вчера'
        assert _spans(detector, text, allow_list=['Ромашка']) == []

    def test_company_name_different_case_form(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Письмо в ООО Ромашки'
        assert _spans(detector, text) != []
        assert _spans(detector, text, allow_list=['Ромашка']) == []

    def test_own_email_any_case(self, detector: PIIDetector) -> None:
        text = 'Пишите на SUPPORT@CORP.EXAMPLE по любым вопросам'
        assert _spans(detector, text) != []
        got = _spans(detector, text, allow_list=['support@corp.example'])
        assert got == []

    def test_partial_match_does_not_suppress(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Звонил Ромашкин Пётр вчера'
        got = _spans(detector, text, allow_list=['Ромашка'])
        assert ('Ромашкин Пётр', 'PER') in got


class TestDenyViaDetector:
    def test_project_name_in_any_case_form(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'Внутри проекта Заря и ПРОЕКТ ЗАРЯ идут работы'
        got = _spans(detector, text, deny_list=['Проект Заря'])
        assert got == [('проекта Заря', 'CUSTOM'), ('ПРОЕКТ ЗАРЯ', 'CUSTOM')]

    def test_does_not_match_inside_longer_word(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _spans(
            detector, 'Нужна зарядка для телефона', deny_list=['Заря']
        )
        assert got == []

    def test_overrides_confidence_threshold(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'внутренний код 1234567890 в системе'
        assert _spans(detector, text) == []
        got = _spans(detector, text, deny_list=['1234567890'])
        assert got == [('1234567890', 'CUSTOM')]

    def test_overlap_with_person_keeps_person_type_and_union(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _spans(detector, 'Иван Иванов пришёл', deny_list=['Иванов'])
        assert got == [('Иван Иванов', 'PER')]

    def test_overlap_extends_coverage_to_union(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'звонил Иван Иванов Петрович'
        got = _spans(detector, text, deny_list=['Иванов Петрович'])
        assert len(got) == 1
        assert got[0][0].startswith('Иван Иванов')
        assert got[0][0].endswith('Петрович')
        assert got[0][1] == 'PER'

    def test_other_word_forms_of_entry_word_do_not_overmatch(
        self,
        detector: PIIDetector,
    ) -> None:
        text = 'видели Ивана рядом'
        got = _spans(detector, text, deny_list=['Иванов'])
        assert all(t != 'Ивана' or e != 'CUSTOM' for t, e in got)

    def test_masked_text_hides_deny_match(self, detector: PIIDetector) -> None:
        text = 'Внутри проекта Заря и ПРОЕКТ ЗАРЯ идут работы'
        result = detector.detect(text, deny_list=['Проект Заря'])
        masked = Masker().mask(text, result.spans)
        assert 'Заря' not in masked
        assert 'ЗАРЯ' not in masked
        assert '<CUSTOM_' in masked


class TestCombination:
    def test_call_list_extends_settings_list(
        self,
        detector: PIIDetector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, 'allow_list', ['support@corp.example'])
        text = 'пишите на support@corp.example и на info@corp.example'
        got = _spans(detector, text, allow_list=['info@corp.example'])
        assert got == []
        # Список вызова не влияет на следующий вызов.
        later = _spans(detector, text)
        assert later == [('info@corp.example', 'EMAIL')]

    def test_call_cannot_remove_configured_deny(
        self,
        detector: PIIDetector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, 'deny_list', ['Проект Заря'])
        got = _spans(detector, 'идёт проект Заря', deny_list=[])
        # Natasha сама помечает «Заря»; deny расширяет охват до «проект Заря»,
        # а тип существующего спана сохраняется.
        assert [t for t, _ in got] == ['проект Заря']

    def test_entry_in_both_lists_is_masked(
        self,
        detector: PIIDetector,
    ) -> None:
        got = _spans(
            detector,
            'внутренний код 1234567890 в системе',
            allow_list=['1234567890'],
            deny_list=['1234567890'],
        )
        assert got == [('1234567890', 'CUSTOM')]

    @pytest.mark.parametrize(
        ('entries', 'reason'),
        [
            ([''], 'empty'),
            (['   '], 'empty'),
            (['«»'], 'empty'),
            (['а' * 201], 'longer'),
            (['x'] * 501, 'more than'),
            ([123], 'not a string'),
        ],
    )
    def test_invalid_entries_rejected_before_any_work(
        self,
        detector: PIIDetector,
        entries: list[object],
        reason: str,
    ) -> None:
        for kwargs in ({'allow_list': entries}, {'deny_list': entries}):
            with pytest.raises(InvalidListEntryError, match=reason):
                detector.detect('текст', **kwargs)  # type: ignore[arg-type]

    def test_error_message_does_not_contain_entry_text(self) -> None:
        secret = 'секрет' * 40
        with pytest.raises(InvalidListEntryError) as info:
            compile_lists([], None, [], [secret])
        assert 'секрет' not in str(info.value)
        assert 'deny_list[0]' in str(info.value)

    def test_list_content_stays_out_of_logs(
        self,
        detector: PIIDetector,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.DEBUG):
            detector.detect(
                'идёт проект Заря и писать на support@corp.example',
                allow_list=['support@corp.example'],
                deny_list=['Проект Заря'],
            )
        joined = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'Заря' not in joined
        assert 'support@corp.example' not in joined
        assert 'allow entries' in joined


class TestWithoutMorphology:
    def test_deny_matches_literal_text_only(self) -> None:
        deny = DenyList(['Проект Заря'], morphology=_unavailable())
        literal = deny.find_spans('Идёт ПРОЕКТ заря сегодня')
        assert [s.text for s in literal] == ['ПРОЕКТ заря']
        assert deny.find_spans('Идёт проекта Заря') == []

    def test_allow_matches_literal_text_only(self) -> None:
        allow = AllowList(['Ромашка'], morphology=_unavailable())
        assert allow.matches('ромашка')
        assert not allow.matches('Ромашки')

    def test_detection_completes_and_returns_other_spans(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        broken = _unavailable()
        monkeypatch.setattr(lists, 'get_morphology', lambda: broken)
        detector = PIIDetector()
        detector.contextual_validator = type(
            detector.contextual_validator,
        )(morphology=broken)
        got = _spans(
            detector,
            'Проект Заря: почта ivan@example.com',
            deny_list=['Проект Заря'],
        )
        assert ('Проект Заря', 'CUSTOM') in got
        assert ('ivan@example.com', 'EMAIL') in got


class TestRepr:
    def test_deny_list_does_not_expose_entries(self) -> None:
        deny = DenyList(['Проект Заря'])
        assert 'Заря' not in repr(deny)
