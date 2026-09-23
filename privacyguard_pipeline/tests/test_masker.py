"""Тесты стабильности токенов Masker для повторяющихся сущностей."""

from __future__ import annotations

import re

from privacyguard_pipeline.detection import PIISpan
from privacyguard_pipeline.masker import Masker

_TOKEN_RE = re.compile(r'<[A-Z]+_[0-9A-F]{8}>')


def _span(text: str, start: int, entity_type: str) -> PIISpan:
    return PIISpan(
        start=start, end=start + len(text), text=text, entity_type=entity_type
    )


class TestRepeatedEntityReusesToken:
    def test_identical_org_mentions_share_one_token(self) -> None:
        text = 'Компания Ромашка работает с 2010 года. Ромашка - лидер рынка.'
        first_start = text.index('Ромашка')
        second_start = text.index('Ромашка', first_start + 1)
        spans = [
            _span('Ромашка', first_start, 'ORG'),
            _span('Ромашка', second_start, 'ORG'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]

        org_entries = [
            e for e in masker.mapping.values() if e.entity_type == 'ORG'
        ]
        assert len(org_entries) == 1

        assert masker.demask(masked) == text


class TestOrgLegalFormAndQuoteNormalization:
    def test_legal_form_and_plain_name_share_token_and_canonical_original(
        self,
    ) -> None:
        text = (
            'ООО «Ромашка» подписала договор. '
            'Ромашка выполнила обязательства в срок.'
        )
        first_text = 'ООО «Ромашка»'
        first_start = text.index(first_text)
        second_text = 'Ромашка'
        second_start = text.index(second_text, first_start + len(first_text))
        spans = [
            _span(first_text, first_start, 'ORG'),
            _span(second_text, second_start, 'ORG'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]

        # Канонический original - текст первого по тексту упоминания.
        expected_demasked = (
            text[:first_start]
            + first_text
            + text[first_start + len(first_text) : second_start]
            + first_text
            + text[second_start + len(second_text) :]
        )
        assert masker.demask(masked) == expected_demasked


class TestCaseAndWhitespaceVariantsShareToken:
    def test_per_span_differing_only_by_case_shares_token(self) -> None:
        text = 'Пациент Иванов, 30 лет. ИВАНОВ повторно обратился в клинику.'
        first_start = text.index('Иванов')
        second_start = text.index('ИВАНОВ')
        spans = [
            _span('Иванов', first_start, 'PER'),
            _span('ИВАНОВ', second_start, 'PER'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]

    def test_org_span_with_surrounding_punctuation_shares_token(
        self,
    ) -> None:
        text = 'Поставщик: Одуванчик. Ранее с «Одуванчик» уже работали.'
        first_start = text.index('Одуванчик')
        second_text = '«Одуванчик»'
        second_start = text.index(second_text)
        spans = [
            _span('Одуванчик', first_start, 'ORG'),
            _span(second_text, second_start, 'ORG'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 2
        assert tokens[0] == tokens[1]


class TestDistinctEntitiesGetDistinctTokens:
    def test_different_org_names_get_different_tokens(self) -> None:
        text = 'Ромашка и Одуванчик подписали совместный договор.'
        first_start = text.index('Ромашка')
        second_start = text.index('Одуванчик')
        spans = [
            _span('Ромашка', first_start, 'ORG'),
            _span('Одуванчик', second_start, 'ORG'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 2
        assert tokens[0] != tokens[1]
        assert masker.mapping_size == 2

    def test_address_components_stay_distinct(self) -> None:
        text = '620004, г. Екатеринбург, ул. Малышева, 101.'
        spans = [
            _span('620004', 0, 'LOC'),
            _span('Екатеринбург', text.index('Екатеринбург'), 'LOC'),
            _span('Малышева', text.index('Малышева'), 'LOC'),
            _span('101', text.index('101'), 'LOC'),
        ]

        masker = Masker()
        masked = masker.mask(text, spans)

        tokens = _TOKEN_RE.findall(masked)
        assert len(tokens) == 4
        assert len(set(tokens)) == 4


class TestClearResetsEntityReuseIndex:
    def test_clear_lets_next_document_mask_and_demask_correctly(
        self,
    ) -> None:
        masker = Masker()

        text1 = 'Ромашка подписала договор.'
        span1 = _span('Ромашка', text1.index('Ромашка'), 'ORG')
        masked1 = masker.mask(text1, [span1])
        assert masker.demask(masked1) == text1
        masker.clear()
        assert masker.mapping_size == 0

        text2 = 'Ромашка снова подписала договор в этом квартале.'
        span2 = _span('Ромашка', text2.index('Ромашка'), 'ORG')
        masked2 = masker.mask(text2, [span2])

        # Если бы индекс переиспользования токенов не очищался в clear(),
        # для этой сущности не создалась бы новая запись в mapping, и
        # demask() не смог бы восстановить токен во втором документе.
        assert masker.mapping_size == 1
        assert masker.demask(masked2) == text2
