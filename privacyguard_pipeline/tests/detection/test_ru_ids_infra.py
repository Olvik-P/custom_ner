"""Каркас новых идентификаторов: окно контекста на тип и тай-брейки."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.constants import (
    OMS_PREFER_CONFIDENCE,
    PATTERN_CONFIDENCE,
)
from privacyguard_pipeline.detection import scoring
from privacyguard_pipeline.detection.common import PIISpan
from privacyguard_pipeline.detection.pattern_matcher import (
    PatternMatcher,
    _prefer_pattern_span,
)
from privacyguard_pipeline.detection.scoring import (
    SCORING_REGISTRY,
    SCORING_WINDOWS,
    ScoringWindow,
    _context_window,
    score_match,
)


def _span(
    entity_type: str,
    confidence: float,
    start: int = 0,
    end: int = 10,
) -> PIISpan:
    return PIISpan(
        start=start,
        end=end,
        text='x' * (end - start),
        entity_type=entity_type,
        source='pattern',
        confidence=confidence,
    )


class TestScoringWindows:
    TEXT = 'БИК 044525225\nр/с 40702810200000012345'

    def _bounds(self) -> tuple[int, int]:
        start = self.TEXT.index('4070')
        return start, start + 20

    def test_default_window_stops_at_line_boundary(self) -> None:
        start, end = self._bounds()
        before, _ = _context_window(self.TEXT, start, end)
        assert before == 'р/с '

    def test_cross_lines_window_sees_previous_line(self) -> None:
        start, end = self._bounds()
        before, _ = _context_window(
            self.TEXT,
            start,
            end,
            ScoringWindow(250, 250, cross_lines=True),
        )
        assert 'БИК 044525225' in before

    def test_score_match_uses_per_type_window(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, str] = {}

        def scorer(raw: str, before: str, after: str) -> float:
            seen['before'] = before
            return 0.5

        monkeypatch.setitem(SCORING_REGISTRY, 'DUMMY', scorer)
        monkeypatch.setitem(
            SCORING_WINDOWS,
            'DUMMY',
            ScoringWindow(250, 250, cross_lines=True),
        )
        start, end = self._bounds()

        score_match('DUMMY', 'x', self.TEXT, start, end)

        assert 'БИК' in seen['before']

    def test_types_without_entry_keep_the_default_window(self) -> None:
        text = 'заказ №\nпаспорт 45 10 123456'
        start = text.index('45 10')
        # Слово «заказ» на предыдущей строке не должно снижать оценку.
        with_order = score_match(
            'PASSPORT', '45 10 123456', text, start, start + 12
        )
        plain = score_match(
            'PASSPORT',
            '45 10 123456',
            'паспорт 45 10 123456',
            len('паспорт '),
            len('паспорт ') + 12,
        )
        assert with_order == plain

    def test_registry_windows_do_not_leak_into_scoring_module_default(
        self,
    ) -> None:
        assert scoring._DEFAULT_WINDOW.cross_lines is False


class TestExactSpanTieRules:
    def test_inn_beats_passport_and_phone(self) -> None:
        for other in ('PASSPORT', 'PHONE'):
            assert (
                _prefer_pattern_span(
                    _span('INN', 0.9), _span(other, 0.9)
                ).entity_type
                == 'INN'
            )
            assert (
                _prefer_pattern_span(
                    _span(other, 0.9), _span('INN', 0.9)
                ).entity_type
                == 'INN'
            )

    def test_driver_licence_beats_passport_only_with_higher_confidence(
        self,
    ) -> None:
        win = _prefer_pattern_span(
            _span('PASSPORT', 0.55),
            _span('DRIVER_LICENSE', 0.85),
        )
        assert win.entity_type == 'DRIVER_LICENSE'
        win = _prefer_pattern_span(
            _span('DRIVER_LICENSE', 0.45),
            _span('PASSPORT', 0.55),
        )
        assert win.entity_type == 'PASSPORT'

    def test_equal_confidence_keeps_passport(self) -> None:
        for kept, incoming in (
            (_span('PASSPORT', 0.5), _span('DRIVER_LICENSE', 0.5)),
            (_span('DRIVER_LICENSE', 0.5), _span('PASSPORT', 0.5)),
        ):
            assert (
                _prefer_pattern_span(kept, incoming).entity_type == 'PASSPORT'
            )

    def test_oms_beats_card_only_with_policy_evidence(self) -> None:
        strong = _span('OMS', OMS_PREFER_CONFIDENCE)
        weak = _span('OMS', OMS_PREFER_CONFIDENCE - 0.01)
        card = _span('CARD', PATTERN_CONFIDENCE)
        assert _prefer_pattern_span(card, strong).entity_type == 'OMS'
        assert _prefer_pattern_span(strong, card).entity_type == 'OMS'
        assert _prefer_pattern_span(card, weak).entity_type == 'CARD'
        assert _prefer_pattern_span(weak, card).entity_type == 'CARD'

    def test_telegram_beats_url_on_same_range(self) -> None:
        win = _prefer_pattern_span(_span('URL', 0.95), _span('TELEGRAM', 0.8))
        assert win.entity_type == 'TELEGRAM'

    def test_other_pairs_still_prefer_the_longer_span(self) -> None:
        shorter = _span('PHONE', 0.95, 0, 10)
        longer = _span('BANK_ACCOUNT', 0.7, 0, 20)
        assert _prefer_pattern_span(shorter, longer).entity_type == (
            'BANK_ACCOUNT'
        )

    def test_rules_apply_only_to_identical_ranges(self) -> None:
        win = _prefer_pattern_span(
            _span('INN', 0.9, 0, 10),
            _span('PASSPORT', 0.5, 0, 12),
        )
        assert win.entity_type == 'PASSPORT'


def test_merge_overlapping_uses_the_tie_rules() -> None:
    text = 'x' * 20
    merged = PatternMatcher.merge_overlapping(
        [_span('PASSPORT', 0.55, 0, 10), _span('INN', 0.95, 0, 10)],
        text,
    )
    assert [s.entity_type for s in merged] == ['INN']
