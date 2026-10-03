"""Тесты метрик оценки на встроенных мини-фикстурах (без сети)."""

from __future__ import annotations

import pytest

from privacyguard_pipeline.evaluation.dataset import Example, GoldEntity
from privacyguard_pipeline.evaluation.metrics import (
    PredSpan,
    evaluate,
    is_supported,
    map_type,
    overlap_with_union,
    rates,
    union_intervals,
)


def _example(
    entities: list[tuple[int, int, str]],
    text: str = 'x' * 100,
    domain: str = 'D',
    id_: str = 'e1',
) -> Example:
    return Example(
        id=id_,
        domain=domain,
        text=text,
        entities=tuple(GoldEntity(s, e, t) for s, e, t in entities),
    )


class TestTypeMapping:
    @pytest.mark.parametrize(
        ('benchmark', 'ours'),
        [
            ('NAME', 'PER'),
            ('ADDRESS', 'LOC'),
            ('PHONE_NUMBER', 'PHONE'),
            ('BANK_CARD_NUMBER', 'CARD'),
            ('PASSPORT_NUMBER', 'PASSPORT'),
            ('EMAIL', 'EMAIL'),
            ('INN', 'INN'),
            ('SNILS', 'SNILS'),
            ('OGRN', 'OGRN'),
            ('KPP', 'KPP'),
            ('OGRNIP', 'OGRNIP'),
        ],
    )
    def test_supported(self, benchmark: str, ours: str) -> None:
        assert map_type(benchmark) == ours
        assert is_supported(benchmark)

    @pytest.mark.parametrize('benchmark', ['CVC', 'TOKEN'])
    def test_unsupported(self, benchmark: str) -> None:
        assert map_type(benchmark) is None
        assert not is_supported(benchmark)


class TestIntervals:
    def test_union_merges_overlapping_nested_and_touching(self) -> None:
        got = union_intervals([(0, 5), (3, 8), (10, 12), (11, 11), (8, 9)])
        assert got == [(0, 9), (10, 12)]

    def test_union_ignores_empty(self) -> None:
        assert union_intervals([(5, 5)]) == []

    def test_overlap_with_union(self) -> None:
        union = [(0, 5), (10, 20)]
        assert overlap_with_union(3, 12, union) == 4
        assert overlap_with_union(5, 10, union) == 0
        assert overlap_with_union(0, 100, union) == 15


class TestLeakRecall:
    def test_leak_recall_is_type_agnostic(self) -> None:
        example = _example([(10, 20, 'ADDRESS')])
        metrics = evaluate([example], [[PredSpan(10, 20, 'PER')]])
        r = rates(metrics.overall)
        assert r['leak_recall'] == 1.0
        assert r['type_accuracy'] == 0.0
        assert r['strict_recall'] == 0.0
        assert r['relaxed_recall'] == 1.0

    def test_partial_coverage(self) -> None:
        example = _example([(10, 20, 'NAME')])
        metrics = evaluate([example], [[PredSpan(10, 15, 'PER')]])
        r = rates(metrics.overall)
        assert r['leak_recall'] == 0.5
        assert r['relaxed_recall'] == 1.0
        assert r['strict_recall'] == 0.0

    def test_exact_match_is_strict_hit(self) -> None:
        example = _example([(10, 20, 'PHONE_NUMBER')])
        metrics = evaluate([example], [[PredSpan(10, 20, 'PHONE')]])
        r = rates(metrics.overall)
        assert r['strict_recall'] == 1.0
        assert r['strict_precision'] == 1.0
        assert r['type_accuracy'] == 1.0

    def test_miss(self) -> None:
        example = _example([(10, 20, 'EMAIL')])
        metrics = evaluate([example], [[]])
        r = rates(metrics.overall)
        assert r['leak_recall'] == 0.0
        assert r['relaxed_recall'] == 0.0
        assert r['strict_precision'] is None

    def test_union_prevents_double_counting_overlapping_spans(self) -> None:
        example = _example([(0, 10, 'NAME')])
        spans = [PredSpan(0, 6, 'PER'), PredSpan(4, 10, 'PER')]
        metrics = evaluate([example], [spans])
        assert metrics.overall.covered_chars == 10


class TestUnsupportedTypes:
    def test_counts_in_leak_recall_but_not_type_accuracy(self) -> None:
        example = _example([(10, 19, 'CVC')])
        metrics = evaluate([example], [[PredSpan(10, 19, 'INN')]])
        assert metrics.overall.covered_chars == 9
        assert metrics.overall.type_evaluated == 0
        assert rates(metrics.overall)['type_accuracy'] is None
        assert metrics.by_group['unsupported'].gold_entities == 1
        assert 'supported' not in metrics.by_group

    def test_never_strict_hit(self) -> None:
        example = _example([(10, 19, 'CVC')])
        metrics = evaluate([example], [[PredSpan(10, 19, 'CVC')]])
        assert metrics.overall.strict_hits == 0

    def test_kpp_and_ogrnip_are_supported_now(self) -> None:
        example = _example([(10, 19, 'KPP')])
        metrics = evaluate([example], [[PredSpan(10, 19, 'KPP')]])
        assert metrics.overall.strict_hits == 1
        assert metrics.overall.type_correct == 1
        assert 'unsupported' not in metrics.by_group


class TestFalsePositives:
    def test_spans_on_clean_example(self) -> None:
        clean = _example([], id_='c1')
        metrics = evaluate([clean], [[PredSpan(5, 10, 'PER')]])
        o = metrics.overall
        assert o.clean_examples == 1
        assert o.clean_examples_with_fp == 1
        assert o.fp_spans_on_clean == 1
        assert rates(o)['fp_char_rate'] == 1.0

    def test_clean_example_without_spans(self) -> None:
        clean = _example([], id_='c1')
        metrics = evaluate([clean], [[]])
        assert metrics.overall.clean_examples == 1
        assert metrics.overall.clean_examples_with_fp == 0

    def test_fp_chars_outside_gold(self) -> None:
        example = _example([(10, 20, 'NAME')])
        metrics = evaluate([example], [[PredSpan(5, 20, 'PER')]])
        assert metrics.overall.fp_chars == 5
        assert metrics.overall.pred_chars == 15
        assert metrics.overall.relaxed_pred_hits == 1
        assert metrics.overall.strict_pred_hits == 0


class TestGrouping:
    def test_groups_by_type_domain_and_pred_type(self) -> None:
        a = _example([(0, 5, 'NAME')], domain='A', id_='a')
        b = _example([(0, 5, 'EMAIL')], domain='B', id_='b')
        metrics = evaluate(
            [a, b],
            [[PredSpan(0, 5, 'PER')], []],
        )
        assert set(metrics.by_type) == {'NAME', 'EMAIL'}
        assert set(metrics.by_domain) == {'A', 'B'}
        assert set(metrics.by_pred_type) == {'PER'}
        assert metrics.by_domain['A'].covered_chars == 5
        assert metrics.by_domain['B'].covered_chars == 0
        assert metrics.overall.examples == 2

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match='same length'):
            evaluate([_example([])], [])

    def test_to_dict_contains_rates(self) -> None:
        example = _example([(0, 5, 'NAME')])
        data = evaluate([example], [[PredSpan(0, 5, 'PER')]]).to_dict()
        overall = data['overall']
        assert isinstance(overall, dict)
        assert overall['rates']['leak_recall'] == 1.0
