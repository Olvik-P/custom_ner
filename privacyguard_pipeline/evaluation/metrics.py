"""Метрики качества детекции PII по интервалам символов.

Главная метрика для анонимайзера — recall утечки: доля символов золотой
PII-разметки, покрытых хотя бы одним найденным спаном независимо от его
типа. Ошибка типа не делает значение утечкой, а вот пропуск делает.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Iterable, Sequence

from privacyguard_pipeline.evaluation.constants import (
    TYPE_MAP,
    UNSUPPORTED_TYPES,
)
from privacyguard_pipeline.evaluation.dataset import Example, GoldEntity

Interval = tuple[int, int]


def map_type(benchmark_type: str) -> str | None:
    """Тип бенчмарка -> тип пакета; ``None`` для неподдерживаемых типов."""
    return TYPE_MAP.get(benchmark_type)


def is_supported(benchmark_type: str) -> bool:
    """Есть ли у типа бенчмарка аналог в пакете."""
    return benchmark_type not in UNSUPPORTED_TYPES


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Сливает пересекающиеся и соприкасающиеся интервалы."""
    merged: list[Interval] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap_with_union(start: int, end: int, union: Sequence[Interval]) -> int:
    """Число символов отрезка [start, end), лежащих в объединении."""
    total = 0
    for u_start, u_end in union:
        if u_start >= end:
            break
        total += max(0, min(end, u_end) - max(start, u_start))
    return total


@dataclass
class Counts:
    """Аддитивные счётчики одной группы; доли считаются из них."""

    examples: int = 0
    # Золотая сторона (recall).
    gold_entities: int = 0
    gold_chars: int = 0
    covered_chars: int = 0
    strict_hits: int = 0
    relaxed_hits: int = 0
    type_evaluated: int = 0
    type_correct: int = 0
    # Сторона детектора (precision).
    pred_spans: int = 0
    pred_chars: int = 0
    fp_chars: int = 0
    strict_pred_hits: int = 0
    relaxed_pred_hits: int = 0
    # Примеры без PII.
    clean_examples: int = 0
    clean_examples_with_fp: int = 0
    fp_spans_on_clean: int = 0


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def rates(counts: Counts) -> dict[str, float | None]:
    """Производные доли группы; ``None``, если знаменатель нулевой."""
    return {
        'leak_recall': _ratio(counts.covered_chars, counts.gold_chars),
        'strict_recall': _ratio(counts.strict_hits, counts.gold_entities),
        'relaxed_recall': _ratio(counts.relaxed_hits, counts.gold_entities),
        'type_accuracy': _ratio(counts.type_correct, counts.type_evaluated),
        'strict_precision': _ratio(counts.strict_pred_hits, counts.pred_spans),
        'relaxed_precision': _ratio(
            counts.relaxed_pred_hits,
            counts.pred_spans,
        ),
        'fp_char_rate': _ratio(counts.fp_chars, counts.pred_chars),
    }


@dataclass(frozen=True)
class PredSpan:
    """Спан детектора в минимальном виде, достаточном для метрик."""

    start: int
    end: int
    entity_type: str


@dataclass
class Metrics:
    """Счётчики по группам одного прогона."""

    overall: Counts
    by_type: dict[str, Counts]
    by_group: dict[str, Counts]
    by_pred_type: dict[str, Counts]
    by_domain: dict[str, Counts]

    def to_dict(self) -> dict[str, object]:
        """Сериализует счётчики вместе с производными долями."""

        def pack(counts: Counts) -> dict[str, object]:
            return {**asdict(counts), 'rates': rates(counts)}

        return {
            'overall': pack(self.overall),
            'by_type': {k: pack(v) for k, v in sorted(self.by_type.items())},
            'by_group': {k: pack(v) for k, v in sorted(self.by_group.items())},
            'by_pred_type': {
                k: pack(v) for k, v in sorted(self.by_pred_type.items())
            },
            'by_domain': {
                k: pack(v) for k, v in sorted(self.by_domain.items())
            },
        }


def _add(target: Counts, source: Counts) -> None:
    for f in fields(Counts):
        setattr(
            target,
            f.name,
            getattr(target, f.name) + getattr(source, f.name),
        )


def _gold_side(
    entity: GoldEntity,
    spans: Sequence[PredSpan],
    pred_union: Sequence[Interval],
) -> Counts:
    c = Counts(gold_entities=1)
    length = entity.end - entity.start
    c.gold_chars = length
    c.covered_chars = overlap_with_union(entity.start, entity.end, pred_union)
    expected = map_type(entity.type)
    best: PredSpan | None = None
    best_overlap = 0
    for span in spans:
        overlap = min(entity.end, span.end) - max(entity.start, span.start)
        if overlap <= 0:
            continue
        if overlap > best_overlap:
            best, best_overlap = span, overlap
        if (
            span.start == entity.start
            and span.end == entity.end
            and span.entity_type == expected
        ):
            c.strict_hits = 1
    if best is not None:
        c.relaxed_hits = 1
        if expected is not None:
            c.type_evaluated = 1
            c.type_correct = int(best.entity_type == expected)
    return c


def _pred_side(
    span: PredSpan,
    entities: Sequence[GoldEntity],
    gold_union: Sequence[Interval],
) -> Counts:
    length = span.end - span.start
    c = Counts(pred_spans=1, pred_chars=length)
    c.fp_chars = length - overlap_with_union(span.start, span.end, gold_union)
    for entity in entities:
        if min(entity.end, span.end) - max(entity.start, span.start) > 0:
            c.relaxed_pred_hits = 1
            if (
                entity.start == span.start
                and entity.end == span.end
                and map_type(entity.type) == span.entity_type
            ):
                c.strict_pred_hits = 1
                break
    return c


def evaluate(
    examples: Sequence[Example],
    predictions: Sequence[Sequence[PredSpan]],
) -> Metrics:
    """Считает метрики по примерам и спанам детектора.

    Args:
        examples: Размеченные примеры.
        predictions: Спаны детектора, по списку на каждый пример
            (в том же порядке).

    Returns:
        Счётчики в целом и по группам (тип бенчмарка, поддерживаемость,
        тип детектора, сценарий).
    """
    if len(examples) != len(predictions):
        msg = 'examples and predictions must have the same length'
        raise ValueError(msg)
    metrics = Metrics(Counts(), {}, {}, {}, {})
    for example, spans in zip(examples, predictions):
        base = Counts(examples=1)
        pred_union = union_intervals((s.start, s.end) for s in spans)
        gold_union = union_intervals(
            (e.start, e.end) for e in example.entities
        )
        domain = metrics.by_domain.setdefault(example.domain, Counts())
        _add(metrics.overall, base)
        _add(domain, base)
        if not example.entities:
            clean = Counts(
                clean_examples=1,
                clean_examples_with_fp=int(bool(spans)),
                fp_spans_on_clean=len(spans),
            )
            _add(metrics.overall, clean)
            _add(domain, clean)
        for entity in example.entities:
            gold = _gold_side(entity, spans, pred_union)
            group = 'supported' if is_supported(entity.type) else 'unsupported'
            for target in (
                metrics.overall,
                domain,
                metrics.by_type.setdefault(entity.type, Counts()),
                metrics.by_group.setdefault(group, Counts()),
            ):
                _add(target, gold)
        for span in spans:
            pred = _pred_side(span, example.entities, gold_union)
            for target in (
                metrics.overall,
                domain,
                metrics.by_pred_type.setdefault(span.entity_type, Counts()),
            ):
                _add(target, pred)
    return metrics
