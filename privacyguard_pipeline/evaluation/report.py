"""Вывод и сравнение снимков результатов оценки."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Метрика -> (поле числителя, поле знаменателя, больше = лучше).
METRIC_DEFS: dict[str, tuple[str, str, bool]] = {
    'leak_recall': ('covered_chars', 'gold_chars', True),
    'strict_recall': ('strict_hits', 'gold_entities', True),
    'relaxed_recall': ('relaxed_hits', 'gold_entities', True),
    'type_accuracy': ('type_correct', 'type_evaluated', True),
    'strict_precision': ('strict_pred_hits', 'pred_spans', True),
    'relaxed_precision': ('relaxed_pred_hits', 'pred_spans', True),
    'fp_char_rate': ('fp_chars', 'pred_chars', False),
}

# Секция снимка -> метрики, осмысленные для её группировки.
SECTION_METRICS: dict[str, tuple[str, ...]] = {
    'overall': tuple(METRIC_DEFS),
    'by_group': ('leak_recall', 'relaxed_recall'),
    'by_type': ('leak_recall', 'strict_recall', 'type_accuracy'),
    'by_pred_type': ('strict_precision', 'fp_char_rate'),
    'by_domain': ('leak_recall', 'fp_char_rate'),
}

# Ненулевая дельта меньше этой считается шумом округления.
_EPSILON = 1e-9


class IncomparableSnapshotsError(ValueError):
    """Снимки получены на разных данных и несравнимы."""


@dataclass(frozen=True)
class Row:
    """Строка сравнения: одна метрика одной группы в двух снимках."""

    section: str
    key: str
    metric: str
    before: float | None
    after: float | None
    detail_before: str
    detail_after: str

    @property
    def delta(self) -> float | None:
        """Разница ``after - before``; ``None``, если нет значения."""
        if self.before is None or self.after is None:
            return None
        return self.after - self.before

    @property
    def verdict(self) -> str:
        """``worse``, ``better`` или пустая строка при отсутствии сдвига."""
        delta = self.delta
        if delta is None or abs(delta) < _EPSILON:
            return ''
        higher_is_better = METRIC_DEFS[self.metric][2]
        return 'better' if (delta > 0) == higher_is_better else 'worse'


@dataclass
class Comparison:
    """Результат сравнения двух снимков."""

    warnings: list[str] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)


def _detail(group: dict[str, Any], metric: str) -> str:
    num, den, _ = METRIC_DEFS[metric]
    return f'{group[num]}/{group[den]}'


def _rate(group: dict[str, Any], metric: str) -> float | None:
    value = group['rates'][metric]
    return None if value is None else float(value)


def compare_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
) -> Comparison:
    """Сравнивает два снимка.

    Raises:
        IncomparableSnapshotsError: Если ревизия или конфигурация
            датасета различаются.
    """
    mb, ma = before['meta'], after['meta']
    for key in ('dataset_revision', 'dataset_config'):
        if mb[key] != ma[key]:
            msg = f'{key} differs: {mb[key]!r} vs {ma[key]!r}'
            raise IncomparableSnapshotsError(msg)
    comparison = Comparison()
    for key in ('min_confidence', 'natasha_available', 'example_count'):
        if mb[key] != ma[key]:
            comparison.warnings.append(
                f'{key} differs: {mb[key]!r} vs {ma[key]!r}'
            )
    for section, metrics in SECTION_METRICS.items():
        groups_b = _groups(before['metrics'], section)
        groups_a = _groups(after['metrics'], section)
        for key in sorted(set(groups_b) | set(groups_a)):
            gb, ga = groups_b.get(key), groups_a.get(key)
            for metric in metrics:
                row = Row(
                    section=section,
                    key=key,
                    metric=metric,
                    before=None if gb is None else _rate(gb, metric),
                    after=None if ga is None else _rate(ga, metric),
                    detail_before='' if gb is None else _detail(gb, metric),
                    detail_after='' if ga is None else _detail(ga, metric),
                )
                if row.before is None and row.after is None:
                    continue
                comparison.rows.append(row)
    return comparison


def _groups(metrics: dict[str, Any], section: str) -> dict[str, Any]:
    if section == 'overall':
        return {'all': metrics['overall']}
    groups: dict[str, Any] = metrics[section]
    return groups


def _pct(value: float | None) -> str:
    return '   n/a' if value is None else f'{value * 100:6.2f}%'


def format_snapshot(snapshot: dict[str, Any]) -> str:
    """Таблица метрик одного снимка."""
    meta = snapshot['meta']
    lines = [
        f'package {meta["package_version"]}  commit '
        f'{str(meta.get("git_commit"))[:10]}'
        f'{" (dirty)" if meta.get("git_dirty") else ""}',
        f'dataset {meta["dataset_repo"]}@{meta["dataset_revision"][:10]} '
        f'config={meta["dataset_config"]} n={meta["example_count"]}',
        f'min_confidence={meta["min_confidence"]} '
        f'natasha={meta["natasha_available"]}',
        '',
    ]
    for section, metrics in SECTION_METRICS.items():
        groups = _groups(snapshot['metrics'], section)
        lines.append(f'[{section}]')
        header = f'{"":18}' + ''.join(f'{m:>21}' for m in metrics)
        lines.append(header)
        for key, group in groups.items():
            cells = ''.join(
                f'{_pct(_rate(group, m))} {_detail(group, m):>11}  '
                for m in metrics
            )
            lines.append(f'{key:18}{cells}')
        lines.append('')
    overall = snapshot['metrics']['overall']
    lines.append(
        f'clean examples: {overall["clean_examples"]}, with false '
        f'positives: {overall["clean_examples_with_fp"]}, '
        f'fp spans: {overall["fp_spans_on_clean"]}'
    )
    return '\n'.join(lines)


def format_comparison(comparison: Comparison) -> str:
    """Таблица дельт; ухудшения помечены WORSE, улучшения — better."""
    lines = [f'WARNING: {w}' for w in comparison.warnings]
    if lines:
        lines.append('')
    current = None
    for row in comparison.rows:
        if row.section != current:
            current = row.section
            lines.append(f'[{current}]')
        delta = row.delta
        delta_text = '     n/a' if delta is None else f'{delta * 100:+7.2f}pp'
        mark = {'worse': ' WORSE', 'better': ' better'}.get(row.verdict, '')
        lines.append(
            f'  {row.key:18}{row.metric:18}'
            f'{_pct(row.before)} -> {_pct(row.after)} {delta_text}{mark}'
            f'   ({row.detail_before} -> {row.detail_after})'
        )
    worse = sum(1 for r in comparison.rows if r.verdict == 'worse')
    better = sum(1 for r in comparison.rows if r.verdict == 'better')
    lines.append('')
    lines.append(f'worse: {worse}, better: {better}')
    return '\n'.join(lines)
