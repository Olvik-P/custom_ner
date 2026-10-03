"""CLI оценки качества детекции: run / show / compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from privacyguard_pipeline.evaluation.constants import (
    DATASET_CONFIGS,
    DEFAULT_CONFIG,
    RESULTS_DIR,
)
from privacyguard_pipeline.evaluation.dataset import (
    EvaluationDependencyError,
    load_examples,
)
from privacyguard_pipeline.evaluation.report import (
    IncomparableSnapshotsError,
    compare_snapshots,
    format_comparison,
    format_snapshot,
)
from privacyguard_pipeline.evaluation.runner import run_detector
from privacyguard_pipeline.evaluation.snapshot import (
    build_snapshot,
    read_snapshot,
    write_snapshot,
)

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_SHORT_COMMIT_LENGTH = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='python -m privacyguard_pipeline.evaluation',
        description='Оценка качества детекции PII на pii-bench.',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='прогнать детектор и сохранить снимок')
    run.add_argument(
        '--config',
        choices=DATASET_CONFIGS,
        default=DEFAULT_CONFIG,
        help='конфигурация датасета (по умолчанию domain)',
    )
    run.add_argument(
        '--min-confidence',
        type=float,
        default=None,
        help='порог уверенности (по умолчанию из настроек)',
    )
    run.add_argument(
        '--output',
        type=Path,
        default=None,
        help='куда записать снимок (по умолчанию в .eval_cache/results/)',
    )
    run.add_argument(
        '--dump-errors',
        type=Path,
        default=None,
        help='записать промахи (id, тип, оффсеты; без текстов) в JSON',
    )

    show = sub.add_parser('show', help='вывести таблицу одного снимка')
    show.add_argument('snapshot', type=Path)

    compare = sub.add_parser('compare', help='сравнить два снимка')
    compare.add_argument('before', type=Path)
    compare.add_argument('after', type=Path)
    return parser


def _default_output(snapshot: dict[str, object], config: str) -> Path:
    meta = snapshot['meta']
    assert isinstance(meta, dict)
    commit = str(meta.get('git_commit') or 'nogit')[:_SHORT_COMMIT_LENGTH]
    return RESULTS_DIR / f'{meta["package_version"]}-{commit}-{config}.json'


def _cmd_run(args: argparse.Namespace) -> int:
    examples = load_examples(args.config)
    result = run_detector(examples, min_confidence=args.min_confidence)
    if not result.natasha_available:
        print(
            'WARNING: Natasha недоступна — цифры по PER/LOC несопоставимы '
            'с прогонами при доступной Natasha.',
            file=sys.stderr,
        )
    snapshot = build_snapshot(
        result,
        config=args.config,
        example_count=len(examples),
        repo_dir=_PACKAGE_DIR,
    )
    output = args.output or _default_output(snapshot, args.config)
    write_snapshot(snapshot, output)
    if args.dump_errors:
        args.dump_errors.parent.mkdir(parents=True, exist_ok=True)
        args.dump_errors.write_text(
            json.dumps(result.errors, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
    print(format_snapshot(snapshot))
    print(f'\nснимок записан: {output}')
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI; возвращает код завершения."""
    args = _build_parser().parse_args(argv)
    try:
        if args.command == 'run':
            return _cmd_run(args)
        if args.command == 'show':
            print(format_snapshot(read_snapshot(args.snapshot)))
            return 0
        comparison = compare_snapshots(
            read_snapshot(args.before),
            read_snapshot(args.after),
        )
        print(format_comparison(comparison))
        return 0
    except EvaluationDependencyError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    except IncomparableSnapshotsError as exc:
        print(f'ERROR: снимки несравнимы: {exc}', file=sys.stderr)
        return 2
