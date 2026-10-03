"""Тесты снимков, сравнения, загрузки датасета и прогона (без сети)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from privacyguard_pipeline.detection.common import DetectionResult, PIISpan
from privacyguard_pipeline.evaluation import dataset
from privacyguard_pipeline.evaluation.constants import DATASET_REVISION
from privacyguard_pipeline.evaluation.dataset import Example, GoldEntity
from privacyguard_pipeline.evaluation.metrics import PredSpan, evaluate
from privacyguard_pipeline.evaluation.report import (
    IncomparableSnapshotsError,
    compare_snapshots,
    format_comparison,
    format_snapshot,
)
from privacyguard_pipeline.evaluation.runner import RunResult, run_detector
from privacyguard_pipeline.evaluation.snapshot import (
    build_snapshot,
    read_snapshot,
    write_snapshot,
)

_EXAMPLES = [
    Example('a', 'D', 'Иван Петров', (GoldEntity(0, 11, 'NAME'),)),
    Example('b', 'D', 'просто текст', ()),
]


def _snapshot(
    spans: list[list[PredSpan]],
    tmp_path: Path,
    min_confidence: float = 0.5,
) -> dict[str, Any]:
    result = RunResult(
        metrics=evaluate(_EXAMPLES, spans),
        min_confidence=min_confidence,
        natasha_available=True,
        errors=[],
    )
    return build_snapshot(result, 'domain', len(_EXAMPLES), tmp_path)


class TestSnapshot:
    def test_meta_and_roundtrip(self, tmp_path: Path) -> None:
        snap = _snapshot([[PredSpan(0, 11, 'PER')], []], tmp_path)
        meta = snap['meta']
        for key in (
            'package_version',
            'git_commit',
            'git_dirty',
            'dataset_revision',
            'dataset_config',
            'min_confidence',
            'natasha_available',
            'example_count',
        ):
            assert key in meta
        path = tmp_path / 'sub' / 'snap.json'
        write_snapshot(snap, path)
        assert read_snapshot(path) == snap

    def test_snapshot_has_no_example_text(self, tmp_path: Path) -> None:
        snap = _snapshot([[PredSpan(0, 11, 'PER')], []], tmp_path)
        dumped = json.dumps(snap, ensure_ascii=False)
        assert 'Иван' not in dumped
        assert 'просто текст' not in dumped


class TestCompare:
    def test_comparable_snapshots_show_deltas(self, tmp_path: Path) -> None:
        before = _snapshot([[], []], tmp_path)
        after = _snapshot([[PredSpan(0, 11, 'PER')], []], tmp_path)
        comparison = compare_snapshots(before, after)
        assert comparison.warnings == []
        leak = next(
            r
            for r in comparison.rows
            if r.section == 'overall' and r.metric == 'leak_recall'
        )
        assert leak.before == 0.0
        assert leak.after == 1.0
        assert leak.verdict == 'better'
        assert 'better' in format_comparison(comparison)

    def test_regression_is_marked_worse(self, tmp_path: Path) -> None:
        before = _snapshot([[PredSpan(0, 11, 'PER')], []], tmp_path)
        after = _snapshot(
            [[PredSpan(0, 11, 'PER')], [PredSpan(0, 5, 'PER')]],
            tmp_path,
        )
        comparison = compare_snapshots(before, after)
        fp = next(
            r
            for r in comparison.rows
            if r.section == 'overall' and r.metric == 'fp_char_rate'
        )
        assert fp.verdict == 'worse'
        assert 'WORSE' in format_comparison(comparison)

    @pytest.mark.parametrize('key', ['dataset_revision', 'dataset_config'])
    def test_different_dataset_refused(
        self,
        key: str,
        tmp_path: Path,
    ) -> None:
        before = _snapshot([[], []], tmp_path)
        after = copy.deepcopy(before)
        after['meta'][key] = 'other'
        with pytest.raises(IncomparableSnapshotsError, match=key):
            compare_snapshots(before, after)

    def test_different_threshold_warns(self, tmp_path: Path) -> None:
        before = _snapshot([[], []], tmp_path, min_confidence=0.5)
        after = _snapshot([[], []], tmp_path, min_confidence=0.7)
        comparison = compare_snapshots(before, after)
        assert any('min_confidence' in w for w in comparison.warnings)
        assert comparison.rows

    def test_format_snapshot_renders(self, tmp_path: Path) -> None:
        snap = _snapshot([[PredSpan(0, 11, 'PER')], []], tmp_path)
        text = format_snapshot(snap)
        assert 'leak_recall' in text
        assert 'config=domain' in text


class _FakeDetector:
    def __init__(self) -> None:
        self.natasha_ner = type('N', (), {'is_available': False})()
        self.seen: list[float | None] = []

    def detect(
        self,
        text: str,
        min_confidence: float | None = None,
    ) -> DetectionResult:
        self.seen.append(min_confidence)
        if 'Иван' in text:
            return DetectionResult(
                spans=[PIISpan(0, 11, text[:11], 'PER')],
            )
        return DetectionResult()


class _SilentDetector(_FakeDetector):
    def detect(
        self,
        text: str,
        min_confidence: float | None = None,
    ) -> DetectionResult:
        return DetectionResult()


class TestRunner:
    def test_run_with_fake_detector(self) -> None:
        detector = _FakeDetector()
        result = run_detector(
            _EXAMPLES,
            min_confidence=0.6,
            detector=detector,  # type: ignore[arg-type]
        )
        assert detector.seen == [0.6, 0.6]
        assert result.natasha_available is False
        assert result.min_confidence == 0.6
        assert result.metrics.overall.covered_chars == 11
        assert result.errors == []

    def test_errors_list_missed_entities_without_text(self) -> None:
        result = run_detector(
            _EXAMPLES,
            detector=_SilentDetector(),  # type: ignore[arg-type]
        )
        assert result.errors == [
            {'id': 'a', 'type': 'NAME', 'start': 0, 'end': 11}
        ]


class TestDataset:
    def test_unknown_config_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='unknown config'):
            dataset.ensure_downloaded('nope', tmp_path)

    def test_downloads_pinned_revision_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def fake_get(url: str, **kwargs: Any) -> httpx.Response:
            calls.append(url)
            return httpx.Response(
                200,
                content=b'PARQUET',
                request=httpx.Request('GET', url),
            )

        monkeypatch.setattr(httpx, 'get', fake_get)
        first = dataset.ensure_downloaded('domain', tmp_path)
        second = dataset.ensure_downloaded('domain', tmp_path)
        assert first == second
        assert first.read_bytes() == b'PARQUET'
        assert len(calls) == 1
        assert DATASET_REVISION in calls[0]
        assert not list(tmp_path.rglob('*.part'))

    def test_cache_hit_makes_no_network_call(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = dataset.cache_path('entity', tmp_path)
        target.parent.mkdir(parents=True)
        target.write_bytes(b'cached')

        def boom(*args: Any, **kwargs: Any) -> None:
            raise AssertionError('network must not be used')

        monkeypatch.setattr(httpx, 'get', boom)
        assert dataset.ensure_downloaded('entity', tmp_path) == target

    def test_rows_to_examples(self) -> None:
        rows: list[dict[str, Any]] = [
            {
                'id': 'x',
                'domain': 'S',
                'text': 'abc',
                'entities': [
                    {'start': 0, 'end': 3, 'type': 'NAME', 'text': 'abc'}
                ],
            },
            {'id': 'y', 'domain': 'S', 'text': 'q', 'entities': None},
        ]
        got = dataset.rows_to_examples(rows)
        assert got[0].entities == (GoldEntity(0, 3, 'NAME'),)
        assert got[1].entities == ()

    def test_missing_pyarrow_gives_clear_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name.startswith('pyarrow'):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        with pytest.raises(dataset.EvaluationDependencyError, match='eval'):
            dataset.load_examples('domain', tmp_path)
