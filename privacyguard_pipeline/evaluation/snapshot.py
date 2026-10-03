"""Снимок результатов оценки: сборка, запись, чтение."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from privacyguard_pipeline import __version__
from privacyguard_pipeline.evaluation.constants import (
    DATASET_REPO,
    DATASET_REVISION,
    SNAPSHOT_FORMAT_VERSION,
)
from privacyguard_pipeline.evaluation.runner import RunResult

_GIT_TIMEOUT_SECONDS = 10


def _git(*args: str, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ['git', *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state(cwd: Path) -> dict[str, object]:
    """Текущий коммит и признак незакоммиченных изменений (None — нет git)."""
    commit = _git('rev-parse', 'HEAD', cwd=cwd)
    status = _git('status', '--porcelain', cwd=cwd)
    return {
        'git_commit': commit,
        'git_dirty': None if status is None else bool(status),
    }


def build_snapshot(
    result: RunResult,
    config: str,
    example_count: int,
    repo_dir: Path,
) -> dict[str, Any]:
    """Собирает снимок: метаданные воспроизводимости и метрики.

    В снимок не попадают тексты примеров — только числа и метаданные.
    """
    return {
        'format_version': SNAPSHOT_FORMAT_VERSION,
        'meta': {
            'package_version': __version__,
            **git_state(repo_dir),
            'dataset_repo': DATASET_REPO,
            'dataset_revision': DATASET_REVISION,
            'dataset_config': config,
            'example_count': example_count,
            'min_confidence': result.min_confidence,
            'natasha_available': result.natasha_available,
        },
        'metrics': result.metrics.to_dict(),
    }


def write_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    """Записывает снимок в JSON (UTF-8, с отступами)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )


def read_snapshot(path: Path) -> dict[str, Any]:
    """Читает снимок из JSON."""
    data: dict[str, Any] = json.loads(path.read_text(encoding='utf-8'))
    return data
