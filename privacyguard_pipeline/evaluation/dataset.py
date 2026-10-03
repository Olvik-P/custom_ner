"""Загрузка и чтение датасета pii-bench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from privacyguard_pipeline.evaluation.constants import (
    CACHE_DIR,
    DATASET_CONFIGS,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_URL_TEMPLATE,
    DOWNLOAD_TIMEOUT_SECONDS,
)


class EvaluationDependencyError(RuntimeError):
    """Не установлена экстра ``eval`` (pyarrow)."""


@dataclass(frozen=True)
class GoldEntity:
    """Золотая сущность примера.

    Attributes:
        start: Индекс начального символа.
        end: Индекс конечного символа (не включительно).
        type: Тип в терминах бенчмарка (NAME, PHONE_NUMBER и т.д.).
    """

    start: int
    end: int
    type: str


@dataclass(frozen=True)
class Example:
    """Размеченный пример бенчмарка."""

    id: str
    domain: str
    text: str
    entities: tuple[GoldEntity, ...]


def cache_path(
    config: str,
    cache_dir: Path = CACHE_DIR,
    revision: str = DATASET_REVISION,
) -> Path:
    """Путь к закэшированному parquet-файлу конфигурации."""
    return cache_dir / revision / f'{config}.parquet'


def ensure_downloaded(
    config: str,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Возвращает путь к parquet конфигурации, скачав его при пустом кэше.

    Скачивает строго закреплённую ревизию. При заполненном кэше не
    обращается к сети.

    Raises:
        ValueError: Если конфигурации нет в датасете.
    """
    if config not in DATASET_CONFIGS:
        msg = f'unknown config {config!r}, expected one of {DATASET_CONFIGS}'
        raise ValueError(msg)
    target = cache_path(config, cache_dir)
    if target.exists():
        return target
    url = DATASET_URL_TEMPLATE.format(
        repo=DATASET_REPO,
        revision=DATASET_REVISION,
        config=config,
    )
    response = httpx.get(
        url,
        follow_redirects=True,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    # Запись через временный файл, чтобы оборванная загрузка не оставила
    # в кэше битый parquet, который потом сочтут готовым.
    tmp = target.with_suffix('.parquet.part')
    tmp.write_bytes(response.content)
    tmp.replace(target)
    return target


def rows_to_examples(rows: list[dict[str, Any]]) -> list[Example]:
    """Приводит строки таблицы к датаклассам ``Example``."""
    return [
        Example(
            id=str(row['id']),
            domain=str(row['domain']),
            text=str(row['text']),
            entities=tuple(
                GoldEntity(
                    start=int(ent['start']),
                    end=int(ent['end']),
                    type=str(ent['type']),
                )
                for ent in (row.get('entities') or [])
            ),
        )
        for row in rows
    ]


def load_examples(
    config: str,
    cache_dir: Path = CACHE_DIR,
) -> list[Example]:
    """Загружает примеры конфигурации (скачивая датасет при необходимости).

    Raises:
        EvaluationDependencyError: Если не установлена экстра ``eval``.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        msg = (
            'Оценка качества требует экстру eval: '
            'pip install -e ".[eval]" (пакет pyarrow)'
        )
        raise EvaluationDependencyError(msg) from exc
    path = ensure_downloaded(config, cache_dir)
    rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    return rows_to_examples(rows)
