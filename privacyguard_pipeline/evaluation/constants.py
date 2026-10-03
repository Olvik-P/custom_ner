"""Константы оценки качества детекции на pii-bench."""

from __future__ import annotations

from pathlib import Path

# Ревизия датасета hivetrace/pii-bench (commit hash). Датасет — held-out
# тест: на нём нельзя обучать и под него нельзя подгонять детектор.
DATASET_REPO = 'hivetrace/pii-bench'
DATASET_REVISION = 'cd6a18ace16daf23e79247ccf1e2b245d4054654'
DATASET_URL_TEMPLATE = (
    'https://huggingface.co/datasets/{repo}/resolve/{revision}'
    '/data/{config}-00000-of-00001.parquet'
)
DATASET_CONFIGS = ('domain', 'entity')
DEFAULT_CONFIG = 'domain'

# Кэш лежит рядом с pyproject.toml пакета и игнорируется git.
CACHE_DIR = Path(__file__).resolve().parents[1] / '.eval_cache'
RESULTS_DIR = CACHE_DIR / 'results'
BASELINES_DIR = Path(__file__).resolve().parent / 'baselines'

DOWNLOAD_TIMEOUT_SECONDS = 60.0

SNAPSHOT_FORMAT_VERSION = 1

# Тип бенчмарка -> тип пакета. ADDRESS идёт как LOC: отдельного типа
# адреса в пакете нет, Natasha AddrExtractor помечает адреса как LOC.
TYPE_MAP = {
    'NAME': 'PER',
    'ADDRESS': 'LOC',
    'PHONE_NUMBER': 'PHONE',
    'EMAIL': 'EMAIL',
    'BANK_CARD_NUMBER': 'CARD',
    'PASSPORT_NUMBER': 'PASSPORT',
    'INN': 'INN',
    'SNILS': 'SNILS',
    'OGRN': 'OGRN',
    'KPP': 'KPP',
    'OGRNIP': 'OGRNIP',
}
# Типы бенчмарка без аналога в пакете (нет проверяемого формата).
UNSUPPORTED_TYPES = frozenset({'CVC', 'TOKEN'})
