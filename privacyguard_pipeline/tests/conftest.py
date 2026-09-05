"""Общие фикстуры pytest для тестов privacyguard_pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from privacyguard_pipeline.detection import PIIDetector

_CYRILLIC_FONT_CANDIDATES = [
    Path('C:/Windows/Fonts/arial.ttf'),
    Path('C:/Windows/Fonts/calibri.ttf'),
    Path('C:/Windows/Fonts/tahoma.ttf'),
]


@pytest.fixture(scope='session')
def cyrillic_font_path() -> str:
    """Путь к TrueType-шрифту, способному корректно вывести/извлечь кириллицу.

    Встроенные base-14 шрифты PyMuPDF (например, "helv") не могут
    корректно отрисовывать/извлекать кириллицу, поэтому PDF-фикстуры
    для этих тестов вместо этого встраивают настоящий системный шрифт.
    """
    for candidate in _CYRILLIC_FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    pytest.skip('No Cyrillic-capable TrueType font found on this system')


@pytest.fixture(scope='session')
def detector() -> PIIDetector:
    """Общий экземпляр PIIDetector (загружает модели Natasha один раз)."""
    return PIIDetector()
