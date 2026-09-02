"""Shared pytest fixtures for privacyguard_pipeline tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_CYRILLIC_FONT_CANDIDATES = [
    Path('C:/Windows/Fonts/arial.ttf'),
    Path('C:/Windows/Fonts/calibri.ttf'),
    Path('C:/Windows/Fonts/tahoma.ttf'),
]


@pytest.fixture(scope='session')
def cyrillic_font_path() -> str:
    """Path to a TrueType font capable of round-tripping Cyrillic text.

    PyMuPDF's built-in base-14 fonts (e.g. "helv") cannot render/extract
    Cyrillic correctly, so PDF fixtures for these tests embed a real
    system font instead.
    """
    for candidate in _CYRILLIC_FONT_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    pytest.skip('No Cyrillic-capable TrueType font found on this system')
