"""FontCache must key its extracted-font cache by xref, not font name.

Two pages can embed distinct font *objects* (different xrefs) under the
same base font name — e.g. two different glyph subsets of "LiberationSerif".
Caching by name alone would let one page's redaction silently reuse the
wrong page's font bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from privacyguard_pipeline.pdf import renderer


class _FakePage:
    def __init__(self, fonts: list[tuple[Any, ...]]) -> None:
        self._fonts = fonts

    def get_fonts(self, full: bool = True) -> list[tuple[Any, ...]]:
        return self._fonts


class _FakeDoc:
    def __init__(self, font_bytes_by_xref: dict[int, bytes]) -> None:
        self._bytes = font_bytes_by_xref

    def extract_font(
        self,
        xref: int,
    ) -> tuple[str, str, str, bytes]:
        return ('embedded', 'ttf', 'TrueType', self._bytes[xref])


class TestFontCacheKeyedByXref:
    def test_same_font_name_different_xref_extracts_separately(
        self,
    ) -> None:
        doc = _FakeDoc({10: b'font-bytes-page-1', 20: b'font-bytes-page-2'})
        cache = renderer.FontCache(doc)
        try:
            page1 = _FakePage(
                [(10, 'ttf', 'Type0', 'AAAAAA+LiberationSerif', 'F1', '', 0)],
            )
            page2 = _FakePage(
                [(20, 'ttf', 'Type0', 'BBBBBB+LiberationSerif', 'F1', '', 0)],
            )

            path1 = cache.path_for(page1, 'LiberationSerif')
            path2 = cache.path_for(page2, 'LiberationSerif')

            assert path1 is not None
            assert path2 is not None
            assert path1 != path2
            assert Path(path1).read_bytes() == b'font-bytes-page-1'
            assert Path(path2).read_bytes() == b'font-bytes-page-2'
        finally:
            cache.cleanup()

    def test_same_xref_looked_up_twice_extracts_only_once(self) -> None:
        calls: list[int] = []
        doc = _FakeDoc({10: b'font-bytes'})
        real_extract_font = doc.extract_font

        def counting_extract_font(
            xref: int,
        ) -> tuple[str, str, str, bytes]:
            calls.append(xref)
            return real_extract_font(xref)

        doc.extract_font = counting_extract_font  # type: ignore[method-assign]
        cache = renderer.FontCache(doc)
        try:
            page = _FakePage(
                [(10, 'ttf', 'Type0', 'AAAAAA+LiberationSerif', 'F1', '', 0)],
            )

            path_a = cache.path_for(page, 'LiberationSerif')
            path_b = cache.path_for(page, 'LiberationSerif')

            assert path_a == path_b
            assert calls == [10]
        finally:
            cache.cleanup()
