"""A PII value wrapped across a line break (hyphen + line-break) must
still reach detection as one contiguous token, not two fragments split
by "-\\n".
"""

from __future__ import annotations

from privacyguard_pipeline.detection.pattern_matcher import PatternMatcher
from privacyguard_pipeline.pdf.text_extractor import (
    RawWord,
    group_words_into_blocks,
)


def _raw(text: str, line_no: int, word_no: int, x: float = 0.0) -> RawWord:
    return RawWord(
        text=text,
        block_no=0,
        line_no=line_no,
        word_no=word_no,
        bbox=(x, 0.0, x + 10.0, 10.0),
    )


class TestHyphenWrapNormalization:
    def test_hyphen_wrapped_word_joins_without_hyphen_or_break(
        self,
    ) -> None:
        raw_words = [
            _raw('Фамилия:', 0, 0),
            _raw('Александро-', 0, 1),
            _raw('ва', 1, 0),
            _raw('здесь', 1, 1),
        ]

        blocks = group_words_into_blocks(0, raw_words)

        assert len(blocks) == 1
        assert blocks[0].text == 'Фамилия: Александрова здесь'

    def test_standalone_dash_token_at_line_end_still_breaks(self) -> None:
        # A "-" that is its own whole word (not attached to a preceding
        # word) is a standalone dash/list-marker, not a wrap-hyphen —
        # the line break must be preserved for it.
        raw_words = [
            _raw('Пункт', 0, 0),
            _raw('-', 0, 1),
            _raw('Далее', 1, 0),
        ]

        blocks = group_words_into_blocks(0, raw_words)

        assert blocks[0].text == 'Пункт -\nДалее'

    def test_offsets_stay_consistent_with_joined_text(self) -> None:
        raw_words = [
            _raw('Александро-', 0, 0),
            _raw('ва', 1, 0),
        ]

        block = group_words_into_blocks(0, raw_words)[0]

        for word in block.words:
            assert block.text[word.start : word.end] == word.text

    def test_hyphenated_inn_detected_as_one_span(self) -> None:
        # Real (checksum-valid) 10-digit organization INN, wrapped
        # across a line break in the middle of the digit run.
        raw_words = [
            _raw('ИНН:', 0, 0),
            _raw('770708-', 0, 1),
            _raw('3893', 1, 0),
        ]
        block = group_words_into_blocks(0, raw_words)[0]

        spans = PatternMatcher().detect(block.text)

        inn_spans = [s for s in spans if s.entity_type == 'INN']
        assert len(inn_spans) == 1
        assert inn_spans[0].text == '7707083893'
