## 1. Fix address extraction

- [x] 1.1 In `detection/natasha_ner.py`, replace the `self._addr_tagger.find(text)` + `for match in addr_matches: ... match.span.start ...` block with iteration over `self._addr_tagger(text)` (the extractor called directly as a match generator), reading `match.start`/`match.stop` instead of `match.span.start`/`match.span.stop`.
- [x] 1.2 Keep the existing containment check that skips an address match already covered by a previously added span, updated to the corrected attribute names.
- [x] 1.3 Keep the surrounding `try/except Exception: logger.debug(...)` around the address-extraction block so a failure there still can't raise out of `detect()` or discard spans already produced by the pattern/NER layers for that call.
- [x] 1.4 Add `NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE` to `constants.py` (lower than `NATASHA_ADDR_CONFIDENCE`) for spans found by the new heuristic below.
- [x] 1.5 In `detection/natasha_ner.py`, add a regex anchored on each `AddrExtractor` match whose `fact.type == 'улица'`: match immediately after that match's `stop` for an optional separator, an optional `д`/`д.`/`дом` marker, and a digit group (allowing `NN/NN`) — covering both a bare number with no marker (`ул. Малышева, 101`) and the abbreviated marker without a period (`ул Бутырский Вал, д 68/70`), which `AddrExtractor`'s own "дом" grammar misses. Add a `PIISpan` for a match, reusing the existing containment check to skip it if already covered by another span.
- [x] 1.6 Extend `detection/natasha_ner.py` with a second anchored regex for a `помещ.`/`помещение` (room/premises) qualifier immediately following a house number — whether that house number came from `AddrExtractor`'s own `'дом'` match or from the 1.5 heuristic — since `AddrExtractor` has no `'помещение'` part type at all, in any spelling (unlike `'офис'`, which it does recognize). Restructure the per-match loop so the containment check for a match's own base span no longer short-circuits the house/room continuation checks below it.

## 2. Tests

- [x] 2.1 Create `tests/` (per CLAUDE.md, it does not exist yet) and `tests/detection/test_natasha_ner.py` covering: an address with index+city+street+house all producing separate spans; an address with building/office qualifiers (`д 68/70 стр 1, офис 54`) producing a span per qualifier; two distinct organizations' addresses in one text block producing spans that don't merge across the intervening non-address text; address extraction raising an exception not propagating out of `detect()` and not discarding spans from other layers (mock `_addr_tagger` to raise); the house-number heuristic catching a bare number with no marker (`ул. Малышева, 101`) and the abbreviated marker without a period (`ул Бутырский Вал, д 68/70`); the heuristic NOT firing on a street not followed by a number-shaped token shortly after.
- [x] 2.2 Add or extend a `PIIDetector`-level test confirming the fixed address spans survive `ContextualValidator`'s merge/dedup step end to end (e.g. using the `620004, г. Екатеринбург, ул. Малышева, 101` sample), asserting each address component appears as its own masked token.

## 3. Verification

- [x] 3.1 Run `uv run ruff format .`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest` from `privacyguard_pipeline/` and fix any issues.
- [x] 3.2 Re-run `PDFAnonymizer` against `docs/Решение (безбумажное).pdf` and confirm the postal indices, house/building/office numbers that previously leaked through (`620004`, `101`, `127055`, `д 68/70 стр 1, офис 54`, `125315`, `д. 14, помещ. 1/1`) no longer appear in the redacted output's extracted text.

## 4. Documentation

- [x] 4.1 Update the worked example in `README.md` ("Пример работы" section, lines ~140-152): after the fix, `д.5` and `кв.17` in the sample address are also detected and masked as separate `LOC` tokens — regenerate the "Обезличенный текст" block from the actual fixed pipeline output rather than hand-editing placeholder tokens.
