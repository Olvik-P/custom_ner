## Why

Structured Russian postal addresses (index, city, street, house, building,
office, apartment, district) leak through masking almost completely — in
both the text pipeline and the PDF anonymizer — because `NatashaNER`'s
integration with Natasha's `AddrExtractor` is broken: it calls the wrong
API method (`.find(text)`, which returns one pre-merged match) and reads a
non-existent `.span` attribute on the result, so the surrounding
`try/except` silently discards every match on the first iteration. This has
been a complete no-op since the code was written. What currently gets
masked in addresses is only the incidental byproduct of the generic
PER/LOC/ORG tagger (city/street names, sometimes misclassified as PER),
never postal indices or house/building/office numbers.

## What Changes

- Fix `NatashaNER`'s address extraction (`detection/natasha_ner.py`) to
  call `AddrExtractor` as a match generator and read `start`/`stop`
  directly, instead of the broken `.find(text)` + `.span` usage.
- Emit one `PIISpan` per matched address part (index/city/street/house/
  building/office/apartment/district/region) rather than the single
  merged span the old (non-functional) code assumed — so addresses with
  multiple parts on one line are all individually maskable, and two
  separate addresses in the same text block don't get merged into one
  giant span.
- Add a small regex-based post-processing step, anchored on each
  `AddrExtractor`-detected street ("улица") component, that catches
  the house/building number immediately following it when
  `AddrExtractor`'s own "дом" grammar misses it — specifically a bare
  number with no marker word at all (e.g. "ул. Малышева, 101") or the
  abbreviated marker without a period (e.g. "ул Бутырский Вал,
  д 68/70"), both confirmed present in real documents and both left
  unmasked even with the `AddrExtractor` integration fixed.
- Add a second, analogous regex step, anchored on a house number
  (however it was found — `AddrExtractor`'s own "дом" match or the
  heuristic above), that catches a following "помещ."/"помещение"
  (room/premises) qualifier and its number (e.g. "д. 14, помещ.
  1/1") — `AddrExtractor` has no "помещение" part type at all, in any
  spelling, unlike "офис" which it does recognize.
- Document the known remaining gap: free-form landmark-style location
  descriptions that aren't in postal-address format (e.g. "17-ый км"
  highway marker) are outside `AddrExtractor`'s grammar and stay
  undetected — out of scope for this fix.

## Capabilities

### New Capabilities
- `pii-detection`: Detection of structured Russian postal address
  components (index, city, street, house, building, office, apartment,
  district, region) as PII spans, usable by both the text pipeline and
  the PDF anonymizer.

### Modified Capabilities
(none — `pdf-anonymization`'s existing requirements describe redaction
mechanics given whatever spans `PIIDetector` returns; this change fixes
what spans are returned, not how PDF redaction consumes them)

## Impact

- `privacyguard_pipeline/detection/natasha_ner.py` — the only code change.
- Affects both `PrivacyGuardPipeline.process()` (text/LLM masking) and
  `PDFAnonymizer.anonymize()` (PDF redaction), since both consume
  `PIIDetector.detect()`.
- No new dependencies, no config/API surface changes, no breaking changes
  — `detect()`'s return shape is unchanged, it simply returns the address
  spans it was always supposed to.
