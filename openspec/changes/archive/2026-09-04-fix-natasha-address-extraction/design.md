## Context

See `proposal.md` for the "why". Technical background needed here:

- `NatashaNER.detect()` (`detection/natasha_ner.py`) already wraps its
  address-extraction call in `try/except Exception: logger.debug(...)` —
  the same graceful-degradation pattern the module uses for model-load
  failures. That's why the bug was invisible: it fails silently every
  time instead of raising or logging at a visible level.
- Natasha's `Extractor` base class (which `AddrExtractor` inherits) has
  two different ways to get matches, confirmed by reading the installed
  library's source and running both against sample text:
  - `extractor(text)` — the extractor is callable and yields one `Match`
    per matched grammar rule (e.g. one for "индекс", one for "город", one
    for "улица", one for "дом"), each `Match` a plain `(start, stop,
    fact)` namedtuple.
  - `extractor.find(text)` — convenience method that collects all matches
    from `extractor(text)`, sorts them, and returns a *single* `Match`
    spanning from the first match's start to the last match's stop, with
    `fact` set to a combined `Addr(parts=[...])`. Returns `None` if there
    are no matches.
  - Current code calls `.find(text)` (the merged single-match form) but
    then iterates it with `for match in addr_matches`, which iterates the
    3 *fields* of that one namedtuple in turn (start, then stop, then
    fact) instead of iterating multiple matches, and then dereferences a
    `.span` attribute that doesn't exist on any of them — so the very
    first field (`start`, a plain `int`) throws `AttributeError` on
    `match.span.start`, caught by the surrounding `except Exception`.
- `ContextualValidator.validate()` (`detection/contextual_validator.py`)
  already deduplicates overlapping spans by sorting on `(start, -end)`
  and dropping any span whose start falls before the previous kept span's
  end (Requirement: existing behavior, unchanged by this fix). Adjacent,
  non-overlapping spans (e.g. index immediately followed by city) are
  both kept as-is — no special-casing needed for the multi-span output
  this fix introduces.
- **Discovered while implementing** (not known when this design was
  first drafted): `AddrExtractor`'s "дом" (house number) grammar rule
  only fires when the number is preceded by an explicit marker written
  as `д.` (with a period) or the full word `дом` — confirmed by probing
  the library directly. A bare number with no marker at all (`ул.
  Малышева, 101`) or the common abbreviated marker without a period
  (`ул Бутырский Вал, д 68/70`, as literally written in the sample
  document) is not recognized as a "дом" part by `AddrExtractor`, even
  once the extractor is called correctly. Both forms occur in the
  document that motivated this change.
- **Also discovered while implementing**: `AddrExtractor` has no
  `'помещение'` (room/premises) part type at all, in any spelling
  (`помещ.`, `помещение`) — unlike `'офис'`, which it does recognize.
  `125315, г. Москва, ул. Балтийская, д. 14, помещ. 1/1` in the sample
  document leaves `помещ. 1/1` completely unmatched by the library
  even after both the API fix and the house-number heuristic.
- Downstream, both `Masker` (text pipeline) and `pdf/anonymizer.py`
  (`_collect_matched_words`, word-overlap test `word.start < span.end and
  word.end > span.start`) already operate on arbitrary lists of
  independent `PIISpan`s with no assumption that address components form
  one contiguous span — so no downstream change is needed once
  `NatashaNER` emits correct spans.

## Goals / Non-Goals

**Goals:**
- Make `NatashaNER`'s address extraction actually run and return one
  `PIISpan` per matched address component.
- Keep the existing graceful-degradation contract: a failure extracting
  addresses from one input must not raise out of `detect()` or discard
  spans already found by the pattern/NER layers.

**Non-Goals:**
- Detecting free-form landmark-style locations that aren't in postal
  address form (e.g. "17-ый км" highway markers) — `AddrExtractor`'s
  grammar doesn't cover these; see proposal.md's "What Changes" for why
  this is deliberately out of scope.
- Changing `PIISpan.entity_type` granularity — every address component
  keeps mapping to `'LOC'`, same as today, so downstream masking/audit
  code (which only knows `'LOC'`, not "index" vs "street") needs no
  changes. Splitting `LOC` into finer-grained types is a separate
  decision not needed to fix this bug.
- Any change to `PatternMatcher`, `ContextualValidator`'s merge logic, or
  the PDF-anonymizer's word-to-span mapping — all already handle
  independent multi-span output correctly (see Context).

## Decisions

### Call the extractor as a match generator, not `.find()`
Use `self._addr_tagger(text)` (iterating matches directly) instead of
`self._addr_tagger.find(text)`. This is the only way to get one match per
address component instead of one artificially merged match per call —
required by both new spec requirements (per-component spans, and
independent handling of multiple addresses in one block). Each yielded
`Match` exposes `.start`/`.stop` directly (no `.span`), which also fixes
the `AttributeError`.

Alternative considered: keep `.find(text)` and derive per-component spans
from `match.fact.parts`. Rejected — `AddrPart` (the `fact.parts` element
type) carries `value`/`type` but not each part's own offsets, only the
merged match carries `start`/`stop` for the whole run; splitting it back
into components would mean re-searching `value` substrings inside the
merged span, which is strictly more code and more failure-prone (e.g.
ambiguous if a value string repeats) than using the per-component matches
the library already produces via the generator form.

### No overlap re-checking against already-added Natasha spans beyond what exists today
The existing loop already skips an address match fully contained within
a previously added span (the `any(s.start <= match.span.start and s.end
>= match.span.stop ...)` containment check) — kept, adjusted to the fixed
attribute names. This still only prevents exact containment duplicates
(e.g. NER's generic `LOC` tag for a city name overlapping the
`AddrExtractor` city-part match for the same text); it does not need to
handle partial overlaps because `ContextualValidator`'s final merge pass
already drops any span that starts before the previously kept span ends
(see Context). Adding a second overlap-resolution pass here would be
redundant.

### Regex fallback for house numbers `AddrExtractor` misses, anchored on the street match
Immediately after each `AddrExtractor` match whose `fact.type == 'улица'`,
try a small regex against the text right after that match's `stop`:
optional separator/comma, an optional marker (`д`/`д.`/`дом`), then a
digit group (allowing a `NN/NN` building-slash form). This single pattern
covers both discovered gaps — bare number and marker-without-period — in
one place, because the marker is optional in the pattern either way. If
it matches, and the matched range isn't already covered by an existing
span (reusing the same containment check as the main address loop), add
one more `PIISpan` for it.

Alternatives considered:
- **Fix it inside `AddrExtractor`'s own grammar** — rejected: that
  grammar lives inside the third-party `natasha` library, not this
  codebase; patching it would mean vendoring or monkeypatching Yargy
  grammar rules, far more invasive than a small anchored regex for a
  documented gap.
- **Generic "number after any address part" regex, not anchored on
  street specifically** — rejected: anchoring on `'улица'` keeps the
  heuristic narrowly scoped to exactly the position a house number is
  expected to appear (right after the street name), instead of risking
  matches on random numbers following an index or a district name
  elsewhere in the text.
- **Widen `AddrExtractor`'s дом detection window across the whole
  block instead of anchoring on the immediately preceding street match**
  — rejected: house numbers separated from their street by unrelated
  text (e.g. a comma-separated aside) aren't part of the same address in
  practice; anchoring tightly (regex must match starting exactly at the
  street match's `stop`) avoids over-redacting adjacent non-address
  numbers.

New confidence constant `NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE` (lower
than `NATASHA_ADDR_CONFIDENCE`, which is reserved for `AddrExtractor`'s
own grammar-validated matches) marks spans found this way as a
regex heuristic, not a validated address-grammar match — same rationale
`PatternMatcher` vs. `NatashaNER` already use differing confidence
tiers for.

### A second, symmetric regex fallback for the room/premises qualifier
Same anchoring approach as the house-number regex, one step further:
right after a house number is established — whether that's an
`AddrExtractor` `'дом'` match or the 1.5 heuristic's own span — try a
regex for `помещ.`/`помещение` plus a number immediately following it.
Reuses `NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE` rather than a separate
constant: the distinction that constant marks is "regex heuristic vs.
grammar-validated match," not "house vs. room," so a new constant would
add a name without adding information.

The per-match loop is restructured so the "already covered by an
existing span" containment check no longer `continue`s past the
house/room logic below it — previously that `continue` meant a
`'дом'` match arriving *after* the street-anchored heuristic had
already produced an equivalent span (common, since `AddrExtractor`
still yields its own `'дом'` match separately even when the heuristic
also caught it) would skip the room check entirely, because the whole
rest of that loop iteration was skipped. The containment check now only
gates whether the match's own base span gets added, not whether its
house/room continuation is checked.

## Risks / Trade-offs

- **[Risk] The house-number regex fallback can over-match.** Anchoring
  strictly on `AddrExtractor`'s own street match bounds the blast radius,
  but a street followed by an unrelated number shortly after (e.g. a
  street name immediately followed by a date or an unrelated reference
  number, with no real house number in between) could still be
  mis-redacted as a house number. Accepted: consistent with the
  project's existing over-redaction-over-under-redaction stance (see the
  archived PDF-anonymizer design's word-boundary risk) — spuriously
  redacting a non-PII number next to a street name is a much smaller
  cost than leaving a real house number exposed.

- **[Risk] More spans per address means more redaction boundaries in
  PDF output.** Each address component (index, city, street, house,
  building, office) now redacts as its own PDF span/rectangle instead of
  the previous coarse city/street-only spans. This is the intended fix,
  but changes the visual output of previously-generated redacted PDFs —
  not a regression, but worth calling out since it changes existing
  golden-file-style expectations if any manual test fixtures compare
  redacted PDF output byte-for-byte.
- **[Risk, accepted] Landmark-style non-postal addresses remain
  undetected**, e.g. "Екатеринбургская кольцевая автомобильная дорога,
  17-ый км, 1" — `AddrExtractor` only recognizes postal-format
  components. See proposal.md; explicitly out of scope for this change.
- **[Risk] `AddrPart.type` values are Russian nouns tied to the library's
  internal grammar** (e.g. `'индекс'`, `'улица'`, `'дом'`) and aren't
  currently surfaced anywhere (entity_type stays `'LOC'` for all of
  them) — no risk today, but if a future change wants per-component
  granularity in `redacted_by_type` stats, those Russian labels would
  need translating to stable identifiers first.

## Migration Plan

Bugfix to unreleased/internal behavior — no data migration, no config
change, no version bump beyond normal release notes. Existing callers of
`PIIDetector.detect()`, `PrivacyGuardPipeline.process()`, and
`PDFAnonymizer.anonymize()` see no API change, only more complete
address spans in their results.
