# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo contains a single package, `privacyguard_pipeline/` — a Russian-language PII
anonymization pipeline that sits in front of LLM API calls (OpenAI-compatible or Claude).
It masks PII before sending text to an LLM, then demasks the response. The top-level
`test.py` is a manual smoke-test script, not part of a test suite.

## Commands

All commands run from `privacyguard_pipeline/` (the `pyproject.toml` root), using `uv`:

```powershell
cd privacyguard_pipeline
uv sync                       # install deps (incl. dev extras: uv sync --extra dev)
uv run python main.py "text with PII"                       # CLI run
uv run python main.py --system "Be concise" -- "text"       # with system prompt
uv run ruff format .          # format (line-length 79, single-quote strings)
uv run ruff check .           # lint (line-length 79, rules: E,F,W,I,Q)
uv run mypy .                 # type check (strict mode)
uv run pytest                 # run tests (testpaths = tests/ — directory does not exist yet)
uv run pytest tests/test_foo.py::test_bar   # run a single test, once tests exist
```

There is no `tests/` directory yet even though `pyproject.toml` points `testpaths` at it —
create it before adding the first test. `pytest-asyncio` is configured with `asyncio_mode = "auto"`,
so `async def test_*` functions work without extra decorators.

The package requires Python >=3.12. A `.env` file (copy from `.env.example`) supplies API keys
and settings — see `config.py` for all fields (`OPENAI_API_KEY`/`CLAUDE_API_KEY`, `LLM_PROVIDER`,
`LLM_MODEL`, `MAX_TEXT_LENGTH`, `NATASHA_MODEL_DIR`, `LOG_LEVEL`). On first run, Natasha downloads
its NER model (~100 MB) to `NATASHA_MODEL_DIR` (default `~/.natasha`).

## Session workflow

After archiving an OpenSpec change (`/opsx:archive`) or before ending a work session,
commit all pending changes — both the code changes and the moved
`openspec/changes/archive/...` directory — so the repo doesn't sit with uncommitted
work between sessions.

## Architecture

Pipeline flow, orchestrated by `pipeline.py::PrivacyGuardPipeline.process()`:

```
text -> PIIDetector.detect() -> Masker.mask() -> LLMProxy.send() -> Masker.demask() -> Masker.clear()
```

### Detection: three layers merged into one `DetectionResult`

`detection/` is a subpackage holding all three detection layers plus their shared
dataclasses. `detection/__init__.py` re-exports the public surface (`PIIDetector`,
`PIISpan`, `DetectionResult`) — import from `privacyguard_pipeline.detection`, not from
the individual submodules below, which are internal to detection.

1. **`detection/pattern_matcher.py` (`PatternMatcher`)** — regex layer. Patterns live in
   `detection/patterns.py` (`PATTERN_REGISTRY`), per-type validators (e.g. Luhn check for
   cards, `phonenumbers` for phones) live in `detection/validators.py`
   (`VALIDATOR_REGISTRY`). Overlapping matches are merged, keeping the longer span.
2. **`detection/natasha_ner.py` (`NatashaNER`)** — neural NER via the Natasha library
   (PER/LOC/ORG, plus address extraction via `AddrExtractor`). Loads models lazily in
   `__init__` and sets `is_available = False` on any failure, so the pipeline degrades
   gracefully (pattern-only) if Natasha/its models can't load — never raises.
3. **`detection/contextual_validator.py` (`ContextualValidator`)** — conflict resolution:
   pattern spans always win over overlapping Natasha spans; NER `PER` hits against
   `WHITELIST` (`detection/common.py` — common words that look like names, e.g. "Ромашка")
   are dropped; ambiguous `PER` spans are reclassified as `LOC` using a small heuristic
   (preceding preposition/movement-verb lookback in `_LOC_PREPOSITIONS`/`_LOC_VERBS`).

`detection/detector.py::PIIDetector` orchestrates the three layers in sequence.
When adding a new PII type: add its regex to `detection/patterns.py`, an optional
validator to `detection/validators.py`, and it's automatically picked up by
`PatternMatcher` — no changes needed in `detection/detector.py`.

### Constants (`constants.py`)

Named constants for literals used inside detection/masking/LLM-proxy/logging logic
(confidence scores per detection layer, context lookback window, Luhn/IP/passport/INN
validator thresholds, token length, HTTP timeouts, `temperature`/`max_tokens`, text
truncation sizes for panels/logs/audit previews). `config.py`'s Pydantic `Settings`
defaults are intentionally *not* duplicated here — they're already named via
`Field(default=...)` and are public configuration, not internal algorithm literals.
Regex digit-length literals in `detection/patterns.py` are also left inline (extracting
them into the pattern string would hurt readability more than it helps).

### Masking (`masker.py`)

`Masker` replaces each `PIISpan` with a token `<TYPE_UUID8HEX>` and keeps a `token -> MappingEntry`
dict **in memory only**. Replacement is done end-to-start (spans sorted by `end` descending) so
earlier indices stay valid as later ones are substituted. `demask()` reverses this via literal
string replacement of tokens found in the LLM response. `clear()` is called by the pipeline after
every request — the mapping must never persist or leak, since it holds raw PII values.

### LLM proxy (`llm_proxy.py`)

`LLMProxy` supports two backends selected by `settings.llm_provider` (`"openai"` or `"claude"`),
both funneled through a shared `_post_request` helper that takes an `extractor` callable for
pulling text out of the differently-shaped response JSON. Missing/invalid API keys and
connection failures raise `LLMAuthenticationError`/`LLMConnectionError` (`exceptions.py`);
`pipeline.py` catches these and degrades gracefully (empty response, error recorded in audit
log and returned in `stats`) rather than failing the whole pipeline.

### Audit (`audit.py`)

`AuditLogger` (module-level singleton `audit_logger`) logs **entity types and counts only, never
PII values** — this is a hard security invariant, not just a style choice (see README's
"Безопасность" section). It uses `structlog` for structured logs (`logs/privacyguard.log`) and
tracks in-memory session stats (`get_stats()`) that flow into the `stats` dict returned by
`pipeline.process()`.

### Errors (`exceptions.py`)

All custom exceptions derive from `PrivacyGuardError`. `main.py`/`pipeline.py` distinguish
`TextTooLongError` (input validation, raised early) from other `PrivacyGuardError` subclasses.
