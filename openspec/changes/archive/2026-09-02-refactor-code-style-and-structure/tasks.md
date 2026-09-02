## 1. Detection layer -> `detection/` subpackage

- [x] 1.1 Create `privacyguard_pipeline/detection/` with `__init__.py`.
- [x] 1.2 Move `_common.py` -> `detection/common.py`.
- [x] 1.3 Move `_patterns.py` -> `detection/patterns.py`.
- [x] 1.4 Move `_validators.py` -> `detection/validators.py`.
- [x] 1.5 Move `_pattern_matcher.py` -> `detection/pattern_matcher.py`.
- [x] 1.6 Move `_natasha_ner.py` -> `detection/natasha_ner.py`.
- [x] 1.7 Move `_contextual_validator.py` -> `detection/contextual_validator.py`.
- [x] 1.8 Move `pii_detector.py` -> `detection/detector.py`.
- [x] 1.9 Update internal imports between the moved files (e.g.
      `detection/pattern_matcher.py` importing `detection/common.py`,
      `detection/patterns.py`, `detection/validators.py`) to the new
      module names/paths.
- [x] 1.10 `detection/__init__.py` re-exports `PIIDetector`, `PIISpan`,
      `DetectionResult` (same public names `pii_detector.py` exported
      before).
- [x] 1.11 Update `pipeline.py` import
      (`from privacyguard_pipeline.pii_detector import DetectionResult,
      PIIDetector` -> `from privacyguard_pipeline.detection import
      DetectionResult, PIIDetector`).
- [x] 1.12 Update `masker.py` import (`from privacyguard_pipeline._common
      import PIISpan` -> `from privacyguard_pipeline.detection import
      PIISpan`).
- [x] 1.13 Grep the whole package (and `test.py`) for any remaining
      references to the old module paths (`_common`, `_patterns`,
      `_validators`, `_pattern_matcher`, `_natasha_ner`,
      `_contextual_validator`, `pii_detector`) and fix them.
- [x] 1.14 Run `uv run mypy .` and fix any import-resolution errors.
- [x] 1.15 Smoke-test: `uv run python main.py "Иванов Пётр, +7(916)123-45-67"`
      (or `python test.py` if an LLM key is configured) — confirm detection,
      masking, and demasking still work end-to-end.

## 2. Extract magic numbers into `constants.py`

- [x] 2.1 Create `privacyguard_pipeline/constants.py` with sections
      (as comments): Detection confidence, Validators, Masking, LLM proxy,
      Text truncation.
- [x] 2.2 Detection confidence: `PATTERN_CONFIDENCE = 0.95`,
      `NATASHA_NER_CONFIDENCE = 0.85`, `NATASHA_ADDR_CONFIDENCE = 0.75`,
      `CONTEXT_RESOLVED_CONFIDENCE = 0.7`, `CONTEXT_LOOKBACK_WORDS = 3`.
- [x] 2.3 Validators: `INN_VALID_LENGTHS = (10, 12)`,
      `PASSPORT_DIGIT_COUNT = 10`, `IP_OCTET_MIN = 0`, `IP_OCTET_MAX = 255`,
      `LUHN_MODULO = 10`, `LUHN_DOUBLE_SUBTRACT = 9`.
- [x] 2.4 Masking: `TOKEN_HEX_LENGTH = 8`.
- [x] 2.5 LLM proxy: `HTTP_TIMEOUT_SECONDS = 120.0`,
      `HTTP_CONNECT_TIMEOUT_SECONDS = 30.0`, `DEFAULT_TEMPERATURE = 0.3`,
      `DEFAULT_MAX_TOKENS = 4096`, `CLAUDE_API_VERSION = '2023-06-01'`,
      `HTTP_STATUS_UNAUTHORIZED = 401`.
- [x] 2.6 Text truncation: `PANEL_PREVIEW_CHARS = 500`,
      `LOG_PREVIEW_CHARS = 100`, `AUDIT_PREVIEW_CHARS = 200`.
- [x] 2.7 Wire `detection/pattern_matcher.py` to `PATTERN_CONFIDENCE`.
- [x] 2.8 Wire `detection/natasha_ner.py` to `NATASHA_NER_CONFIDENCE` /
      `NATASHA_ADDR_CONFIDENCE`.
- [x] 2.9 Wire `detection/contextual_validator.py` to
      `CONTEXT_RESOLVED_CONFIDENCE` / `CONTEXT_LOOKBACK_WORDS` (the
      `tokens_before[-3:]` slice).
- [x] 2.10 Wire `detection/validators.py` to `INN_VALID_LENGTHS`,
      `PASSPORT_DIGIT_COUNT`, `IP_OCTET_MIN`/`IP_OCTET_MAX`, `LUHN_MODULO`,
      `LUHN_DOUBLE_SUBTRACT` (leave the `% 2 == 1` parity check as-is —
      it's structural to the Luhn algorithm, not a named value).
- [x] 2.11 Wire `masker.py` to `TOKEN_HEX_LENGTH` (the `uuid.uuid4().hex[:8]`
      slice).
- [x] 2.12 Wire `llm_proxy.py` to `HTTP_TIMEOUT_SECONDS`,
      `HTTP_CONNECT_TIMEOUT_SECONDS`, `DEFAULT_TEMPERATURE`,
      `DEFAULT_MAX_TOKENS`, `CLAUDE_API_VERSION`,
      `HTTP_STATUS_UNAUTHORIZED`.
- [x] 2.13 Wire `main.py` (`result["anonymized_text"][:500]` /
      `llm_response[:500]`) to `PANEL_PREVIEW_CHARS`.
- [x] 2.14 Wire `pipeline.py` (`anonymized_text[:100]` debug log) to
      `LOG_PREVIEW_CHARS`.
- [x] 2.15 Wire `audit.py` (`anonymized_text[:200]` preview) to
      `AUDIT_PREVIEW_CHARS`.
- [x] 2.16 Confirm `config.py` is untouched (no constants extracted from
      `Settings` field defaults).
- [x] 2.17 Confirm regex digit counts in `detection/patterns.py` are left
      untouched (not extracted into constants).
- [x] 2.18 Run `uv run mypy .` and a smoke-test run to confirm behavior
      (detected entities, confidences, token format) is unchanged.

## 3. Unify string quotes to `'`

- [x] 3.1 Add `[tool.ruff.format]` with `quote-style = "single"` to
      `pyproject.toml`.
- [x] 3.2 Add `"Q"` to `[tool.ruff.lint].select` in `pyproject.toml`
      (flake8-quotes, enforced going forward on top of the one-time
      format pass).
- [x] 3.3 Run `uv run ruff format .` across the whole package.
- [x] 3.4 Spot-check that triple-double-quoted docstrings (module/class/
      function) were left as `"""..."""` and not converted to `'''...'''`
      in a couple of representative files (`main.py`, `pipeline.py`).
- [x] 3.5 Run `uv run ruff check .` and fix any remaining `Q`
      violations that `ruff format` didn't already resolve.
- [x] 3.6 Review the full diff for unintended changes (formatter
      reflowing lines beyond quote style) before considering this step
      done.

## 4. Final verification

- [x] 4.1 Run `uv run ruff check .` (clean).
- [x] 4.2 Run `uv run mypy .` (clean, strict mode).
- [x] 4.3 Run `uv run pytest` if a `tests/` directory exists by this
      point; otherwise smoke-test via `uv run python main.py "..."` /
      `python test.py`.
- [x] 4.4 Re-read `README.md`'s architecture section/module references
      and update any mentions of moved module paths (e.g. `pii_detector`)
      if they no longer match the new `detection/` layout.
- [x] 4.5 Update `CLAUDE.md`'s Architecture section to reflect the new
      `detection/` subpackage layout and module names.
