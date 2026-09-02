## Why

Пакет `privacyguard_pipeline/` вырос без единого стиля кода и без физического
разделения слоёв: детекция PII (три под-слоя) лежит плоско рядом с
маскированием, LLM-прокси и оркестрацией; магические числа (confidence,
таймауты, размеры обрезки текста и т.д.) разбросаны по модулям без имён;
строковые литералы написаны в двойных кавычках без единого правила в линтере.
Это усложняет навигацию по коду и добавление новых типов PII/слоёв.

## What Changes

- Детекция PII выносится в подпакет `detection/` (Вариант A): `_common.py`,
  `_patterns.py`, `_validators.py`, `_pattern_matcher.py`, `_natasha_ner.py`,
  `_contextual_validator.py`, `pii_detector.py` переезжают внутрь и теряют
  `_`-префикс (инкапсуляция становится структурной, через подпакет, а не
  через нейминг); `detection/__init__.py` реэкспортирует публичный API
  (`PIIDetector`, `PIISpan`, `DetectionResult`). `masker.py`, `llm_proxy.py`,
  `audit.py`, `pipeline.py`, `main.py`, `config.py`, `exceptions.py`
  остаются на верхнем уровне пакета (каждый уже соответствует одному слою).
- Все строковые литералы во всём пакете переводятся с `"` на `'`;
  `[tool.ruff.lint]` в `pyproject.toml` дополняется правилом `Q`
  (flake8-quotes) с `quote-style = "single"`, чтобы стиль не размылся в
  будущих правках.
- Магические числа/литералы, используемые внутри алгоритмов (confidence
  detection-слоёв, длина токена маскирования, окно контекста в
  `ContextualValidator`, параметры Луна, таймауты/temperature/max_tokens
  LLM-прокси, версия Claude API, HTTP-код 401, размеры обрезки текста для
  панели/логов/аудита), выносятся в новый модуль `constants.py` с именованными
  константами. Regex-паттерны в `_patterns.py`/`patterns.py` не трогаются —
  цифры длины остаются встроенными в регулярные выражения как есть.
- `config.py` (Pydantic `Settings` и её дефолты) не затрагивается — его
  значения уже именованы через `Field(default=...)` и являются частью
  публичного конфига, а не внутренней логики.

**BREAKING**: внутренние пути импорта меняются (`privacyguard_pipeline._pattern_matcher`
→ `privacyguard_pipeline.detection.pattern_matcher` и т.п.). Публичный API
пакета (`main.process`, `PrivacyGuardPipeline`) не меняется.

## Capabilities

Чисто внутренний рефакторинг (расположение файлов, стиль кавычек, именование
констант) — наблюдаемое поведение пайплайна (детекция → маскирование → LLM →
демаскирование → аудит) не меняется. Спецификации не создаются и не
изменяются (`skip_specs: true`).

### New Capabilities
_нет_

### Modified Capabilities
_нет_

## Impact

- Затронуты все файлы `privacyguard_pipeline/*.py` (перенос детекции,
  замена кавычек) и `pyproject.toml` (правило `ruff` `Q`).
- Импорты, ссылающиеся на переехавшие модули: `pii_detector.py` (реэкспорт
  `DetectionResult`/`PIISpan`), `pipeline.py` (`from ... import PIIDetector`),
  `masker.py` (`from ... import PIISpan`) — все обновляются на новые пути
  внутри `detection/`.
- `test.py` (ручной smoke-test) и `README.md` — путей импорта не используют
  напрямую, но диаграмма в README ссылается на `pii_detector` как модуль;
  стоит сверить актуальность после переноса.
- Внешние потребители пакета (если есть) не затрагиваются: публичный API
  (`privacyguard_pipeline.main.process`, `PrivacyGuardPipeline`) не меняется.
