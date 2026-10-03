"""Именованные константы для PrivacyGuard Pipeline.

Централизует магические числа/литералы, используемые в логике
детекции, маскирования, LLM-прокси и логирования. Значения по
умолчанию Pydantic ``Settings`` в ``config.py`` намеренно оставлены
нетронутыми — они уже именованы через ``Field(default=...)`` и
являются частью публичной поверхности конфигурации, а не внутренними
литералами алгоритма.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Уверенность детекции
# ---------------------------------------------------------------------------
PATTERN_CONFIDENCE = 0.95
NATASHA_NER_CONFIDENCE = 0.85
NATASHA_ADDR_CONFIDENCE = 0.75
NATASHA_ADDR_HOUSE_HEURISTIC_CONFIDENCE = 0.6
CONTEXT_RESOLVED_CONFIDENCE = 0.7
CONTEXT_LOOKBACK_WORDS = 3

# ---------------------------------------------------------------------------
# Морфология и списки allow/deny (detection/morphology.py, lists.py)
# ---------------------------------------------------------------------------
# Маркеры персоны: одиночное словарное слово после них остаётся PER
# (например, «гражданин Гусь»). Сравниваются в нижнем регистре без
# окружающей пунктуации.
PERSON_MARKERS = frozenset(
    {
        'гражданин',
        'гражданина',
        'гражданину',
        'гражданином',
        'гражданке',
        'гражданка',
        'гражданки',
        'гражданку',
        'гражданкой',
        'граждане',
        'г-н',
        'г-жа',
        'г-на',
        'г-же',
        'господин',
        'господина',
        'господину',
        'господином',
        'госпожа',
        'госпожи',
        'госпоже',
        'госпожу',
        'пациент',
        'пациента',
        'пациенту',
        'пациентом',
        'пациентка',
        'пациентки',
        'пациентке',
        'пациентку',
        'клиент',
        'клиента',
        'клиенту',
        'клиентом',
        'клиентка',
        'клиентки',
        'клиентке',
        'клиентку',
        'фио',
        'ф.и.о',
        'ф.и.о.',
    }
)
# Юридические формы, снимаемые в начале записи/спана при сравнении со
# списками allow/deny.
LEGAL_FORMS = frozenset({'ооо', 'ао', 'пао', 'зао', 'оао', 'ип'})
MAX_LIST_ENTRIES = 500
MAX_LIST_ENTRY_CHARS = 200
# Строка из такого числа цифр и больше сравнивается только по цифрам.
LIST_DIGITS_ONLY_THRESHOLD = 7
CUSTOM_ENTITY_TYPE = 'CUSTOM'
CUSTOM_SOURCE = 'deny'
DENY_CONFIDENCE = 1.0

# ---------------------------------------------------------------------------
# Скоринг pattern-совпадений (detection/scoring.py)
# ---------------------------------------------------------------------------
# Окно контекста вокруг совпадения в символах; не пересекает перевод строки.
SCORING_WINDOW_BEFORE = 40
SCORING_WINDOW_AFTER = 15

SCORE_PASSPORT_BASE = 0.30
SCORE_PASSPORT_KEYWORD = 0.30
SCORE_PASSPORT_PAIRED_SERIES = 0.15
SCORE_PASSPORT_PLAUSIBLE_YEAR = 0.10
# Серия паспорта: 3-4 цифры - год выдачи (две цифры), правдоподобный
# диапазон 97..99 и 00..PASSPORT_YEAR_MAX.
PASSPORT_YEAR_MAX = 30
PASSPORT_YEAR_MIN_LEGACY = 97

SCORE_SNILS_BASE = 0.30
SCORE_SNILS_CHECKSUM = 0.40
SCORE_SNILS_KEYWORD = 0.30
SCORE_SNILS_FORMATTED = 0.10

SCORE_OGRN_BASE = 0.30
SCORE_OGRN_CHECKSUM = 0.40
SCORE_OGRN_KEYWORD = 0.30

SCORE_COORDS_DECIMAL_BASE = 0.30
SCORE_COORDS_DMS_BASE = 0.70
SCORE_COORDS_KEYWORD = 0.30
SCORE_COORDS_IN_RANGE = 0.10
SCORE_COORDS_PRECISE = 0.15
COORDS_PRECISE_MIN_DECIMALS = 4
COORDS_LATITUDE_MAX = 90.0
COORDS_LONGITUDE_MAX = 180.0

SCORE_IP_BASE = 0.45
SCORE_IP_KEYWORD = 0.30

SCORE_ANTI_CONTEXT_PENALTY = 0.40

# SNILS: контрольное число проверяется только для номеров выше этого.
SNILS_CHECKSUM_MIN_NUMBER = 1_001_998
SNILS_DIGIT_COUNT = 11
SNILS_CHECKSUM_MODULO = 101
OGRN_13_LENGTH = 13
OGRN_15_LENGTH = 15
OGRN_13_MODULO = 11
OGRN_15_MODULO = 13
OGRN_CHECKSUM_DIGIT_MODULO = 10

# ---------------------------------------------------------------------------
# Валидаторы
# ---------------------------------------------------------------------------
INN_VALID_LENGTHS = (10, 12)
# Таблицы весов контрольной суммы ФНС (взвешенная сумма по модулю 11, 10).
INN_10_CHECKSUM_WEIGHTS = (2, 4, 10, 3, 5, 9, 4, 6, 8)
INN_12_CHECKSUM_WEIGHTS_1 = (7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
INN_12_CHECKSUM_WEIGHTS_2 = (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8)
INN_CHECKSUM_MODULO = 11
INN_CHECKSUM_DIGIT_MODULO = 10
PASSPORT_DIGIT_COUNT = 10
IP_OCTET_MIN = 0
IP_OCTET_MAX = 255
LUHN_MODULO = 10
LUHN_DOUBLE_SUBTRACT = 9

# ---------------------------------------------------------------------------
# Маскирование
# ---------------------------------------------------------------------------
TOKEN_HEX_LENGTH = 8

# ---------------------------------------------------------------------------
# LLM-прокси
# ---------------------------------------------------------------------------
HTTP_TIMEOUT_SECONDS = 120.0
HTTP_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 4096
CLAUDE_API_VERSION = '2023-06-01'
HTTP_STATUS_UNAUTHORIZED = 401

# ---------------------------------------------------------------------------
# Усечение текста
# ---------------------------------------------------------------------------
PANEL_PREVIEW_CHARS = 500
LOG_PREVIEW_CHARS = 100
AUDIT_PREVIEW_CHARS = 200

# ---------------------------------------------------------------------------
# Анонимизация PDF
# ---------------------------------------------------------------------------
PDF_OCR_LANGUAGE = 'rus'
PDF_OCR_ZOOM = 3.0
PDF_REDACTION_PADDING = 2.0

# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
MAX_PDF_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 МБ

# ---------------------------------------------------------------------------
# MCP-сервер
# ---------------------------------------------------------------------------
# Как часто фоновая задача проверяет реестр handle -> Masker на просроченные
# по TTL записи (см. Settings.mcp_handle_ttl_seconds в config.py — сам TTL
# настраиваемый и потому там, а не здесь).
MCP_REGISTRY_SWEEP_INTERVAL_SECONDS = 30.0
