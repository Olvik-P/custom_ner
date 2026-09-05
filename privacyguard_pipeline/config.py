"""Модуль конфигурации для PrivacyGuard Pipeline.

Использует Pydantic Settings для загрузки и валидации конфигурации из
переменных окружения.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из файла .env.

    Attributes:
        openai_api_key: Ключ API для OpenAI-совместимых LLM-провайдеров.
        openai_base_url: Базовый URL эндпоинта API OpenAI.
        llm_model: Имя модели для запросов к LLM.
        llm_provider: Тип провайдера LLM ('openai' или 'claude').
        claude_api_key: Ключ API для Claude API.
        claude_api_url: Базовый URL для Claude API.
        log_level: Уровень логирования приложения.
        max_text_length: Максимальная длина текста для обработки.
        api_key: Общий секрет, который клиенты HTTP API должны
            передавать через X-API-Key.
        api_key_required: Отклоняет ли HTTP API запросы без валидного
            API-ключа. Отключение предназначено только для
            локальной/dev-разработки.
        api_port: Порт, который слушает сервер HTTP API.
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
    )

    # OpenAI / Universal LLM
    openai_api_key: str = Field(
        default='',
        validation_alias='OPENAI_API_KEY',
    )
    openai_base_url: str = Field(
        default='https://api.openai.com/v1',
        validation_alias='OPENAI_BASE_URL',
    )
    llm_model: str = Field(
        default='gpt-4o-mini',
        validation_alias='LLM_MODEL',
    )
    llm_provider: Literal['openai', 'claude'] = Field(
        default='openai',
        validation_alias='LLM_PROVIDER',
    )

    # Claude
    claude_api_key: str = Field(
        default='',
        validation_alias='CLAUDE_API_KEY',
    )
    claude_api_url: str = Field(
        default='https://api.anthropic.com/v1/messages',
        validation_alias='CLAUDE_API_URL',
    )

    # Application
    log_level: str = Field(default='INFO', validation_alias='LOG_LEVEL')
    max_text_length: int = Field(
        default=100_000,
        validation_alias='MAX_TEXT_LENGTH',
    )

    # HTTP API
    api_key: str = Field(default='', validation_alias='API_KEY')
    api_key_required: bool = Field(
        default=True,
        validation_alias='API_KEY_REQUIRED',
    )
    api_port: int = Field(default=8420, validation_alias='API_PORT')

    # ------------------------------------------------------------------
    # Валидаторы
    # ------------------------------------------------------------------

    @field_validator('openai_api_key', 'claude_api_key', mode='before')
    @classmethod
    def _validate_api_keys(cls, v: str) -> str:
        """Убирает пробелы по краям ключей API."""
        return v.strip() if isinstance(v, str) else ''

    @model_validator(mode='after')
    def _validate_provider_key(self) -> Settings:
        """Проверяет, что ключ API выбранного провайдера не пуст.

        Предупреждает (проверкой в духе assert), если ключ активного
        провайдера отсутствует — но не блокирует старт, так как
        пайплайн обрабатывает отсутствующие ключи через корректную
        деградацию.
        """
        if self.llm_provider == 'openai' and not self.openai_api_key:
            self._log_missing_key('OPENAI_API_KEY', 'openai')
        elif self.llm_provider == 'claude' and not self.claude_api_key:
            self._log_missing_key('CLAUDE_API_KEY', 'claude')
        return self

    @staticmethod
    def _log_missing_key(env_var: str, provider: str) -> None:
        """Логирует предупреждение об отсутствующем ключе провайдера."""
        logging.getLogger(__name__).warning(
            '%s is not set — LLM requests to %s will be skipped',
            env_var,
            provider,
        )

    # ------------------------------------------------------------------
    # Свойства
    # ------------------------------------------------------------------

    @property
    def has_openai_key(self) -> bool:
        """Проверяет, настроен ли ключ API OpenAI."""
        return bool(self.openai_api_key)

    @property
    def has_claude_key(self) -> bool:
        """Проверяет, настроен ли ключ API Claude."""
        return bool(self.claude_api_key)

    @property
    def has_any_llm_key(self) -> bool:
        """Проверяет, настроен ли хоть один ключ API LLM."""
        return self.has_openai_key or self.has_claude_key


settings = Settings()
