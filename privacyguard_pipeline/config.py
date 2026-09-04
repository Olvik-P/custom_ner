"""Configuration module for PrivacyGuard Pipeline.

Uses Pydantic Settings to load and validate configuration
from environment variables.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    Attributes:
        openai_api_key: API key for OpenAI-compatible LLM providers.
        openai_base_url: Base URL for the OpenAI API endpoint.
        llm_model: Model name to use for LLM requests.
        llm_provider: LLM provider type ('openai' or 'claude').
        claude_api_key: API key for Claude API.
        claude_api_url: Base URL for Claude API.
        log_level: Logging level for the application.
        max_text_length: Maximum text length for processing.
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

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator('openai_api_key', 'claude_api_key', mode='before')
    @classmethod
    def _validate_api_keys(cls, v: str) -> str:
        """Strip whitespace from API keys."""
        return v.strip() if isinstance(v, str) else ''

    @model_validator(mode='after')
    def _validate_provider_key(self) -> Settings:
        """Ensure the API key for the selected provider is not empty.

        Warns (via assertion-like check) if the active provider's key
        is missing — but does not block startup, since the pipeline
        handles missing keys via graceful degradation.
        """
        if self.llm_provider == 'openai' and not self.openai_api_key:
            self._log_missing_key('OPENAI_API_KEY', 'openai')
        elif self.llm_provider == 'claude' and not self.claude_api_key:
            self._log_missing_key('CLAUDE_API_KEY', 'claude')
        return self

    @staticmethod
    def _log_missing_key(env_var: str, provider: str) -> None:
        """Log a warning about a missing API key for the active provider."""
        logging.getLogger(__name__).warning(
            '%s is not set — LLM requests to %s will be skipped',
            env_var,
            provider,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured."""
        return bool(self.openai_api_key)

    @property
    def has_claude_key(self) -> bool:
        """Check if Claude API key is configured."""
        return bool(self.claude_api_key)

    @property
    def has_any_llm_key(self) -> bool:
        """Check if any LLM API key is configured."""
        return self.has_openai_key or self.has_claude_key


settings = Settings()
