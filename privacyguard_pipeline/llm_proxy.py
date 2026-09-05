"""Модуль LLM Proxy для PrivacyGuard Pipeline.

Обрабатывает взаимодействие с API OpenAI/Claude.
Отправляет анонимизированный текст и получает ответы, возможно
содержащие маскирующие токены.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.constants import (
    CLAUDE_API_VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    HTTP_CONNECT_TIMEOUT_SECONDS,
    HTTP_STATUS_UNAUTHORIZED,
    HTTP_TIMEOUT_SECONDS,
)
from privacyguard_pipeline.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
)

logger = logging.getLogger(__name__)

# Reasoning-модели DeepSeek выдают отдельный блок reasoning_content и
# тратят на него лишние токены/время, если это явно не отключено через
# `thinking: {"type": "disabled"}` (их задокументированное OpenAI-
# совместимое расширение - см. https://api-docs.deepseek.com). Этот
# пайплайн всегда потребляет только финальный ответ, поэтому поле
# отправляется, только когда настроенный endpoint действительно
# DeepSeek - настоящие OpenAI-совместимые API могут отклонить
# нераспознанное поле тела запроса.
_DEEPSEEK_HOST_MARKER = 'deepseek'


class LLMProxy:
    """Прокси для отправки анонимизированного текста в API LLM и приёма ответа.

    Поддерживает OpenAI-совместимые API и Claude API.

    Использование:
        proxy = LLMProxy()
        response = await proxy.send("anonymized text")
        await proxy.close()
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider
        self._model = settings.llm_model
        self._http_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Управление HTTP-клиентом
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Возвращает или создаёт HTTP-клиент (ленивая инициализация)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    HTTP_TIMEOUT_SECONDS,
                    connect=HTTP_CONNECT_TIMEOUT_SECONDS,
                ),
            )
        return self._http_client

    async def close(self) -> None:
        """Закрывает HTTP-клиент и освобождает ресурсы."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    async def send(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Отправляет анонимизированный текст в API LLM и возвращает ответ.

        Args:
            anonymized_text: Текст с PII, заменённым на маскирующие
                токены.
            system_prompt: Опциональный системный промпт для LLM.

        Returns:
            Текст ответа LLM (может содержать маскирующие токены).

        Raises:
            LLMConnectionError: Если API недоступен.
            LLMAuthenticationError: Если ключ API невалиден.
        """
        if self._provider == 'claude':
            return await self._send_claude(anonymized_text, system_prompt)
        return await self._send_openai(anonymized_text, system_prompt)

    # ------------------------------------------------------------------
    # OpenAI-совместимый API
    # ------------------------------------------------------------------

    async def _send_openai(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Отправляет запрос в OpenAI-совместимый API."""
        if not settings.has_openai_key:
            raise LLMAuthenticationError(
                'OPENAI_API_KEY is not configured in .env',
            )

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})

        messages.append({'role': 'user', 'content': anonymized_text})

        json_body: dict[str, Any] = {
            'model': self._model,
            'messages': messages,
            'temperature': DEFAULT_TEMPERATURE,
            'max_tokens': DEFAULT_MAX_TOKENS,
        }
        if _DEEPSEEK_HOST_MARKER in settings.openai_base_url.lower():
            json_body['thinking'] = {'type': 'disabled'}

        return await self._post_request(
            url=f'{settings.openai_base_url.rstrip("/")}/chat/completions',
            headers={
                'Authorization': f'Bearer {settings.openai_api_key}',
                'Content-Type': 'application/json',
            },
            json_body=json_body,
            extractor=lambda data: data['choices'][0]['message']['content'],
            auth_error_msg='Invalid OpenAI API key',
            connection_error_msg='OpenAI API error',
            unreachable_error_msg='OpenAI API unreachable',
        )

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    async def _send_claude(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Отправляет запрос в Claude API."""
        if not settings.has_claude_key:
            raise LLMAuthenticationError(
                'CLAUDE_API_KEY is not configured in .env',
            )

        body: dict[str, Any] = {
            'model': self._model,
            'max_tokens': DEFAULT_MAX_TOKENS,
            'messages': [
                {'role': 'user', 'content': anonymized_text},
            ],
        }

        if system_prompt:
            body['system'] = system_prompt

        return await self._post_request(
            url=settings.claude_api_url,
            headers={
                'x-api-key': settings.claude_api_key,
                'anthropic-version': CLAUDE_API_VERSION,
                'Content-Type': 'application/json',
            },
            json_body=body,
            extractor=lambda data: data['content'][0]['text'],
            auth_error_msg='Invalid Claude API key',
            connection_error_msg='Claude API error',
            unreachable_error_msg='Claude API unreachable',
        )

    # ------------------------------------------------------------------
    # Общая HTTP-логика
    # ------------------------------------------------------------------

    async def _post_request(
        self,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        extractor: Callable[[dict[str, Any]], str],
        auth_error_msg: str,
        connection_error_msg: str,
        unreachable_error_msg: str,
    ) -> str:
        """Отправляет HTTP POST-запрос и извлекает ответ.

        Args:
            url: URL запроса.
            headers: HTTP-заголовки.
            json_body: Тело запроса в формате JSON.
            extractor: Функция для извлечения текста ответа из JSON.
            auth_error_msg: Сообщение для ошибок 401.
            connection_error_msg: Сообщение для HTTP-ошибок.
            unreachable_error_msg: Сообщение для ошибок соединения.

        Returns:
            Извлечённый текст ответа.

        Raises:
            LLMAuthenticationError: При статусе 401.
            LLMConnectionError: При других HTTP-ошибках или ошибках
                соединения.
        """
        client = await self._get_client()

        try:
            response = await client.post(
                url,
                headers=headers,
                json=json_body,
            )
            response.raise_for_status()
            data = response.json()
            return extractor(data)

        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == HTTP_STATUS_UNAUTHORIZED:
                raise LLMAuthenticationError(auth_error_msg) from exc
            raise LLMConnectionError(
                f'{connection_error_msg}: {exc.response.status_code}',
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f'{unreachable_error_msg}: {exc}',
            ) from exc
        except (KeyError, IndexError, TypeError) as exc:
            # Ответ 200 OK, тело которого не соответствует ожидаемой
            # форме (например, пустой массив `choices`/`content` из-за
            # отфильтрованного safety-фильтром или обрезанного
            # completion'а) — та же классификация, что и у сбоя на
            # уровне транспорта.
            raise LLMConnectionError(
                f'{connection_error_msg}: malformed response body ({exc})',
            ) from exc
