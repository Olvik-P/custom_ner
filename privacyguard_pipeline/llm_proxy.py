"""LLM Proxy module for PrivacyGuard Pipeline.

Handles communication with OpenAI/Claude APIs.
Sends anonymized text and receives responses with possible masking tokens.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import httpx

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.exceptions import (
    LLMAuthenticationError,
    LLMConnectionError,
)

logger = logging.getLogger(__name__)


class LLMProxy:
    """Proxy for sending anonymized text to LLM APIs and receiving responses.

    Supports OpenAI-compatible APIs and Claude API.

    Usage:
        proxy = LLMProxy()
        response = await proxy.send("anonymized text")
        await proxy.close()
    """

    def __init__(self) -> None:
        self._provider = settings.llm_provider
        self._model = settings.llm_model
        self._http_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # HTTP client management
    # ------------------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy initialisation)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=30.0),
            )
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def send(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send anonymized text to the LLM API and return the response.

        Args:
            anonymized_text: Text with PII replaced by masking tokens.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            LLM response text (may contain masking tokens).

        Raises:
            LLMConnectionError: If the API is unreachable.
            LLMAuthenticationError: If API key is invalid.
        """
        if self._provider == "claude":
            return await self._send_claude(anonymized_text, system_prompt)
        return await self._send_openai(anonymized_text, system_prompt)

    # ------------------------------------------------------------------
    # OpenAI-compatible API
    # ------------------------------------------------------------------

    async def _send_openai(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send request to OpenAI-compatible API."""
        if not settings.has_openai_key:
            raise LLMAuthenticationError(
                "OPENAI_API_KEY is not configured in .env",
            )

        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": anonymized_text})

        return await self._post_request(
            url=f"{settings.openai_base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json_body={
                "model": self._model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            extractor=lambda data: data["choices"][0]["message"]["content"],
            auth_error_msg="Invalid OpenAI API key",
            connection_error_msg="OpenAI API error",
            unreachable_error_msg="OpenAI API unreachable",
        )

    # ------------------------------------------------------------------
    # Claude API
    # ------------------------------------------------------------------

    async def _send_claude(
        self,
        anonymized_text: str,
        system_prompt: str | None = None,
    ) -> str:
        """Send request to Claude API."""
        if not settings.has_claude_key:
            raise LLMAuthenticationError(
                "CLAUDE_API_KEY is not configured in .env",
            )

        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [
                {"role": "user", "content": anonymized_text},
            ],
        }

        if system_prompt:
            body["system"] = system_prompt

        return await self._post_request(
            url=settings.claude_api_url,
            headers={
                "x-api-key": settings.claude_api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json_body=body,
            extractor=lambda data: data["content"][0]["text"],
            auth_error_msg="Invalid Claude API key",
            connection_error_msg="Claude API error",
            unreachable_error_msg="Claude API unreachable",
        )

    # ------------------------------------------------------------------
    # Shared HTTP logic
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
        """Send an HTTP POST request and extract the response.

        Args:
            url: Request URL.
            headers: HTTP headers.
            json_body: JSON body payload.
            extractor: Callable to extract the response text from JSON.
            auth_error_msg: Message for 401 errors.
            connection_error_msg: Message for HTTP errors.
            unreachable_error_msg: Message for connection errors.

        Returns:
            Extracted response text.

        Raises:
            LLMAuthenticationError: On 401 status.
            LLMConnectionError: On other HTTP or connection errors.
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
            if exc.response.status_code == 401:
                raise LLMAuthenticationError(auth_error_msg) from exc
            raise LLMConnectionError(
                f"{connection_error_msg}: {exc.response.status_code}",
            ) from exc
        except httpx.RequestError as exc:
            raise LLMConnectionError(
                f"{unreachable_error_msg}: {exc}",
            ) from exc
