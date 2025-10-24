"""
HTTP implementation for OpenAI-compatible chat endpoints.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests

from .base import LLMClient, LLMError, LLMResponse
from ..core.messages import ChatMessage


class OpenAICompatibleClient(LLMClient):
    """
    Generic client that targets OpenAI-style chat completion APIs.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key_env: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        organization: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: float = 0.2,
    ) -> None:
        super().__init__(model, temperature=temperature, max_output_tokens=max_output_tokens)
        self._api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(
                f"API key is required. Provide via constructor or set the {api_key_env} environment variable."
            )
        self.timeout = timeout
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if organization:
            headers["OpenAI-Organization"] = organization
        if default_headers:
            headers.update(default_headers)
        self.headers = headers

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload = self.chat_as_dicts(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        response = requests.post(self.endpoint, json=payload, headers=self.headers, timeout=self.timeout)
        if response.status_code >= 400:
            raise LLMError(f"LLM request failed ({response.status_code}): {response.text}")
        body: Dict[str, Any] = response.json()
        try:
            choice = body["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Malformed response structure: {body}") from exc
        content = message.get("content", "")
        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            usage=body.get("usage"),
            raw=body,
        )
