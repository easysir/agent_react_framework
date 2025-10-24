"""
Provider registry and OpenAI-compatible client implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

from .base import LLMClient, LLMError, LLMResponse
from ..core.primitives.messages import ChatMessage


@dataclass(frozen=True)
class ProviderSpec:
    """
    Minimal configuration required to talk to an OpenAI-compatible endpoint.
    """

    name: str
    api_key_env: str
    default_base_url: str
    base_url_env: Optional[str] = None
    organization_env: Optional[str] = None
    header_env_map: Optional[Dict[str, str]] = None  # header -> env var
    default_headers: Optional[Dict[str, str]] = None

    def resolve_base_url(self, explicit: Optional[str] = None) -> str:
        env_value = os.getenv(self.base_url_env) if self.base_url_env else None
        return (explicit or env_value or self.default_base_url).rstrip("/")

    def resolve_headers(self, explicit: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = dict(self.default_headers or {})
        if self.header_env_map:
            for header_name, env_var in self.header_env_map.items():
                value = os.getenv(env_var)
                if value:
                    headers[header_name] = value
        if explicit:
            headers.update(explicit)
        return headers

    def resolve_organization(self, explicit: Optional[str] = None) -> Optional[str]:
        if explicit is not None:
            return explicit
        if self.organization_env:
            return os.getenv(self.organization_env)
        return None


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


_PROVIDER_REGISTRY: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        api_key_env="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
        base_url_env="OPENAI_BASE_URL",
        organization_env="OPENAI_ORG_ID",
    ),
    "deepseek": ProviderSpec(
        name="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
        default_base_url="https://api.deepseek.com/v1",
        base_url_env="DEEPSEEK_BASE_URL",
    ),
    "qwen": ProviderSpec(
        name="qwen",
        api_key_env="QWEN_API_KEY",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        base_url_env="QWEN_BASE_URL",
        header_env_map={"X-DashScope-Workspace": "QWEN_WORKSPACE"},
    ),
}


def register_provider(spec: ProviderSpec) -> None:
    """
    Allow users to register additional OpenAI-compatible providers at runtime.
    """

    _PROVIDER_REGISTRY[spec.name] = spec


def list_providers() -> Iterable[str]:
    return tuple(_PROVIDER_REGISTRY.keys())


def create_openai_compatible_client(
    provider: str,
    model: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    organization: Optional[str] = None,
    timeout: float = 30.0,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> OpenAICompatibleClient:
    """
    Instantiate an OpenAI-compatible client for a registered provider.
    """

    try:
        spec = _PROVIDER_REGISTRY[provider]
    except KeyError as exc:
        raise ValueError(f"Unknown provider '{provider}'. Available: {list_providers()}") from exc

    resolved_base_url = spec.resolve_base_url(base_url)
    resolved_headers = spec.resolve_headers(headers)
    resolved_org = spec.resolve_organization(organization)

    return OpenAICompatibleClient(
        model,
        api_key_env=spec.api_key_env,
        base_url=resolved_base_url,
        api_key=api_key,
        timeout=timeout,
        default_headers=resolved_headers if resolved_headers else None,
        organization=resolved_org,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
