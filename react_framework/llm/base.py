"""
Base interfaces for LLM chat clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.messages import ChatMessage, coerce_messages


class LLMError(RuntimeError):
    """Raised when an LLM request fails."""


@dataclass
class LLMResponse:
    content: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """
    Abstract base class for all chat-completion clients.
    """

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.2,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    def _resolve_kwargs(
        self,
        *,
        temperature: Optional[float],
        max_output_tokens: Optional[int],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        final_max_tokens = max_output_tokens if max_output_tokens is not None else self.max_output_tokens
        if final_max_tokens is not None:
            payload["max_tokens"] = final_max_tokens
        return payload

    def chat_as_dicts(self, messages: List[ChatMessage], **kwargs: Any) -> Dict[str, Any]:
        """
        Utility for subclasses that send HTTP requests with JSON bodies.
        """
        payload = self._resolve_kwargs(
            temperature=kwargs.pop("temperature", None),
            max_output_tokens=kwargs.pop("max_output_tokens", None),
        )
        payload.update(kwargs)
        payload["messages"] = coerce_messages(messages)
        return payload
