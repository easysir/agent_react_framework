"""
Convenience exports for built-in LLM clients.
"""

from .base import LLMClient, LLMError, LLMResponse
from .providers import (
    OpenAICompatibleClient,
    ProviderSpec,
    create_chat_completion_client,
    list_providers,
    register_provider,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "OpenAICompatibleClient",
    "ProviderSpec",
    "create_chat_completion_client",
    "list_providers",
    "register_provider",
]
