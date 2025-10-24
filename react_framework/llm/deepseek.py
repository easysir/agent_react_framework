"""
Client for DeepSeek's OpenAI-compatible chat API.
"""

from __future__ import annotations

import os
from typing import Optional

from .http_client import OpenAICompatibleClient

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekChatClient(OpenAICompatibleClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)
        super().__init__(
            model,
            api_key_env="DEEPSEEK_API_KEY",
            api_key=api_key,
            base_url=resolved_base_url,
            **kwargs,
        )
