"""
Client for the official OpenAI API.
"""

from __future__ import annotations

import os
from typing import Optional

from .http_client import OpenAICompatibleClient

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIChatClient(OpenAICompatibleClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)
        super().__init__(
            model,
            api_key_env="OPENAI_API_KEY",
            api_key=api_key,
            base_url=resolved_base_url,
            organization=organization or os.getenv("OPENAI_ORG_ID"),
            **kwargs,
        )
