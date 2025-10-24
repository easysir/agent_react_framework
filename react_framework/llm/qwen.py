"""
Client for Qwen's OpenAI-compatible interface (DashScope compatible mode).
"""

from __future__ import annotations

import os
from typing import Optional

from .http_client import OpenAICompatibleClient

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class QwenChatClient(OpenAICompatibleClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        workspace: Optional[str] = None,
        **kwargs,
    ) -> None:
        resolved_base_url = base_url or os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL)
        resolved_workspace = workspace or os.getenv("QWEN_WORKSPACE")
        headers = {"X-DashScope-Workspace": resolved_workspace} if resolved_workspace else None
        super().__init__(
            model,
            api_key_env="QWEN_API_KEY",
            api_key=api_key,
            base_url=resolved_base_url,
            default_headers=headers,
            **kwargs,
        )
