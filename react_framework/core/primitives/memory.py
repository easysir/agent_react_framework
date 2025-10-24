"""
Simple in-memory storage for agent conversation state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .messages import ChatMessage


@dataclass
class ConversationMemory:
    """
    Container that records the entire conversation between the agent and the
    language model, including tool results.
    """

    messages: List[ChatMessage] = field(default_factory=list)

    def append(self, message: ChatMessage) -> None:
        """追加一条消息到会话记忆。"""
        self.messages.append(message)

    def extend(self, new_messages: Iterable[ChatMessage]) -> None:
        """批量追加多条消息。"""
        self.messages.extend(list(new_messages))

    def last(self) -> Optional[ChatMessage]:
        """返回最近一条消息，没有则返回 None。"""
        if not self.messages:
            return None
        return self.messages[-1]

    def snapshot(self) -> List[ChatMessage]:
        """返回当前消息列表的浅拷贝，避免外部直接修改内部状态。"""
        return list(self.messages)
