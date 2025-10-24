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
        self.messages.append(message)

    def extend(self, new_messages: Iterable[ChatMessage]) -> None:
        self.messages.extend(list(new_messages))

    def last(self) -> Optional[ChatMessage]:
        if not self.messages:
            return None
        return self.messages[-1]

    def snapshot(self) -> List[ChatMessage]:
        return list(self.messages)
