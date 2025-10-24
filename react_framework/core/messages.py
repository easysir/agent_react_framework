"""
Core message primitives shared across the agent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional


class MessageRole(str, Enum):
    """Canonical chat roles accepted by the framework."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """
    Minimal representation of a chat message.

    The structure mirrors common OpenAI-compatible schemas and is easily
    serializable to JSON for prompt construction.
    """

    role: MessageRole
    content: str
    name: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        payload: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def system_message(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.SYSTEM, content=content)


def user_message(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=content)


def assistant_message(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=content)


def tool_message(content: str, name: str, *, metadata: Optional[Mapping[str, Any]] = None) -> ChatMessage:
    return ChatMessage(role=MessageRole.TOOL, content=content, name=name, metadata=metadata or {})


def coerce_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert a list of message objects into dictionaries."""
    return [message.to_dict() for message in messages]
