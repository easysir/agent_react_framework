"""
Core message primitives shared across the agent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


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
    tool_call_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_calls: Optional[Sequence[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        payload: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        if self.tool_calls:
            payload["tool_calls"] = list(self.tool_calls)
        return payload


def system_message(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.SYSTEM, content=content)


def user_message(content: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=content)


def assistant_message(
    content: str = "",
    *,
    tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        metadata=metadata or {},
        tool_calls=tool_calls,
    )


def tool_message(
    content: str,
    name: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    tool_call_id: Optional[str] = None,
) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.TOOL,
        content=content,
        name=name,
        metadata=metadata or {},
        tool_call_id=tool_call_id,
    )


def coerce_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """Convert a list of message objects into dictionaries."""
    return [message.to_dict() for message in messages]
