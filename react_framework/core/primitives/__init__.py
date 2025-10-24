"""
Foundational data structures shared across the framework.
"""

from .actions import AgentAction, AgentFinish
from .memory import ConversationMemory
from .messages import (
    ChatMessage,
    MessageRole,
    assistant_message,
    coerce_messages,
    system_message,
    tool_message,
    user_message,
)
from .tools import Tool, ToolCallable, ToolExecutionError, ToolRegistry, ToolResult

__all__ = [
    "AgentAction",
    "AgentFinish",
    "ConversationMemory",
    "ChatMessage",
    "MessageRole",
    "assistant_message",
    "coerce_messages",
    "system_message",
    "tool_message",
    "user_message",
    "Tool",
    "ToolCallable",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResult",
]
