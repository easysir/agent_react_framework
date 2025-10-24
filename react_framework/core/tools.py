"""
Utilities for registering and invoking tools in the agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


class ToolExecutionError(RuntimeError):
    """Raised when a tool invocation fails."""


@dataclass
class ToolResult:
    """Structured response produced by a tool."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


ToolCallable = Callable[[Dict[str, Any]], ToolResult]


@dataclass
class Tool:
    name: str
    description: str
    func: ToolCallable
    schema: Optional[Dict[str, Any]] = None
    return_direct: bool = False

    def __call__(self, arguments: Dict[str, Any]) -> ToolResult:
        return self.func(arguments)


class ToolRegistry:
    """In-memory registry responsible for resolving tool instances."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool named '{tool.name}' already registered.")
        self._tools[tool.name] = tool

    def update(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool)

    def remove(self, name: str) -> None:
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool '{name}'.") from exc

    def names(self) -> Iterable[str]:
        return list(self._tools.keys())

    def describe(self) -> str:
        """Return a prompt-friendly description of registered tools."""
        blocks = []
        for tool in self._tools.values():
            schema_part = f"\n  schema: {tool.schema}" if tool.schema else ""
            blocks.append(f"- {tool.name}: {tool.description}{schema_part}")
        return "\n".join(blocks)
