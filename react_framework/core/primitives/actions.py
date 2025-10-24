"""
Definitions of agent actions emitted by the planner/executor loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AgentAction:
    """
    Indicates that the agent should execute a tool invocation.
    """

    tool: str
    input: Dict[str, Any]
    log: str


@dataclass
class AgentFinish:
    """
    Signals that the agent has produced a final answer.
    """

    output: str
    log: Optional[str] = None
