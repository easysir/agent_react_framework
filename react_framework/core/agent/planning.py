"""
Task planning utilities that break user goals into manageable steps.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence

from ..primitives.memory import ConversationMemory
from ..primitives.tools import ToolRegistry
from ..primitives.messages import ChatMessage, MessageRole, system_message, user_message
from ...llm import LLMClient


@dataclass
class PlanStep:
    index: int
    description: str


@dataclass
class PlanResult:
    steps: List[PlanStep] = field(default_factory=list)

    def describe(self) -> str:
        return "\n".join(f"{step.index + 1}. {step.description}" for step in self.steps)

    def is_empty(self) -> bool:
        return len(self.steps) == 0


class TaskPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        task: str,
        *,
        memory: ConversationMemory,
        tools: ToolRegistry,
    ) -> PlanResult:
        raise NotImplementedError


PLANNER_SYSTEM_PROMPT = """You are an expert operations planner for a ReAct-style AI assistant.
Break down the user's task into concise, non-overlapping steps that can be executed sequentially.
Return the plan as JSON with a `steps` array where each entry is an object containing a `description` field.
If the task is already simple, you may return a single step."""


class LLMTaskPlanner(TaskPlanner):
    def __init__(
        self,
        llm: LLMClient,  # 用于生成计划的语言模型客户端
        *,
        max_context_messages: int = 10,  # 规划时附带的历史消息上限
        system_prompt: str = PLANNER_SYSTEM_PROMPT,  # 规划器使用的系统提示词
    ) -> None:
        self.llm = llm
        self.max_context_messages = max_context_messages
        self.system_prompt = system_prompt

    def plan(
        self,
        task: str,
        *,
        memory: ConversationMemory,
        tools: ToolRegistry,
    ) -> PlanResult:
        # 获取最近的 max_context_messages 条消息作为上下文注入
        context_messages = memory.snapshot()[-self.max_context_messages :]
        messages: List[ChatMessage] = [system_message(self.system_prompt)]
        if context_messages:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "Here is the conversation so far for context:\n"
                        + "\n".join(f"{msg.role.value}: {msg.content}" for msg in context_messages)
                    ),
                )
            )
        tool_listing = tools.describe()
        tool_section = f"\nAvailable tools:\n{tool_listing}" if tool_listing else ""
        messages.append(
            user_message(
                (
                    f"Create a plan for the following task: {task}."
                    + tool_section
                    + "\nRespond with JSON using the schema {\"steps\": [{\"description\": string}]}."
                )
            )
        )
        response = self.llm.chat(messages)
        return self._parse_plan(response.content)

    def _parse_plan(self, content: str) -> PlanResult:
        content = content.strip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fall back to parsing numbered list
            steps = [
                PlanStep(index=i, description=line.strip(" -*"))
                for i, line in enumerate(content.splitlines())
                if line.strip()
            ]
            return PlanResult(steps=steps)
        raw_steps = data.get("steps", [])
        parsed_steps: List[PlanStep] = []
        for idx, step in enumerate(raw_steps):
            if isinstance(step, dict):
                description = step.get("description") or ""
            else:
                description = str(step)
            description = description.strip()
            if description:
                parsed_steps.append(PlanStep(index=idx, description=description))
        return PlanResult(steps=parsed_steps)
