"""
Execution loop that drives the ReAct agent using an LLM and registered tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from ..primitives.actions import AgentAction, AgentFinish
from ..primitives.memory import ConversationMemory
from ..primitives.messages import (
    ChatMessage,
    MessageRole,
    assistant_message,
    system_message,
    tool_message,
)
from ..primitives.tools import ToolExecutionError, ToolRegistry
from ...llm import LLMClient
from .parsers import ReActOutputParser
from .planning import PlanResult
from .prompts import DEFAULT_SYSTEM_PROMPT, build_react_prompt


@dataclass
class ExecutorConfig:
    max_turns: int = 12
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    observation_as_metadata: bool = True


class AgentExecutor:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        *,
        parser: Optional[ReActOutputParser] = None,
        config: Optional[ExecutorConfig] = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.parser = parser or ReActOutputParser()
        self.config = config or ExecutorConfig()

    def run(
        self,
        task: str,
        *,
        memory: ConversationMemory,
        plan: PlanResult,
    ) -> AgentFinish:
        self._prime_memory(memory, task, plan)
        for _ in range(self.config.max_turns):
            response = self.llm.chat([system_message(self.config.system_prompt)] + memory.snapshot())
            memory.append(assistant_message(response.content))
            parsed = self._interpret_response(response.content)
            if isinstance(parsed, AgentFinish):
                return parsed
            action_result = self._execute_action(parsed, memory)
            if isinstance(action_result, AgentFinish):
                return action_result
        raise RuntimeError("Max turns exceeded without reaching a final answer.")

    def _prime_memory(self, memory: ConversationMemory, task: str, plan: PlanResult) -> None:
        for msg in memory.messages:
            metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
            if metadata.get("agent_task_prompt") and metadata.get("task") == task:
                return
        plan_text = plan.describe() if not plan.is_empty() else ""
        tool_descriptions = self.tools.describe()
        prompt = build_react_prompt(task, plan=plan_text, tool_descriptions=tool_descriptions)
        memory.append(
            ChatMessage(
                role=MessageRole.USER,
                content=prompt,
                metadata={"agent_task_prompt": True, "task": task},
            )
        )

    def _interpret_response(self, content: str) -> Union[AgentAction, AgentFinish]:
        try:
            return self.parser.parse(content)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse model response: {content}") from exc

    def _execute_action(
        self,
        action: AgentAction,
        memory: ConversationMemory,
    ) -> Optional[AgentFinish]:
        tool = self.tools.get(action.tool)
        try:
            result = tool(action.input)
        except Exception as exc:
            raise ToolExecutionError(f"Tool '{tool.name}' failed: {exc}") from exc
        observation_text = result.content
        memory.append(
            tool_message(
                observation_text,
                name=tool.name,
                metadata=result.metadata,
            )
        )
        if tool.return_direct:
            return AgentFinish(output=observation_text, log=action.log)
        return None
