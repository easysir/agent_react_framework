"""
High-level Agent interface that coordinates planning, tool execution, and LLM interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .core.actions import AgentFinish
from .core.memory import ConversationMemory
from .core.messages import ChatMessage, MessageRole
from .core.tools import Tool, ToolRegistry
from .executor import AgentExecutor, ExecutorConfig
from .llm import LLMClient
from .output_parser import ReActOutputParser
from .planning import LLMTaskPlanner, PlanResult, TaskPlanner


@dataclass
class AgentRunResult:
    task: str
    final_answer: str
    plan: PlanResult
    memory: Sequence[ChatMessage]
    reasoning_log: Optional[str] = None


@dataclass
class AgentConfig:
    llm: LLMClient
    tools: Iterable[Tool] = field(default_factory=list)
    planner: Optional[TaskPlanner] = None
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    parser: Optional[ReActOutputParser] = None
    reset_memory_each_run: bool = True


class Agent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.llm = config.llm
        self.tools = ToolRegistry()
        self.tools.update(config.tools)
        self.planner = config.planner or LLMTaskPlanner(self.llm)
        self.executor = AgentExecutor(
            self.llm,
            self.tools,
            parser=config.parser,
            config=config.executor_config,
        )
        self._memory = ConversationMemory()

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def add_tool(self, tool: Tool) -> None:
        self.tools.register(tool)

    def list_tools(self) -> Sequence[str]:
        return list(self.tools.names())

    def run(
        self,
        task: str,
        *,
        memory: Optional[ConversationMemory] = None,
    ) -> AgentRunResult:
        if memory is not None:
            working_memory = memory
        elif self.config.reset_memory_each_run:
            working_memory = ConversationMemory()
        else:
            working_memory = self._memory
        plan = self.planner.plan(task, memory=working_memory, tools=self.tools)
        finish = self.executor.run(task, memory=working_memory, plan=plan)
        self._record_final_answer(working_memory, finish)
        if self.config.reset_memory_each_run:
            self._memory = working_memory
        return AgentRunResult(
            task=task,
            final_answer=finish.output,
            plan=plan,
            memory=working_memory.snapshot(),
            reasoning_log=finish.log,
        )

    def _record_final_answer(self, memory: ConversationMemory, finish: AgentFinish) -> None:
        memory.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=finish.output,
                metadata={"agent_final_answer": True, "log": finish.log},
            )
        )
