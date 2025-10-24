"""
High-level Agent interface that coordinates planning, tool execution, and LLM interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..primitives.actions import AgentFinish
from ..primitives.memory import ConversationMemory
from ..primitives.messages import ChatMessage, MessageRole
from ..primitives.tools import Tool, ToolRegistry
from .executor import AgentExecutor, ExecutorConfig
from .parsers import ReActOutputParser
from .planning import LLMTaskPlanner, PlanResult, TaskPlanner
from ...llm import LLMClient


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
        task: str,  # 用户希望智能体完成的任务描述
        *,
        memory: Optional[ConversationMemory] = None,  # 可选外部记忆实例，用于复用上下文
    ) -> AgentRunResult:
        if memory is not None:
            # 调用方提供现成的记忆实例，则直接复用
            working_memory = memory
        elif self.config.reset_memory_each_run:
            # 配置为每次运行重置记忆，创建全新会话
            working_memory = ConversationMemory()
        else:
            # 否则沿用 Agent 内部累计的历史记忆，实现多轮连续对话
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
