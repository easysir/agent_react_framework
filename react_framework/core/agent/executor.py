"""
Execution loop that drives the ReAct agent using an LLM and registered tools.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import json
import re
from typing import Dict, List, Optional, Sequence, Union

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
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        task: str,
        *,
        memory: ConversationMemory,
        plan: PlanResult,
    ) -> AgentFinish:
        # 先把任务说明、计划和工具清单写入记忆，确保模型上下文完整
        self._prime_memory(memory, task, plan)

        self._logger.info(
            "\n%s\n[EXECUTION START]\nTask: %s\nMax turns: %d\n%s",
            "=" * 80,
            task,
            self.config.max_turns,
            "=" * 80,
        )
        self._logger.info(
            "\n%s\n[MEMORY SNAPSHOT]\n%s\n%s",
            "-" * 80,
            self._format_memory(memory.messages),
            "-" * 80,
        )

        for turn in range(1, self.config.max_turns + 1):
            # 每轮都携带系统提示 + 历史记忆请求 LLM
            response = self.llm.chat([system_message(self.config.system_prompt)] + memory.snapshot())
            self._logger.info(
                "\n%s\n[TURN %d] RAW LLM RESPONSE\n%s\n%s",
                "-" * 80,
                turn,
                response.content.strip(),
                "-" * 80,
            )
            # 尝试解析模型返回的 JSON，判断是工具调用还是最终回答
            parsed = self._interpret_response(response.content)
            if isinstance(parsed, AgentFinish):
                parsed = self._maybe_augment_final_answer(parsed, memory)
                self._logger.info(
                    "\n%s\n[TURN %d] FINAL ANSWER RECEIVED\n%s\n%s",
                    "=" * 80,
                    turn,
                    parsed.output.strip(),
                    "=" * 80,
                )
                memory.append(
                    assistant_message(
                        parsed.output,
                        metadata={"raw_response": response.content},
                    )
                )
                # 模型已给出最终答案，直接收尾
                return parsed
            # 模型选择调用工具，执行并将观察结果写回记忆
            self._logger.info(
                "\n%s\n[TURN %d] TOOL ACTION\nTool: %s\nInput: %s\n%s",
                "-" * 80,
                turn,
                parsed.tool,
                json.dumps(parsed.input, ensure_ascii=False, indent=2),
                "-" * 80,
            )
            tool_call_payload = parsed.tool_call_payload or {
                "id": parsed.call_id or f"tool_call_{turn}",
                "type": "function",
                "function": {
                    "name": parsed.tool,
                    "arguments": json.dumps(parsed.input, ensure_ascii=False),
                },
            }
            memory.append(
                assistant_message(
                    parsed.log or "",
                    tool_calls=[tool_call_payload],
                    metadata={"raw_response": response.content},
                )
            )
            action_result = self._execute_action(parsed, memory)
            if isinstance(action_result, AgentFinish):
                # 如果工具标记 result 直接返回，则提前结束
                return action_result
        # 超出最大轮次仍未产生结果，抛出异常提醒
        raise RuntimeError("Max turns exceeded without reaching a final answer.")

    def _prime_memory(self, memory: ConversationMemory, task: str, plan: PlanResult) -> None:
        for msg in memory.messages:
            metadata = msg.metadata if isinstance(msg.metadata, dict) else {}
            if metadata.get("agent_task_prompt") and metadata.get("task") == task:
                return
        plan_text = plan.describe() if not plan.is_empty() else ""
        tool_descriptions = self.tools.describe()
        prompt = build_react_prompt(task, plan=plan_text, tool_descriptions=tool_descriptions)
        self._logger.info(
            "\n%s\n[PRIME MEMORY]\nTask: %s\nPlan Summary:\n%s\nTools:\n%s\nPrimer Prompt:\n%s\n%s",
            "=" * 80,
            task,
            plan_text.strip() or "(no explicit plan)",
            tool_descriptions.strip() or "(no tools available)",
            prompt,
            "=" * 80,
        )
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
                tool_call_id=action.call_id,
            )
        )
        self._logger.info(
            "\n%s\n[TOOL RESULT] %s\n%s\n%s",
            "-" * 80,
            tool.name,
            observation_text.strip(),
            "-" * 80,
        )
        if tool.return_direct:
            return AgentFinish(output=observation_text, log=action.log)
        return None

    def _maybe_augment_final_answer(self, finish: AgentFinish, memory: ConversationMemory) -> AgentFinish:
        if finish.log != "__fallback_plain__":
            return finish
        summary = self._build_tool_summary(memory)
        if summary:
            finish.output = summary
            finish.log = "auto_summarized"
        return finish

    def _build_tool_summary(self, memory: ConversationMemory) -> Optional[str]:
        traces = self._collect_tool_traces(memory.messages)
        if not traces:
            return None
        lines = []
        final_result: Optional[str] = None
        for index, trace in enumerate(traces, start=1):
            tool_name = trace.get("tool", "tool")
            expression = trace.get("expression") or trace.get("arguments_raw") or ""
            result = trace.get("result", "")
            lines.append(f"步骤 {index}: 使用 {tool_name} 计算 {expression} → {result}")
            final_result = result
        if not lines:
            return None
        summary = "\n".join(lines)
        result_value = self._extract_result_value(final_result) if final_result else None
        if result_value:
            summary += f"\n\n最终答案: {result_value}"
        elif final_result:
            summary += f"\n\n最终答案: {final_result}"
        return summary

    def _collect_tool_traces(self, messages: Sequence[ChatMessage]) -> Sequence[dict]:
        pending: Dict[str, dict] = {}
        traces: List[dict] = []
        for message in messages:
            if message.role is MessageRole.ASSISTANT and message.tool_calls:
                for call in message.tool_calls:
                    if call.get("type") != "function":
                        continue
                    call_id = call.get("id")
                    function_block = call.get("function", {})
                    arguments_text = function_block.get("arguments") or ""
                    try:
                        arguments = json.loads(arguments_text) if arguments_text else {}
                    except json.JSONDecodeError:
                        arguments = {"__raw": arguments_text}
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments}
                    pending[call_id] = {
                        "tool": function_block.get("name", "tool"),
                        "arguments": arguments,
                        "arguments_raw": arguments_text,
                    }
            elif message.role is MessageRole.TOOL and message.tool_call_id:
                trace = pending.get(message.tool_call_id)
                if trace:
                    trace_with_result = dict(trace)
                    trace_with_result["result"] = message.content.strip()
                    trace_with_result["call_id"] = message.tool_call_id
                    expression = trace_with_result["arguments"].get("expression")
                    if expression:
                        trace_with_result["expression"] = expression
                    traces.append(trace_with_result)
        return traces

    def _extract_result_value(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        match = re.search(r"(-?\d+(?:\.\d+)?)", text)
        if match:
            return match.group(1)
        return None

    def _format_memory(self, messages: Sequence[ChatMessage]) -> str:
        if not messages:
            return "(memory is empty)"
        lines = []
        for index, message in enumerate(messages, start=1):
            snippet = message.content.strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = f"{snippet[:157]}..."
            lines.append(f"{index:02d}. {message.role.value.upper()}: {snippet}")
        return "\n".join(lines)
