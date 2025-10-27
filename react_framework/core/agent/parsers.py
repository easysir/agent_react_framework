"""
Parser for interpreting LLM responses within the ReAct loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Union
from uuid import uuid4

from ..primitives.actions import AgentAction, AgentFinish


LOGGER = logging.getLogger(__name__)


class ReActOutputParser:
    """
    Expects the LLM to return JSON containing at least a `type` field.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        text = text.strip()
        try:
            data = json.loads(text)
            LOGGER.info("Parser received JSON: %s", data)
        except json.JSONDecodeError:
            fallback = self._parse_deepseek_markup(text)
            if fallback is not None:
                LOGGER.info("Parser used DeepSeek markup fallback for text: %s", text)
                return fallback
            stripped = text.strip()
            if stripped:
                LOGGER.info("Parser treating plain text as final answer: %s", stripped)
                return AgentFinish(output=stripped, log="__fallback_plain__")
            raise ValueError(
                "Unable to parse LLM response. Ensure the model is configured to output JSON."
            )

        kind = data.get("type")

        # Gracefully handle responses that omit the explicit `type` field but
        # still follow an "action"/"final_answer" structure.
        if not kind:
            if "action" in data or "tool" in data:
                kind = "tool"
            elif "final_answer" in data or "output" in data:
                kind = "finish"
        LOGGER.info("Parser resolved response type: %s", kind)
        if kind == "tool":
            action_block = data.get("action", {})
            tool_payload = data.get("tool")
            tool_name: Optional[str] = None
            if isinstance(tool_payload, str):
                tool_name = tool_payload
            elif isinstance(tool_payload, dict):
                tool_name = tool_payload.get("name") or tool_payload.get("tool")
            if not tool_name:
                tool_name = action_block.get("tool") or action_block.get("name")
            if not tool_name:
                raise ValueError(f"Missing tool name in response: {data}")
            tool_input: Any = (
                data.get("input")
                or action_block.get("input")
                or action_block.get("parameters")
                or (tool_payload.get("input") if isinstance(tool_payload, dict) else None)
                or (tool_payload.get("parameters") if isinstance(tool_payload, dict) else None)
                or data.get("parameters")
                or {}
            )
            if not isinstance(tool_input, dict):
                raise ValueError("Tool input must be an object/dict.")
            log = data.get("thought") or data.get("reasoning") or ""
            thoughts_block = data.get("thoughts")
            if not log and isinstance(thoughts_block, dict):
                log = thoughts_block.get("reasoning") or thoughts_block.get("plan") or ""
            if not log and thoughts_block:
                log = json.dumps(thoughts_block, ensure_ascii=False)
            call_id = (
                data.get("id")
                or action_block.get("id")
                or (tool_payload.get("id") if isinstance(tool_payload, dict) else None)
                or f"tool_call_{uuid4().hex}"
            )
            payload = self._build_tool_call_payload(call_id, tool_name, tool_input)
            LOGGER.info(
                "Parser constructed AgentAction: tool=%s call_id=%s input=%s",
                tool_name,
                call_id,
                tool_input,
            )
            return AgentAction(
                tool=tool_name,
                input=tool_input,
                log=log,
                call_id=call_id,
                tool_call_payload=payload,
            )
        if kind == "finish":
            output = data.get("final_answer") or data.get("output")
            if output is None:
                raise ValueError(f"Missing final answer in response: {data}")
            log = data.get("thought")
            LOGGER.info("Parser constructed AgentFinish with output length %d", len(str(output)))
            return AgentFinish(output=str(output), log=log)
        raise ValueError(f"Unsupported response type: {kind}")

    def _parse_deepseek_markup(self, text: str) -> Optional[AgentAction]:
        marker_begin = "<｜tool▁calls▁begin｜>"
        marker_end = "<｜tool▁calls▁end｜>"
        call_begin = "<｜tool▁call▁begin｜>"
        call_sep = "<｜tool▁sep｜>"
        call_end = "<｜tool▁call▁end｜>"
        if marker_begin not in text:
            return None
        prefix, remainder = text.split(marker_begin, 1)
        thought = prefix.strip()
        body = remainder
        if marker_end in body:
            body, _ = body.split(marker_end, 1)
        segments = []
        while call_begin in body:
            before, body = body.split(call_begin, 1)
            if call_sep not in body:
                break
            name, body = body.split(call_sep, 1)
            if call_end not in body:
                break
            args_str, body = body.split(call_end, 1)
            tool_name = name.strip()
            args_text = args_str.strip()
            if not tool_name:
                continue
            try:
                tool_input = json.loads(args_text) if args_text else {}
            except json.JSONDecodeError:
                tool_input = {"__raw": args_text}
            if not isinstance(tool_input, dict):
                tool_input = {"value": tool_input}
            call_id = f"tool_call_{uuid4().hex}"
            payload = self._build_tool_call_payload(call_id, tool_name, tool_input)
            return AgentAction(
                tool=tool_name,
                input=tool_input,
                log=thought,
                call_id=call_id,
                tool_call_payload=payload,
            )
        return None

    def _build_tool_call_payload(self, call_id: str, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(tool_input, ensure_ascii=False),
            },
        }

    def format_instruction(self) -> str:
        schema = {
            "type": "tool|finish",
            "tool": "string (required when type=tool)",
            "input": "object (arguments for the selected tool)",
            "thought": "string explaining reasoning",
            "final_answer": "string (required when type=finish)",
        }
        return json.dumps(schema, indent=2)
