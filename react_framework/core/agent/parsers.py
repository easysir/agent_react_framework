"""
Parser for interpreting LLM responses within the ReAct loop.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Union

from ..primitives.actions import AgentAction, AgentFinish


class ReActOutputParser:
    """
    Expects the LLM to return JSON containing at least a `type` field.
    """

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        text = text.strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(
                "Unable to parse LLM response. Ensure the model is configured to output JSON."
            )

        kind = data.get("type")
        if kind == "tool":
            tool_name = data.get("tool")
            if not tool_name:
                raise ValueError(f"Missing tool name in response: {data}")
            tool_input = data.get("input") or {}
            if not isinstance(tool_input, dict):
                raise ValueError("Tool input must be an object/dict.")
            log = data.get("thought", "")
            return AgentAction(tool=tool_name, input=tool_input, log=log)
        if kind == "finish":
            output = data.get("final_answer") or data.get("output")
            if output is None:
                raise ValueError(f"Missing final answer in response: {data}")
            log = data.get("thought")
            return AgentFinish(output=str(output), log=log)
        raise ValueError(f"Unsupported response type: {kind}")

    def format_instruction(self) -> str:
        schema = {
            "type": "tool|finish",
            "tool": "string (required when type=tool)",
            "input": "object (arguments for the selected tool)",
            "thought": "string explaining reasoning",
            "final_answer": "string (required when type=finish)",
        }
        return json.dumps(schema, indent=2)
