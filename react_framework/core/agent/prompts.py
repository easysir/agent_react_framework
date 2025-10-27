"""
Prompt templates for the ReAct agent loop.
"""

from __future__ import annotations

from textwrap import dedent


DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are an AI assistant that must reason about tasks and optionally call tools.
    Follow the ReAct pattern: think, decide on an action, observe results, and iterate until you can finish.

    Output Requirements:
    - Reply with a single JSON object only (no prose before or after).
    - Valid tool call response (example):
      {
        "type": "tool",
        "thought": "I should use the calculator to add the numbers.",
        "tool": "calculator",
        "input": {
          "expression": "24 + 18"
        }
      }
    - Valid final answer response (example):
      {
        "type": "finish",
        "thought": "I have the final result and can explain it.",
        "final_answer": "Step 1: 24 + 18 = 42. Step 2: 42 * 0.75 = 31.5."
      }
    - Do not emit any other keys. Ensure the JSON is valid (double quotes, proper commas).
    """
).strip()


def build_react_prompt(
    task: str,
    *,
    plan: str,
    tool_descriptions: str,
) -> str:
    tool_section = tool_descriptions.strip() if tool_descriptions else "No tools available."
    plan_section = plan.strip() if plan else "No explicit plan was created."
    return dedent(
        f"""
        Task: {task}

        Current plan:
        {plan_section}

        Available tools:
        {tool_section}

        Respond with one JSON object that follows the schema in the system instructions.
        """
    ).strip()
