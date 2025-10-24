"""
Prompt templates for the ReAct agent loop.
"""

from __future__ import annotations

from textwrap import dedent


DEFAULT_SYSTEM_PROMPT = dedent(
    """
    You are an AI assistant that can reason about complex tasks and call tools.
    Follow the ReAct pattern: think about what to do, decide whether to call a tool,
    observe the result, and continue until you can provide a final answer.
    Always respond strictly in JSON matching the documented schema.
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

        Respond with JSON describing either a tool call or the final answer.
        """
    ).strip()
