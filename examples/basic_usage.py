"""
Basic usage example for the ReAct agent framework.
"""

import logging

from react_framework import Agent, AgentConfig, Tool
from react_framework.core.primitives import ToolResult
from react_framework.llm import create_chat_completion_client


def calculator(arguments: dict) -> ToolResult:
    expression = arguments["expression"]
    try:
        # Unsafe eval should be replaced with a safe parser in production.
        value = eval(expression, {"__builtins__": {}})
    except Exception as exc:  # pragma: no cover - demo code
        return ToolResult(content=f"无法计算表达式: {exc}")
    return ToolResult(content=f"计算结果: {value}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    llm = create_chat_completion_client("deepseek", model="deepseek-chat")
    agent = Agent(
        AgentConfig(
            llm=llm,
            tools=[
                Tool(
                    name="calculator",
                    description="计算简单的数学表达式",
                    schema={"type": "object", "properties": {"expression": {"type": "string"}}},
                    func=calculator,
                )
            ],
        )
    )
    task = "请计算(24 + 18) * 0.75，然后解释每一步推导。"
    result = agent.run(task)
    print("Final answer:", result.final_answer)
    print("Plan:")
    for step in result.plan.steps:
        print(f"- {step.description}")


if __name__ == "__main__":
    main()
