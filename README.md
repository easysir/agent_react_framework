# ReAct Agent Framework

一个轻量级、模块化的 ReAct 智能体框架，支持：
- 任务拆分（LLM 驱动的计划器）
- 工具注册与调用
- 多轮 ReAct 推理循环（思考 → 工具调用 → 观察 → 再思考）
- 多模型适配（OpenAI、DeepSeek、Qwen 等 OpenAI-Compatible Chat Completion API）

> 所有 API Key 会自动从环境变量中读取，无需硬编码。

## 快速开始

1. 安装依赖：
   ```bash
   pip install requests
   ```
   （如需调用 OpenAI 官方 SDK，可自行替换成 `openai` 版本）

2. 设置你要使用的模型对应的环境变量，例如：
   ```bash
   export OPENAI_API_KEY=sk-...
   export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
   ```

3. 构建并运行智能体：
   ```python
   from react_framework import Agent, AgentConfig, Tool
   from react_framework.core.tools import ToolResult
   from react_framework.llm import OpenAIChatClient


   def search_tool(arguments: dict) -> ToolResult:
       query = arguments["query"]
       # 这里可以接入真实的搜索服务
       return ToolResult(content=f"搜索到的结果: {query[:20]} ...")


   llm = OpenAIChatClient(model="gpt-4o-mini")
   agent = Agent(
       AgentConfig(
           llm=llm,
           tools=[
               Tool(
                   name="search",
                   description="进行网络搜索",
                   schema={"type": "object", "properties": {"query": {"type": "string"}}},
                   func=search_tool,
               )
           ],
       )
   )

   result = agent.run("帮我找出最新的三条 AI 新闻，并总结要点")
   print(result.final_answer)
   ```

## 目录结构

```
react_framework/
  agent.py        # 对外暴露的 Agent API
  executor.py     # ReAct 循环执行器
  planning.py     # 任务规划器（LLM 驱动）
  prompts.py      # 系统 / ReAct Prompt 模板
  output_parser.py# 模型输出解析器
  core/           # 基础类型（消息、工具、内存等）
  llm/            # 不同模型客户端适配
```

## 主要模块说明

- `Agent`：高层接口，组合计划器、工具注册表、执行器与对话记忆。
- `LLMTaskPlanner`：使用 LLM 将用户任务拆分为步骤计划。
- `AgentExecutor`：驱动 ReAct 循环，解析模型动作、执行工具、收集观察。
- `ToolRegistry`：管理工具生命周期，支持灵活扩展。
- `OpenAIChatClient` / `DeepSeekChatClient` / `QwenChatClient`：针对常见兼容接口的 LLM 客户端。

## 扩展与自定义

- **自定义计划器**：实现 `TaskPlanner` 并传入 `AgentConfig`。
- **替换解析逻辑**：实现自己的输出解析器，确保模型遵循对应 JSON 协议。
- **新增模型**：继承 `LLMClient` 或基于 `OpenAICompatibleClient` 封装新的 HTTP 客户端。
- **状态持久化**：`Agent.run` 可传入或复用 `ConversationMemory`，实现跨轮对话。

## 环境变量约定

| 模型     | 必填键             | 可选键                     |
|----------|--------------------|----------------------------|
| OpenAI   | `OPENAI_API_KEY`   | `OPENAI_BASE_URL`, `OPENAI_ORG_ID` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL`        |
| Qwen     | `QWEN_API_KEY`     | `QWEN_BASE_URL`, `QWEN_WORKSPACE` |

## 示例

更多可拓展示例，可参考 `examples/` 目录，或直接在业务代码中按需组合工具与模型。

