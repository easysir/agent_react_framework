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
   pip install -r requirements.txt
   ```

2. 设置你要使用的模型对应的环境变量，例如：
   ```bash
   export OPENAI_API_KEY=sk-...
   export DEEPSEEK_API_KEY=sk-...
   export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选
   ```

3. 运行示例（推荐使用包模块方式，避免路径问题）：
   ```bash
   python -m examples.basic_usage
   ```
   该脚本会：
   - 构建一个带 `calculator` 工具的 Agent；
   - 使用 LLM 先生成计划，再迭代执行 ReAct 循环；
   - 在控制台输出增强后的结构化日志与最终总结。

## 目录结构

```
react_framework/
  core/
    agent/        # Agent 主流程、计划器、执行器、解析器
    primitives/   # 消息、工具、记忆等基础数据结构
  llm/            # 不同模型客户端适配
  __init__.py     # 对外暴露 API
```

## 主要模块说明

- `Agent`：高层接口，组合计划器、工具注册表、执行器与对话记忆。
- `LLMTaskPlanner`：使用规划提示词，要求模型返回 `{"steps": [...]}` JSON，并在解析失败时回退到编号列表。
- `AgentExecutor`：驱动 ReAct 循环，解析模型动作、执行工具、收集观察，并产生结构化运行日志。
- `ToolRegistry`：管理工具生命周期，支持灵活扩展。
- `create_chat_completion_client`：快速实例化指向 OpenAI / DeepSeek / Qwen 等兼容接口的客户端，支持自定义或扩展新的 provider。

## 响应协议与兼容性

- **标准 JSON 协议**：系统提示会明确要求模型仅返回单个 JSON 对象：
  - 工具调用示例：
    ```json
    {
      "type": "tool",
      "thought": "I should use the calculator to add the numbers.",
      "tool": "calculator",
      "input": { "expression": "24 + 18" }
    }
    ```
  - 最终回答示例：
    ```json
    {
      "type": "finish",
      "thought": "I have the final result.",
      "final_answer": "Step 1: 24 + 18 = 42. Step 2: 42 * 0.75 = 31.5."
    }
    ```
- **DeepSeek 兼容**：解析器 `_parse_deepseek_markup` 对 DeepSeek 专用的 `<｜tool▁calls▁begin｜>` 标记做了适配，可自动还原为标准工具调用。
- **纯文本兜底**：若模型最终返回自然语言而非 JSON，解析器会将其视为 `AgentFinish`，执行器会回溯工具轨迹生成带步骤与结果的最终总结。
- **日志增强**：运行过程中会输出 `[PLAN]`、`[TURN n] TOOL ACTION`、`[LLM REQUEST]` 等分隔块，辅助排查与回放。

## 扩展与自定义

- **自定义计划器**：实现 `TaskPlanner` 并传入 `AgentConfig`。
- **替换解析逻辑**：实现自己的输出解析器，确保模型遵循对应 JSON 协议。
- **新增模型**：继承 `LLMClient` 或基于 `OpenAICompatibleClient` 封装新的 HTTP 客户端。
- **扩展 Provider**：调用 `register_provider(ProviderSpec(...))` 注册自定义的 OpenAI-Compatible 服务端。
- **状态持久化**：`Agent.run` 可传入或复用 `ConversationMemory`，实现跨轮对话。
- **Prompt 定制**：若需要进一步约束模型输出，可修改 `react_framework/core/agent/prompts.py` 的系统提示或在构造 `AgentConfig` 时自定义 `ExecutorConfig.system_prompt`。

## 环境变量约定

| 模型     | 必填键             | 可选键                     |
|----------|--------------------|----------------------------|
| OpenAI   | `OPENAI_API_KEY`   | `OPENAI_BASE_URL`, `OPENAI_ORG_ID` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL`        |
| Qwen     | `QWEN_API_KEY`     | `QWEN_BASE_URL`, `QWEN_WORKSPACE` |

## 示例

更多可拓展示例，可参考 `examples/` 目录，或直接在业务代码中按需组合工具与模型。
