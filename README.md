# Prometheus

Prometheus 是一个面向分析师的 Agent 服务。分析师负责判断和最终决定；Agent 调用配置的聊天模型生成回答，并在返回前让同一个模型检查回答的格式、工具使用和结果是否合理。当前项目是可运行、可扩展的基础框架，还没有接入具体的分析业务。

## 当前功能与边界

- `GET /health` 返回服务状态；`POST /chat` 接收消息，返回回答和 `thread_id`。
- Agent 图支持“模型调用工具 → 工具结果返回模型”的循环，但工具注册表 `TOOLS` 目前为空，服务中还没有业务工具。
- 模型生成最终回答后会进入 `reflect` 节点。只有审查结果明确以 `REVISE` 开头时才会修改回答；每轮最多审查 3 次，达到上限后返回当前回答。审查由同一个模型完成，不保证回答一定正确。
- `thread_id` 可以由请求提供，也可以由服务生成。目前服务没有接入对话记忆；重复使用同一个 `thread_id` 不会自动带入之前的对话。
- 测试使用脚本化假模型验证接口和流程，不发送真实模型请求，也不验证真实模型的回答质量。

旧版 v0.1 报告审阅流程已存档。背景与取回方式见 [docs/decisions.md](docs/decisions.md) 的 ADR-001；后续改名和当前仓库的说明见 ADR-006。

## 本地运行

需要 Python 3.12 和 uv。首次运行时复制配置文件，并在 `.env` 中填写真实密钥。示例配置使用 Moonshot 的接口和 `kimi-k2.6`；使用其他 OpenAI 兼容接口时，同时修改 `LLM_BASE_URL` 和 `LLM_MODEL`。

```bash
cp .env.example .env
uv sync --locked
uv run --locked pytest -q
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
```

服务启动后，可以在另一个终端检查接口：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","thread_id":"demo"}'
```

`/chat` 的 `message` 必填，长度为 1–20,000 个字符；`thread_id` 可省略，填写时只能使用字母、数字、下划线和连字符，长度为 1–64 个字符。成功响应形如 `{"thread_id":"demo","reply":"..."}`。没有密钥或密钥为空时返回 503 `missing_api_key`；上游请求超时返回 504 `upstream_timeout`，上游 API 报错返回 502 `upstream_error`，超过图步数上限返回 500 `step_limit_reached`。服务不会在模型调用失败时改用假模型。

## 代码结构

```text
app/
  config.py            模型配置和每轮图步数上限
  llm.py               构造 OpenAI 兼容的聊天模型
  agent/
    graph.py           Agent、工具循环和回答审查；build_graph 预留记忆参数
    prompts.py         系统提示、审查提示和修改提示
    tools/__init__.py  工具注册表 TOOLS（当前为空）
  main.py              FastAPI 接口：GET /health、POST /chat
tests/                 使用脚本化假模型的离线测试
docs/decisions.md      项目决策记录
```

要添加工具，在 `app/agent/tools/` 下编写带类型注解和文档字符串的函数，用 `langchain_core.tools.tool` 装饰，再加入 `TOOLS`，并在 `tests/` 中验证工具调用与结果返回模型的流程。

`build_graph` 已接受 `checkpointer` 和 `store` 参数，测试验证了 `InMemorySaver` 的线程内历史保存与隔离。`/chat` 目前调用图时没有传入这两个参数，因此服务还没有启用对话记忆。
