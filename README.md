# Prometheus

Prometheus 是一个面向分析师的 Agent 服务。分析师负责判断和最终决定；Agent 调用配置的聊天模型生成回答，并在返回前让同一个模型检查回答的格式、工具使用和结果是否合理。当前项目是可运行、可扩展的基础框架，还没有接入具体的分析业务。

## 当前功能与边界

- `GET /health` 返回服务状态；`POST /chat` 接收消息，返回回答和 `thread_id`。
- Agent 图支持“模型调用工具 → 工具结果返回模型”的循环，但工具注册表 `TOOLS` 目前为空，服务中还没有业务工具。
- 模型生成的回答先暂存在 `draft`，再进入 `reflect` 节点。审查输入和输出均使用 JSON，输出经过 Pydantic 校验后按 `verdict` 决定通过或修改；每轮最多完成 3 次有效审查，达到上限后即使仍要求修改，也返回当前回答。审查结束时才把最终选用的回答写入消息历史，中间废稿和审查意见不会加入 `messages`；实际工具调用及结果仍会保留。审查由同一个模型完成，不保证回答一定正确。
- 审查输出格式不合法时，基于同份材料最多补问一次；再次失败返回明确错误，不默认通过，也不提交草稿。格式补问不计入有效审查次数，不触发回答重写。
- `thread_id` 可以由请求提供，也可以由服务生成。目前服务没有接入对话记忆；重复使用同一个 `thread_id` 不会自动带入之前的对话。
- 测试使用脚本化假模型验证接口和流程，不发送真实模型请求，也不验证真实模型的回答质量。

## 本地运行

需要 Python 3.12 和 uv。首次运行时复制配置文件，并在 `.env` 中填写真实密钥。示例配置使用 Moonshot 的接口和 `kimi-k2.6`；使用其他 OpenAI 兼容接口时，同时修改 `LLM_BASE_URL` 和 `LLM_MODEL`。

```bash
cp .env.example .env
uv sync --locked
uv run --locked pytest -q
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

服务启动后，可以在另一个终端检查接口：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Hello","thread_id":"demo"}'
```

`/chat` 的 `message` 必填，长度为 1–20,000 个字符；`thread_id` 可省略，填写时只能使用字母、数字、下划线和连字符，长度为 1–64 个字符。成功响应形如 `{"thread_id":"demo","reply":"..."}`。没有密钥或密钥为空时返回 503 `missing_api_key`；上游请求超时返回 504 `upstream_timeout`，上游 API 报错返回 502 `upstream_error`，审查结果经一次格式补问后仍不合法返回 502 `review_output_invalid`，超过图步数上限返回 500 `step_limit_reached`。服务不会在模型调用失败时改用假模型。

## 运行日志

服务启动时配置 `prometheus` logger，向标准输出写入逐行 JSON。默认级别为 `INFO`，可以在
`.env` 中设置 `PROMETHEUS_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR|CRITICAL`。不新增日志文件或依赖；
容器/Databricks 等宿主负责收集、保存和轮转标准输出。上面的启动命令关闭 Uvicorn access log，
避免重复请求日志以及原始 URL/query 被默认 access log 记录。

- `service.started/stopped`：服务生命周期，启动记录模型名、工具数量和图步数上限。
- `http.started/completed`：每个请求的服务端 `request_id`、方法、路由模板、HTTP 状态码和耗时。
  响应头 `X-Request-ID` 返回同一个 ID，可据此查找整次请求的日志。未知路径只记录 `<unmatched>`。
- `chat.started/completed/failed`：增加 `thread_id`，记录审核轮数、耗时及现有 API 错误码。
  相同 `thread_id` 的不同请求仍有不同 `request_id`；这不意味着服务已启用对话记忆。
- `llm.started/completed/failed`：记录实际模型调用的 span ID、耗时、异常类型，以及供应商提供时的
  数字 token 用量。一次事件代表一次逻辑模型调用，不单独展开 SDK 内部的网络重试。
- `tool.started/completed/failed`：记录工具名、span ID 和耗时；ToolNode 将失败转换为错误结果时也记录为失败。
- `review.completed/format_retry/format_failed/limit_reached`：区分有效审查、格式补问、格式失败和到达审查上限。
  `limit_reached` 为 WARNING，表示仍要求修改但按现有规则提交当前回答，不表示审查通过。
- `http.failed`：未映射异常的类型和栈位置（文件、行号、函数），不包含异常正文或局部变量。
  `http.cancelled`：请求任务被取消；完成日志中的 499 是诊断标记，不是向断开的客户端发送响应。

示例（字段值仅作说明）：

```json
{"timestamp":"2026-10-06T12:00:00+00:00","level":"INFO","event":"chat.completed","request_id":"abc123","thread_id":"demo","duration_ms":125.4,"review_count":1}
```

应用事件不记录请求/响应正文、提示词、草稿、审查意见、工具参数/结果、密钥或上游异常正文。
Uvicorn 与第三方 SDK 的日志是独立来源，不由此 JSON formatter 接管；不要为真实业务开启会输出
完整 HTTP 请求/响应的 SDK debug 日志，宿主收集未捕获异常回溯时也需要检查其中的敏感内容。

## 代码结构

```text
app/
  config.py            模型配置和每轮图步数上限
  llm.py               构造 OpenAI 兼容的聊天模型
  schemas.py           接口请求、响应字段及 Pydantic 校验规则
  runtime_logging.py   JSON 运行日志、请求上下文与模型/工具回调
  agent/
    state.py           完整 State 字段、消息合并规则和每轮输入初始化
    graph.py           Agent、工具循环和回答审查；build_graph 预留记忆参数
    review.py          审查 JSON 输入、结果校验和有限格式重试
    prompts.py         系统提示、审查提示和修改提示
    tools/__init__.py  工具注册表 TOOLS（当前为空）
  main.py              FastAPI 接口：GET /health、POST /chat
tests/                 使用脚本化假模型的离线测试
```

要添加工具，在 `app/agent/tools/` 下编写带类型注解和文档字符串的函数，用 `langchain_core.tools.tool` 装饰，再加入 `TOOLS`，并在 `tests/` 中验证工具调用与结果返回模型的流程。

`build_graph` 已接受 `checkpointer` 和 `store` 参数，测试验证了 `InMemorySaver` 的线程内历史保存与隔离。`/chat` 目前调用图时没有传入这两个参数，因此服务还没有启用对话记忆。

## 审查协议

输入由程序生成 JSON 对象，`messages` 保存历史消息和工具证据，`draft` 单独保存当前待审回答。消息保留角色和原始内容；工具调用保留名称、参数和调用 ID，工具结果保留对应的 `tool_call_id`。这些材料不会包含无关的 token 用量和响应元数据。

模型必须返回仅含两个字段的 JSON 对象：`{"verdict":"PASS","feedback":""}` 或 `{"verdict":"REVISE","feedback":"具体问题和修改建议"}`。`verdict` 只允许这两个值；`feedback` 必须是字符串，去掉首尾空白后，PASS 必须为空、REVISE 必须非空。缺字段、多余字段、错误类型，以及 JSON 外的 Markdown 围栏或说明文字都会导致校验失败。修改意见本身可以包含 Markdown。

当前通过提示要求 JSON，并在本地严格校验；没有启用供应商专属的 JSON mode 或 JSON Schema 接口。输出结构合法不代表审查判断正确。格式补问在 `reflect` 内部单独限次，网络和 API 错误不走格式补问流程；图的 `recursion_limit` 不负责限制节点内部的补问次数。

## 字段维护

接口字段和校验规则集中在 `app/schemas.py`。`app/agent/state.py` 显式列出全部内部状态字段；`TypedDict` 提供结构声明，不会执行运行时校验，也不会自动赋默认值。

| State 字段 | 含义与更新方式 |
| --- | --- |
| `messages` | 消息历史，保留用户消息、工具调用与结果，以及每轮最终选用的回答；`add_messages` 合并新消息，遇到相同消息 ID 时替换原消息。 |
| `draft` | 当前待审回答，保留完整 `AIMessage`；新稿覆盖旧稿，调用工具期间保留，每轮初始化和最终提交后为 `None`。 |
| `answer` | 最近一次最终提交的回答文本，由 `reflect` 在审查通过或达到次数上限时写入；首次提交前可以不存在。 |
| `review` | 最近一次有效审查的 `feedback` 文本，PASS 时为空字符串；新一轮初始化为 `None`。 |
| `decision` | `continue` 表示继续修改，`stop` 表示接受回答；新一轮初始化为 `None`。 |
| `iteration` | 本轮已完成的有效审查次数，不包含格式补问；新一轮初始化为 `0`。 |

`create_turn_state(message)` 只构造当前问题的输入，并重置本轮草稿和审查字段，`/chat` 已使用它。接入记忆后，checkpointer 负责恢复历史状态，`add_messages` 负责合并本轮消息。直接调用带 checkpointer 的图时，也应通过这个函数开始新问题；仅提交 `messages` 不会自动清除旧草稿和审查状态。修订时通过临时提示传入当前草稿及最新审查意见，新的完整回答只覆盖 `draft`，直到审查结束才提交到历史。新增内部字段时，在 `state.py` 声明，并按用途更新初始化逻辑和读写该字段的节点；需要对外暴露时再更新 `schemas.py`。

checkpointer 保存 State 的快照，包括草稿和审查字段；中间快照仍可能包含废稿，但废稿不会进入正式消息历史。不需要重复声明一份业务字段清单。相关定位配置如下：

| checkpoint 配置项 | 用途 |
| --- | --- |
| `thread_id` | 区分会话，位于调用配置的 `configurable` 中。 |
| `checkpoint_ns` | checkpoint 命名空间，普通调用通常使用框架默认值。 |
| `checkpoint_id` | 定位具体快照，继续最新对话时通常不手动指定。 |

`recursion_limit` 是调用配置中的图执行步数上限，不是 State 或 checkpoint 定位字段。默认值为 20；没有工具调用时，首次审查通过至少需要 3，完整执行 3 次审查至少需要 7，工具循环还会消耗额外步数。格式补问发生在同一个 `reflect` 节点内，不消耗额外图步数。较低的自定义上限可能让本轮在完成审查前以 `step_limit_reached` 结束。

当前没有独立的 saver 创建逻辑；实际接入记忆时，再用 `app/agent/checkpointer.py` 集中管理后端及 saver 的创建、复用和关闭。
