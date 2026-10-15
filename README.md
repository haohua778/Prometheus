# Prometheus

Prometheus 是一个面向分析师的 Agent 服务。分析师负责判断和最终决定；Agent 调用配置的聊天模型生成回答，并在返回前让同一个模型检查回答的格式、工具使用和结果是否合理。当前项目是可运行、可扩展的基础框架，还没有接入具体的分析业务。

## 当前功能与边界

- `GET /health` 返回服务状态；`POST /chat` 接收消息，返回回答和 `thread_id`。
- Agent 图支持“模型调用工具 → 工具结果返回模型”的循环，但工具注册表 `TOOLS` 目前为空，服务中还没有业务工具。
- 模型生成的回答先暂存在 `draft`，再进入 `reflect` 节点。审查输入和输出均使用 JSON，输出经过 Pydantic 校验后按 `verdict` 决定通过或修改；每轮最多完成 3 次有效审查，达到上限后即使仍要求修改，也返回当前回答。审查结束时才把最终选用的回答写入消息历史，中间废稿和审查意见不会加入 `messages`；实际工具调用及结果仍会保留。审查由同一个模型完成，不保证回答一定正确。
- 审查输出格式不合法时，基于同份材料最多补问一次；再次失败返回明确错误，不默认通过，也不提交草稿。格式补问不计入有效审查次数，不触发回答重写。
- `thread_id` 可以由请求提供，也可以由服务生成。目前服务没有接入对话记忆；重复使用同一个 `thread_id` 不会自动带入之前的对话。
- 现有接口和图流程测试使用脚本化模型；日志测试使用真实 HTTP 服务和真实工具调用。真实模型日志测试需要显式开启，不评估回答质量。

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

### 输出与配置

服务启动时配置 `prometheus` logger，每个事件输出一行 JSON 到 `stdout`（标准输出）。
本地运行时显示在启动后端的终端中；应用默认不创建 `.log` 文件，也不按会话创建文件。
日志使用 Python 标准库，模型和工具事件通过现有 LangChain 回调记录。

在 `.env` 中配置日志级别，修改后重新启动服务：

```env
PROMETHEUS_LOG_LEVEL=INFO
```

默认 `INFO` 保留正常执行记录、提醒和错误。可选值为 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。
选择 `WARNING` 或更高等级会过滤正常执行记录。此配置只控制应用自己的 logger。
启动命令中的 `--no-access-log` 关闭 Uvicorn access log，避免重复记录请求和原始 URL/query。

### 一次请求怎样记录

1. HTTP 中间件生成 `request_id`，记录请求开始和处理结束，并在正常返回的响应头中附上 `X-Request-ID`。
2. `/chat` 确定 `thread_id`，记录对话开始，并为这次图执行注册 `RuntimeLogCallback`。
3. 框架在模型、工具调用开始或结束时触发回调，自动记录执行结果和耗时。生成回答与 Reflection 的模型调用都会记录。
4. 审查代码记录审查结论、格式补问或审查上限；对话完成时记录处理耗时和审查轮数。
5. 请求结束时收尾尚未完成的调用记录，清理关联 ID。服务进程退出或被强制终止时，无法保证最后的事件已输出。

每次 HTTP 请求会产生多条日志。通过同一个 `request_id` 查找这一轮的全部事件；
通过 `thread_id` 查找同一对话线程的多次请求。当前服务尚未启用对话记忆。
`http.*` 事件的 `thread_id` 通常为 `null`，仍可通过 `request_id` 与对话事件关联。

| 字段 | 含义 |
| --- | --- |
| `timestamp`、`level`、`event` | UTC 时间、日志级别和事件名称，每条日志都包含。 |
| `request_id` | 服务端为每次 HTTP 请求生成的 ID；服务启动/停止事件为 `null`。 |
| `thread_id` | 对话线程 ID，由请求提供或由服务生成；使用非敏感标识。 |
| `span_id`、`parent_span_id` | 模型或工具调用的 ID；开始事件还记录框架提供的上级调用 ID。 |
| `duration_ms` | 结束事件中的耗时，单位为毫秒。HTTP 耗时表示后端请求处理时间，不包含完整网络传输时间。 |
| `tool_name` | 工具名称，工具开始和结束事件都包含。 |
| `error_code`、`error_type`、`stack` | 已知 API 错误码、异常类型和异常调用位置；按事件提供。 |
| `input_tokens`、`output_tokens`、`total_tokens` | 模型供应商提供时记录的数字 token 用量。 |

示例日志（字段值仅作说明）：

```json
{"timestamp":"2026-10-06T12:00:00+00:00","level":"INFO","event":"chat.completed","request_id":"abc123","thread_id":"demo","duration_ms":125.4,"review_count":1}
```

### 事件与排错

| 事件 | 用途 |
| --- | --- |
| `service.started/stopped` | 服务生命周期。启动记录模型名、工具数量和图步数上限。 |
| `http.started/completed` | HTTP 请求及其处理结果。结束记录方法、路由模板、状态码和耗时；未知路径使用 `<unmatched>`。 |
| `chat.started/completed/failed` | 对话开始、完成或已知错误。完成记录审查轮数，失败记录 API 错误码。 |
| `llm.started/completed/failed` | 一次逻辑模型调用及其耗时、token 用量或异常位置；SDK 内部网络重试包含在该调用中。 |
| `tool.started/completed/failed` | 工具执行结果和耗时。ToolNode 将异常转换为错误结果时也记录为失败，`outcome=error_result`。 |
| `review.completed` | 有效审查结论和轮数。`verdict=PASS` 表示模型接受回答，`REVISE` 表示需要修改；结论不保证回答正确。 |
| `review.format_retry/format_failed` | 审查格式不合法时的补问与最终失败。 |
| `review.limit_reached` | WARNING：达到审查次数上限，按业务规则提交当前回答；此时仍可能是 REVISE。 |
| `http.failed/cancelled` | 未映射异常或请求任务取消。取消对应的 499 仅作为日志诊断标记。 |

请求结束时仍未收到完成回调的模型或工具调用记录为失败，`outcome=interrupted`。
例如，取消请求可能跳过框架的结束回调。该字段表示调用记录未完整结束。

排错时查找出错响应的 `X-Request-ID`，定位同 ID 的 `chat.failed`、`llm.failed`、`tool.failed` 或 `http.failed`。
未预期异常可能由服务器直接生成响应，响应头不保证包含该 ID，此时从日志中的时间和请求记录定位。
已知错误码对应以下检查位置：

| 错误码 | HTTP 状态 | 检查位置 |
| --- | --- | --- |
| `missing_api_key` | 503 | `.env` 的模型密钥配置。 |
| `upstream_timeout` | 504 | 模型接口耗时和 `LLM_TIMEOUT_SECONDS`。 |
| `upstream_error` | 502 | 模型服务连接、认证和调用配置；关联 `llm.failed` 查看异常类型及代码位置。 |
| `review_output_invalid` | 502 | 审查输出协议，以及 `review.format_retry/format_failed` 事件。 |
| `step_limit_reached` | 500 | 图执行流程和 `PROMETHEUS_MAX_STEPS`。 |

### 新增工具和业务事件

通过 LangChain 工具机制注册到 `TOOLS`、由当前图执行的工具，会自动触发 `RuntimeLogCallback`，
通常不需要修改日志模块。通过普通 Python 调用直接执行的函数，需要在其执行入口接入记录。

需要记录业务自己掌握的结果时，调用 `log_event()`。例如，文档解析完成处可采用以下写法：

```python
from app.runtime_logging import log_event

log_event('document.parsed', page_count=12)
```

示例中的页数仅作说明，实际使用解析结果。业务位置提供事件名和必要字段；
JSON 格式、输出方式和调用计时集中在 `app/runtime_logging.py`，HTTP 请求记录集中在 `app/main.py`。
调整日志保存方式应修改输出配置或采集配置。

### 敏感信息与保存范围

应用事件不记录请求/响应正文、提示词、草稿、审查意见、工具参数/结果、密钥或上游异常正文。
异常只记录类型和代码位置，省略局部变量。新增业务事件也需要遵守这个字段范围。
Uvicorn 与第三方 SDK 的日志独立于此 JSON formatter；启用完整 HTTP debug 输出或收集服务器异常回溯时，
需要检查其中的敏感内容。

Databricks Apps 可以采集部署进程的 stdout/stderr，在应用 Logs 页面展示。
默认 Logs 页面在应用计算资源关闭后不保留日志；长期保存需要额外配置平台采集与持久存储。
当前代码没有配置 Databricks 持久化或 LangSmith。平台接入时需确认自定义应用日志实际进入保存位置。
参考 [Apps 日志说明](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/monitor) 和
[App Telemetry 配置](https://docs.databricks.com/aws/en/dev-tools/databricks-apps/observability)。

当前记录单次耗时，p95/p99 需要在保存的日志或监控系统中汇总计算。
日志测试产生的 `.runtime/` 文件是检查产物，已由 `.gitignore` 忽略。

### 验证日志

运行真实 Uvicorn HTTP 请求和真实工具检查：

```bash
uv run --locked pytest tests/test_runtime_logging.py -q
```

默认跳过真实模型测试。填写 `.env` 后，可在 PowerShell 显式开启：

```powershell
$env:PROMETHEUS_LIVE_LOG_TEST = '1'
uv run --locked pytest tests/test_runtime_logging.py -q
Remove-Item Env:PROMETHEUS_LIVE_LOG_TEST
```

真实模型测试执行一次对话及其 Reflection，会产生供应商用量费用。

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
tests/                 接口、图流程与运行日志测试
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

## Coding 计算核查子 Agent

通过 `app.agent.coding.run_check` 接收结构化核查任务，基于授权 CSV/JSON 数据执行 Python，返回结论与证据。接口仍为本地草案，需要与 Review Agent 双方确认。真实 Docker 容器与真实模型的核查测试已在 Linux 上通过。

主 Agent 需要通过工具选择计算核查时，使用 `make_check_tool(authorized_files=...)` 创建本次审核专用的 LangChain 工具，再传给主 Agent 的工具列表。授权文件映射由后台提供，模型只提交 `CheckRequest` 字段。该工具返回完整 `CheckResult`；固定审核节点可以直接调用 `run_check`。当前聊天服务尚未注册此工具。
