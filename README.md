# Charon

Charon 是一个和分析师配合工作的 agent。名字来自冥王星的卫星卡戎：它大到两者围绕一个位于二者之间的共同质心运行，并且始终以同一面相对。分析师是主体，做判断和最终决定；Charon 分担工作，并让每一步都能被看见、被核对。

## 当前状态

从零开始的 agent 骨架：一个模型 ⇄ 工具的循环、一个聊天接口、离线测试。还没有任何业务工具。

v0.1 的报告审阅流程（逐句切分、flag 规则、复核）已完整存档，见 [docs/decisions.md](docs/decisions.md) 的 ADR-001。

## 结构

```text
app/
  config.py            设置：模型、接口地址、是否允许真实调用、步数上限
  llm.py               唯一构造聊天模型的地方（langchain-openai，OpenAI 兼容接口）
  agent/
    graph.py           build_graph(model, tools, checkpointer=None, store=None)
    prompts.py         系统提示、审查提示、修改提示
    tools/__init__.py  工具注册表 TOOLS
  main.py              FastAPI：GET /health，POST /chat
tests/                 离线测试，用脚本化假模型
docs/decisions.md      决策记录
```

图的结构：`START → agent`；agent 要调工具就进 `tools`，执行完回到 `agent`；不调工具时，最终回答进入 `reflect`，由模型审查输出格式、工具调用和结果是否合理。审查结论是 `REVISE` 就把意见交回 `agent` 修改，否则结束。每轮最多审查 3 次，之后直接返回当前回答。没有工具时只有 `agent` 和 `reflect` 两个节点。

## 运行

需要 Python 3.12 和 uv。

```bash
uv sync --locked
uv run --locked pytest -q
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
```

默认不调用真实模型，`/chat` 返回 503 `live_calls_disabled`。要真实调用，复制 `.env.example` 为 `.env`，填入密钥，并设置 `CHARON_ALLOW_LIVE=true`。

```bash
curl http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hello","thread_id":"demo"}'
```

`thread_id` 可省略，服务会生成一个并在响应中返回。目前还没有接入记忆，同一 thread_id 的多次请求互不相关。

## 加一个工具

1. 在 `app/agent/tools/` 下新建模块，用 `langchain_core.tools.tool` 装饰一个带类型注解和文档字符串的函数。文档字符串就是模型看到的工具说明。
2. 把它加入 `app/agent/tools/__init__.py` 的 `TOOLS`。
3. 在 `tests/` 里用 `ScriptedChatModel` 写一个“调用工具 → 结果回到模型”的测试。

## 记忆（下一步）

`build_graph` 已预留两个参数：`checkpointer` 按 thread_id 保存对话状态（线程内记忆），`store` 保存跨线程的长期信息。`tests/test_graph.py` 已验证 `InMemorySaver` 能保留同一线程的历史、隔离不同线程。
