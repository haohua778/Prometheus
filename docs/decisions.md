# 决策记录

每条记录写清楚当时的背景、决定和后果。新决定追加在最后，旧记录不改；推翻旧决定时新写一条并注明。

## ADR-001 存档 v0.1 审阅流程，从零开始（2026-09-29）

**背景**：与客户的需求会议显示，实际工作流是从大量数据中核对数据集、询问代码问题、直接对数据集写 SQL。原先的假设（单份报告的逐句 flag 审阅）和 Architecture B 约定（固定 DAG、无 RAG、无数据库）已不适用。

**决定**
- v0.1 审阅流程完整存档：提交 `0d98553`，标签 `review-pipeline-v0.1`，分支 `archive/review-pipeline`，远程私有仓库 `haohua778/Charon`。存档时离线测试 266 passed，并从 GitHub 重新 clone 验证过。
- 主线清空，从零搭建通用 agent 骨架。Architecture B 约定废弃，原 AGENTS.md 不再适用。
- 旧待办 `docs/backlog.md` 随存档保留。其中仍适用于新框架的只有两项：模型供应商与数据出境（并入 ADR-002），以及 MinerU 文档解析（确认需要处理 PDF 时再取回）。

**取回旧代码**
- 整体查看：`git switch archive/review-pipeline`
- 取回单个模块（例如以后把审阅做成 tool）：`git checkout review-pipeline-v0.1 -- app/segment.py`

## ADR-002 模型接口用 langchain-openai（2026-09-29）

**决定**：聊天模型通过 `langchain-openai` 的 `ChatOpenAI` 接入任意 OpenAI 兼容接口，只在 `app/llm.py` 一处构造，换提供方不改图。工具调用用 LangGraph 自带的 `ToolNode`。

**待定**：客户真实数据能发给哪个模型（合规）。当前默认值 kimi-k2.6 只用于开发。

## ADR-003 默认不调用真实模型，每轮有步数上限（2026-09-29）

**决定**：`CHARON_ALLOW_LIVE` 默认 false，此时 `/chat` 返回 503 `live_calls_disabled`。测试全部用离线假模型。每轮对话的图步数上限 `CHARON_MAX_STEPS`（默认 12），防止工具调用死循环。

## 下一步

- 记忆：`build_graph` 已接受 `checkpointer`（线程内记忆）和 `store`（跨线程长期记忆），测试已验证 `InMemorySaver` 能按 thread_id 保留历史。接入时要在进程内共享一个 checkpointer，而不是每个请求新建。
- 工具：按客户真实任务逐个加入 `app/agent/tools/`。
