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

## ADR-004 反思：最终回答先经模型审查（2026-10-01）

**背景**：agent 的最终回答直接返回给分析师，格式、工具调用和结果是否自洽都没有检查。

**决定**
- agent 不再调工具时，回答先进入 `reflect` 节点，由同一个模型（不绑定工具）审查。审查看完整对话（XML 格式，含工具调用和结果），只检查三点：输出格式、工具调用、结果是否合理。
- 结论用文本给出：第一行 `PASS` 或 `REVISE`，`REVISE` 后逐条写问题和改法。只有明确的 `REVISE` 才退回，其余都放行（fail-open）。
- 退回时审查意见只加进 agent 这一次的提示，不写入消息历史；草稿回答仍保留在消息历史中。
- 每轮最多审查 `MAX_ITERATIONS = 3` 次（最多修改 2 次），第 3 次后不论结论都返回当前回答。
- 图的状态从 `MessagesState` 换成 `AgentState`，在 `messages` 之外加 `answer`（当前回答）、`review`（审查意见）、`decision`（`continue`/`stop`）、`iteration`（本轮已审查次数）。`/chat` 返回 `answer`。`/chat` 每轮的输入都把 `iteration`、`review`、`decision` 重置为 0 / None，接入 checkpointer 后也不会沿用上一轮的审查状态。

**后果**：每轮最多多 3 次模型调用；审查和修改也占用图步数，计入 `CHARON_MAX_STEPS`。原默认值 12 在最坏情况（3 次回答前各调一次工具）下会超限，因此默认值改为 20（取代 ADR-003 中的 12）。

## 下一步

- 记忆：`build_graph` 已接受 `checkpointer`（线程内记忆）和 `store`（跨线程长期记忆），测试已验证 `InMemorySaver` 能按 thread_id 保留历史。接入时要在进程内共享一个 checkpointer，而不是每个请求新建。
- 工具：按客户真实任务逐个加入 `app/agent/tools/`。

## ADR-005 聊天直接使用配置的模型（2026-10-01）

**背景**：ADR-003 的 `CHARON_ALLOW_LIVE=false` 让填好密钥的 Agent 仍默认拒绝聊天，也容易被误解为模型故障后的兜底机制。

**决定**：废止 ADR-003 中默认关闭真实模型调用的开关，移除 `CHARON_ALLOW_LIVE` 和 `live_calls_disabled`。`/chat` 有密钥就使用配置的模型；密钥缺失或空白时返回 503 `missing_api_key`。真实调用失败时继续明确返回上游错误，不切换到假模型。测试继续使用脚本化假模型，不发起真实请求。图步数上限仍保留，默认值沿用 ADR-004 的 20。
