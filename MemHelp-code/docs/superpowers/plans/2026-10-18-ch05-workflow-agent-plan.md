# Workflow 确定性编排 + 主力 Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 LangGraph 搭出「指代消解 → 意图识别 → 四出口分流 → 知识检索 → 置信度闸 → ReAct Agent → 日志」的客服骨架，并把手写 Agent 循环、SQLite checkpointer、前端双按钮一起落地。

**Architecture:** 薄图 + 厚节点。外层图只做确定性流程与分流；ReAct 循环封装在 `agent` 节点内部，节点内用 `get_stream_writer()` 发 token/status。State 全程一份 TypedDict，checkpointer 落 SQLite；MySQL 仍是聊天记录/页面回读的权威副本。

**Tech Stack:** Python 3.12、FastAPI、LangChain 1.x、LangGraph 1.2.15、langgraph-checkpoint-sqlite、SQLAlchemy/MySQL、原生 HTML/CSS/JS。

**Spec:** `MemHelp-code/docs/superpowers/specs/2026-10-18-ch05-workflow-agent-design.md`

## Global Constraints

- 代码只写 `MemHelp-code/`；测试配置两份（`pyproject.toml` 与根 `pytest.ini`）保持一致。
- 上游导入只允许 `app/llm.py` import `langchain_openai`；配置只允许 `app/config.py` 读环境。
- 错误文本一律过 `redact()`；响应体统一 `{"error": {"type", "message"}}`。
- 不新写业务工具；Agent 查询工具复用 `app/tools/registry.py` 的 `query_order/query_product/query_logistics/query_faq`，**不含 `create_ticket`**。
- `create_ticket` 只在 `POST /api/actions/create-ticket` 里经 `execute_tool` 调用。
- 转人工是纯前端模拟，不发后端请求。
- 测试不依赖 `tmp_path` / 新建目录；需要文件时用已存在目录，用完删除。
- 纯 Prompt 任务（意图分类）用 `evals/ch05_intent_cases.jsonl` 跑真实模型验证，不用 TDD 代替。
- 所有新配置写 `.env.example`，测试的 `Settings` 显式覆盖。
- 提交前检查 `.env` / `.venv` / `.uv-cache` / `.idea` 未进暂存区。

## Review Focus

1. **流到一半上游挂掉**：已发出 token 后抛错，不能把半截回答写进 MySQL 或 checkpoint；接口要给 `error` 帧，历史里只留用户轮。
2. **「在吗，退货政策是什么」这类“寒暄+真问题”**：快速通道不能把整句吞成 chitchat；必须仍走意图分类/知识检索。
3. **模型一次返回多个 tool_calls，或意图 JSON 非法**：只执行第一个工具、裁掉多余 tool_call（避免孤儿 ToolMessage）；意图必须确定性兜底，不 500。
4. **重复点「建工单」/刷新后再点**：不能重复写 tickets；前端要一次性锁定，后端要能承受重复请求（至少返回明确结果）。
5. **重排降级（没有 rerank 分）时**：闸不能拿 RRF 的 0.03 分去比 0.3 阈值，否则知识类每轮都被拒。

---

### Task 0: 真机探针（SqliteSaver / 自定义流 / chunk 合并）

**Files:**
- Create: `MemHelp-code/scripts/_probe_ch05.py`（临时，探针结束删除）
- Modify: `MemHelp-code/pyproject.toml`、`MemHelp-code/uv.lock`（新增 `langgraph-checkpoint-sqlite`）
- Modify: `MemHelp-code/dev-notes/ch05.md`（记录结论）

**Interfaces:**
- Produces: 三条被证实的 API 用法，供 Task 3 / Task 8 / Task 11 使用。

- [ ] **Step 1: 加依赖**

```powershell
$env:UV_CACHE_DIR = "F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache"
& "C:\Users\dbq\.local\bin\uv.exe" add --directory MemHelp-code langgraph-checkpoint-sqlite
```

- [ ] **Step 2: 写探针脚本**，覆盖：
  - `from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver`
  - `async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:` + `await saver.setup()`
  - 编译一个单节点图，`ainvoke` 后用 `aget_state` 读回字段
  - 异步节点签名 `async def node(state, writer: StreamWriter)` 与 `graph.astream(..., stream_mode="custom")` 的输出形态
  - `AIMessageChunk(content="", tool_call_chunks=[...])` 相加后 `tool_calls` 是否完整
  - 真模型流式响应里 `usage_metadata` 出现在哪个 chunk

- [ ] **Step 3: 运行探针**

Run: `& $uv run --directory MemHelp-code python scripts/_probe_ch05.py`
Expected: 打印五段 `OK`；任何一段失败，先改本计划/接口，不进入 Task 1。

- [ ] **Step 4: 记录结论到 `dev-notes/ch05.md`，删除探针脚本，提交**

```powershell
Remove-Item MemHelp-code/scripts/_probe_ch05.py
git add MemHelp-code/pyproject.toml MemHelp-code/uv.lock MemHelp-code/dev-notes/ch05.md
git commit -m "ch05: 增加 SqliteSaver 依赖与真机探针结论"
```

---

### Task 1: 配置项与 checkpointer 工厂

**Files:**
- Modify: `MemHelp-code/app/config.py`
- Modify: `MemHelp-code/.env.example`
- Create: `MemHelp-code/app/agent/__init__.py`
- Create: `MemHelp-code/app/agent/checkpoint.py`
- Modify: `MemHelp-code/.gitignore`
- Test: `MemHelp-code/tests/test_ch05_config_checkpoint.py`

**Interfaces:**
- Consumes: Task 0 的 `AsyncSqliteSaver` 用法。
- Produces:
  - `Settings.checkpoint_db_path: str = "data/checkpoints.sqlite"`
  - `Settings.agent_max_steps: int = 6`
  - `Settings.agent_token_budget: int = 6000`
  - `Settings.turn_log_path: str = "logs/ch05-turns.jsonl"`
  - `open_checkpointer(path: str | Path) -> AsyncContextManager[BaseCheckpointSaver]`
  - `memory_checkpointer() -> InMemorySaver`

- [ ] **Step 1: 写失败测试**

```python
async def test_open_checkpointer_persists_state_across_invocations():
    async with open_checkpointer(":memory:") as saver:
        graph = _tiny_graph(saver)
        await graph.ainvoke({"value": 1}, {"configurable": {"thread_id": "t1"}})
        snap = await graph.aget_state({"configurable": {"thread_id": "t1"}})
        assert snap.values["value"] == 1

def test_settings_have_ch05_defaults(test_settings):
    assert test_settings.agent_max_steps == 6
    assert test_settings.turn_log_path == ""
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_config_checkpoint.py -q`
Expected: `ImportError` / `AttributeError`。

- [ ] **Step 3: 实现**

`checkpoint.py`：
- `open_checkpointer` 用 `@asynccontextmanager`；路径不是 `:memory:` 时 `Path(path).parent.mkdir(parents=True, exist_ok=True)`，再 `AsyncSqliteSaver.from_conn_string(str(path))`，`await saver.setup()`。
- `memory_checkpointer()` 返回 `InMemorySaver()`。
- 路径相对 `PROJECT_ROOT` 解析。

`config.py` 加四个字段（`agent_max_steps` ge=1 le=12；`agent_token_budget` ge=500；`turn_log_path` 允许空串）。`.env.example` 补注释。`.gitignore` 加 `data/`、`logs/`。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_config_checkpoint.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/config.py MemHelp-code/app/agent MemHelp-code/tests/test_ch05_config_checkpoint.py MemHelp-code/.env.example MemHelp-code/.gitignore
git commit -m "ch05: 配置项与 SqliteSaver 工厂"
```

---

### Task 2: `messages.actions` 数据层 + 最后一条用户消息

**Files:**
- Modify: `MemHelp-code/app/db/models.py`
- Modify: `MemHelp-code/app/db/repositories.py`
- Create: `MemHelp-code/sql/ch05-ddl.sql`
- Test: `MemHelp-code/tests/test_ch05_repositories.py`
- Modify: `MemHelp-code/tests/test_db_models.py`（如需列对齐）

**Interfaces:**
- Produces:
  - `Message.actions: Mapped[list | None]`
  - `MessageRow.actions: list[dict] | None`
  - `MessageRepository.append(..., actions: Sequence[dict] | None = None)`
  - `MessageRepository.append_many(conversation_id: int, rows: Sequence[dict]) -> None`
  - `MessageRepository.last_user_content(session_id: str) -> str | None`

- [ ] **Step 1: 写失败测试**

```python
async def test_actions_roundtrip(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-actions", "anonymous")
    await fake_repositories.messages.append(
        cid, role="assistant", content="可以帮你转人工或建工单。",
        actions=[{"action": "transfer_human", "ticket_type": "投诉"}],
    )
    rows = await fake_repositories.messages.list_for_conversation(cid)
    assert rows[-1].actions == [{"action": "transfer_human", "ticket_type": "投诉"}]

async def test_append_many_is_one_transaction(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-many", "anonymous")
    await fake_repositories.messages.append_many(cid, [
        {"role": "user", "content": "我要投诉"},
        {"role": "assistant", "content": "抱歉…"},
    ])
    assert [r.role for r in await fake_repositories.messages.list_for_conversation(cid)] == ["user", "assistant"]

async def test_last_user_content(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-last", "anonymous")
    await fake_repositories.messages.append_many(cid, [
        {"role": "user", "content": "第一句"},
        {"role": "assistant", "content": "回答"},
        {"role": "user", "content": "订单 1001 的物流"},
    ])
    assert await fake_repositories.messages.last_user_content("s-last") == "订单 1001 的物流"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_repositories.py -q`
Expected: `AttributeError` / `TypeError`。

- [ ] **Step 3: 实现**

- ORM 加 `actions` JSON 列（放在 `citations` 后）。
- `_append` / `append` 加 `actions` 参数，写入 `Message.actions`。
- `_list` 返回 `MessageRow(actions=row.actions)`。
- `_append_many`：校验所有 role，`session.add` 全部，最后 `commit()` 一次；`append_many` 用 `_run`。
- `_last_user_content`：join `Conversation`，`role == "user"`，`order_by(Message.id.desc()).limit(1)`；返回 `content`。
- `sql/ch05-ddl.sql`：

```sql
ALTER TABLE messages ADD COLUMN actions JSON NULL AFTER citations;
```

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_repositories.py tests/test_db_repositories.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/db MemHelp-code/sql/ch05-ddl.sql MemHelp-code/tests/test_ch05_repositories.py MemHelp-code/tests/test_db_models.py
git commit -m "ch05: messages.actions 列与多行事务写入"
```

---

### Task 3: State、意图/路由常量、纯函数分流

**Files:**
- Create: `MemHelp-code/app/agent/state.py`
- Test: `MemHelp-code/tests/test_ch05_state.py`

**Interfaces:**
- Produces:
  - `Intent = Literal["logistics","order","product_consult","refund_return","after_sales","complaint","chitchat"]`
  - `Route = Literal["knowledge","business","complaint","chitchat"]`
  - `INTENT_TO_ROUTE: dict[Intent, Route]`
  - `route_for(intent: str) -> Route`（未知回落 `business`）
  - `AgentState` TypedDict（字段见 spec §5）

- [ ] **Step 1: 写失败测试**

```python
def test_seven_intents_map_to_four_exits():
    assert route_for("product_consult") == "knowledge"
    assert route_for("refund_return") == "knowledge"
    assert route_for("logistics") == "business"
    assert route_for("order") == "business"
    assert route_for("after_sales") == "business"
    assert route_for("complaint") == "complaint"
    assert route_for("chitchat") == "chitchat"
    assert route_for("unknown") == "business"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_state.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现 `state.py`**，字段严格按 spec §5，包含 `pending_messages`、`ticket_type`、`trace`。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_state.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/agent/state.py MemHelp-code/tests/test_ch05_state.py
git commit -m "ch05: State 契约与七类意图分流表"
```

---

### Task 4: 意图分类（Prompt + 严格 JSON + 启发式 + 寒暄快速通道）

**Files:**
- Create: `MemHelp-code/app/agent/prompts.py`
- Create: `MemHelp-code/app/agent/intent.py`
- Test: `MemHelp-code/tests/test_ch05_intent.py`

**Interfaces:**
- Produces:
  - `IntentResult(intent: Intent, confidence: float, raw: str, source: Literal["llm","heuristic","rule"])`
  - `is_trivial_chitchat(message: str) -> bool`
  - `heuristic_intent(message: str) -> Intent`
  - `parse_intent_reply(reply: str) -> IntentResult | None`
  - `IntentClassifier(model, *, timeout: float = 5.0).classify(message: str) -> IntentResult`

- [ ] **Step 1: 写失败测试**

```python
def test_parse_intent_json_tolerates_code_fence():
    result = parse_intent_reply('```json\n{"intent":"logistics","confidence":0.91}\n```')
    assert result.intent == "logistics" and result.source == "llm"

def test_parse_invalid_returns_none():
    assert parse_intent_reply("我觉得是物流") is None

def test_trivial_chitchat_is_exact_not_prefix():
    assert is_trivial_chitchat("你好")
    assert is_trivial_chitchat("在吗？")
    assert not is_trivial_chitchat("在吗，退货政策是什么")

def test_heuristic_priority_complaint_over_refund():
    assert heuristic_intent("收到的杯子是碎的，我要投诉并退款") == "complaint"

async def test_classifier_falls_back_on_upstream_error():
    class Boom:
        async def ainvoke(self, *a, **k): raise RuntimeError("boom")
    result = await IntentClassifier(Boom()).classify("我要投诉")
    assert result.intent == "complaint" and result.source == "heuristic"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_intent.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现**

- `prompts.py`：`INTENT_PROMPT`（七类定义 + 边界 + 只输出 JSON），`TRIVIAL_CHITCHAT` 集合。
- `intent.py`：
  - `is_trivial_chitchat`：去空白/标点后**整句**命中集合，最长 6 字；不做前缀匹配。
  - `heuristic_intent` 优先级：投诉 > 退款退货 > 物流 > 订单 > 售后 > 商品咨询；默认 `after_sales`。
  - `parse_intent_reply`：去 ``` 围栏、`json.loads`、校验枚举、confidence 夹到 0..1。
  - `IntentClassifier.classify`：快速通道 → `asyncio.wait_for(model.ainvoke([...]), timeout)` → 解析；异常/解析失败 → heuristic，`raw` 保留原文。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_intent.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/agent/prompts.py MemHelp-code/app/agent/intent.py MemHelp-code/tests/test_ch05_intent.py
git commit -m "ch05: 意图分类 prompt、严格 JSON 与启发式兜底"
```

---

### Task 5: `RagPipeline.retrieve_only()` + 置信度闸

**Files:**
- Modify: `MemHelp-code/app/knowledge/pipeline.py`
- Create: `MemHelp-code/app/agent/gate.py`
- Test: `MemHelp-code/tests/test_ch05_gate.py`

**Interfaces:**
- Produces:
  - `RagPipeline.retrieve_only(query: str, *, category: str | None = None) -> RetrievalOutcome`
  - `GateDecision(passed: bool, reason: str, score: float, evidence_count: int)`
  - `evaluate_gate(*, refused: bool, evidence_count: int, top_score: float, rerank_degraded: bool, min_score: float) -> GateDecision`

- [ ] **Step 1: 写失败测试**

```python
async def test_retrieve_only_does_not_call_sufficiency():
    judge = _ExplodingJudge()
    pipeline = RagPipeline(understander=_FakeUnderstander(), retriever=_FakeRetriever(hits=1),
                           sufficiency=judge, min_score=0.3, top_n=10)
    outcome = await pipeline.retrieve_only("退货政策")
    assert outcome.refused is False
    assert judge.calls == 0

def test_gate_rejects_low_score():
    d = evaluate_gate(refused=False, evidence_count=3, top_score=0.12,
                      rerank_degraded=False, min_score=0.3)
    assert d.passed is False and "低于阈值" in d.reason

def test_gate_passes_when_rerank_degraded_but_evidence_exists():
    d = evaluate_gate(refused=False, evidence_count=2, top_score=0.0,
                      rerank_degraded=True, min_score=0.3)
    assert d.passed is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_gate.py -q`
Expected: `AttributeError: retrieve_only`。

- [ ] **Step 3: 实现**

- `retrieve_only`：复制 `answer_context` 的理解/检索/0 命中/阈值段，**不调用** `self._sufficiency.judge`；返回 `RetrievalOutcome`。
- `gate.py`：`evaluate_gate` 纯函数；拒绝原因写清 `0 命中` / 分数低于阈值 / 检索层拒答。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_gate.py tests/test_knowledge_sufficiency.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/knowledge/pipeline.py MemHelp-code/app/agent/gate.py MemHelp-code/tests/test_ch05_gate.py
git commit -m "ch05: 只检索入口与置信度闸"
```

---

### Task 6: `suggest_actions` 控制面工具 + Agent 工具集过滤 + 证据/引用助手

**Files:**
- Create: `MemHelp-code/app/agent/tools.py`
- Create: `MemHelp-code/app/agent/evidence.py`
- Test: `MemHelp-code/tests/test_ch05_agent_tools.py`

**Interfaces:**
- Produces:
  - `SuggestActionsArgs(actions: list[Literal["transfer_human","create_ticket"]], reason: str = "")`
  - `make_suggest_actions_tool(record: list[str]) -> BaseTool`
  - `agent_tools(repositories, retriever, *, include_faq: bool, record_actions: list[str]) -> list[BaseTool]`
  - `evidence_from_tool_result(result: ToolResult, repositories: Repositories) -> tuple[Evidence, ...]`
  - `citations_for_answer(answer: str, evidence: Sequence[Evidence]) -> list[dict]`

- [ ] **Step 1: 写失败测试**

```python
def test_agent_tools_never_include_create_ticket(fake_repositories, empty_retriever):
    names = [t.name for t in agent_tools(fake_repositories, empty_retriever,
                                         include_faq=True, record_actions=[])]
    assert "create_ticket" not in names
    assert "suggest_actions" in names

def test_knowledge_tools_drop_query_faq(...):
    names = ...
    assert "query_faq" not in names

async def test_suggest_actions_records_without_writing(...):
    record = []
    tool = make_suggest_actions_tool(record)
    await tool.ainvoke({"actions": ["transfer_human"], "reason": "用户要求"})
    assert record == ["transfer_human"]

def test_citations_only_include_cited_numbers():
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_agent_tools.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现**

- `tools.py`：Pydantic schema + `@tool("suggest_actions")` 闭包；描述里写「只记录建议，不执行任何动作」。`agent_tools` 调 `get_tools` 后过滤。
- `evidence.py`：从 `query_faq` 出参还原证据（复用 `evidence_from_payload`），`truncated` 回库取全文；`citations_for_answer` 用 `parse_citations` + `evidence_payload` 过滤。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_agent_tools.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/agent/tools.py MemHelp-code/app/agent/evidence.py MemHelp-code/tests/test_ch05_agent_tools.py
git commit -m "ch05: 建议动作控制面工具与 Agent 工具集"
```

---

### Task 7: ReAct 循环（流式、单工具步、停止条件、token 预算）

**Files:**
- Create: `MemHelp-code/app/agent/react.py`
- Create: `MemHelp-code/tests/fakes_agent.py`（`ScriptedChatModel`）
- Test: `MemHelp-code/tests/test_ch05_react.py`

**Interfaces:**
- Produces:
  - `AgentOutcome(answer, pending_messages, tool_traces, tool_records, citations, suggested_actions, agent_steps, llm_calls, token_usage, stop_reason)`
  - `run_react_agent(*, model, messages, repositories, retriever, include_faq, evidence, max_steps, token_budget, timeout, max_retries, emit, executor=None) -> AgentOutcome`
  - `emit` 接受 `{"type": "token", "delta": ...}` 或 `{"type": "status", ...}`

- [ ] **Step 1: 写失败测试**（用 `ScriptedChatModel` 按调用顺序返回脚本）

```python
async def test_one_tool_call_then_final(...):
    model = ScriptedChatModel(script=[
        tool_call_chunks("query_logistics", {"order_id": "1001"}, "call_1"),
        text_chunks(["已发货，正在派送。"]),
    ])
    out = await run_react_agent(...)
    assert out.agent_steps == 1
    assert [m.type for m in out.pending_messages] == ["ai", "tool", "ai"]
    assert out.stop_reason == "final"
    assert "已发货" in out.answer

async def test_complex_question_takes_two_tool_steps(...):
    script = [tool_call_chunks("query_order", ...), tool_call_chunks("query_logistics", ...),
              text_chunks(["订单已付款，物流运输中。"])]
    out = await run_react_agent(...)
    assert out.agent_steps == 2
    assert len(out.tool_records) == 2

async def test_multiple_calls_in_one_step_only_first_runs(...):
    model = ScriptedChatModel(script=[multi_tool_call_chunks([...]), text_chunks(["好了"])])
    out = await run_react_agent(...)
    assert out.agent_steps == 1
    assert all(m.tool_calls == [] for m in out.pending_messages if m.type == "tool")
    assert len([m for m in out.pending_messages if m.type == "tool"]) == 1

async def test_max_steps_returns_handoff(...):
    out = await run_react_agent(..., max_steps=1)
    assert out.stop_reason == "max_steps"
    assert out.suggested_actions == ["transfer_human", "create_ticket"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_react.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现 `ScriptedChatModel`**

- `bind_tools` 记录工具名并返回共享 `ScriptState` 的副本。
- `_generate` 返回脚本下一条 `AIMessage`。
- `_astream` 逐块吐 `AIMessageChunk`；tool_call 用 `tool_call_chunks=[{"name","args","id","index","type"}]`。
- 每条响应带 `usage_metadata={"input_tokens":..., "output_tokens":..., "total_tokens":...}`。

- [ ] **Step 4: 实现 `react.py`**

按 spec §7：`astream` → 合并 chunk → `_message_text` 发 token → `to_ai_message` → 无 tool_calls 收敛；有则 `first_call_only`、发 status、`execute_tool` 或 `suggest_actions`、回灌 ToolMessage；累计 usage；max_steps/token_budget 刹车；`pending_messages` 只放本轮新消息。

- [ ] **Step 5: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_react.py -q`
Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add MemHelp-code/app/agent/react.py MemHelp-code/tests/fakes_agent.py MemHelp-code/tests/test_ch05_react.py
git commit -m "ch05: ReAct 循环、流式事件与停止条件"
```

---

### Task 8: 图节点、条件边、投诉/闲聊/兜底、日志节点

**Files:**
- Create: `MemHelp-code/app/agent/graph.py`
- Create: `MemHelp-code/app/agent/coref.py`
- Create: `MemHelp-code/app/agent/persist.py`
- Create: `MemHelp-code/app/agent/logging.py`
- Test: `MemHelp-code/tests/test_ch05_graph.py`

**Interfaces:**
- Consumes: Task 3–7 的 `AgentState`、`IntentClassifier`、`retrieve_only`、`evaluate_gate`、`run_react_agent`、`agent_tools`。
- Produces:
  - `build_agent_graph(*, checkpointer=None) -> CompiledStateGraph`
  - `persist_turn(*, repositories, session_id, user_message, pending_messages, citations, actions, ticket_type) -> None`
  - `persist_user_only(repositories, session_id, message) -> None`
  - `TurnLogger(path).write(record: dict) -> None`
  - `build_turn_log(state, latency_ms) -> dict`

- [ ] **Step 1: 写失败测试**

```python
async def test_policy_question_goes_through_retrieval_and_gate(...):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    events = await _run(graph, "退货政策是什么", model=_intent_then_text("refund_return", "7 天无理由…"))
    values = await _state(graph, "s-policy")
    assert values["route"] == "knowledge"
    assert "knowledge_retrieve" in values["trace"]
    assert values["retrieval_attempted"] is True
    assert values["gate_passed"] is True

async def test_gate_failure_does_not_call_agent(...):
    # retriever 返回 0 证据；模型脚本里只有意图 JSON，若 Agent 被调用会耗尽脚本
    values = ...
    assert values["answer"] == REFUSAL_TEXT
    assert values["agent_steps"] == 0

async def test_complaint_has_two_actions_and_no_agent(...):
    values = ...
    assert values["suggested_actions"] == ["transfer_human", "create_ticket"]
    assert values["agent_steps"] == 0

async def test_trivial_chitchat_has_no_model_call(...):
    values = ...
    assert values["intent_source"] == "rule"
    assert values["answer"] == CHITCHAT_REPLY

async def test_greeting_plus_real_question_is_not_swallowed_by_fast_path(...):
    # 「在吗，退货政策是什么」不是逐字寒暄，必须仍走意图分类 + 知识检索
    values = ...
    assert values["route"] == "knowledge"
    assert values["retrieval_attempted"] is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_graph.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现 `persist.py` / `logging.py`**

- `persist.py`：用 `append_many` 单事务；`pending_messages` → rows；`tool_records` 按 `tool_call_id` 填 `tool_traces`；`actions` 只挂最终 assistant。
- `logging.py`：空路径 return；否则 `mkdir(parents=True, exist_ok=True)`，追加 UTF-8 JSONL；异常只 warning。

- [ ] **Step 4: 实现 `graph.py`**

- 节点严格按 spec §6；每个节点 `{"trace": [*state.get("trace", []), "node"]}`。
- `config["configurable"]` 取 `model/repositories/retriever/settings`。
- `agent_node`：`build_context` + `build_agent_system_prompt` + `run_react_agent`，把 `emit` 包成 writer。
- `log_turn`：幂等 → `persist_turn` → `TurnLogger` → 返回 `messages=pending_messages, last_committed_turn_id=turn_id, turn_log=...`。
- 条件边按 `route` 与 `gate_passed`。

- [ ] **Step 5: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_graph.py -q`
Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add MemHelp-code/app/agent/graph.py MemHelp-code/app/agent/persist.py MemHelp-code/app/agent/logging.py MemHelp-code/tests/test_ch05_graph.py
git commit -m "ch05: LangGraph 图节点、分流与日志"
```

---

### Task 9: Runner + API 接线（SSE / 非流式 / done 扩展）

**Files:**
- Create: `MemHelp-code/app/agent/runner.py`
- Modify: `MemHelp-code/app/api.py`
- Modify: `MemHelp-code/app/schemas.py`
- Modify: `MemHelp-code/tests/conftest.py`
- Modify: `MemHelp-code/tests/fakes.py`（`EmptyRetriever.retrieve_only`；`FakeChatModel` 的 `tool_calls_then_stream` 只在 `bind_tools` 之后才触发，否则意图分类会把工具调用吃掉）
- Test: `MemHelp-code/tests/test_ch05_api.py`

**Interfaces:**
- Produces:
  - `astream_turn(graph, *, session_id, message, model, repositories, retriever, settings, turn_id=None) -> AsyncIterator[dict]`
  - `complete_turn(...) -> tuple[str, dict]`
  - `create_app(..., checkpointer=None, agent_graph=None)`
  - `done` 新增 `actions: list[dict]` / `intent` / `route` / `agent_steps` / `stop_reason` / `gate`
  - `ChatResponse` 新增同名字段（默认值）

- [ ] **Step 1: 写失败测试**

```python
def test_sse_done_has_actions_and_route(test_settings, fake_repositories):
    client = TestClient(main.create_app(model=..., settings=test_settings,
                                         checkpointer=memory_checkpointer()))
    text = _stream(client, {"session_id": "s", "message": "我要投诉"})
    done = _events(text)[-1][1]
    assert done["route"] == "complaint"
    assert [a["action"] for a in done["actions"]] == ["transfer_human", "create_ticket"]
    assert done["agent_steps"] == 0

def test_non_stream_returns_actions(...):
    resp = client.post(..., json={"stream": False, "message": "我要投诉"})
    assert resp.json()["actions"][0]["action"] == "transfer_human"

def test_upstream_failure_persists_only_user(...):
    # model 在意图调用就抛；断言 /messages 只有 user 一行
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_api.py -q`
Expected: `KeyError: "actions"` / 旧编排仍被调用。

- [ ] **Step 3: 实现**

- `runner.py`：`graph.astream(input, config, stream_mode="custom")`，兼容 `{"type":"custom","data":...}` 与裸 payload；完成后 `aget_state`；`done` 从 state 组装。
- `api.py`：
  - `create_app` 加 `checkpointer` / `agent_graph` 参数；lifespan 用 `AsyncExitStack` 管理 `open_checkpointer`，编译图。
  - `_chat_stream_body` 改调 `astream_turn`；`meta` 不变；`done` 用 runner 的字段。
  - 非流式改 `complete_turn`。
  - 异常路径：`persist_user_only` 包 try/except，再抛原异常。
- `schemas.py`：`ChatResponse` 加 `actions: list[dict] = []` 等默认字段。
- `conftest.py`：`_wrapped` 默认 `kwargs.setdefault("checkpointer", memory_checkpointer())`；`test_settings` 加 `turn_log_path=""`；`EmptyRetriever.retrieve_only` 返回 `retrieval_outcome(query, (), refused=False)`。
- 现有 HTTP 测试：把断言从「chat_tools 的行为」改成「图的行为」；保留 `tests/test_chat_tools.py` 不动。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_api.py tests/test_api_sse.py tests/test_api_tools.py -q`
Expected: PASS（旧 HTTP 用例按图契约调整后）。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/app/agent/runner.py MemHelp-code/app/api.py MemHelp-code/app/schemas.py MemHelp-code/tests/conftest.py MemHelp-code/tests/fakes.py MemHelp-code/tests/test_ch05_api.py MemHelp-code/tests/test_api_sse.py MemHelp-code/tests/test_api_tools.py
git commit -m "ch05: 图 runner 与聊天 API 接线"
```

---

### Task 10: 工单端点 + 前端双按钮 + 历史还原

**Files:**
- Modify: `MemHelp-code/app/api.py`
- Modify: `MemHelp-code/app/schemas.py`
- Modify: `MemHelp-code/web/app.js`
- Modify: `MemHelp-code/web/style.css`
- Test: `MemHelp-code/tests/test_ch05_ticket_action.py`

**Interfaces:**
- Produces:
  - `POST /api/actions/create-ticket` → `{"ok": true, "ticket_no", "ticket_type", "status", "message"}`
  - 前端 `attachActions(target, actions, {route, sourceText})`
  - `/messages` 的 assistant 行带 `actions`

- [ ] **Step 1: 写失败测试**

```python
def test_create_ticket_endpoint_writes_tickets_table(...):
    resp = client.post("/api/actions/create-ticket", json={
        "session_id": "s-ticket", "description": "我要投诉", "ticket_type": "投诉"})
    body = resp.json()
    assert resp.status_code == 200 and body["ticket_no"].startswith("T")
    # 用 SQLite 仓库查 tickets 行

def test_create_ticket_falls_back_to_last_user_message(...):
    # 先发一条投诉，再不带 description 调端点
    ...

def test_invalid_ticket_type_is_422(...):
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_ticket_action.py -q`
Expected: 404。

- [ ] **Step 3: 实现后端**

- `schemas.py`：`CreateTicketRequest` / `CreateTicketResponse`。
- `api.py`：端点；`description` 缺省用 `last_user_content`；调 `execute_tool("create_ticket", ...)`；`ok=false` → 502 `tool_error`；`ValueError` → 422。
- `session_messages`：assistant 行有 `actions` 就带上。

- [ ] **Step 4: 实现前端**

- `send()` 把 `userText` 存进 `ctx`；`done` 调 `attachActions(ctx.bot, data.actions, {route: data.route, sourceText: userText})`。
- `restoreHistory` 记录 `lastUserText`，bot 行有 `actions` 时同样渲染。
- `renderActions`：两个独立按钮；localStorage 键 `mewhelp.action.<session>.<botIndex>.<action>`；已点状态还原。
- 「转人工」：不发请求，追加「已转接人工客服」提示 + 客服小猫问候气泡。
- 「建工单」：`POST /api/actions/create-ticket`，成功后显示工单号，失败提示。
- CSS 补 `.chat-actions` / `.action-btn` / `.action-note`。

- [ ] **Step 5: 跑测试 + 手工检查**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_ch05_ticket_action.py -q`
Expected: PASS。
浏览器手工：投诉 → 两个按钮；点转人工只出现前端模拟；点建工单后 tickets 表有行；刷新后按钮仍在。

- [ ] **Step 6: 提交**

```powershell
git add MemHelp-code/app/api.py MemHelp-code/app/schemas.py MemHelp-code/web MemHelp-code/tests/test_ch05_ticket_action.py
git commit -m "ch05: 建单端点与转人工/建工单前端双按钮"
```

---

### Task 11: 祛魅热身脚本 + 测试

**Files:**
- Create: `MemHelp-code/scripts/bare_agent_loop.py`
- Test: `MemHelp-code/tests/test_bare_agent_loop.py`

**Interfaces:**
- Produces:
  - `BareResult(answer, messages, steps, tool_calls)`
  - `async def run_bare_agent(model, user_message, tools, *, max_steps=5) -> BareResult`
  - CLI `python scripts/bare_agent_loop.py --message "订单 1001 到哪了"` / `--fake`

- [ ] **Step 1: 写失败测试**

```python
async def test_bare_loop_executes_tool_and_feeds_back():
    model = ScriptedChatModel(script=[tool_call_chunks("query_order", {"order_id": "1001"}, "c1"),
                                      text_chunks(["订单 1001 已发货。"])])
    out = await run_bare_agent(model, "订单 1001 怎么样", [query_order_tool()])
    assert out.steps == 1
    assert out.answer == "订单 1001 已发货。"
    assert [m.type for m in out.messages] == ["human", "ai", "tool", "ai"]

async def test_bare_loop_stops_at_max_steps():
    out = await run_bare_agent(model, "x", [tool], max_steps=1)
    assert out.steps == 1
```

- [ ] **Step 2: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_bare_agent_loop.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 3: 实现**

- 脚本内 `@tool` 包 `query_order`；`run_bare_agent` 用 `ainvoke` 循环（非流式，保持“最裸”）；
  每次工具调用打印 `[step N] tool(args)`；`--fake` 用本地脚本模型，`--message` 走真模型。
- 不 import LangGraph。

- [ ] **Step 4: 跑测试确认通过**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_bare_agent_loop.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add MemHelp-code/scripts/bare_agent_loop.py MemHelp-code/tests/test_bare_agent_loop.py
git commit -m "ch05: 手写 Agent 循环热身脚本"
```

---

### Task 12: 意图标注集 + 评估脚本

**Files:**
- Create: `MemHelp-code/evals/ch05_intent_cases.jsonl`（约 56–60 条）
- Create: `MemHelp-code/evals/run_ch05_intent_evals.py`
- Test: `MemHelp-code/tests/test_run_ch05_intent_evals.py`

**Interfaces:**
- Produces:
  - CLI `python evals/run_ch05_intent_evals.py [--limit N] [--json PATH] [--markdown PATH] [--self-test]`
  - 指标：`accuracy` / `route_accuracy` / `per_intent_recall` / `confusion`
  - 退出码 0/1/2

- [ ] **Step 1: 写标注集**，七类各 8 条，覆盖：
  - 订单 vs 物流边界；投诉 vs 售后；退款退货 vs 商品咨询；闲聊；省略/指代。
  - 每条带 `expected_intent` / `expected_route` / `bucket` / `note`。

- [ ] **Step 2: 写失败测试**

```python
def test_aggregate_computes_accuracy_and_missing_predictions():
    report = aggregate([Case(...), Case(...)])
    assert report["accuracy"] == 0.5
    assert report["route_accuracy"] == 0.5

def test_exit_code_requires_overall_and_per_class_threshold():
    assert exit_code({"accuracy": 0.9, "per_intent_recall": {...0.7}}) == 0
    assert exit_code({"accuracy": 0.9, "per_intent_recall": {"chitchat": 0.4}}) == 1
```

- [ ] **Step 3: 跑测试确认失败**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_run_ch05_intent_evals.py -q`
Expected: `ModuleNotFoundError`。

- [ ] **Step 4: 实现脚本**

- 读 JSONL；`IntentClassifier` 跑真模型；记录 `predicted_intent / route / source / raw / latency_ms`。
- `--self-test` 用「按 expected 返回」的假分类器，只验证聚合/退出码。
- Markdown 输出混淆矩阵与失败明细；JSON 输出逐条。
- 不达标时打印「先改 Prompt / 示例，不改标签」。

- [ ] **Step 5: 跑离线测试 + 真机评估**

Run: `& $uv run --directory MemHelp-code python -m pytest tests/test_run_ch05_intent_evals.py -q`
Run: `& $uv run --directory MemHelp-code python evals/run_ch05_intent_evals.py --self-test`
Run（真机）: `& $uv run --directory MemHelp-code python evals/run_ch05_intent_evals.py --json dev-notes/ch05-intent-evals.json --markdown dev-notes/ch05-intent-eval-report.md`
Expected: 退出码 0（整体 ≥0.85，每类 ≥0.60）；不达标先改 Prompt 再跑。

- [ ] **Step 6: 提交**

```powershell
git add MemHelp-code/evals/ch05_intent_cases.jsonl MemHelp-code/evals/run_ch05_intent_evals.py MemHelp-code/tests/test_run_ch05_intent_evals.py MemHelp-code/dev-notes/ch05-intent-evals.json MemHelp-code/dev-notes/ch05-intent-eval-report.md
git commit -m "ch05: 意图标注集与评估脚本"
```

---

### Task 13: 验收演示脚本 + 真机验收

**Files:**
- Create: `MemHelp-code/scripts/demo-ch05.py`
- Modify: `MemHelp-code/dev-notes/ch05.md`

**Interfaces:**
- Produces: `dev-notes/ch05-acceptance-output.txt`，五条验收的实测结果。

- [ ] **Step 1: 实现脚本**

- 用 `httpx` 调 `http://127.0.0.1:8000`（可用 `--base-url` 覆盖）。
- 解析 SSE；场景：
  1. 政策问题 → 断言日志 `trace` 含 `knowledge_retrieve`；
  2. `订单 1001 的物流到哪了` → `done.tools` 含 `query_logistics`；
  3. `我要投诉` → `done.actions` 两个；调 create-ticket → 查 `tickets` 表有行；
  4. `你好` → 固定话术 + 日志 `intent_source=rule`；
  5. 复杂问题 → `agent_steps>=2`。
- 输出每场景 PASS/FAIL 与关键字段；退出码 0/1。

- [ ] **Step 2: 真机跑**

先起服务（8000 被占用就用 8010，并传 `--base-url`）：

```powershell
& $uv run --directory MemHelp-code uvicorn app.main:app --port 8010
& $uv run --directory MemHelp-code python scripts/demo-ch05.py --base-url http://127.0.0.1:8010 --out dev-notes/ch05-acceptance-output.txt
```

Expected: 五条全 PASS，退出码 0。

- [ ] **Step 3: 补 `dev-notes/ch05.md` 验收段，提交**

```powershell
git add MemHelp-code/scripts/demo-ch05.py MemHelp-code/dev-notes/ch05.md MemHelp-code/dev-notes/ch05-acceptance-output.txt
git commit -m "ch05: 五条验收真机跑通"
```

---

### Task 14: README / 配置 / 真库迁移

**Files:**
- Modify: `MemHelp-code/README.md`
- Modify: `MemHelp-code/.env.example`
- Modify: `MemHelp-code/dev-notes/ch05.md`

- [ ] **Step 1: 文档**

README 加第五章：新配置、建表迁移、起服务、热身脚本、意图评估、验收脚本、留痕路径。

- [ ] **Step 2: 真库迁移**

```powershell
& $uv run --directory MemHelp-code python -c "from app.config import get_settings; from sqlalchemy import create_engine, text; e=create_engine(get_settings().database_url()); c=e.connect(); c.execute(text('ALTER TABLE messages ADD COLUMN actions JSON NULL AFTER citations')); c.commit(); c.close()"
```

Expected: 无报错；`SHOW COLUMNS FROM messages LIKE 'actions'` 有行。若列已存在，先查再执行，不重复加。

- [ ] **Step 3: 全量测试 + 提交**

```powershell
& $uv run --directory MemHelp-code python -m pytest -q
git add MemHelp-code/README.md MemHelp-code/.env.example MemHelp-code/dev-notes/ch05.md
git commit -m "ch05: 运行说明与真库迁移"
```

---

### Task 15: 独立 code review + 逐条处置

**Files:**
- Modify: 按 review 结论
- Modify: `MemHelp-code/dev-notes/ch05.md`

- [ ] **Step 1: 派独立 reviewer**（全新上下文）复查 `git diff 3cacfa3..HEAD`，要求：
  - 五条验收是否都有测试/真机证据；
  - 半截回答、孤儿 ToolMessage、重复建单、闸降级、超大工具结果；
  - 新旧契约兼容、`create_ticket` 是否真的不在 Agent 工具集；
  - 日志是否含敏感信息。

- [ ] **Step 2: 逐条处置** Critical/Important 必须修；Minor 记 deferred。每条补回归测试。

- [ ] **Step 3: 跑全量 + 提交**

```powershell
& $uv run --directory MemHelp-code python -m pytest -q
git add MemHelp-code
git commit -m "ch05: 处置独立评审（Critical/Important）"
```

---

### Task 16: 收尾

**Files:**
- Modify: `MemHelp-code/dev-notes/ch05.md`
- Modify: `MemHelp-code/dev-notes/DELIVERY.md`（如存在交付汇总，补 ch05）

- [ ] **Step 1: 补「finish」段**：最终验证命令、测试数字、验收输出、遗留项。

- [ ] **Step 2: 最终验证**

```powershell
& $uv run --directory MemHelp-code python -m pytest -q
& $uv run --directory MemHelp-code python evals/run_ch05_intent_evals.py --self-test
git status --short
```

Expected: 全绿；无 `.env` / `.venv` / `.uv-cache` 进暂存区。

- [ ] **Step 3: 提交收尾**

```powershell
git add MemHelp-code/dev-notes
git commit -m "ch05: 验收总结与交付留痕"
```
