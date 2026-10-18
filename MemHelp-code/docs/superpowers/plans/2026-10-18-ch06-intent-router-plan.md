# 第六章「正式版分流器」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch05 的占位分流器替换成正式版：LLM 指代消解/Query 改写、八类意图（含 `other`）、退款/售后确定性子流程（订单槽位 → Query 扩写 → 强制政策检索 → 主力 Agent 判断）、聊天流订单卡片与退款表单。

**Architecture:** 在现有 LangGraph 薄图上新增 `refund` 出口；退款子流程用 `interrupt()` 暂停等订单选择，`Command(resume=...)` 从 checkpoint 续跑；扩写只在检索侧，`RagPipeline.retrieve_many` 合并去重；意图默认质量档，配置可切成本档；前端沿用现有 SSE/静态页，无构建步骤。

**Tech Stack:** Python 3.12、FastAPI、LangGraph 1.2.15、LangChain、SQLAlchemy、Milvus（现有）、原生 JS/CSS。

**Spec:** `docs/superpowers/specs/2026-10-18-ch06-intent-router-design.md`

## Global Constraints

- 所有代码写入 `MemHelp-code/`；不新建仓库、不把 `.venv` / `.uv-cache` 搬进代码目录。
- 不新增组件/框架：沿用现有 FastAPI + LangGraph + SQLAlchemy + Milvus + 原生前端。
- 不训练/微调意图分类模型，不做 BERT；不跨会话记忆。
- 扩写只在检索侧；入库侧不拆存多份知识。
- 前端是 Vibe Coding 例外：不套 TDD / code review；以浏览器实际效果为准。
- 涉及 LangGraph / FastAPI / SQLAlchemy / LangChain / Milvus 的新用法，先用 Context7 查官方文档；本计划已对 `interrupt/Command` 做真机探针。
- 配置只从 `app/config.py` 读；`app/llm.py` 是唯一 import `langchain_openai` 的模块（`tests/test_smoke.py` 例外）。
- 所有回给客户端的错误文本必须过 `redact()`；错误外壳统一 `{"error": {"type", "message"}}`。
- 不把 `create_ticket` 放回 Agent 工具集；退款单由用户在前端点提交。
- 提交前跑：`git diff --cached --name-only | Select-String "\.env$"`，应无输出。
- 测试命令默认在 `MemHelp-code/` 下执行：`uv run python -m pytest ... -q --color=no`。
- 每完成一个任务，往 `dev-notes/ch06.md` 追加一段（用户原话 / 产物 / 纠偏 / 翻车），不许收尾补记。

**留痕节奏（每个 Task 的最后一个动作）**：在该 Task 的 commit 之后，立即往
`dev-notes/ch06.md` 追加「阶段 N · Task X」四段式记录，再开始下一个 Task。不要攒到 Task 15。

## Review Focus

1. **订单选择器暂停时用户又发新消息**：不能把新消息塞进暂停中的图，也不能丢消息；应重新发 `awaiting_input` 并保持 checkpoint 可续跑。测试放在 Task 9。
2. **coref 模型编造订单号/型号**：补全结果里新出现的编号必须来自原话或历史，否则整句回退。测试放在 Task 3。
3. **政策检索整体失败（Milvus 未起）**：退款轮仍要进 Agent 并明确说无法确认，不能 500、不能编造规则。测试放在 Task 6/8。
4. **意图模型返回坏 JSON / 空回复 / `other`**：必须可解析或走兜底，不能把异常冒到 SSE。测试放在 Task 4。
5. **用户重复点订单卡片/重复提交退款单**：不能重复建单、不能把续跑流接错气泡。测试放在 Task 10/11。

---

### Task 0: 基线、Context7 与 interrupt 真机探针

**Files:**
- Create: `scripts/probe_ch06_interrupt.py`
- Test: 无（探针脚本）
- Modify: `dev-notes/ch06.md`（追加探针结果）

**Interfaces:**
- Consumes: `app.agent.checkpoint.open_checkpointer`、`langgraph.types.interrupt/Command`
- Produces: 探针结论；后续任务据此使用 `interrupt()` / `Command(resume=...)`

- [ ] **Step 1: 跑基线测试**

Run:
```powershell
cd MemHelp-code
uv run python -m pytest -q --color=no
```

Expected: 全绿（ch05 收尾基线；记录 passed/skipped 数字）。

- [ ] **Step 2: 查 Context7 的 LangGraph interrupt 文档**

用 Context7 查 `/websites/langchain_oss_python_langgraph` 的 “interrupt / Command resume / human-in-the-loop” 页面；记录：
`interrupt(value)` 暂停、`Command(resume=...)` 恢复、同一 `thread_id`、`astream(stream_mode="custom")` 的消费方式。

- [ ] **Step 3: 写探针脚本**

`scripts/probe_ch06_interrupt.py` 用 `AsyncSqliteSaver`（`open_checkpointer("data/probe-ch06.sqlite")`）编译一个小图：

```python
# prepare 节点发 {"type":"order_picker","orders":["A1","B2"]}
# await_order 节点：无 order 时 interrupt({"kind":"order_picker","orders":["A1","B2"]})
# finish 节点发 token
```

断言并打印：
- 第一次 `astream` 收到 `order_picker`；
- `snapshot.next == ("await_order",)`；
- `snapshot.tasks[0].interrupts[0].value["orders"] == ["A1","B2"]`；
- `Command(resume={"order_id":"A1"})` 后收到 token，最终 state `order == "A1"`。

- [ ] **Step 4: 运行探针**

Run:
```powershell
cd MemHelp-code
uv run python scripts/probe_ch06_interrupt.py
```

Expected: 四行断言全部 `True`；脚本退出码 0。

- [ ] **Step 5: 清理探针数据库并记录**

删除 `MemHelp-code/data/probe-ch06.sqlite`（确认路径后删除）；把探针输出与 Context7 结论追加到 `dev-notes/ch06.md` 阶段 2。

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/scripts/probe_ch06_interrupt.py MemHelp-code/dev-notes/ch06.md
git commit -m "ch06: interrupt/resume 真机探针与基线"
```

---

### Task 1: 配置与模型工厂

**Files:**
- Modify: `app/config.py`（新增 `INTENT_*` / `COREF_*` / `QUERY_EXPANSION_*` / `REFUND_ORDER_COUNT`）
- Modify: `app/llm.py`（`get_chat_model(settings=None, model_name=None)`）
- Modify: `.env.example`
- Test: `tests/test_ch06_config.py`、`tests/test_llm.py`（补充）

**Interfaces:**
- Consumes: `Settings`
- Produces:
  - `Settings.intent_model: str`、`intent_cheap_model: str`、`intent_cost_mode: bool`、`intent_confidence_threshold: float`、`intent_timeout: float`
  - `Settings.coref_enabled: bool`、`coref_timeout: float`、`coref_history_messages: int`
  - `Settings.query_expansion_enabled: bool`、`query_expansion_timeout: float`、`query_expansion_max_queries: int`
  - `Settings.refund_order_count: int`
  - `get_chat_model(settings: Settings | None = None, model_name: str | None = None) -> BaseChatModel`

- [ ] **Step 1: 写失败测试**

```python
from app.config import Settings
from app.llm import get_chat_model

def _settings(**overrides):
    base = dict(
        openai_base_url="https://example.com/v1",
        openai_api_key="sk-test",
        openai_model="base-model",
        _env_file=None,
    )
    base.update(overrides)
    return Settings(**base)

def test_ch06_config_defaults():
    s = _settings()
    assert s.intent_model == ""
    assert s.intent_cost_mode is False
    assert s.intent_confidence_threshold == 0.6
    assert s.coref_enabled is True
    assert s.coref_history_messages == 8
    assert s.query_expansion_enabled is True
    assert s.query_expansion_max_queries == 4
    assert s.refund_order_count == 3

def test_chat_model_name_can_be_overridden():
    model = get_chat_model(_settings(), model_name="intent-big")
    assert model.model_name == "intent-big"

def test_chat_model_defaults_to_openai_model():
    model = get_chat_model(_settings())
    assert model.model_name == "base-model"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_config.py -q --color=no`
Expected: FAIL（字段/参数不存在）。

- [ ] **Step 3: 实现配置与模型工厂**

在 `Settings` 中加字段，全部带默认值与 `Field(ge/le/gt)` 校验；`intent_confidence_threshold` 允许 `0.0–1.0`；`query_expansion_max_queries` 允许 `2–6`；`refund_order_count` 允许 `1–5`。
`get_chat_model` 用 `model_name or s.openai_model` 传给 `GatewayChatOpenAI`，其余不变。
`.env.example` 加一段带注释的 ch06 配置。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_config.py tests/test_llm.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/config.py MemHelp-code/app/llm.py MemHelp-code/.env.example \
        MemHelp-code/tests/test_ch06_config.py MemHelp-code/tests/test_llm.py
git commit -m "ch06: 意图/指代/扩写配置与模型工厂"
```

---

### Task 2: State 契约与八类→五出口映射

**Files:**
- Modify: `app/agent/state.py`
- Test: `tests/test_ch05_state.py`（更新为新契约；保留文件名避免无谓移动）

**Interfaces:**
- Consumes: 无
- Produces:
  - `Intent = Literal[..., "other"]`（8 项）
  - `Route = Literal["knowledge", "business", "refund", "complaint", "chitchat"]`
  - `ALL_INTENTS: tuple[Intent, ...]`（8 项，顺序固定）
  - `INTENT_TO_ROUTE: dict[Intent, Route]`
  - `AgentState` 新增 §5 的退款/扩写/coref 字段

- [ ] **Step 1: 写失败测试**

```python
from app.agent.state import ALL_INTENTS, INTENT_TO_ROUTE, AgentState, route_for

def test_eight_intents_map_to_five_exits():
    assert len(ALL_INTENTS) == 8
    assert INTENT_TO_ROUTE == {
        "logistics": "business",
        "order": "business",
        "product_consult": "knowledge",
        "refund_return": "refund",
        "after_sales": "refund",
        "complaint": "complaint",
        "chitchat": "chitchat",
        "other": "business",
    }

def test_unknown_route_falls_back_to_business():
    assert route_for("unknown") == "business"
    assert route_for("") == "business"
    assert route_for("other") == "business"

def test_refund_state_fields_exist():
    fields = set(AgentState.__annotations__)
    for name in ("resolved_message", "coref_changed", "coref_source",
                 "refund_order_id", "refund_order_candidates", "refund_order",
                 "refund_order_error", "expanded_queries",
                 "query_expansion_degraded", "refund_form",
                 "intent_llm_calls", "coref_llm_calls"):
        assert name in fields, name
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch05_state.py -q --color=no`
Expected: FAIL（8 vs 7、refund 缺失）。

- [ ] **Step 3: 实现**

按 spec §5 更新 `state.py`；保留 ch05 已有字段；`Stop` 的 `ticket_type` 不变。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch05_state.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/state.py MemHelp-code/tests/test_ch05_state.py
git commit -m "ch06: 八类意图五出口与退款 state 契约"
```

---

### Task 3: 指代消解 + Query 改写

**Files:**
- Modify: `app/agent/coref.py`（重写）
- Modify: `app/agent/prompts.py`（新增 `COREF_PROMPT`）
- Test: `tests/test_ch06_coref.py`

**Interfaces:**
- Consumes: `langchain_core.messages.BaseMessage`、`HumanMessage/AIMessage`、`TRIVIAL_CHITCHAT`/`is_trivial_chitchat`
- Produces:
  - `@dataclass(frozen=True) CorefResult(resolved: str, changed: bool, raw: str, source: Literal["llm","passthrough","fallback"], reason: str = "")`
  - `needs_coref(message: str, history: Sequence[BaseMessage]) -> bool`
  - `format_history(messages: Sequence[BaseMessage], limit: int) -> str`
  - `parse_coref_reply(reply: str, original: str, history_text: str) -> tuple[str, bool] | None`
  - `class CorefResolver: __init__(model, *, timeout=5.0, history_limit=8, enabled=True); async resolve(message, history) -> CorefResult`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.agent.coref import CorefResolver, needs_coref, parse_coref_reply

class _Model:
    def __init__(self, reply): self.reply, self.calls = reply, 0
    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)

def test_complete_question_passthrough():
    assert needs_coref("退货政策是什么", []) is False
    assert needs_coref("你好", []) is False

def test_anaphora_needs_coref():
    history = [HumanMessage(content="订单 20240115001 的物流到哪了"),
               AIMessage(content="运输中。")]
    assert needs_coref("那它能退吗", history) is True

def test_colloquial_needs_coref():
    assert needs_coref("钱啥时候回来", []) is True

def test_parse_rejects_missing_resolved():
    assert parse_coref_reply('{"changed": true}', "原话", "") is None

def test_parse_tolerates_code_fence():
    parsed = parse_coref_reply(
        '```json\n{"resolved":"订单 20240115001 能退吗","changed":true}\n```',
        "那它能退吗",
        "订单 20240115001 的物流到哪了",
    )
    assert parsed == ("订单 20240115001 能退吗", True)

async def test_resolver_uses_history():
    history = [HumanMessage(content="订单 20240115001 的物流到哪了"),
               AIMessage(content="运输中。")]
    model = _Model('{"resolved":"订单 20240115001 能退吗","changed":true}')
    result = await CorefResolver(model).resolve("那它能退吗", history)
    assert result.resolved == "订单 20240115001 能退吗"
    assert result.changed is True
    assert result.source == "llm"
    assert model.calls == 1

async def test_hallucinated_order_number_falls_back():
    model = _Model('{"resolved":"订单 20249999999 能退吗","changed":true}')
    result = await CorefResolver(model).resolve("那它能退吗", [])
    assert result.resolved == "那它能退吗"
    assert result.source == "fallback"

async def test_timeout_falls_back():
    class _Slow:
        async def ainvoke(self, messages):
            import asyncio; await asyncio.sleep(1); return AIMessage(content="{}")
    result = await CorefResolver(_Slow(), timeout=0.01).resolve("钱啥时候回来", [])
    assert result.resolved == "钱啥时候回来"
    assert result.source == "fallback"

def test_format_history_ignores_tool_and_limits():
    msgs = [HumanMessage(content="a"), ToolMessage(content="tool", tool_call_id="1"),
            AIMessage(content="b"), HumanMessage(content="c")]
    text = format_history(msgs, limit=2)
    assert "tool" not in text
    assert text.count("\n") <= 2
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_coref.py -q --color=no`
Expected: FAIL（函数/类不存在）。

- [ ] **Step 3: 实现**

- `COREF_PROMPT` 按 spec §7.1；包含 three-shot。
- 触发条件按 spec §6.1；`needs_coref` 先判 `is_trivial_chitchat`。
- 解析容忍代码块/前后解释；`parse_coref_reply` 负责 JSON 与 `resolved` 非空。
- 防幻觉：订单号 8–20 位数字、型号 `[A-Za-z]{2,}-\d{2,}[A-Za-z]?`；新 token 不在 `original + history_text` 中 → 返回 `None`，由 resolver 回退。
- `format_history` 只取 Human/AI 文本，每条截 200 字，取最后 `limit` 条。
- resolver 异常/超时/None → `CorefResult(original, False, raw, "fallback", reason)`。
- `changed` 以去空白后比较为准。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_coref.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/coref.py MemHelp-code/app/agent/prompts.py MemHelp-code/tests/test_ch06_coref.py
git commit -m "ch06: LLM 指代消解与 Query 改写"
```

---

### Task 4: 正式意图识别（八类 + 成本档）

**Files:**
- Modify: `app/agent/intent.py`
- Modify: `app/agent/prompts.py`（重写 `INTENT_PROMPT`）
- Test: `tests/test_ch05_intent.py`（更新）、`tests/test_ch06_intent.py`（新增）

**Interfaces:**
- Consumes: `Intent`/`ALL_INTENTS`（Task 2）、`INTENT_PROMPT`
- Produces:
  - `IntentResult(intent, confidence, raw, source, llm_calls=0)`
  - `class IntentClassifier: __init__(primary_model, *, cheap_model=None, timeout=5.0, confidence_threshold=0.6, cost_mode=False); async classify(message: str) -> IntentResult`
  - `parse_intent_reply(reply) -> IntentResult | None`（新增 `other` 别名）
  - `heuristic_intent(message) -> Intent`（默认 `other`）

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.messages import AIMessage
from app.agent.intent import IntentClassifier, heuristic_intent, parse_intent_reply

class _Model:
    def __init__(self, *replies): self.replies, self.calls = list(replies), 0
    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.replies.pop(0))

def test_parse_other():
    r = parse_intent_reply('{"intent":"other","confidence":0.42}')
    assert r and r.intent == "other"

def test_heuristic_default_is_other():
    assert heuristic_intent("这个东西怎么弄") == "other"
    assert heuristic_intent("我要投诉并退款") == "complaint"

async def test_quality_mode_only_calls_primary():
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    r = await IntentClassifier(primary).classify("这个能退吗")
    assert r.source == "llm" and r.llm_calls == 1
    assert primary.calls == 1

async def test_cost_mode_escalates_low_confidence():
    cheap = _Model('{"intent":"logistics","confidence":0.3}')
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    r = await IntentClassifier(cheap, cheap_model=cheap, cost_mode=True,
                               confidence_threshold=0.6).classify("那它能退吗")
    assert r.intent == "refund_return"
    assert r.source == "llm_escalated" and r.llm_calls == 2

async def test_cost_mode_accepts_high_confidence_cheap():
    cheap = _Model('{"intent":"order","confidence":0.95}')
    primary = _Model('{"intent":"refund_return","confidence":0.9}')
    r = await IntentClassifier(cheap, cheap_model=cheap, cost_mode=True).classify("订单 1001 多少钱")
    assert r.intent == "order" and r.source == "llm_cheap"
    assert primary.calls == 0

async def test_bad_cheap_json_escalates():
    cheap = _Model("我觉得是物流")
    primary = _Model('{"intent":"logistics","confidence":0.9}')
    r = await IntentClassifier(cheap, cheap_model=cheap, cost_mode=True).classify("到哪了")
    assert r.source == "llm_escalated" and r.intent == "logistics"

async def test_bad_primary_falls_back_to_heuristic():
    primary = _Model("nonsense")
    r = await IntentClassifier(primary).classify("我要投诉")
    assert r.source == "heuristic" and r.intent == "complaint"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch05_intent.py tests/test_ch06_intent.py -q --color=no`
Expected: FAIL（`other`、成本档、`llm_calls` 不存在）。

- [ ] **Step 3: 实现**

- 重写 `INTENT_PROMPT`：八类枚举、强制 JSON、边界 few-shot（至少 8 组）、`other` 兜底；明确不要 markdown。
- `_ALIASES` 加 `"其他": "other"`；`ALL_INTENTS` 校验。
- `IntentResult` 加 `llm_calls`；`_call_model` 返回 parsed 或 None。
- 质量档/成本档按 spec §6.2；cheap 的 confidence 阈值用 `>=`；坏 JSON/异常都升级。
- `heuristic_intent` 默认 `other`；关键词表不变。
- 保留寒暄快速通道与 `is_trivial_chitchat` 的整句匹配语义。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch05_intent.py tests/test_ch06_intent.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/intent.py MemHelp-code/app/agent/prompts.py \
        MemHelp-code/tests/test_ch05_intent.py MemHelp-code/tests/test_ch06_intent.py
git commit -m "ch06: 八类意图 prompt 与成本档升级"
```

---

### Task 5: Query 扩写（强制 JSON `queries`）

**Files:**
- Create: `app/agent/query_expansion.py`
- Modify: `app/agent/prompts.py`（新增 `QUERY_EXPANSION_PROMPT`）
- Test: `tests/test_ch06_query_expansion.py`

**Interfaces:**
- Consumes: 聊天模型、`settings.query_expansion_*`
- Produces:
  - `@dataclass(frozen=True) ExpansionResult(queries: tuple[str, ...], raw: str, degraded: bool)`
  - `parse_expansion_reply(reply: str, question: str, *, max_queries: int) -> tuple[str, ...] | None`
  - `class QueryExpander: __init__(model, *, timeout=5.0, max_queries=4, enabled=True); async expand(question: str, *, order: Mapping[str, Any] | None = None) -> ExpansionResult`

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.messages import AIMessage
from app.agent.query_expansion import QueryExpander, parse_expansion_reply

class _Model:
    def __init__(self, reply): self.reply, self.calls = reply, 0
    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.reply)

def test_parse_queries_and_dedupe():
    q = parse_expansion_reply(
        '{"queries":["退货时间窗口","退货运费谁承担","退货时间窗口"]}',
        "这个能退吗", max_queries=4,
    )
    assert q == ("退货时间窗口", "退货运费谁承担")

def test_parse_rejects_missing_or_empty():
    assert parse_expansion_reply('{"q":["a"]}', "问题", max_queries=4) is None
    assert parse_expansion_reply('{"queries":[]}', "问题", max_queries=4) is None

def test_parse_caps_max_and_keeps_original():
    q = parse_expansion_reply(
        '{"queries":["a","b","c","d","e"]}', "原问题", max_queries=3,
    )
    assert len(q) == 3
    assert "原问题" in q

async def test_expand_includes_order_context():
    model = _Model('{"queries":["退货时间窗口","退货运费谁承担"]}')
    result = await QueryExpander(model).expand(
        "这个能退吗", order={"order_id": "20240115001", "product_name": "耳机"},
    )
    assert result.degraded is False
    assert len(result.queries) == 2
    assert model.calls == 1

async def test_expand_falls_back_on_bad_json():
    result = await QueryExpander(_Model("不是 JSON")).expand("这个能退吗")
    assert result.queries == ("这个能退吗",)
    assert result.degraded is True
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_query_expansion.py -q --color=no`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

- Prompt 按 spec §7.3；订单上下文只放 `order_id/product_name/status/amount_yuan`。
- `parse_expansion_reply`：容忍代码块/前后解释；只认 `queries` 数组；去空、去重、截断；若原问题不在结果里，插到第一条，再截断。
- `expand`：`enabled=False` 或模型失败 → `(question,)` + `degraded`；成功 → 解析结果。
- 超时/异常不抛。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_query_expansion.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/query_expansion.py MemHelp-code/app/agent/prompts.py \
        MemHelp-code/tests/test_ch06_query_expansion.py
git commit -m "ch06: 退款/售后 Query 扩写"
```

---

### Task 6: 多查询检索合并

**Files:**
- Modify: `app/knowledge/pipeline.py`
- Test: `tests/test_ch06_retrieval_many.py`

**Interfaces:**
- Consumes: 现有 `RagPipeline._retrieve`、`RetrievedChunk`、`RetrievalOutcome`
- Produces:
  - `RetrievalOutcome` 新增 `queries: tuple[UnderstoodQuery, ...] = ()`
  - `async RagPipeline.retrieve_many(queries: Sequence[str], *, category: str | None = None) -> RetrievalOutcome`

- [ ] **Step 1: 写失败测试**

```python
import pytest
from app.knowledge.hybrid import RetrievedChunk, StrategyResult
from app.knowledge.pipeline import RagPipeline
from app.knowledge.query import UnderstoodQuery

class _Understander:
    async def understand(self, query):
        return UnderstoodQuery(original=query, standard=query, bm25_text=query)

class _Retriever:
    def __init__(self, mapping, *, fail=()): self.mapping, self.fail, self.calls = mapping, set(fail), []
    async def retrieve(self, strategy, query, *, category=None):
        self.calls.append(query.original)
        if query.original in self.fail: raise RuntimeError("milvus down")
        chunks = self.mapping.get(query.original, ())
        return StrategyResult(chunks=chunks, strategy="hybrid_rerank")

def _chunk(cid, score):
    return RetrievedChunk(chunk_id=cid, question="q", answer="a", category="c",
                          section_path="s", content_type="manual", score=score, source="rerank")

async def test_merge_dedupes_and_keeps_max_score():
    retriever = _Retriever({"a": (_chunk(1, 0.4), _chunk(2, 0.9)),
                            "b": (_chunk(1, 0.8), _chunk(3, 0.7))})
    outcome = await RagPipeline(understander=_Understander(), retriever=retriever,
                                sufficiency=object(), min_score=0.3, top_n=3).retrieve_many(["a", "b"])
    assert [c.chunk_id for c in outcome.evidence] == [2, 1, 3]
    assert outcome.top_score == 0.9
    assert outcome.queries[0].original == "a"

async def test_single_query_failure_does_not_abort():
    retriever = _Retriever({"a": (_chunk(1, 0.9),)}, fail={"b"})
    outcome = await RagPipeline(understander=_Understander(), retriever=retriever,
                                sufficiency=object(), top_n=3).retrieve_many(["a", "b"])
    assert outcome.refused is False
    assert [c.chunk_id for c in outcome.evidence] == [1]

async def test_all_fail_returns_refused():
    retriever = _Retriever({}, fail={"a", "b"})
    outcome = await RagPipeline(understander=_Understander(), retriever=retriever,
                                sufficiency=object(), top_n=3).retrieve_many(["a", "b"])
    assert outcome.refused is True
    assert outcome.evidence == ()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_retrieval_many.py -q --color=no`
Expected: FAIL（`retrieve_many` 不存在）。

- [ ] **Step 3: 实现**

- 循环 `await self._retrieve(query, category=category)`；单条 `Exception` 记 warning 并继续。
- 以 `chunk_id` 去重取最高分；按分降序；截断 `top_n`。
- `top_score = max`；`rerank_degraded` 任一为真；`queries` 收集。
- 0 命中或（未降级且 top_score < min_score）→ `refused=True`、`refusal_source="retrieval_low_conf"`；全失败 reason 写「多查询检索全部失败」。
- 保留 `query=queries[0]`（空输入时用空字符串理解结果或直接 refused；实现时给空序列一个安全分支）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_retrieval_many.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/knowledge/pipeline.py MemHelp-code/tests/test_ch06_retrieval_many.py
git commit -m "ch06: 多查询检索去重合并"
```

---

### Task 7: 演示订单、订单号抽取与退款原因

**Files:**
- Create: `app/agent/refund_orders.py`
- Test: `tests/test_ch06_refund_orders.py`

**Interfaces:**
- Consumes: `app.tools.business.query_order/query_product`、`derive_rng`
- Produces:
  - `REFUND_REASONS: tuple[str, ...]`（六个固定类目）
  - `extract_order_id(text: str) -> str | None`
  - `list_demo_orders(session_id: str, count: int = 3) -> list[dict[str, Any]]`
  - `build_refund_form(order: Mapping[str, Any]) -> dict[str, Any]`

- [ ] **Step 1: 写失败测试**

```python
import json
from app.agent.refund_orders import REFUND_REASONS, build_refund_form, extract_order_id, list_demo_orders

def test_extract_order_id_patterns():
    assert extract_order_id("订单 20240115001 的物流到哪了") == "20240115001"
    assert extract_order_id("订单号：1001") == "1001"
    assert extract_order_id("单号 2024-1001-AB") == "2024-1001-AB"
    assert extract_order_id("这个能退吗") is None

def test_demo_orders_are_stable_and_complete():
    first = list_demo_orders("s-demo", 3)
    second = list_demo_orders("s-demo", 3)
    assert first == second
    assert len(first) == 3
    assert len({o["order_id"] for o in first}) == 3
    for order in first:
        assert order["order_id"] and order["title"] and order["status"]
        assert order["amount_yuan"]
        assert order["product_name"] and order["product_model"]

def test_refund_reasons_are_fixed_six():
    assert REFUND_REASONS == (
        "商品质量问题", "尺寸/颜色/型号不合适", "与描述不符",
        "不想要了（七天无理由）", "物流破损/少件", "其他",
    )

def test_build_refund_form_shape():
    form = build_refund_form({"order_id": "1001", "product_name": "耳机",
                              "product_model": "A1", "amount_yuan": "99.00"})
    assert form["order_id"] == "1001"
    assert form["reasons"] == list(REFUND_REASONS)
    assert form["submit_url"] == "/api/actions/create-refund"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_refund_orders.py -q --color=no`
Expected: FAIL（模块不存在）。

- [ ] **Step 3: 实现**

- `extract_order_id`：先匹配 `(?:订单(?:号)?|单号)[：:\s]*([A-Za-z0-9-]{4,32})`；再匹配 8–20 位独立数字；返回第一个。
- `list_demo_orders`：`derive_rng("demo-orders", session_id)`；生成 `2024` + 7 位数字；去重；每单调 `query_order`/`query_product` 解析；组装 `order_id/title/amount_yuan/status/created_at/product_id/product_name/product_model`。
- `title = f"{product_name} {product_model}"`。
- `build_refund_form`：`order_label = f"{product_name} {product_model} · ¥{amount_yuan}"`；返回 `order_id/order_label/reasons/submit_url`。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_refund_orders.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/refund_orders.py MemHelp-code/tests/test_ch06_refund_orders.py
git commit -m "ch06: 演示订单选择器与固定退款原因"
```

---

### Task 8: 图接线与退款确定性子流程

**Files:**
- Modify: `app/agent/graph.py`
- Modify: `app/agent/prompts.py`（Agent Prompt 增加 refund/other/resolved 段落）
- Modify: `app/agent/logging.py`（`build_turn_log` 新增字段）
- Modify: `tests/test_ch05_graph.py`（更新四出口/七类断言）
- Test: `tests/test_ch06_refund_graph.py`

> `run_react_agent` 的签名不变：它在 `graph._agent` 里已经接收拼好的 `messages`，system prompt 由 graph 构造，不需要改 `react.py`。

**Interfaces:**
- Consumes: Task 2–7 的所有接口
- Produces:
  - `GraphDeps(model, intent_model, intent_cheap_model, repositories, retriever, settings)`（`graph.py` 内部）
  - 节点：`resolve_coref`、`classify_intent`、`route_intent`、`knowledge_retrieve`、`confidence_gate`、`refund_prepare_order`、`refund_await_order`、`refund_expand_queries`、`refund_retrieve_policies`、`refund_agent`、`agent`、`complaint`、`chitchat`、`fallback`、`log_turn`
  - `build_agent_graph(*, checkpointer=None)` 不变签名
  - `build_agent_system_prompt(session_id, evidence=(), *, resolved_question="", refund_order=None, refund_mode=False, other_intent=False)`

- [ ] **Step 1: 写失败测试**

用 `ScriptedChatModel` 与假仓库；复用 `tests/test_ch05_graph.py` 的 `_run`/`_evidence`：

```python
import json
import pytest
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.knowledge.citations import build_evidence
from app.knowledge.hybrid import RetrievedChunk
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from tests.fakes_agent import ScriptedChatModel, text_turn
from tests.test_ch05_graph import _evidence, _run

def _coref(resolved: str, changed: bool):
    return text_turn(json.dumps({"resolved": resolved, "changed": changed}, ensure_ascii=False))

def _intent(intent: str, confidence: float = 0.9):
    return text_turn(json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False))

def _expand(*queries: str):
    return text_turn(json.dumps({"queries": list(queries)}, ensure_ascii=False))

class _RefundRetriever:
    def __init__(self, *, evidence=_evidence(), refused=False):
        self.evidence, self.refused, self.many_calls = evidence, refused, []

    async def retrieve_only(self, query, *, category=None):
        return RetrievalOutcome(query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
                                evidence=self.evidence, refused=self.refused)

    async def retrieve_many(self, queries, *, category=None):
        self.many_calls.append(list(queries))
        return RetrievalOutcome(query=UnderstoodQuery(original=queries[0], standard=queries[0], bm25_text=queries[0]),
                                queries=tuple(UnderstoodQuery(original=q, standard=q, bm25_text=q) for q in queries),
                                evidence=self.evidence, top_score=0.9, refused=self.refused)

async def test_refund_with_order_goes_through_deterministic_subflow(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[
        _intent("logistics"), text_turn("运输中。"),
        _coref("订单 20240115001 能退吗", True), _intent("refund_return"),
        _expand("订单 20240115001 的退货条件", "退货运费谁承担"),
        text_turn("可以退，按政策…… [1]"),
    ])
    retriever = _RefundRetriever()

    await _run(graph, message="订单 20240115001 的物流到哪了", model=model,
               repositories=fake_repositories, retriever=retriever,
               settings=test_settings, session="s-refund")
    values, _, _ = await _run(graph, message="那它能退吗", model=model,
                              repositories=fake_repositories, retriever=retriever,
                              settings=test_settings, session="s-refund")

    assert values["route"] == "refund"
    assert values["intent"] == "refund_return"
    assert values["coref_changed"] is True
    assert values["refund_order"]["order_id"] == "20240115001"
    for node in ("refund_prepare_order", "refund_await_order", "refund_expand_queries",
                 "refund_retrieve_policies", "refund_agent"):
        assert node in values["trace"]
    assert retriever.many_calls and len(retriever.many_calls[0]) == 2
    assert values["refund_form"]["reasons"][:2] == ["商品质量问题", "尺寸/颜色/型号不合适"]

async def test_missing_order_emits_picker_and_interrupts(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    values, events, config = await _run(
        graph, message="这个能退吗", model=model, repositories=fake_repositories,
        retriever=_RefundRetriever(), settings=test_settings, session="s-picker", turn_id="t-picker",
    )
    assert [e["type"] for e in events].count("order_picker") == 1
    assert len(values["refund_order_candidates"]) == 3
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("refund_await_order",)
    assert snapshot.tasks[0].interrupts[0].value["orders"] == values["refund_order_candidates"]

async def test_resume_with_selected_order_finishes(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    values, _, config = await _run(
        graph, message="这个能退吗", model=model, repositories=fake_repositories,
        retriever=_RefundRetriever(), settings=test_settings, session="s-resume", turn_id="t-resume",
    )
    picked = values["refund_order_candidates"][0]["order_id"]
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])

    events = []
    async for event in graph.astream(Command(resume={"order_id": picked}), config, stream_mode="custom"):
        events.append(event)

    snapshot = await graph.aget_state(config)
    final = snapshot.values
    assert final["answer"] == "可以退。"
    assert final["refund_order"]["order_id"] == picked
    assert "refund_agent" in final["trace"]
    assert not any(e.get("type") == "order_picker" for e in events)

async def test_invalid_selection_reinterrupts(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    _, _, config = await _run(
        graph, message="这个能退吗", model=model, repositories=fake_repositories,
        retriever=_RefundRetriever(), settings=test_settings, session="s-bad-pick", turn_id="t-bad",
    )
    await graph.astream(Command(resume={"order_id": "not-in-list"}), config, stream_mode="custom")
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("refund_await_order",)

async def test_policy_retrieval_empty_still_calls_agent(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[
        _intent("refund_return"), _expand("退货条件", "退货运费"), text_turn("这项政策我暂时无法确认。"),
    ])
    retriever = _RefundRetriever(evidence=(), refused=True)
    values, _, _ = await _run(
        graph, message="订单 20240115001 能退吗", model=model,
        repositories=fake_repositories, retriever=retriever,
        settings=test_settings, session="s-empty-policy",
    )
    assert values["answer"] == "这项政策我暂时无法确认。"
    assert "refund_agent" in values["trace"]
    assert values["stop_reason"] != "gate_failed"

async def test_other_intent_goes_business_and_asks_clarification(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[
        _coref("你们公司股票能买吗", False), _intent("other"), text_turn("你想咨询哪方面？"),
    ])
    values, _, _ = await _run(
        graph, message="你们公司股票能买吗", model=model,
        repositories=fake_repositories, retriever=_RefundRetriever(),
        settings=test_settings, session="s-other",
    )
    assert values["route"] == "business"
    assert values["intent"] == "other"
    system = model.state.calls[-1][0].content
    assert "意图未确定" in system
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_refund_graph.py -q --color=no`
Expected: FAIL（节点/路由不存在）。

- [ ] **Step 3: 实现图**

- `_deps` 改为 `GraphDeps`；`config` 键：`model/intent_model/intent_cheap_model/repositories/retriever/settings`。
- `_resolve_coref`：历史 = `messages[:-1]`；调用 `CorefResolver`；发 `coref` 事件；写 state。
- `_classify_intent`：按 settings 构造 `IntentClassifier`；发 `intent` 事件；写 state；`llm_calls` 加 `result.llm_calls`。
- `_route_intent`：`route_for`，`ticket_type`。
- `_refund_prepare_order`：按 Task 7；发 `order_picker`；写 state。
- `_refund_await_order`：按 spec §6.5；`interrupt`；查订单；组装 `refund_order`。
- `_refund_expand_queries`：`QueryExpander`；写 state；`llm_calls`。
- `_refund_retrieve_policies`：`retrieve_many` 或 fallback；写 evidence；不 gate。
- `_refund_agent`：调用共享 `_run_agent`，`refund_mode=True`，`include_faq=False`，`refund_form`；trace。
- `_agent`：最后一轮问题用 `resolved_message` 构造模型输入；system prompt 传 `resolved_question`、`refund_order`、`refund_mode`、`other_intent`；落库仍用原话。
- 边：`route_intent` 加 `refund` 分支；`refund_await_order` 条件边（有 `refund_order_error`/无订单 → `refund_agent`，否则 `refund_expand_queries`）；`refund_retrieve_policies` → `refund_agent`；`refund_agent` → `log_turn`。
- `build_agent_system_prompt` 增加对应段落；`other` 段落一句话澄清；refund 段落订单数据/政策/不重复查/证据不足不编造。
- `logging.build_turn_log` 增加 `resolved_message/coref_changed/coref_source/intent_source/expanded_queries/query_expansion_degraded/refund_order_id`。

- [ ] **Step 4: 更新 ch05 图测试**

- `test_policy_question_goes_through_retrieval_then_gate` 改用 `product_consult`（如「XH-3000A 续航多久」），不再用退款问题。
- 所有退款/售后断言改为 `route == "refund"` 或移到新退款测试。
- 完整问句仍走 coref 透传（零调用），脚本无需额外加 coref 回复；只有指代用例加 coref 回复。

- [ ] **Step 5: 跑测试确认通过**

Run:
```powershell
uv run python -m pytest tests/test_ch06_refund_graph.py tests/test_ch05_graph.py -q --color=no
```
Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/app/agent/graph.py MemHelp-code/app/agent/prompts.py \
        MemHelp-code/tests/test_ch05_graph.py MemHelp-code/tests/test_ch06_refund_graph.py
git commit -m "ch06: 退款确定性子流程与图接线"
```

---

### Task 9: Runner 暂停/续跑与 done 契约

**Files:**
- Modify: `app/agent/runner.py`
- Test: `tests/test_ch06_runner.py`

**Interfaces:**
- Consumes: `build_agent_graph`、`Command`
- Produces:
  - `pending_interrupt(snapshot) -> dict | None`
  - `astream_turn(..., intent_model=None, intent_cheap_model=None)` 在暂停时发 `awaiting_input`，不发 `done`
  - `astream_resume(graph, *, session_id, order_id, model, intent_model=None, intent_cheap_model=None, repositories, retriever, settings) -> AsyncIterator[dict]`
  - `complete_turn(...)` 暂停时返回 `(text, {"awaiting_input": {...}, "stop_reason": "awaiting_input", ...})`
  - `done_payload` 增加 `resolved_message/coref_changed/query_expansion/refund_form/trace`

- [ ] **Step 1: 写失败测试**

```python
import json
from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.agent.runner import astream_resume, astream_turn, complete_turn
from app.knowledge.pipeline import RetrievalOutcome
from app.knowledge.query import UnderstoodQuery
from tests.fakes_agent import ScriptedChatModel, text_turn
from tests.test_ch05_graph import _evidence

def _coref(resolved: str, changed: bool):
    return text_turn(json.dumps({"resolved": resolved, "changed": changed}, ensure_ascii=False))

def _intent(intent: str):
    return text_turn(json.dumps({"intent": intent, "confidence": 0.9}, ensure_ascii=False))

def _expand(*queries: str):
    return text_turn(json.dumps({"queries": list(queries)}, ensure_ascii=False))

class _Retriever:
    async def retrieve_only(self, query, *, category=None):
        return RetrievalOutcome(query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
                                evidence=_evidence(), top_score=0.9)

    async def retrieve_many(self, queries, *, category=None):
        return RetrievalOutcome(query=UnderstoodQuery(original=queries[0], standard=queries[0], bm25_text=queries[0]),
                                queries=tuple(UnderstoodQuery(original=q, standard=q, bm25_text=q) for q in queries),
                                evidence=_evidence(), top_score=0.9)

async def _collect(agen):
    return [event async for event in agen]

async def test_astream_turn_pauses_without_done(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    events = await _collect(astream_turn(
        graph, session_id="r1", message="这个能退吗", model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    ))
    assert any(e["type"] == "order_picker" for e in events)
    assert any(e["type"] == "awaiting_input" for e in events)
    assert not any(e["type"] == "done" for e in events)

async def test_astream_resume_finishes(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    events = await _collect(astream_turn(
        graph, session_id="r2", message="这个能退吗", model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    ))
    first_order = next(e for e in events if e["type"] == "awaiting_input")["payload"]["orders"][0]["order_id"]
    model.script.extend([_expand("退货条件", "退货运费"), text_turn("可以退。")])

    resumed = await _collect(astream_resume(
        graph, session_id="r2", order_id=first_order, model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    ))
    assert any(e["type"] == "token" for e in resumed)
    done = [e for e in resumed if e["type"] == "done"][-1]["payload"]
    assert done["refund_form"]["order_id"] == first_order
    assert "refund_agent" in done["trace"]

async def test_astream_turn_reprompts_when_pending(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    await _collect(astream_turn(
        graph, session_id="r3", message="这个能退吗", model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    ))
    calls_before = model.state.index
    events = await _collect(astream_turn(
        graph, session_id="r3", message="那先查物流", model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    ))
    assert [e["type"] for e in events] == ["awaiting_input"]
    assert model.state.index == calls_before

async def test_complete_turn_returns_awaiting_payload(test_settings, fake_repositories):
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(script=[_coref("这个商品能退吗", True), _intent("refund_return")])
    text, done = await complete_turn(
        graph, session_id="r4", message="这个能退吗", model=model,
        repositories=fake_repositories, retriever=_Retriever(), settings=test_settings,
    )
    assert text == "请选择要办理的订单。"
    assert done["awaiting_input"]["orders"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_runner.py -q --color=no`
Expected: FAIL（无 `awaiting_input`/`astream_resume`）。

- [ ] **Step 3: 实现**

- `_config` 加 `intent_model`/`intent_cheap_model`。
- `pending_interrupt(snapshot)`：遍历 `snapshot.tasks`，返回第一个 `interrupts[0].value`；无则 `None`。
- `astream_turn`：开头 `aget_state` 检查 pending；有则 `awaiting_input` 并 return；流结束后再检查；完成才发 tools/citations/done。
- `awaiting_payload(values, interrupt_value)`：`{"kind":"order_picker","turn_id":...,"orders":[...],"message":"请选择要办理的订单。"}`。
- `astream_resume`：`Command(resume={"order_id": order_id})`；同样处理暂停/完成；不调用 `begin_turn`。
- `complete_turn`：收集 token 与 awaiting；暂停时返回提示文本与合成 done。
- `done_payload` 加新字段。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_runner.py tests/test_ch05_graph.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/agent/runner.py MemHelp-code/tests/test_ch06_runner.py
git commit -m "ch06: runner 暂停续跑与 awaiting_input 契约"
```

---

### Task 10: API/SSE/退款单接口

**Files:**
- Modify: `app/api.py`
- Modify: `app/schemas.py`
- Test: `tests/test_ch06_api.py`、`tests/test_ch05_api.py`（更新）

**Interfaces:**
- Consumes: Task 9 runner、Task 8 图
- Produces:
  - `ChatResponse.awaiting_input: dict | None`、`resolved_message: str | None`、`coref_changed: bool`
  - `RefundSelectRequest(session_id, order_id)`
  - `CreateRefundRequest(session_id, order_id, reason, idempotency_key=None)`
  - `CreateRefundResponse(ok, refund_no, ticket_no, order_id, reason, status, message)`
  - `POST /api/chat/refund/select`（SSE）
  - `POST /api/actions/create-refund`
  - SSE 白名单加 `coref/intent/order_picker/awaiting_input`

- [ ] **Step 1: 写失败测试**

```python
def test_chat_stream_emits_order_picker_and_awaiting(client):
    # POST /api/chat/completions {"message":"这个能退吗","stream":True}
    # 解析 SSE 文本：含 event: order_picker、event: awaiting_input；不含 event: done

def test_refund_select_resumes_and_done(client):
    # 先拿 order_picker 的第一张卡；POST /api/chat/refund/select
    # SSE 含 token 与 done；done.refund_form.order_id == 卡号

def test_refund_select_without_pending_returns_409(client):
    resp = client.post("/api/chat/refund/select", json={"session_id":"nope","order_id":"1001"})
    assert resp.status_code == 409
    assert resp.json()["error"]["type"] == "no_pending_refund"

def test_create_refund_validates_reason(client):
    resp = client.post("/api/actions/create-refund", json={
        "session_id":"s1","order_id":"1001","reason":"瞎编的原因"})
    assert resp.status_code == 422

def test_create_refund_idempotent(client):
    body = {"session_id":"s1","order_id":"1001","reason":"商品质量问题",
            "idempotency_key":"refund-1"}
    first = client.post("/api/actions/create-refund", json=body).json()
    second = client.post("/api/actions/create-refund", json=body).json()
    assert first["refund_no"] == second["refund_no"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_ch06_api.py -q --color=no`
Expected: FAIL（端点不存在）。

- [ ] **Step 3: 实现**

- `create_app` 增加 `app.state.intent_model` / `intent_cheap_model` 注入；`get_intent_model`/`get_intent_cheap_model` 依赖：注入优先，否则 `get_chat_model(settings, name or None)`；缓存默认实例。
- `chat_completions` 传入 intent 模型；SSE 白名单加四类事件；`stream=false` 处理 `awaiting_input`。
- `/api/chat/refund/select`：校验 pending（`graph.aget_state` 的 `next` 包含 `refund_await_order`）；返回 `StreamingResponse(_refund_resume_body(...), media_type=SSE_MEDIA_TYPE)`。
- `/api/actions/create-refund`：`reason` Literal；`execute_tool("create_ticket", ...)`；错误映射 502/503；返回 `refund_no=ticket_no`。
- `schemas.py` 加请求/响应模型；`ChatResponse` 加字段。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run python -m pytest tests/test_ch06_api.py tests/test_ch05_api.py tests/test_api_sse.py -q --color=no`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/app/api.py MemHelp-code/app/schemas.py \
        MemHelp-code/tests/test_ch06_api.py MemHelp-code/tests/test_ch05_api.py
git commit -m "ch06: 订单续跑与退款单 API/SSE"
```

---

### Task 11: 前端（Vibe Coding 例外）

**Files:**
- Modify: `web/app.js`
- Modify: `web/style.css`
- Maybe: `web/index.html`（只加容器/说明时）
- Test: 浏览器验收（不写前端单测）

**Interfaces:**
- Consumes: Task 10 的 SSE 事件与端点
- Produces: 聊天流订单卡片、续跑、退款表单、coref/intent 提示

- [ ] **Step 1: 实现事件处理**

`applyFrame` 增加：
- `coref`：在 bot 气泡上方显示「已补全：{resolved}」；`changed=false` 不显示。
- `intent`：显示小徽章 `意图：{intent} · {confidence}`。
- `order_picker`：在 bot 气泡内渲染 `.order-picker`，每张卡显示 `title / order_id / ¥amount / status / created_at`。
- `awaiting_input`：标记本轮暂停，状态栏「等待选择订单」；不要显示「（客服没有返回内容）」。

- [ ] **Step 2: 实现选单续跑**

- 点击卡片：禁用整组卡片，显示「已选择订单 …」；`POST /api/chat/refund/select`，请求体 `{session_id, order_id}`。
- 复用同一个 SSE 解析/`applyFrame`，把 token 接到**同一张 bot 气泡**；续跑 `done` 后渲染工具/引用/退款表单。
- 失败：恢复卡片可点，显示 notice。

- [ ] **Step 3: 实现退款表单**

- `done.refund_form` 存在时渲染 `.refund-form`：订单只读、`<select>` 六类原因、提交按钮。
- 提交 `POST /api/actions/create-refund`，带 `idempotency_key`（可用 `sessionId + order_id` 生成）；成功显示 `refund_no` 并禁用表单；失败显示错误。
- 样式沿用现有气泡/圆角/配色。

- [ ] **Step 4: 本地跑起来**

Run:
```powershell
cd MemHelp-code
uv run uvicorn app.main:app --port 8000
```

打开 `http://127.0.0.1:8000/chat/`。

- [ ] **Step 5: 浏览器验收**

1. 输入「订单 20240115001 的物流到哪了」→ 意图 logistics，回答物流。
2. 输入「那它能退吗」→ 看到「已补全」、意图 refund_return、退款子流程回答。
3. 新会话输入「这个能退吗」→ 订单卡片出现；点一张 → 同一气泡继续输出 → 退款表单出现。
4. 选原因提交 → 显示工单号。
5. 截图保存 `dev-notes/ch06-order-picker.png`、`dev-notes/ch06-refund-form.png`。

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/web/app.js MemHelp-code/web/style.css MemHelp-code/dev-notes/ch06-order-picker.png MemHelp-code/dev-notes/ch06-refund-form.png
git commit -m "ch06: 前端订单卡片与退款表单"
```

---

### Task 12: 多轮标注用例与评估脚本

**Files:**
- Create: `evals/ch06_multiturn_cases.jsonl`
- Create: `evals/run_ch06_evals.py`
- Test: `tests/test_run_ch06_evals.py`
- Output: `dev-notes/ch06-evals.json`、`dev-notes/ch06-eval-report.md`

**Interfaces:**
- Consumes: `CorefResolver`、`IntentClassifier`、`QueryExpander`、真实 `Settings`/模型
- Produces:
  - `validate_case(case: dict) -> None`（非法 intent/缺字段抛 `ValueError`）
  - `summarise(rows: Sequence[dict]) -> dict`；每行含 `intent_ok/coref_ok/json_ok/intent`
  - 可重复运行的评估命令与报告

- [ ] **Step 1: 写标注用例**

至少 8 个 session / 24 轮，字段：

```json
{"id":"M01","turns":[
  {"user":"订单 20240115001 的物流到哪了","intent":"logistics",
   "coref_contains":["20240115001"],"coref_forbidden":["它","那个"]},
  {"user":"那它能退吗","intent":"refund_return",
   "coref_contains":["20240115001"],"coref_forbidden":["它"]},
  {"user":"那现在到哪了","intent":"logistics",
   "coref_contains":["20240115001"],"coref_forbidden":["它"]}
]}
```

覆盖：物流→退款→物流、商品→退款、售后、投诉优先、寒暄、怪问题→other。
怪问题至少 4 条：`你们公司股票能买吗`、`帮我看看那个`、`这个怎么弄`、`今天天气怎么样`。

- [ ] **Step 2: 写离线测试（解析/指标，不连模型）**

```python
def test_load_cases_rejects_unknown_intent():
    # 临时不写文件；直接调用 validate_case(dict) 返回错误
    with pytest.raises(ValueError):
        validate_case({"id":"X","turns":[{"user":"a","intent":"nope"}]})

def test_metrics_counts_other_and_coref():
    rows = [
        {"intent": "logistics", "intent_ok": True, "coref_ok": True, "json_ok": True},
        {"intent": "other", "intent_ok": True, "coref_ok": False, "json_ok": True},
        {"intent": "refund_return", "intent_ok": False, "coref_ok": True, "json_ok": True},
    ]
    report = summarise(rows)
    assert report["total"] == 3
    assert report["intent_accuracy"] == pytest.approx(2/3)
    assert report["coref_accuracy"] == pytest.approx(2/3)
    assert report["json_failures"] == 0
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run python -m pytest tests/test_run_ch06_evals.py -q --color=no`
Expected: FAIL（模块不存在）。

- [ ] **Step 4: 实现评估脚本**

- `validate_case`：intent 必须在 `ALL_INTENTS`；每轮必须有 `user/intent`。
- `summarise`：intent accuracy、other accuracy、coref pass rate、JSON parse failures、失败明细。
- `main`：读 `.env` 真实模型；每个 session 维护历史；逐轮 `CorefResolver.resolve` + `IntentClassifier.classify`；退款/售后轮再调 `QueryExpander` 检查 JSON/条数/含原问题；把结果写 JSON + Markdown。
- 不连 Milvus/MySQL；只评估 coref/intent/扩写。
- 阈值：intent accuracy ≥ 0.85，coref pass ≥ 0.85，JSON 解析失败 = 0；不达标不静默改模型，报告失败明细。

- [ ] **Step 5: 跑真机评估**

Run:
```powershell
cd MemHelp-code
uv run python evals/run_ch06_evals.py
```

Expected: 生成 `dev-notes/ch06-eval-report.md` 与 JSON；记录准确率、失败明细、模型名。

- [ ] **Step 6: 提交**

```bash
git add MemHelp-code/evals/ch06_multiturn_cases.jsonl \
        MemHelp-code/evals/run_ch06_evals.py \
        MemHelp-code/tests/test_run_ch06_evals.py \
        MemHelp-code/dev-notes/ch06-evals.json MemHelp-code/dev-notes/ch06-eval-report.md
git commit -m "ch06: 多轮 coref/意图/扩写评估集与报告"
```

---

### Task 13: 全量回归、演示脚本与文档

**Files:**
- Create: `scripts/demo-ch06.py`
- Modify: `README.md`
- Modify: `dev-notes/ch06.md`
- Test: `tests/test_demo_ch06.py`（只测纯函数/参数解析，不连真实模型）

**Interfaces:**
- Consumes: 全部功能
- Produces:
  - `parse_case(value: str) -> str`（`all|multiturn|other|refund|picker`，非法值抛 `ValueError`）
  - 可复制的演示命令、验收输出

- [ ] **Step 1: 全量回归**

Run:
```powershell
cd MemHelp-code
uv run python -m pytest -q --color=no
```

Expected: 全绿。失败逐条修；不改测试来掩盖真实回归。

- [ ] **Step 2: 写演示脚本 + 纯函数测试**

先写 `tests/test_demo_ch06.py`：

```python
import pytest
from scripts.demo_ch06 import parse_case

def test_parse_case_accepts_known_values():
    assert parse_case("all") == "all"
    assert parse_case("picker") == "picker"

def test_parse_case_rejects_unknown():
    with pytest.raises(ValueError):
        parse_case("nope")
```

`scripts/demo-ch06.py` 支持 `--case all|multiturn|other|refund|picker`：

- `multiturn`：用真实模型跑「物流→退款→物流」三轮，打印每轮 `user/resolved/intent/confidence`。
- `other`：跑 4 条怪问题，打印 intent；断言 JSON 可解析。
- `refund`：带订单号问「这个能退吗」，打印 coref、expanded_queries、trace、订单快照、政策证据数。
- `picker`：无订单号，打印订单卡片；模拟选第一张，续跑打印 done/refund_form。

真实依赖不可用（MySQL/Milvus/模型）时，打印清晰错误并退出码 1，不伪造结果。

- [ ] **Step 3: 跑演示**

Run:
```powershell
cd MemHelp-code
uv run python scripts/demo-ch06.py --case all
```

Expected: 四条验收各自打印 PASS/关键数字；保存输出到 `dev-notes/ch06-acceptance-output.txt`。

- [ ] **Step 4: 更新 README**

加：
- ch06 新配置；
- `POST /api/chat/refund/select`、`POST /api/actions/create-refund`；
- 演示命令；
- 订单选择器/退款表单的操作步骤；
- 评估命令。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/scripts/demo-ch06.py MemHelp-code/tests/test_demo_ch06.py \
        MemHelp-code/README.md MemHelp-code/dev-notes/ch06-acceptance-output.txt
git commit -m "ch06: 演示脚本、验收输出与 README"
```

---

### Task 14: 独立 code review 与处置

**Files:**
- Modify: 按 review 结论
- Modify: `dev-notes/ch06.md`

**Interfaces:**
- Consumes: 全量 diff、spec、plan
- Produces: review 结论与修复提交

- [ ] **Step 1: 加载 requesting-code-review 技能并派独立 reviewer**

给 reviewer：spec、plan、`git diff main...HEAD`、测试命令、验收标准。要求按 Critical / Important / Minor 分级，重点看：
- interrupt/resume 幂等与暂停轮落库；
- coref 防幻觉、prompt 注入；
- 多查询检索去重/降级；
- 退款单重复提交；
- 前端续跑流状态；
- 旧测试是否被削弱。

- [ ] **Step 2: 逐条核实并处置**

Critical/Important 必须修；Minor 说明取舍。不盲从 review：每条先在代码/测试里验证。

- [ ] **Step 3: 跑全量测试 + 评估**

Run:
```powershell
cd MemHelp-code
uv run python -m pytest -q --color=no
uv run python evals/run_ch06_evals.py
```

Expected: 全绿；评估达标或失败明细已记录。

- [ ] **Step 4: 提交 review 修复**

```bash
# 只加已跟踪文件的改动 + 本次明确改动的文件；不要 git add -A
git add -u MemHelp-code/app MemHelp-code/tests MemHelp-code/web
git add MemHelp-code/dev-notes/ch06.md
git commit -m "ch06: 处置独立评审（Critical/Important + 关键 Minor）"
```

---

### Task 15: 收尾与 finish

**Files:**
- Modify: `dev-notes/ch06.md`（finish 段）
- Maybe: `dev-notes/DELIVERY.md`

**Interfaces:**
- Consumes: 全部
- Produces: 最终交付说明与验收证据

- [ ] **Step 1: 加载 finishing-a-development-branch 技能**

按技能检查：分支状态、测试、评估、演示、文档、未提交文件、密钥。

- [ ] **Step 2: 最终验证**

Run:
```powershell
cd MemHelp-code
uv run python -m pytest -q --color=no
uv run python scripts/demo-ch06.py --case all
uv run python evals/run_ch06_evals.py
```

Expected: 全绿；演示四条 PASS；评估达标。

- [ ] **Step 3: 浏览器最终验收**

按 Task 11 的四步再走一遍；确认截图/输出是最新代码。

- [ ] **Step 4: 写 finish 留痕**

`dev-notes/ch06.md` 记：用户原话、最终产物（spec/plan/报告/截图路径）、评审结论、翻车与返工、交付命令。

- [ ] **Step 5: 提交**

```bash
git add MemHelp-code/dev-notes/ch06.md MemHelp-code/dev-notes/DELIVERY.md
git commit -m "ch06: 验收总结与交付留痕"
```

---

## 计划自查

- **Spec 覆盖**：§1 目标 → Task 3/4/5/8；§2 决策 → Task 2/8/9；§5 State → Task 2；§6 节点 → Task 8；§7 Prompt → Task 3/4/5/8；§8 多查询 → Task 6；§9 SSE/API/前端 → Task 9/10/11；§10 配置 → Task 1；§11 失败路径 → Task 3/4/5/6/8/9/10；§12 日志 → Task 8/9；§13 测试 → 各 Task + Task 12/13；§14 验收 → Task 11/13/15。
- **占位**：无 TBD/TODO；示例订单号统一 `20240115001`。
- **类型一致**：`CorefResult` / `IntentResult` / `ExpansionResult` / `RetrievalOutcome` / `GraphDeps` 的字段在 Task 间一致；`retrieve_many` 在 Task 6 定义、Task 8 使用。
- **Review Focus**：5 条均有对应测试（Task 3/4/6/8/9/10/11）。
- **比例**：计划只固定接口、测试断言与命令；实现体由执行者按现有代码风格写。
