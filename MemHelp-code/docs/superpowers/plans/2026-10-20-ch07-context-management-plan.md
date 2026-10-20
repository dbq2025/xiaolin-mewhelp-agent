# 第七章「会话上下文管理」实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ch01 的一层 `trim_messages` 裁剪升级成三层上下文管理：近处原文不压、中间按规则截短、远处由后台异步摘要代表；预算从模型窗口倒推；两个锚点（`summary_upto_msg_id` / `layer1_from_msg_id`）只挪 id 不搬数据；每轮把实际发出的上下文原样落 `log/app.log`；前端加多会话侧栏。

**Architecture:** `app/context.py` 升级为 `app/context/` 包，拆成「校准计数器 / 预算 / 分层 / 拼装」四个纯函数模块，输入是纯数据（`MessageRow` 列表 + 两个 int），不碰 DB 与网络，可全离线测。摘要成独立模块 `app/agent/summarizer.py`，用 `asyncio.create_task` 在请求之外跑，自己开 Session 落库，全程 try/except 只落日志。数据侧给 `conversations` 加两个锚点列 + 新开 `conversation_summaries` 表（一段一行，只 append）。可观测性单独 `app/agent/ctxlog.py`。前端沿用现有静态页，无构建步骤。

**Tech Stack:** Python 3.13（`MemHelp-code/.venv`）、FastAPI、LangGraph（`State.messages` + `add_messages` + checkpointer）、LangChain（`trim_messages`）、SQLAlchemy 2.x、MySQL、原生 JS/CSS。

**Spec:** `docs/superpowers/specs/2026-10-20-ch07-context-management-design.md`

## Global Constraints

- **测试与脚本一律用 `MemHelp-code/.venv` 的 python**，不要用 `uv run`：本机沙箱里 `uv` 初始化缓存直接失败（`failed to open file ...sdists-v9\.git: 拒绝访问。 (os error 5)`）。命令模板：`cd MemHelp-code; .\.venv\Scripts\python.exe -m pytest ...`。
- **不改 `agent_max_steps` 的默认值（6）**：`tests/test_ch05_config_checkpoint.py:45` 钉死它。`MAX_AGENT_STEPS` 只用 `AliasChoices` 兼容，与本字段同为 6 的默认值。
- **`tests/test_context.py` 的 13 个用例一行都不许改**：`build_context(...)` 是 ch01–ch06 的兼容外壳，行为必须逐字节一致（含 `used_tokens` 仍用 approximate 口径）。
- **两个口径必须同源**：预算里所有常量按校准口径标定，新路径的 `used_tokens` 也必须用校准计数器算；legacy 外壳继续用 approximate，互不串。
- **`SystemMessage` 只有第 0 条，且同一会话内每轮字节一致**：梗概与证据不许进 system，也不许单开 system；上游 chat template 会把所有 system 上提合并，工具定义被挤到可变内容之后，前缀缓存整段作废。
- **`sql/` 下每个 `.sql` 都必须含 `SET NAMES utf8mb4`**（`tests/test_repo_conventions.py` 静态检查）。ch07 迁移照 `sql/ch05-ddl.sql` 的 `information_schema` + `PREPARE` 幂等写法。
- **不新增依赖**：不装 tiktoken、不装 `langchain` 的中间件包；摘要用现有 `BaseChatModel`。
- **不换方案**：三层结构、两个锚点、摘要单表、后台异步、固定拼装顺序都是用户点名的选型，走不通就停下来问。
- **前端是例外**：`web/` 不套 TDD / code review，以浏览器实际效果为准（沿用 ch06 惯例）。
- **涉及库用法先查 Context7**：`trim_messages` 的 `token_counter` 注入、`add_messages` reducer、FastAPI 路径参数、SQLAlchemy 2.x `mapped_column` 都已在本计划写前核过；新出现未核过的 API 时先查再写。
- **每完成一个 Task，立刻往 `dev-notes/ch07.md` 追加一段**（我这一步发的关键原话 / 你的关键产出 / 我拒绝或纠偏了什么 / 翻车与返工），不许收尾补记。
- 提交前跑 `git diff --cached --name-only | Select-String "\.env$"`，应无输出。

**留痕节奏（每个 Task 的最后一个动作）**：commit 之后立即追加 `dev-notes/ch07.md`，再开下一个 Task。

## Review Focus

> 五条 spec 隐含、但不会被「照着 spec 写」自然覆盖到的输入类风险。每条都指定了必须补测试的 Task。

1. **降级只能前进，不能后退**：`layer1_from_msg_id` 若在某轮被算小（例如那一轮历史被外部删了几条），层 1 会去读已被摘要覆盖的区间 —— 同一批事实被二次有损压缩，订单号开始变形。要求 `set_anchors` 保证 `layer1_from_msg_id >= summary_upto_msg_id + 1` 且单调不减。测试放 **Task 3 + Task 6**。
2. **多进程下 in-flight 去重失效**：`_inflight` 是进程内集合，两个 uvicorn worker 会各自起一次摘要，压同一段。兜底必须是数据库唯一键 `UNIQUE(conversation_id, segment_no)`：冲突时放弃并留痕，不许重试改号。测试放 **Task 6 + Task 8**。
3. **摘要把「旧段背景」当成本次事实写进新段**：提示词里旧段是背景，但模型完全可能把背景里的订单号写进新段，于是同一事实被反复搬运、逐次变形。要求一条测试：新段内容里不出现只存在于旧段的事实。测试放 **Task 8**。
4. **system prompt 每轮漂移**：只要有人把「当前时间」「摘要长度」之类塞进 system，前缀缓存就整段作废，而且不会有任何报错。要求一条测试：同一会话两轮（历史不同、证据不同）产出的 `messages[0].content` **完全相等**。测试放 **Task 4 + Task 9**。
5. **回载会话后 session_id 串台**：侧栏点击只换了视图、没换 `sessionStorage.session_id` 时，用户以为在跟旧会话聊，消息其实落到新会话 —— 数据错位且无报错。测试放 **Task 12**（UI 检查脚本断言回载后的 `session_id` 与列表项一致）。

---

### Task 1: 校准 token 计数器

**Files:**
- Create: `MemHelp-code/app/context/tokens.py`
- Create: `MemHelp-code/app/context/__init__.py`（把旧 `app/context.py` 的 `ContextResult` / `build_context` **原样搬进来**）
- Delete: `MemHelp-code/app/context.py`
- Test: `MemHelp-code/tests/test_ch07_tokens.py`

> **为什么 Task 1 就要搬家**：Python 里同名的包目录会**遮蔽**同名模块（`FileFinder` 先找目录）。
> 只建 `app/context/tokens.py` 而留着 `app/context.py`，`app/context/` 一旦成为包，
> 旧的 `build_context` 就再也 import 不到，`tests/test_context.py` 13 条立刻全红。
> 所以「建包」与「搬旧实现」必须同一步完成。

**Interfaces:**
- Consumes: `langchain_core.messages.BaseMessage`
- Produces:
  - `CJK_CHARS_PER_TOKEN_DEFAULT: float = 1.0`、`OTHER_CHARS_PER_TOKEN_DEFAULT: float = 3.5`、`MESSAGE_OVERHEAD_TOKENS_DEFAULT: int = 3`
  - `count_text(text: str, *, cjk_chars_per_token=1.0, other_chars_per_token=3.5) -> int`
  - `count_tokens(messages: Sequence[BaseMessage], *, cjk_chars_per_token=1.0, other_chars_per_token=3.5, message_overhead=3) -> int`（满足 `trim_messages(token_counter=...)` 的 `Callable[[list[BaseMessage]], int]` 签名）
  - `make_token_counter(*, cjk_chars_per_token=1.0, other_chars_per_token=3.5, message_overhead=3) -> Callable[[Sequence[BaseMessage]], int]`
  - `is_cjk(ch: str) -> bool`

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_tokens.py`：

```python
"""校准 token 口径。

为什么不能用 `count_tokens_approximately` 的默认口径：实测 50 字中文（28 个汉字）
只算 17 tokens，真实分词约 40–50 —— 低估近 3 倍，方向是危险的那一侧（会撑爆窗口）。
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.context.tokens import count_text, count_tokens, make_token_counter


def test_chinese_costs_about_one_token_per_char():
    assert count_text("你好世界") == 4          # 4 汉字 × 1.0


def test_ascii_costs_chars_over_3_5_rounded_up():
    assert count_text("hello world") == 4       # ceil(11 / 3.5) = 4


def test_mixed_text_adds_both_sides():
    assert count_text("你好abc") == 3           # 2 + ceil(3/3.5)=1


def test_message_overhead_is_per_message():
    one = count_tokens([HumanMessage("你好世界")])
    two = count_tokens([HumanMessage("你好世界"), AIMessage("你好世界")])
    assert one == 4 + 3
    assert two == 2 * (4 + 3)


def test_role_name_is_not_counted():
    """role 名不计字符：否则同一句正文当 user 与当 ai 会算出不同 token。"""
    assert count_tokens([HumanMessage("你好世界")]) == count_tokens([AIMessage("你好世界")])


def test_non_string_content_does_not_raise():
    """多模态 content 是 list；这里只要求稳定，不要求精确。"""
    msg = HumanMessage(content=[{"type": "text", "text": "你好"}])
    assert count_tokens([msg]) >= 1


def test_make_token_counter_matches_count_tokens():
    msgs = [HumanMessage("你好世界"), AIMessage("hello world")]
    assert make_token_counter()(msgs) == count_tokens(msgs)


def test_calibrated_counter_is_stricter_than_approximate_for_chinese():
    """同一条中文，校准口径必须比 approx 大 —— 这是「不低估」的最低要求。"""
    from langchain_core.messages.utils import count_tokens_approximately

    msgs = [HumanMessage("我上周买的耳机还没发货，订单号 A123456789")]
    assert count_tokens(msgs) > count_tokens_approximately(msgs)


def test_empty_list_is_zero():
    assert count_tokens([]) == 0
```

- [ ] **Step 2: 跑测试确认全红**

Run:
```powershell
cd MemHelp-code
.\.venv\Scripts\python.exe -m pytest tests/test_ch07_tokens.py -q --color=no
```
Expected: `ModuleNotFoundError: No module named 'app.context.tokens'`。

- [ ] **Step 3: 实现**

要点：CJK 判定用 `"\u4e00" <= ch <= "\u9fff"` 加全角标点区 `\u3000-\u303f` / `\uff00-\uffef`；`count_text` 用 `math.ceil`；`count_tokens` 对每条取 `str(content)`（list content 走 `str()`，只求稳定）。**不要**在 `count_tokens` 内部读 `Settings` —— 保持纯函数，参数由调用方注入，否则 `trim_messages` 的二分里会反复读配置。

- [ ] **Step 4: 跑测试确认全绿 + 跑全量基线**

Run:
```powershell
cd MemHelp-code
.\.venv\Scripts\python.exe -m pytest tests/test_ch07_tokens.py -q --color=no
.\.venv\Scripts\python.exe -m pytest -q --color=no
```
Expected: 新用例全绿；全量仍是 `858 passed, 39 skipped`（此时 `app/context` 还是旧文件，包尚未接管 —— 若 import 冲突，本步只跑新用例并说明）。

- [ ] **Step 5: commit + 追记 dev-notes**

```powershell
git add MemHelp-code/app/context/tokens.py MemHelp-code/tests/test_ch07_tokens.py MemHelp-code/dev-notes/ch07.md
git commit -m "ch07: 校准 token 计数器（中文按字数折 token）"
```
然后往 `dev-notes/ch07.md` 追加「阶段 2 · Task 1」。

---

### Task 2: 预算倒推

**Files:**
- Create: `MemHelp-code/app/context/budget.py`
- Modify: `MemHelp-code/app/config.py`（新增 ch07 配置键，含 `AliasChoices`）
- Modify: `MemHelp-code/tests/conftest.py`（env 清理清单追加新键）
- Test: `MemHelp-code/tests/test_ch07_budget.py`、`MemHelp-code/tests/test_ch07_config.py`

**Interfaces:**
- Consumes: `app.config.Settings`
- Produces:
  - `@dataclass(frozen=True) class BudgetReport`：`model_context_window / output_reserve / react_peak / safety_margin / system_reserve / evidence_reserve / summary_reserve / overhead / window_available / desired_history / history_budget / layer1_budget / layer2_budget / ok: bool / reason: str`
  - `compute_budget(settings) -> BudgetReport`
  - `format_budget_report(report) -> str`（一行摘要，供 lifespan 打 INFO）
  - `INSUFFICIENT_BUDGET_PREFIX = "上下文预算不足"`

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_config.py`（配置键存在性与别名）：

```python
from app.config import Settings


def _base(**over):
    data = dict(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    data.update(over)
    return Settings(**data)


def test_ch07_defaults():
    s = _base()
    assert s.model_context_window == 32000
    assert s.max_output_tokens == 4000
    assert s.max_user_input_tokens == 2000
    assert s.tool_result_max_tokens == 1200
    assert s.safety_margin_ratio == 0.10
    assert s.system_prompt_reserve_tokens == 1200
    assert s.evidence_tokens_per_item == 250
    assert s.summary_reserve_tokens == 500
    assert s.history_desired_turns == 20
    assert s.history_tokens_per_turn == 400
    assert s.layer1_ratio == 0.7
    assert s.layer2_ratio == 0.3
    assert s.assistant_preview_chars == 60
    assert s.summary_enabled is True
    assert s.summary_model == ""
    assert s.summary_timeout_seconds == 20
    assert s.summary_max_tokens == 300
    assert s.cjk_chars_per_token == 1.0
    assert s.other_chars_per_token == 3.5
    assert s.message_overhead_tokens == 3
    assert s.app_log_path == "log/app.log"


def test_max_agent_steps_accepts_both_env_names(monkeypatch):
    """新名 MAX_AGENT_STEPS 与旧名 AGENT_MAX_STEPS 都要能读；默认仍是 6。"""
    assert _base().agent_max_steps == 6
    monkeypatch.setenv("MAX_AGENT_STEPS", "3")
    assert _base().agent_max_steps == 3
    monkeypatch.delenv("MAX_AGENT_STEPS")
    monkeypatch.setenv("AGENT_MAX_STEPS", "4")
    assert _base().agent_max_steps == 4


def test_rerank_top_k_accepts_both_env_names(monkeypatch):
    assert _base().rag_rerank_top_n == 10
    monkeypatch.setenv("RERANK_TOP_K", "5")
    assert _base().rag_rerank_top_n == 5
    monkeypatch.delenv("RERANK_TOP_K")
    monkeypatch.setenv("RAG_RERANK_TOP_N", "7")
    assert _base().rag_rerank_top_n == 7


def test_ch07_budget_settings_reject_invalid_values():
    import pytest
    from pydantic import ValidationError

    for bad in (
        {"model_context_window": 0},
        {"laywr1_ratio": 0.0},
        {"safety_margin_ratio": 1.5},
        {"cjk_chars_per_token": 0.0},
    ):
        with pytest.raises(ValidationError):
            _base(**bad)
```

`tests/test_ch07_budget.py`：

```python
"""预算倒推：演示配置必须逐项对上验收第 2 条的数字。"""
from app.config import Settings
from app.context.budget import INSUFFICIENT_BUDGET_PREFIX, compute_budget


def _demo(**over):
    data = dict(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        model_context_window=18000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        tool_result_max_tokens=1200,
        agent_max_steps=3,
        rag_rerank_top_n=5,
    )
    data.update(over)
    return compute_budget(Settings(**data))


def test_demo_configuration_reproduces_the_documented_numbers():
    r = _demo()
    assert (r.output_reserve, r.react_peak) == (2000, 5600)
    assert r.safety_margin == 1800
    assert (r.system_reserve, r.evidence_reserve, r.summary_reserve) == (1200, 1250, 500)
    assert r.overhead == 12350
    assert r.window_available == 5650
    assert r.desired_history == 8000
    assert r.history_budget == 5650
    assert r.layer2_budget == 1695
    assert r.layer1_budget == 3954
    assert r.ok is True


def test_default_window_is_healthy():
    r = compute_budget(Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m"))
    assert r.ok is True
    assert r.history_budget >= r.desired_history  # 默认窗口装得下 20 轮 → 取 desired


def test_too_small_window_reports_insufficient_budget():
    r = _demo(model_context_window=8000)
    assert r.ok is False
    assert r.reason.startswith(INSUFFICIENT_BUDGET_PREFIX)
    assert r.window_available <= 0


def test_history_budget_never_below_one_turn_when_ok():
    """ok=True 时至少要能装下一轮；否则 ok 必须是 False。"""
    for window in (12000, 15000, 18000, 32000):
        r = _demo(model_context_window=window)
        if r.ok:
            assert r.history_budget >= 400


def test_layer_budgets_split_the_history_budget():
    r = _demo()
    assert r.layer1_budget + r.layer2_budget + 1 == r.history_budget


def test_format_report_contains_the_key_numbers():
    text = compute_budget(Settings(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")).reason if False else None
    from app.context.budget import format_budget_report
    line = format_budget_report(_demo())
    for piece in ("window=18000", "history=5650", "layer1=3954", "layer2=1695"):
        assert piece in line
```

- [ ] **Step 2: 跑测试确认全红**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ch07_budget.py tests/test_ch07_config.py -q --color=no`
Expected: `ImportError: cannot import name 'compute_budget'` / 属性不存在。

- [ ] **Step 3: 实现**

`app/config.py` 三处改动：

1. 顶部 `from pydantic import AliasChoices`（若未导入）。
2. `agent_max_steps` 的 `Field` 加 `validation_alias=AliasChoices("AGENT_MAX_STEPS", "MAX_AGENT_STEPS")`；`rag_rerank_top_n` 加 `validation_alias=AliasChoices("RAG_RERANK_TOP_N", "RERANK_TOP_K")`。**默认值一个字不改。**
3. 文件末段（`refund_order_count` 之后）追加 ch07 一整块字段，全部带中文 `description`，取值约束按 Step 1 的用例。

`app/context/budget.py` 严格照 spec §4 的公式，`layer1_budget = history_budget - layer2_budget - 1`。`ok=False` 的两种情形与 `reason` 文本按 spec §4.2。`format_budget_report` 输出形如：

```
window=18000 overhead=12350 history=5650 layer1=3954 layer2=1695
```

`tests/conftest.py` 的 env 清理元组追加：`MODEL_CONTEXT_WINDOW / MAX_OUTPUT_TOKENS / MAX_USER_INPUT_TOKENS / TOOL_RESULT_MAX_TOKENS / MAX_AGENT_STEPS / RERANK_TOP_K / SAFETY_MARGIN_RATIO / SYSTEM_PROMPT_RESERVE_TOKENS / EVIDENCE_TOKENS_PER_ITEM / SUMMARY_RESERVE_TOKENS / HISTORY_DESIRED_TURNS / HISTORY_TOKENS_PER_TURN / LAYER1_RATIO / LAYER2_RATIO / ASSISTANT_PREVIEW_CHARS / SUMMARY_ENABLED / SUMMARY_MODEL / SUMMARY_TIMEOUT_SECONDS / SUMMARY_MAX_TOKENS / CJK_CHARS_PER_TOKEN / OTHER_CHARS_PER_TOKEN / MESSAGE_OVERHEAD_TOKENS / APP_LOG_PATH`。

- [ ] **Step 4: 跑测试全绿 + 全量不回归**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ch07_budget.py tests/test_ch07_config.py tests/test_config.py tests/test_ch05_config_checkpoint.py -q --color=no
.\.venv\Scripts\python.exe -m pytest -q --color=no
```

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 预算从模型窗口倒推（演示配置逐项对上）`。

---

### Task 3: 三层划分与规则截短

**Files:**
- Create: `MemHelp-code/app/context/layers.py`
- Test: `MemHelp-code/tests/test_ch07_layers.py`

**Interfaces:**
- Consumes: `app.db.repositories.MessageRow`、`app.context.tokens.make_token_counter`、`app.context.budget.BudgetReport`、`langchain_core.messages.utils.trim_messages`
- Produces:
  - `LAYER_SUMMARIZED = 3` / `LAYER_TRIMMED = 2` / `LAYER_VERBATIM = 1`
  - `classify(msg_id: int, *, summary_upto: int | None, layer1_from: int | None) -> int`
  - `rule_trim_row(row: MessageRow, *, assistant_preview_chars: int = 60) -> MessageRow`
  - `row_to_message(row: MessageRow) -> BaseMessage`（复用 `_row_to_message` 的语义，但接受已经截短的行）
  - `@dataclass(frozen=True) class LayerPlan`：`rows: list[MessageRow]`（**全部**行，按 id 升序）、`layer2: list[MessageRow]`、`layer1: list[MessageRow]`、`layer1_from_msg_id: int | None`、`summary_upto_msg_id: int | None`、`layer1_tokens: int`、`layer2_tokens: int`、`degraded: bool`、`summary_due: bool`、`summary_rows: list[MessageRow]`、`dropped_layer1_rows: int`
  - `plan_layers(rows, *, summary_upto, layer1_from, layer1_budget, layer2_budget, assistant_preview_chars=60, counter=None) -> LayerPlan`

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_layers.py`（覆盖：分层判定、规则截短三类角色、层 1 降级按整轮切、降级不后退、层 2 超预算置 `summary_due`）：

```python
"""三层划分 + 规则截短。全部纯函数，不碰 DB。"""
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.context.layers import (
    LAYER_SUMMARIZED, LAYER_TRIMMED, LAYER_VERBATIM,
    classify, plan_layers, rule_trim_row,
)


def _row(i, role, content, **over):
    from app.db.repositories import MessageRow
    return MessageRow(id=i, role=role, content=content, tool_calls=None,
                      tool_call_id=None, created_at=datetime.now(timezone.utc), **over)


def _turns(n, *, start=1, user_chars=20, ai_chars=20):
    rows, i = [], start
    for k in range(n):
        rows.append(_row(i, "user", "问" * user_chars + str(k))); i += 1
        rows.append(_row(i, "assistant", "答" * ai_chars + str(k))); i += 1
    return rows


def test_classify_three_layers():
    assert classify(5, summary_upto=5, layer1_from=10) == LAYER_SUMMARIZED
    assert classify(6, summary_upto=5, layer1_from=10) == LAYER_TRIMMED
    assert classify(9, summary_upto=5, layer1_from=10) == LAYER_TRIMMED
    assert classify(10, summary_upto=5, layer1_from=10) == LAYER_VERBATIM


def test_classify_with_null_anchors_puts_everything_in_layer1():
    assert classify(1, summary_upto=None, layer1_from=None) == LAYER_VERBATIM


def test_rule_trim_keeps_user_text_verbatim():
    long = "我上周买的耳机还没发货" * 20
    assert rule_trim_row(_row(1, "user", long)).content == long


def test_rule_trim_shortens_assistant_to_preview():
    row = _row(2, "assistant", "好" * 200)
    out = rule_trim_row(row, assistant_preview_chars=60)
    assert out.content == "好" * 60 + "…"
    assert len(out.content) == 61


def test_rule_trim_leaves_short_assistant_untouched():
    row = _row(2, "assistant", "好的呢")
    assert rule_trim_row(row, assistant_preview_chars=60).content == "好的呢"


def test_rule_trim_keeps_empty_assistant_body_empty():
    """纯工具调用那行正文是空串，截短不能把它变成「…」。"""
    assert rule_trim_row(_row(3, "assistant", "")).content == ""


def test_rule_trim_replaces_tool_result_with_one_line_marker():
    row = _row(4, "tool", "很长的工具结果" * 100, tool_call_id="call_abc")
    out = rule_trim_row(row)
    assert out.content.startswith("[工具结果 · ")
    assert "call_abc" in out.content
    assert "字已省略]" in out.content
    assert len(out.content) < 60


def test_rule_trim_does_not_mutate_the_input_row():
    row = _row(2, "assistant", "好" * 200)
    rule_trim_row(row)
    assert row.content == "好" * 200


def test_plan_with_no_anchors_returns_everything_in_layer1():
    rows = _turns(3)
    plan = plan_layers(rows, summary_upto=None, layer1_from=None,
                       layer1_budget=99999, layer2_budget=99999)
    assert plan.layer2 == []
    assert [r.id for r in plan.layer1] == [r.id for r in rows]
    assert plan.degraded is False and plan.summary_due is False


def test_plan_splits_by_anchors():
    rows = _turns(5)                       # id 1..10
    plan = plan_layers(rows, summary_upto=4, layer1_from=8,
                       layer1_budget=99999, layer2_budget=99999)
    assert [r.id for r in plan.layer2] == [5, 6, 7]
    assert [r.id for r in plan.layer1] == [8, 9, 10]


def test_layer1_over_budget_moves_the_anchor_forward():
    rows = _turns(10)                      # 20 条
    plan = plan_layers(rows, summary_upto=None, layer1_from=1,
                       layer1_budget=60, layer2_budget=99999)
    assert plan.degraded is True
    assert plan.dropped_layer1_rows > 0
    assert plan.layer1_from_msg_id > 1
    # 按整轮切：新的层 1 第一条必须是 user
    assert plan.layer1[0].role == "user"
    assert plan.layer1[-1].id == rows[-1].id


def test_layer1_degradation_only_moves_forward():
    """Review Focus 1：降级只前进不后退，绝不回到摘要区间里。"""
    rows = _turns(10)
    plan = plan_layers(rows, summary_upto=8, layer1_from=9,
                       layer1_budget=60, layer2_budget=99999)
    assert plan.layer1_from_msg_id >= 9


def test_single_turn_that_does_not_fit_is_kept_and_flagged():
    rows = [_row(1, "user", "问" * 5000), _row(2, "assistant", "答" * 5000)]
    plan = plan_layers(rows, summary_upto=None, layer1_from=None,
                       layer1_budget=10, layer2_budget=99999)
    assert plan.degraded is True
    assert [r.id for r in plan.layer1] == [1, 2]   # 宁可超，也不让模型看不到刚发生的事


def test_layer2_over_budget_is_due_for_summary():
    rows = _turns(5)                       # 层 2 = id 1..8
    plan = plan_layers(rows, summary_upto=0, layer1_from=9,
                       layer1_budget=99999, layer2_budget=50)
    assert plan.summary_due is True
    assert [r.id for r in plan.summary_rows] == list(range(1, 9))
    assert plan.layer2_tokens > 50


def test_layer2_within_budget_is_not_due():
    plan = plan_layers(_turns(5), summary_upto=0, layer1_from=9,
                       layer1_budget=99999, layer2_budget=99999)
    assert plan.summary_due is False
    assert plan.summary_rows == []


def test_layer2_tokens_are_counted_after_rule_trimming():
    """截短必须让层 2 明显变小，否则「按规则截短」没生效。"""
    rows = [_row(1, "user", "问"), _row(2, "assistant", "答" * 400)]
    full = plan_layers(rows, summary_upto=0, layer1_from=3,
                       layer1_budget=99999, layer2_budget=99999)
    trimmed = plan_layers(rows, summary_upto=0, layer1_from=3,
                          layer1_budget=99999, layer2_budget=99999,
                          assistant_preview_chars=10)
    assert trimmed.layer2_tokens < full.layer2_tokens
```

- [ ] **Step 2: 跑测试确认全红**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ch07_layers.py -q --color=no`

- [ ] **Step 3: 实现**

要点：

- `classify` 三行判定，`None` 视为「没有该边界」。
- `rule_trim_row` 用 `dataclasses.replace` 返回**新** `MessageRow`（frozen），不改入参。
- 层 1 降级：把层 1 的 `MessageRow` 转成 `BaseMessage`，用
  `trim_messages(msgs, max_tokens=layer1_budget, token_counter=counter, strategy="last", start_on="human", allow_partial=False, include_system=True)`，
  然后**按内容后缀对齐**回 `MessageRow` 列表（kept 是原列表的前缀截断，取 `rows[-len(kept):]`）。
  若 `kept == []`（一整轮都装不下）→ 保留全部层 1 行并 `degraded=True`。
  新的 `layer1_from_msg_id = layer1[0].id`，且必须 `>= max(layer1_from or 1, (summary_upto or 0) + 1)`。
- 层 2 预算判定用 `counter` 对**截短后**的消息计数。
- `plan_layers` 不写库、不改锚点，只**报告**建议的新锚点值（由调用方决定是否落库）。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 三层划分与规则截短（两个锚点只挪 id）`。

---

### Task 4: 固定顺序拼装 + 兼容外壳

**Files:**
- Create: `MemHelp-code/app/context/assemble.py`
- Modify: `MemHelp-code/app/context/__init__.py`（re-export `ContextResult` / `build_context` / `assemble_context`）
- Delete: `MemHelp-code/app/context.py`
- Test: `MemHelp-code/tests/test_ch07_assemble.py`（`tests/test_context.py` 的 13 个用例**一个字不改**，继续绿）

**Interfaces:**
- Consumes: `app.context.tokens.count_tokens`、`LayerPlan`
- Produces:
  - `@dataclass(frozen=True) class ContextResult`（字段兼容旧 6 个 + 新增带默认值：`summary_tokens=0`、`layer2_messages=0`、`layer1_messages=0`、`layer1_from_msg_id=None`、`summary_upto_msg_id=None`、`degraded=False`、`summary_due=False`）
  - `build_reference_block(*, summary_text="", evidence_text="") -> str`
  - `assemble_context(*, system_message, question, layer2=(), layer1=(), summary_text="", evidence_text="", budget_tokens=0, counter=count_tokens_approximately) -> ContextResult`
  - `build_context(history, question, system_message, budget_tokens) -> ContextResult`（**兼容外壳**，实现为 `assemble_context(layer1=history, counter=count_tokens_approximately)`）

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_assemble.py`：

```python
"""拼装顺序是硬件约束，不是风格偏好。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.context import assemble_context, build_context


SYSTEM = "你是客服小美。"


def _msg(i, role, text):
    return (HumanMessage if role == "user" else AIMessage)(text, id=str(i))


def test_order_is_system_layer2_layer1_question_reference():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer2=[_msg(1, "user", "旧问题"), _msg(2, "assistant", "旧回答…")],
        layer1=[_msg(3, "user", "近问题"), _msg(4, "assistant", "近回答")],
        question=HumanMessage("当前这句"),
        summary_text="用户问过耳机",
        evidence_text="证据1",
    )
    contents = [m.content for m in out.messages]
    assert contents[0] == SYSTEM
    assert contents[1].startswith("旧问题")
    assert contents[3].startswith("近问题")
    assert contents[4] == "当前这句"
    assert "用户问过耳机" in contents[5] and "证据1" in contents[5]


def test_only_one_system_message_and_it_is_first():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer1=[_msg(1, "user", "q"), _msg(2, "assistant", "a")],
        question=HumanMessage("当前"),
        summary_text="梗概", evidence_text="证据",
    )
    systems = [m for m in out.messages if isinstance(m, SystemMessage)]
    assert len(systems) == 1 and out.messages[0].is_same_type(systems[0])


def test_summary_and_evidence_are_merged_into_one_message():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM), question=HumanMessage("当前"),
        summary_text="梗概", evidence_text="证据",
    )
    refs = out.messages[-1]
    assert isinstance(refs, HumanMessage)
    assert "梗概" in refs.content and "证据" in refs.content


def test_no_reference_message_when_both_empty():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM), question=HumanMessage("当前"),
    )
    assert len(out.messages) == 2
    assert out.messages[-1].content == "当前"


def test_question_comes_before_reference():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM), question=HumanMessage("当前"),
        summary_text="梗概",
    )
    assert out.messages[-2].content == "当前"
    assert out.messages[-1].content != "当前"


def test_system_content_is_identical_across_two_turns():
    """Review Focus 4：system 一旦掺进可变内容，前缀缓存整段作废。"""
    first = assemble_context(
        system_message=SystemMessage(SYSTEM), question=HumanMessage("第一轮"),
    )
    second = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer1=[_msg(1, "user", "第一轮"), _msg(2, "assistant", "答")],
        question=HumanMessage("第二轮"),
        summary_text="一段梗概", evidence_text="一条证据",
    )
    assert first.messages[0].content == second.messages[0].content


def test_counts_report_each_layer():
    out = assemble_context(
        system_message=SystemMessage(SYSTEM),
        layer2=[_msg(1, "user", "a")], layer1=[_msg(2, "user", "b")],
        question=HumanMessage("c"),
    )
    assert out.layer2_messages == 1
    assert out.layer1_messages == 1
    assert out.kept_messages == 2
    assert out.trimmed_messages == 0


def test_legacy_shell_still_uses_approximate_counting():
    """兼容外壳必须逐字节复刻 ch01 行为：used_tokens 仍是 approx 口径。"""
    from langchain_core.messages.utils import count_tokens_approximately

    out = build_context(
        history=[HumanMessage("历史问题0"), AIMessage("历史回答0")],
        question=HumanMessage("本轮"), system_message=SystemMessage(SYSTEM),
        budget_tokens=2000,
    )
    assert out.used_tokens == count_tokens_approximately(out.messages[1:-1])
    assert out.budget_tokens == 2000
```

- [ ] **Step 2: 跑测试确认全红**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ch07_assemble.py tests/test_context.py -q --color=no`
Expected: 新用例 import 失败。**同时确认 `tests/test_context.py` 此刻仍绿**（旧 `app/context.py` 还在）。

- [ ] **Step 3: 实现**

- 先写 `assemble.py`，再把旧 `app/context.py` 的内容整体搬进 `app/context/__init__.py` 的 `build_context`（保留原 docstring 与 13 条注释），实现改为一层转调 `assemble_context(layer1=history, counter=count_tokens_approximately)`，**`used_tokens` / `system_prompt_tokens` 仍用 `count_tokens_approximately`**。
- 删 `app/context.py`。
- `build_reference_block` 输出形如 `[会话梗概]\n…\n[检索证据]\n…`，任一段为空则整段省略；两段都空返回 `""`。

- [ ] **Step 4: 跑绿 + 全量（重点：`test_context.py` 13 条一条不红）**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 固定顺序拼装（system 唯一且在首位）+ build_context 兼容外壳`。

---

### Task 5: 数据模型与迁移

**Files:**
- Modify: `MemHelp-code/app/db/models.py`（`Conversation` 加两列；新增 `ConversationSummary`）
- Modify: `MemHelp-code/sql/ch07-ddl.sql`（整份重写，幂等）
- Modify: `MemHelp-code/tests/test_db_models.py`（`test_conversation_columns_match_ddl` 期望列 + 新表用例）
- Modify: `MemHelp-code/tests/test_db_mysql.py`（`KNOWN_SCHEMA_DIVERGENCES["conversations"]["extra_columns"]`）
- Test: `MemHelp-code/tests/test_ch07_ddl.py`（静态检查迁移脚本）

**Interfaces:**
- Produces:
  - `Conversation.summary_upto_msg_id: Mapped[int | None]`、`Conversation.layer1_from_msg_id: Mapped[int | None]`（均可空，`BigInteger`）
  - `class ConversationSummary(Base)`：表 `conversation_summaries`，列 `id / conversation_id / segment_no / from_msg_id / upto_msg_id / content / msg_count / source_tokens / model / elapsed_ms / created_at`，`UniqueConstraint("conversation_id", "segment_no", name="uq_summary_segment")`，`Index("idx_conversation_id", "conversation_id", "segment_no")`
  - `SUMMARY_SEGMENT_UNIQUE = "uq_summary_segment"`

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_ddl.py`（静态检查，模仿 `test_repo_conventions.py` 的风格）：

```python
"""ch07 迁移脚本的静态约束。"""
from pathlib import Path

SQL = Path(__file__).resolve().parents[1] / "sql" / "ch07-ddl.sql"


def test_ddl_exists_and_sets_names():
    text = SQL.read_text(encoding="utf-8")
    assert "SET NAMES utf8mb4" in text


def test_ddl_is_idempotent():
    text = SQL.read_text(encoding="utf-8")
    assert "information_schema.COLUMNS" in text
    assert "PREPARE stmt FROM @ddl" in text
    assert text.count("DEALLOCATE PREPARE stmt") >= 3


def test_ddl_adds_the_two_anchors_and_drops_the_old_summary_column():
    text = SQL.read_text(encoding="utf-8")
    assert "summary_upto_msg_id" in text
    assert "layer1_from_msg_id" in text
    assert "CREATE TABLE IF NOT EXISTS conversation_summaries" in text
    assert "ADD COLUMN summary " not in text      # 旧双层方案不再落列


def test_ddl_creates_the_segment_unique_key():
    text = SQL.read_text(encoding="utf-8")
    assert "uq_summary_segment" in text
    assert "UNIQUE KEY" in text


def test_ddl_quotes_are_balanced():
    for lineno, line in enumerate(SQL.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        assert line.count("'") % 2 == 0, f"第 {lineno} 行单引号不成对: {line}"
```

`tests/test_db_models.py` 追加：

```python
def test_conversation_summary_columns_match_ddl():
    from app.db.models import ConversationSummary

    assert set(ConversationSummary.__table__.c.keys()) == {
        "id", "conversation_id", "segment_no", "from_msg_id", "upto_msg_id",
        "content", "msg_count", "source_tokens", "model", "elapsed_ms", "created_at",
    }


def test_conversation_summary_segment_no_is_unique_per_conversation():
    """Review Focus 2 的兜底：多进程重复压同一段时靠这个唯一键收口。"""
    from app.db.models import ConversationSummary

    names = {
        c.name for c in ConversationSummary.__table__.constraints if getattr(c, "name", None)
    }
    assert "uq_summary_segment" in names


def test_summary_table_is_not_messages():
    """摘要不进 messages 表 —— 否则 history_as_messages 会把它当成一条消息。"""
    from app.db.models import ConversationSummary

    assert ConversationSummary.__tablename__ == "conversation_summaries"
```

同时把 `test_conversation_columns_match_ddl` 的期望集合改成六列 + `summary_upto_msg_id` + `layer1_from_msg_id`。

- [ ] **Step 2: 跑测试确认全红**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ch07_ddl.py tests/test_db_models.py -q --color=no`

- [ ] **Step 3: 实现**

- `sql/ch07-ddl.sql` 整份重写：`SET NAMES utf8mb4;` 开头；三段幂等 DDL（两个 `ADD COLUMN` 走 `information_schema.COLUMNS`，建表用 `CREATE TABLE IF NOT EXISTS`，唯一键随建表语句一起给）；每段 `PREPARE/EXECUTE/DEALLOCATE`。注意**只查列/表是否存在，不 DROP 任何东西**（真库里可能还有历史遗留的 `summary` 列，留着不碍事）。
- `models.py`：`Conversation` 两列加在 `status` 之后；`ConversationSummary` 类放在 `Message` 之后，docstring 写清「为什么单开表」。
- `test_db_mysql.py` 的 `extra_columns` 改为 `{"summary", "summary_upto_msg_id", "layer1_from_msg_id"}`，`note` 说明「`summary` 是旧方案遗留，新代码不读它」。

- [ ] **Step 4: 跑绿 + 全量**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_ch07_ddl.py tests/test_db_models.py tests/test_repo_conventions.py -q --color=no
.\.venv\Scripts\python.exe -m pytest -q --color=no
```

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 两个锚点列 + conversation_summaries 表（幂等迁移）`。

---

### Task 6: 仓库层（锚点 + 摘要表 + 会话列表）

**Files:**
- Modify: `MemHelp-code/app/db/repositories.py`
- Test: `MemHelp-code/tests/test_ch07_repositories.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class AnchorRow`：`conversation_id: int`、`summary_upto_msg_id: int | None`、`layer1_from_msg_id: int | None`
  - `@dataclass(frozen=True) class SummaryRow`：`id / conversation_id / segment_no / from_msg_id / upto_msg_id / content / msg_count / source_tokens / model / elapsed_ms / created_at`
  - `@dataclass(frozen=True) class ConversationRow`：`id / session_id / user_id / status / preview / message_count / summary_segments / created_at / updated_at`
  - `ConversationRepository.get_anchors(conversation_id) -> AnchorRow | None`
  - `ConversationRepository.set_anchors(*, conversation_id, summary_upto_msg_id=None, layer1_from_msg_id=None, only_if_layer1_from=None) -> bool`（`only_if_layer1_from` 给乐观并发用：现值和它不一致就返回 `False` 不写）
  - `ConversationRepository.list_for_user(*, user_id="anonymous", limit=50) -> list[ConversationRow]`（新在前，`preview` = 该会话第一条 user 消息前 40 字，`message_count`、`summary_segments`）
  - `SummaryRepository.append(*, conversation_id, segment_no, from_msg_id, upto_msg_id, content, msg_count, source_tokens, model=None, elapsed_ms=None) -> int`（唯一键冲突抛 `SummarySegmentConflict`）
  - `SummaryRepository.list_for_conversation(conversation_id) -> list[SummaryRow]`（按 `segment_no` 升序）
  - `SummaryRepository.next_segment_no(conversation_id) -> int`
  - `class SummarySegmentConflict(RuntimeError)`
  - `Repositories.summaries: SummaryRepository`（`from_session` 里构造）

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_repositories.py`（用 `tests/conftest.py` 的 `sqlite_session` + `Repositories.from_session`）：

```python
"""锚点与摘要段的仓库层行为。"""
import pytest

from app.db.repositories import Repositories, SummarySegmentConflict


async def _conv(repos, session_id="s1"):
    return await repos.conversations.upsert_by_session(session_id, "anonymous")


async def _msg(repos, cid, role="user", content="你好"):
    return await repos.messages.append(
        conversation_id=cid, role=role, content=content,
        tool_calls=None, tool_call_id=None, tool_traces=[],
    )


async def test_anchors_are_none_on_a_fresh_conversation(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    got = await repos.conversations.get_anchors(cid)
    assert got is not None
    assert got.summary_upto_msg_id is None and got.layer1_from_msg_id is None


async def test_set_anchors_round_trips(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    assert await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=10, layer1_from_msg_id=11
    )
    got = await repos.conversations.get_anchors(cid)
    assert (got.summary_upto_msg_id, got.layer1_from_msg_id) == (10, 11)


async def test_set_anchors_never_lets_layer1_fall_behind_summary(sqlite_session):
    """Review Focus 1：层 1 起点不允许落回摘要区间里。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=20, layer1_from_msg_id=21
    )
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=20, layer1_from_msg_id=15
    )
    got = await repos.conversations.get_anchors(cid)
    assert got.layer1_from_msg_id == 21


async def test_set_anchors_is_monotonic_for_summary_upto(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(conversation_id=cid, summary_upto_msg_id=30)
    await repos.conversations.set_anchors(conversation_id=cid, summary_upto_msg_id=10)
    assert (await repos.conversations.get_anchors(cid)).summary_upto_msg_id == 30


async def test_optimistic_guard_refuses_when_anchor_moved(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=11)
    ok = await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=10,
        layer1_from_msg_id=11, only_if_layer1_from=99,
    )
    assert ok is False
    assert (await repos.conversations.get_anchors(cid)).summary_upto_msg_id is None


async def test_summary_append_and_list(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=4,
        content="用户问过耳机", msg_count=4, source_tokens=120,
    )
    await repos.summaries.append(
        conversation_id=cid, segment_no=2, from_msg_id=5, upto_msg_id=8,
        content="用户报了订单号 A1", msg_count=4, source_tokens=130,
    )
    rows = await repos.summaries.list_for_conversation(cid)
    assert [r.segment_no for r in rows] == [1, 2]
    assert rows[1].content.startswith("用户报了订单号")


async def test_summary_segment_no_is_unique(sqlite_session):
    """Review Focus 2：多进程重复压同一段时数据库收口。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=4,
        content="a", msg_count=4, source_tokens=10,
    )
    with pytest.raises(SummarySegmentConflict):
        await repos.summaries.append(
            conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=4,
            content="b", msg_count=4, source_tokens=10,
        )


async def test_next_segment_no(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    assert await repos.summaries.next_segment_no(cid) == 1
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=2,
        content="a", msg_count=2, source_tokens=5,
    )
    assert await repos.summaries.next_segment_no(cid) == 2


async def test_list_for_user_returns_newest_first_with_preview(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    old = await _conv(repos, "s-old")
    await _msg(repos, old, content="我上周买的耳机还没发货")
    new = await _conv(repos, "s-new")
    await _msg(repos, new, content="我想退货")

    rows = await repos.conversations.list_for_user(user_id="anonymous", limit=50)
    assert [r.session_id for r in rows] == ["s-new", "s-old"]
    assert rows[0].preview == "我想退货"
    assert rows[0].message_count == 1
    assert rows[0].summary_segments == 0


async def test_list_for_user_marks_summarized_conversations(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await _msg(repos, cid)
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=1,
        content="a", msg_count=1, source_tokens=5,
    )
    rows = await repos.conversations.list_for_user(user_id="anonymous", limit=50)
    assert rows[0].summary_segments == 1


async def test_list_for_user_respects_limit(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    for i in range(5):
        await _conv(repos, f"s{i}")
    assert len(await repos.conversations.list_for_user(user_id="anonymous", limit=2)) == 2
```

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

- 全部走 `_run(fn, ...)` 包 `anyio.to_thread.run_sync`（照既有风格）。
- `set_anchors` 的单调性在**应用侧**做（先读现值，取 `max`），不要写 SQL `GREATEST` —— SQLite 没有。`only_if_layer1_from` 在读到的现值不等于它时**不写任何东西**并返回 `False`。
- `SummarySegmentConflict` 由 `IntegrityError` 转换；转换必须只捕获唯一键冲突，其它 `IntegrityError` 原样抛。
- `list_for_user` 的 `preview` / `message_count` / `summary_segments` 用三条聚合 SELECT 拼装，不建 ORM relationship（`expire_on_commit=False` 下 relationship 的懒加载会在线程里踩到 session 已关闭）。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 锚点与摘要段的仓库层（单调 + 唯一键兜底）`。

---

### Task 7: 上下文日志

**Files:**
- Create: `MemHelp-code/app/agent/ctxlog.py`
- Test: `MemHelp-code/tests/test_ch07_ctxlog.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class AppLogger: path: str | Path | None = "log/app.log"`
  - `AppLogger.write(record: str) -> None`（按 `PROJECT_ROOT` 解析相对路径，`mkdir(parents=True, exist_ok=True)`，append + `\n`；`OSError` 只 `logger.warning`）
  - `log_model_ctx(*, logger, session_id, turn, messages, layer1_from_msg_id, summary_upto_msg_id, budget, layer1_tokens, layer2_tokens, summary_texts=()) -> str`
  - `log_history_ctx(*, logger, session_id, turn, summary_texts=(), messages) -> str`
  - `log_summary_event(*, logger, event, session_id, **fields) -> str`（`event` ∈ `trigger/start/done/skip/failed`）

- [ ] **Step 1: 写失败测试**

```python
"""上下文可观测：grep model_ctx / history_ctx 必须一行命中。"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.ctxlog import (
    AppLogger, log_history_ctx, log_model_ctx, log_summary_event,
)


def _read(path):
    return path.read_text(encoding="utf-8")


def test_model_ctx_writes_one_greppable_record(tmp_path):
    path = tmp_path / "app.log"
    line = log_model_ctx(
        logger=AppLogger(path), session_id="s1", turn="t1",
        messages=[SystemMessage("人设"), HumanMessage("我买了耳机"), AIMessage("好的")],
        layer1_from_msg_id=5, summary_upto_msg_id=3,
        budget=type("B", (), {"layer1_budget": 3954, "layer2_budget": 1695,
                              "history_budget": 5650})(),
        layer1_tokens=1200, layer2_tokens=300, summary_texts=["用户问过耳机"],
    )
    text = _read(path)
    assert "model_ctx session=s1 turn=t1" in text
    assert "用户问过耳机" in text
    assert "我买了耳机" in text
    assert "cards=" in text or "messages=" in text
    assert "tokens=" in text
    assert line.splitlines()[0].startswith("model_ctx")


def test_model_ctx_records_both_anchors_and_layer_budgets(tmp_path):
    path = tmp_path / "app.log"
    log_model_ctx(
        logger=AppLogger(path), session_id="s1", turn="t1",
        messages=[HumanMessage("hi")], layer1_from_msg_id=5, summary_upto_msg_id=3,
        budget=type("B", (), {"layer1_budget": 3954, "layer2_budget": 1695,
                              "history_budget": 5650})(),
        layer1_tokens=10, layer2_tokens=2,
    )
    text = _read(path)
    for piece in ("summary_upto=3", "layer1_from=5", "layer1_budget=3954", "layer2_budget=1695"):
        assert piece in text


def test_history_ctx_is_written_even_without_summary(tmp_path):
    path = tmp_path / "app.log"
    log_history_ctx(
        logger=AppLogger(path), session_id="s1", turn="t1",
        messages=[HumanMessage("上一句"), AIMessage("上一答")],
    )
    text = _read(path)
    assert "history_ctx session=s1 turn=t1" in text
    assert "上一句" in text


def test_summary_events_are_tagged(tmp_path):
    path = tmp_path / "app.log"
    log_summary_event(logger=AppLogger(path), event="trigger", session_id="s1",
                      layer2_tokens=1832, budget=1695, msgs=14)
    log_summary_event(logger=AppLogger(path), event="done", session_id="s1",
                      seg=3, chars=118, elapsed_ms=742, upto_msg_id=54)
    text = _read(path)
    assert "summary trigger session=s1" in text
    assert "layer2_tokens=1832 > budget=1695" in text
    assert "summary done session=s1" in text
    assert "elapsed_ms=742" in text


def test_missing_directory_is_created(tmp_path):
    path = tmp_path / "deep" / "nested" / "app.log"
    log_history_ctx(logger=AppLogger(path), session_id="s", turn="t", messages=[])
    assert path.is_file()


def test_write_failure_is_swallowed(tmp_path):
    """日志是旁路：路径不可写时不能让用户拿不到回答。"""
    blocker = tmp_path / "app.log"
    blocker.mkdir()
    AppLogger(blocker).write("x")     # 不抛即通过
```

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

- 记录正文格式：首行 `model_ctx session=… turn=… summary_upto=… layer1_from=… layer1_budget=… layer2_budget=… layer1_tokens=… layer2_tokens=… messages=N tokens=M`，然后每个摘要段一行 `[摘要 N] …`，再逐条 `[层1] user: 正文` / `[层2] ai: 正文`。
- 单条正文超过 500 字要截断（`…`），否则一轮就能把日志写成几 MB。
- **不**用 `logging.FileHandler`：app 里没有任何 `basicConfig`，靠 root logger 落盘不可靠。自己 `open(..., "a")`。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 上下文日志（model_ctx / history_ctx / summary 五态）`。

---

### Task 8: 后台异步摘要

**Files:**
- Create: `MemHelp-code/app/agent/summarizer.py`
- Modify: `MemHelp-code/app/agent/prompts.py`（新增 `SUMMARY_PROMPT`）
- Test: `MemHelp-code/tests/test_ch07_summarizer.py`

**Interfaces:**
- Consumes: `BaseChatModel.ainvoke`、`Repositories`、`BudgetReport`、`LayerPlan`、`AppLogger`
- Produces:
  - `SUMMARY_PROMPT: str`（`app/agent/prompts.py`）
  - `@dataclass(frozen=True) class SummaryJob`：`session_id / conversation_id / segment_no / from_msg_id / upto_msg_id / rows: list[MessageRow] / source_tokens / background: str / expected_layer1_from: int | None`
  - `@dataclass(frozen=True) class SummaryOutcome`：`status: str`（`"done" / "skipped" / "failed"`）、`segment_no: int | None`、`content: str`、`elapsed_ms: int`、`reason: str`
  - `class ContextSummarizer`：`__init__(*, model, repositories_factory, settings, logger=None, clock=None)`、`schedule(*, session_id, conversation_id, plan, background=None) -> bool`、`async run_once(job) -> SummaryOutcome`、`inflight: frozenset[str]`、`async aclose() -> None`
  - `build_summary_job(*, plan, session_id, conversation_id, segment_no, background="", expected_layer1_from=None) -> SummaryJob`

- [ ] **Step 1: 写失败测试**

```python
"""后台摘要：不阻塞、可去重、只 append、失败不冒泡。"""
import asyncio
import time

import pytest

from app.agent.summarizer import ContextSummarizer, build_summary_job
from app.context.layers import plan_layers


class _FakeModel:
    def __init__(self, text="用户问过耳机，报了订单号 A123456789。", delay=0.0):
        self.text, self.delay, self.calls = text, delay, 0

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        from langchain_core.messages import AIMessage
        return AIMessage(self.text)


def _settings(**over):
    from app.config import Settings
    data = dict(openai_base_url="https://x/v1", openai_api_key="k", openai_model="m")
    data.update(over)
    return Settings(**data)


def _rows(n=4):
    from datetime import datetime, timezone
    from app.db.repositories import MessageRow
    now = datetime.now(timezone.utc)
    return [
        MessageRow(id=i + 1, role="user" if i % 2 == 0 else "assistant",
                   content=f"消息{i}", tool_calls=None, tool_call_id=None, created_at=now)
        for i in range(n)
    ]


def _plan(rows):
    return plan_layers(rows, summary_upto=0, layer1_from=len(rows) + 1,
                       layer1_budget=99999, layer2_budget=1)


async def test_schedule_returns_immediately_without_awaiting_the_model(sqlite_session):
    """验收 4：摘要不许阻塞该轮用户回复。"""
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    model = _FakeModel(delay=0.5)
    s = ContextSummarizer(model=model, repositories_factory=lambda: repos, settings=_settings())
    plan = _plan(_rows())
    started = time.perf_counter()
    assert s.schedule(session_id="s1", conversation_id=cid, plan=plan) is True
    assert time.perf_counter() - started < 0.2      # 没等模型
    await s.aclose()
    assert model.calls >= 0


async def test_schedule_is_false_when_summary_not_due(sqlite_session):
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    s = ContextSummarizer(model=_FakeModel(), repositories_factory=lambda: repos, settings=_settings())
    plan = plan_layers(_rows(), summary_upto=0, layer1_from=5,
                       layer1_budget=99999, layer2_budget=99999)
    assert s.schedule(session_id="s1", conversation_id=cid, plan=plan) is False
    await s.aclose()


async def test_concurrent_schedule_for_same_session_is_deduplicated(sqlite_session, tmp_path):
    from app.agent.ctxlog import AppLogger
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    model = _FakeModel(delay=0.3)
    s = ContextSummarizer(model=model, repositories_factory=lambda: repos,
                          settings=_settings(), logger=AppLogger(tmp_path / "app.log"))
    plan = _plan(_rows())
    assert s.schedule(session_id="s1", conversation_id=cid, plan=plan) is True
    assert s.schedule(session_id="s1", conversation_id=cid, plan=plan) is False
    await s.aclose()
    assert "reason=inflight" in (tmp_path / "app.log").read_text(encoding="utf-8")


async def test_run_once_appends_one_segment_and_moves_the_anchor(sqlite_session, tmp_path):
    from app.agent.ctxlog import AppLogger
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    rows = _rows(4)
    s = ContextSummarizer(model=_FakeModel(), repositories_factory=lambda: repos,
                          settings=_settings(), logger=AppLogger(tmp_path / "app.log"))
    job = build_summary_job(plan=_plan(rows), session_id="s1", conversation_id=cid,
                            segment_no=1, expected_layer1_from=5)
    out = await s.run_once(job)
    assert out.status == "done"
    segs = await repos.summaries.list_for_conversation(cid)
    assert len(segs) == 1 and segs[0].content.startswith("用户问过耳机")
    anchors = await repos.conversations.get_anchors(cid)
    assert anchors.summary_upto_msg_id == 4


async def test_run_once_never_rewrites_old_segments(sqlite_session):
    """只 append：压完不回头重写，避免同一事实被反复有损压缩。"""
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    await repos.summaries.append(conversation_id=cid, segment_no=1, from_msg_id=1,
                                 upto_msg_id=2, content="旧段原文", msg_count=2, source_tokens=5)
    rows = _rows(4)
    s = ContextSummarizer(model=_FakeModel("新段"), repositories_factory=lambda: repos,
                          settings=_settings())
    job = build_summary_job(plan=_plan(rows), session_id="s1", conversation_id=cid,
                            segment_no=2, expected_layer1_from=5)
    await s.run_once(job)
    segs = await repos.summaries.list_for_conversation(cid)
    assert segs[0].content == "旧段原文"     # 一个字都没动
    assert segs[1].content == "新段"


async def test_old_segments_are_passed_as_background_not_merged(sqlite_session):
    """Review Focus 3：旧段只作背景，不许把它的事实搬进新段。"""
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    await repos.summaries.append(conversation_id=cid, segment_no=1, from_msg_id=1,
                                 upto_msg_id=2, content="旧段：订单号 OLD-999", msg_count=2,
                                 source_tokens=5)
    captured = {}

    class _Capturing(_FakeModel):
        async def ainvoke(self, messages, **kwargs):
            captured["prompt"] = "\n".join(str(m.content) for m in messages)
            return await super().ainvoke(messages, **kwargs)

    s = ContextSummarizer(model=_Capturing("新段：订单号 NEW-111"), repositories_factory=lambda: repos,
                          settings=_settings())
    job = build_summary_job(plan=_plan(_rows(4)), session_id="s1", conversation_id=cid,
                            segment_no=2, expected_layer1_from=5)
    await s.run_once(job)
    assert "OLD-999" in captured["prompt"]          # 旧段作为背景进去了
    assert "只总结" in captured["prompt"] or "待压缩" in captured["prompt"]
    segs = await repos.summaries.list_for_conversation(cid)
    assert "OLD-999" not in segs[1].content          # 但没被搬进新段


async def test_anchor_moved_during_summary_is_skipped(sqlite_session, tmp_path):
    from app.agent.ctxlog import AppLogger
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=7)
    rows = _rows(4)
    s = ContextSummarizer(model=_FakeModel(), repositories_factory=lambda: repos,
                          settings=_settings(), logger=AppLogger(tmp_path / "app.log"))
    job = build_summary_job(plan=_plan(rows), session_id="s1", conversation_id=cid,
                            segment_no=1, expected_layer1_from=5)
    out = await s.run_once(job)
    assert out.status == "skipped"
    assert await repos.summaries.list_for_conversation(cid) == []
    assert "anchor moved" in (tmp_path / "app.log").read_text(encoding="utf-8")


async def test_model_failure_is_logged_not_raised(sqlite_session, tmp_path):
    from app.agent.ctxlog import AppLogger
    from app.db.repositories import Repositories

    class _Boom:
        async def ainvoke(self, messages, **kwargs):
            raise RuntimeError("上游 500")

    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    s = ContextSummarizer(model=_Boom(), repositories_factory=lambda: repos,
                          settings=_settings(), logger=AppLogger(tmp_path / "app.log"))
    job = build_summary_job(plan=_plan(_rows(4)), session_id="s1", conversation_id=cid,
                            segment_no=1, expected_layer1_from=5)
    out = await s.run_once(job)
    assert out.status == "failed"
    assert "上游 500" in out.reason
    assert "summary failed" in (tmp_path / "app.log").read_text(encoding="utf-8")


async def test_disabled_summarizer_never_calls_the_model(sqlite_session):
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    model = _FakeModel()
    s = ContextSummarizer(model=model, repositories_factory=lambda: repos,
                          settings=_settings(summary_enabled=False))
    assert s.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is False
    await s.aclose()
    assert model.calls == 0


async def test_summary_text_is_within_the_configured_length(sqlite_session):
    """长度要求：几十到一两百字。超长要截断，不能把整段对话贴回来。"""
    from app.db.repositories import Repositories
    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session("s1", "anonymous")
    s = ContextSummarizer(model=_FakeModel("很" * 5000), repositories_factory=lambda: repos,
                          settings=_settings())
    job = build_summary_job(plan=_plan(_rows(4)), session_id="s1", conversation_id=cid,
                            segment_no=1, expected_layer1_from=5)
    out = await s.run_once(job)
    assert len(out.content) <= 300


def test_summary_prompt_forbids_fabrication():
    from app.agent.prompts import SUMMARY_PROMPT

    for piece in ("不许编", "没出现", "寒暄"):
        assert piece in SUMMARY_PROMPT or "编" in SUMMARY_PROMPT
    assert "待压缩" in SUMMARY_PROMPT
```

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

关键点（每一条都有对应测试）：

1. `schedule()` 里 `asyncio.create_task(self._run_guarded(job))` 后**立刻 return True**，任务引用塞进 `self._tasks: set[asyncio.Task]`，`done_callback` 里 `discard` 并从 `_inflight` 移除会话。
2. `_inflight` 是 `set[str]`，键用 `session_id`；重复 `schedule` → 写 `summary skip … reason=inflight` 并返回 `False`。
3. `run_once` 的锚点检查：读 `get_anchors(conversation_id)`；若 `expected_layer1_from` 非空且现值不等 → `skipped`，日志 `reason=anchor moved`。**先检查、后调模型**，省一次无用的上游调用。
4. 落库：`set_anchors(..., summary_upto_msg_id=upto, layer1_from_msg_id=max(现值, upto+1), only_if_layer1_from=expected)`；返回 `False` → 记 `skipped`。`SummarySegmentConflict` → `skipped`（`reason=segment exists`）。
5. 摘要正文 `strip()` 后按 `SUMMARY_MAX_TOKENS` 对应字数上限 `min(600, summary_max_tokens * 2)` 截断（中文 1 字≈1 token，留一倍余量）。
6. `SUMMARY_PROMPT` 必须含「只总结【待压缩对话】」「对话里没出现的内容一个字不许编」「寒暄闲聊不要写」「几十到一两百字」，并把旧段放在「已记录背景（不要重复）」区块。
7. `aclose()`：`asyncio.gather(*tasks, return_exceptions=True)` 等所有任务收尾 —— 供 lifespan / 测试用。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 后台异步摘要（只 append + 乐观并发 + 五态留痕）`。

---

### Task 9: 图接入（三层装配 + 模型上下文日志）

**Files:**
- Modify: `MemHelp-code/app/agent/graph.py`
- Modify: `MemHelp-code/app/agent/runner.py`（`done_payload` 多带几个统计字段）
- Modify: `MemHelp-code/app/schemas.py`（`ContextStats` 加 `layer1_from_msg_id / summary_upto_msg_id / summary_messages / degraded / summary_due`，全部带默认值）
- Test: `MemHelp-code/tests/test_ch07_graph.py`

**Interfaces:**
- Consumes: `assemble_context`、`compute_budget`、`plan_layers`、`ContextSummarizer`、`log_model_ctx`
- Produces:
  - `GraphDeps` 新增 `summarizer: Any = None`（第七个字段，默认 `None` 保持旧构造可用）
  - `_agent_context(state, config) -> tuple[ContextResult, LayerPlan, BudgetReport]`（供 `agent` 与 `refund_agent` 共用）
  - `state["context"]` 新增键：`layer1_from_msg_id / summary_upto_msg_id / layer1_messages / layer2_messages / summary_messages / degraded / summary_due / layer1_tokens / layer2_tokens`

- [ ] **Step 1: 写失败测试**

```python
"""图接入：历史带数据库 id、上下文按三层装配、摘要只调度不等待。"""
import pytest

from app.agent.graph import build_agent_graph
from app.agent.state import AgentState


async def test_agent_node_uses_three_layers_and_logs_model_ctx(
    agent_deps, tmp_path, monkeypatch
):
    """端到端最薄的一条：跑一轮，断言 log/app.log 里有 model_ctx 且 system 在首位。"""
    ...


async def test_context_never_contains_a_second_system_message(...):
    ...


async def test_summary_is_scheduled_but_not_awaited(...):
    ...


async def test_library_empty_falls_back_to_state_messages(...):
    ...
```

> **写这条测试前先读 `tests/test_ch05_graph.py` 的夹具写法**（`agent_deps` / fake model / fake retriever / `memory_checkpointer` 都是现成的），照它的形状写，不要另造一套。四个用例分别对应：日志落盘、system 唯一、摘要不等待、库空回退。

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

`_run_agent_node` 的改造要点：

1. 历史来源优先 `repositories.messages.list_for_conversation(conversation_id)`；`conversation_id` 由 `repositories.conversations.find_id(state["session_id"])` 取（`None` 时说明还没落过库 → 回退 `state.messages`，全算层 1）。
2. `compute_budget(settings)` 每轮算一次（纯函数，成本可忽略）；`plan_layers(...)` 拿 `LayerPlan`。
3. `assemble_context(...)` 产出 `ContextResult`，**只**用它调模型；**不写回** `state["messages"]`。
4. 调模型前 `log_model_ctx(...)`。
5. `plan.summary_due` 且 `deps.summarizer` 非空 → `build_summary_job` + `schedule(...)`，**不 await**。
6. `payload["context"]` 换成 `ContextStats` 那套新字段。
7. `GraphDeps` 加 `summarizer` 字段；`_deps` 从 `configurable["summarizer"]` 读。

`runner._config` 增加 `"summarizer": ...` 透传（`astream_turn` / `complete_turn` 的签名加可选参数，默认 `None`）。

- [ ] **Step 4: 跑绿 + 全量（重点：`test_ch05_graph.py` / `test_ch06_refund_graph.py` 不许红）**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 图接入三层装配 + model_ctx 日志 + 摘要调度`。

---

### Task 10: 指代/意图的历史上下文日志

**Files:**
- Modify: `MemHelp-code/app/agent/graph.py`（`_resolve_coref` / `_classify_intent` 前置 `log_history_ctx`）
- Test: `MemHelp-code/tests/test_ch07_history_ctx.py`

**Interfaces:**
- Produces: 两个节点在调模型前都调 `log_history_ctx(...)`，**每轮必打**（含 `chitchat` / `complaint` / `fallback` 轮，因为它们同样跑了 coref + intent）。

- [ ] **Step 1: 写失败测试**

```python
"""history_ctx 每轮必打，闲聊兜底轮也要有。"""
import pytest


@pytest.mark.parametrize("text,route", [
    ("你好呀", "chitchat"),
    ("我要投诉", "complaint"),
])
async def test_history_ctx_is_logged_on_every_route(...):
    ...


async def test_history_ctx_contains_summary_when_present(...):
    ...


async def test_history_ctx_is_only_written_once_per_turn(...):
    """两个节点都调，但要按 turn 去重 —— 否则一轮写两遍。"""
    ...
```

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

- 在 `_resolve_coref` 里取历史（`_history_before_current`）后先 `log_history_ctx`，把 `state["history_ctx_logged_turn"] = turn_id` 写回；`_classify_intent` 看到已写过就跳过。
- 摘要文本：若 `conversation_id` 可用，取 `summaries.list_for_conversation` 的 `content` 列表；否则空。
- 新增 state 字段 `history_ctx_logged_turn: str`，加进 `AgentState` 与 `_reset_turn` 的清理清单。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: history_ctx 每轮必打（含闲聊兜底轮）`。

---

### Task 11: 两个只读接口

**Files:**
- Modify: `MemHelp-code/app/schemas.py`（`ConversationSummaryOut` / `ConversationListResponse` / `ConversationMessagesResponse`）
- Modify: `MemHelp-code/app/api.py`（两个端点 + lifespan 启动自检 + 注入 summarizer）
- Test: `MemHelp-code/tests/test_ch07_api.py`

**Interfaces:**
- Produces:
  - `GET /api/conversations?user_id=anonymous&limit=50` → `ConversationListResponse`
  - `GET /api/conversations/{conversation_id}/messages` → 与 `SessionMessagesResponse` 同形状；找不到 → 404 + `{"error": {"type": "not_found", "message": ...}}`
  - lifespan 里 `compute_budget` 自检（`ok=False` → ERROR `上下文预算不足：…`）

- [ ] **Step 1: 写失败测试**

```python
"""两个只读接口 + 启动自检。"""
import pytest


async def test_list_conversations_returns_newest_first(client, seeded):
    ...


async def test_list_conversations_marks_summarized(client, seeded_with_summary):
    ...


async def test_list_conversations_limit_is_applied(client, seeded):
    ...


async def test_conversation_messages_round_trips_the_ui_shape(client, seeded):
    """复用 _db_session_messages_payload：role 是 human/ai/tool。"""
    ...


async def test_unknown_conversation_returns_404(client):
    ...


async def test_read_only_endpoints_do_not_mutate_anything(client, seeded, sqlite_session):
    ...


def test_startup_budget_check_logs_error_for_tiny_window(caplog):
    """spec §4.2：连一轮都装不下时打 ERROR，但不阻断启动。"""
    ...
```

> 夹具照 `tests/test_ch05_api.py` / `tests/test_ch06_api.py` 的 `client` 构造方式写（`create_app(...)` 注入 fake model / 内存 sqlite / `repositories`）。

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现**

- 端点函数放 `_lifespan` 附近，和既有只读端点风格一致；`user_id` / `limit` 用 `Query(...)` 带 `ge/le`。
- `{conversation_id}` 是 **`conversations.id`**，不是 `session_id`（docstring 写清，否则前端一定传错）。
- 复用 `_db_session_messages_payload` 需要 `session_id`；先用 `conversations` 行把 id → session_id 转换，再调它（不要复制一份取消息逻辑）。
- lifespan 里：`report = compute_budget(settings)`；`report.ok` → `logger.info("ch07 上下文预算 %s", format_budget_report(report))`，否则 `logger.error("%s：%s", INSUFFICIENT_BUDGET_PREFIX, report.reason)`。整段 try/except，绝不让启动失败。
- 顺手在 lifespan 构造 `ContextSummarizer` 挂到 `app.state.summarizer`，并注入到图依赖；退出时 `await summarizer.aclose()`。

- [ ] **Step 4: 跑绿 + 全量**

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 两个只读会话接口 + 启动预算自检`。

---

### Task 12: 前端会话侧栏（例外：不套 TDD）

**Files:**
- Modify: `MemHelp-code/web/index.html`（`.app` 内加 `<aside class="sidebar">` + 移动端抽屉开关）
- Modify: `MemHelp-code/web/style.css`（侧栏样式 + `<720px` 抽屉）
- Modify: `MemHelp-code/web/app.js`（列表加载、切换、回载、「新对话」、静默降级）
- Create: `MemHelp-code/scripts/ch07_ui_check.py`（无头 HTTP 断言，不依赖浏览器）

**Interfaces:**
- Consumes: `GET /api/conversations`、`GET /api/conversations/{id}/messages`
- Produces: 侧栏 DOM（`.sidebar` / `.sidebar-item[data-conversation-id]` / `.sidebar-item.is-active` / `.sidebar-item.is-summarized`）；`state.conversationId`、`state.sessionId` 两个前端状态

- [ ] **Step 1: 改 `index.html`**

在 `.topbar` 之后、`main.messages` 之前插：

```html
<aside class="sidebar" id="sidebar" aria-label="会话列表">
  <div class="sidebar-head">
    <span class="sidebar-title">会话</span>
    <button class="btn-ghost" id="btn-new-chat" type="button">新对话</button>
  </div>
  <ul class="sidebar-list" id="sidebar-list"></ul>
</aside>
<button class="sidebar-toggle" id="sidebar-toggle" type="button" aria-label="会话列表">☰</button>
```

- [ ] **Step 2: 改 `style.css`**

- `.app` 改成两列 grid（`grid-template-columns: 248px 1fr`），侧栏 `overflow-y: auto`。
- `.sidebar-item.is-active` 高亮；`.is-summarized::after { content: "✂"; }`。
- `@media (max-width: 720px)`：侧栏 `position: fixed` + `transform: translateX(-100%)`，`.is-open` 时滑入；`.sidebar-toggle` 只在窄屏显示。

- [ ] **Step 3: 改 `app.js`**

要点（每条都对应验收或 Review Focus）：

1. `async function loadConversations()` → `fetch("/api/conversations?user_id=anonymous&limit=50")`，**整个函数体 try/catch**，失败只是 `list.innerHTML = ""`，不 toast、不阻塞聊天。
2. `renderConversations(rows)`：新在前（后端已排序），显示 `preview`（空则「（空会话）」）+ 时间；`summary_segments > 0` 加 `is-summarized`。
3. 点击 → `switchConversation(id, sessionId)`：`GET /api/conversations/{id}/messages` → 清空 `#messages` → 逐条 `appendMessage(role, content)`（`human` → user 气泡）× **同时把 `state.sessionId` 换成该会话的 session_id 并写回 `sessionStorage`**（Review Focus 5）。失败则保留原会话不变。
4. 「新对话」→ 生成新 `sessionId`（沿用现有 uuid 生成），清空视图，`state.conversationId = null`，侧栏保持不动。
5. 每轮 `done` 事件之后重新 `loadConversations()`（拿到新的 preview / 已摘要标记）—— 失败静默。
6. 首屏：`loadConversations()` 之后若列表非空**不自动切换**（保持「新对话」语义），只在用户点击时切换。

- [ ] **Step 4: 写 `scripts/ch07_ui_check.py` 并跑**

脚本用 `httpx` 直接打接口（不启浏览器）：
- 建两个会话各聊一句（走 `/api/chat/completions` 非流式，注入 fake model 或真模型均可，脚本参数 `--offline` 用假数据）；
- 断言 `/api/conversations` 返回两条、新在前、`preview` 是首问；
- 断言 `/api/conversations/{id}/messages` 的角色序列是 `human / ai`；
- 打印一份人类可读的对照表。

- [ ] **Step 5: 浏览器实跑 + commit + 追记**

启动服务（managed job），浏览器里实际点一遍：开两个会话 → 切回第一个 → 历史回载 → 接着聊一句 → 侧栏顺序与 preview 更新。把截图路径写进 dev-notes。

commit message：`ch07: 会话侧栏（多会话切换 + 回载续聊 + 静默降级）`。

---

### Task 13: 演示脚本与 20 轮压测

**Files:**
- Create: `MemHelp-code/scripts/demo_ch07.py`
- Create: `MemHelp-code/scripts/ch07_twenty_turns.py`
- Test: `MemHelp-code/tests/test_ch07_demo.py`（只测脚本的纯函数部分，不发网络）

**Interfaces:**
- Produces:
  - `demo_ch07.py`：读 `--window/--output/--user-input/--steps/--tool-result/--rerank` 六个参数（默认演示值 18000/2000/2000/3/1200/5），打印预算表 → 断言 5650/3954/1695 → 用假模型跑一轮触发级联 → `grep` 出 `summary trigger` / `summary done` / `layer1 降级` 三类日志 → 最后问「最开始那个订单后来怎么说」并打印模型的梗概召回证据。
  - `ch07_twenty_turns.py`：默认窗口下跑 20 轮（可用 `--fake` 走假模型、`--real` 走真模型），每轮打印 `layer1_tokens / layer2_tokens / degraded / summary_due`，收尾断言 `degraded` 与 `summary_due` **全程为 False**（验收 3）。

- [ ] **Step 1: 写失败测试**

`tests/test_ch07_demo.py` 只钉脚本里可离线验证的部分：

```python
def test_demo_budget_matches_acceptance_numbers():
    from scripts.demo_ch07 import demo_budget
    r = demo_budget()
    assert (r.history_budget, r.layer1_budget, r.layer2_budget) == (5650, 3954, 1695)


def test_twenty_turns_produces_no_compression_under_default_window():
    from scripts.ch07_twenty_turns import run_twenty_turns_fake
    summary = run_twenty_turns_fake(turns=20)
    assert summary["degraded"] is False
    assert summary["summary_due"] is False
    assert summary["max_layer1_tokens"] <= summary["layer1_budget"]
```

- [ ] **Step 2: 跑测试确认全红**

- [ ] **Step 3: 实现两个脚本**

`demo_ch07.py` 的 `demo_budget(**overrides)` 构造 `Settings(...)` → `compute_budget`。级联演示部分用假模型（继承 `BaseChatModel` 的现成 fake，见 `tests/fakes_agent.py`），这样脚本可离线跑；`--real` 加开关走真模型。

`ch07_twenty_turns.py` 的 `run_twenty_turns_fake(turns)` 造 20 轮递增长度的对话，每轮调 `plan_layers` 并累计最大值，返回 dict（`degraded / summary_due / max_layer1_tokens / layer1_budget / max_layer2_tokens`）。

- [ ] **Step 4: 跑脚本，抓验收证据**

```powershell
cd MemHelp-code
.\.venv\Scripts\python.exe scripts\demo_ch07.py | Tee-Object dev-notes\ch07-acceptance-output.txt
.\.venv\Scripts\python.exe scripts\ch07_twenty_turns.py --fake --turns 20
Select-String -Path log\app.log -Pattern "summary trigger|summary done|layer1 降级|model_ctx|history_ctx" | Select-Object -First 40
```

Expected：
- 预算表 5650 / 3954 / 1695 逐项对上；
- 日志里能看到 `layer1 降级 X→Y`、`summary trigger … layer2_tokens=… > budget=1695`、`summary done … seg=1`；
- 20 轮默认窗口那条里 `degraded` / `summary_due` 全程 False。

- [ ] **Step 5: commit + 追记**

commit message：`ch07: 演示脚本 + 20 轮压测（含验收输出留档）`。

---

### Task 14: 收尾（code review / finish）

**Files:**
- Modify: `MemHelp-code/dev-notes/ch07.md`（补 code review 结论 + finish 段）
- Modify: `MemHelp-code/dev-notes/DELIVERY.md`（追加 ch07 一节）

- [ ] **Step 1: 全量测试**

```powershell
cd MemHelp-code
.\.venv\Scripts\python.exe -m pytest -q --color=no
```
Expected: 全绿，记录 `passed/skipped` 数字；真库用例仍 `skipped`（未设 `MEMHELP_TEST_MYSQL=1`）。

- [ ] **Step 2: 请求 code review**

用 `superpowers:requesting-code-review` 走一遍：重点看
（a）`app/context/` 四个模块有没有偷偷碰 DB/网络；
（b）`summarizer` 的异常面是否真的全吞；
（c）`graph.py` 有没有把裁过的消息写回 `state["messages"]`；
（d）`test_context.py` 是否一个字未改（`git diff` 应为空）。

- [ ] **Step 3: 处理评审意见**

逐条按 `superpowers:receiving-code-review` 的要求「先验证再改」，不盲从。改完重跑 Step 1。

- [ ] **Step 4: 追记 finish 段**

写清：交付命令、测试数字、验收 5 条各自对应哪个证据文件/日志行、dev-notes 路径。

- [ ] **Step 5: commit**

commit message：`ch07: 收尾（code review 结论 + 交付说明）`。
