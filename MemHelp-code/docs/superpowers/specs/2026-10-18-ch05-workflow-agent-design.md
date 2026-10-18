# 第五章设计 · Workflow 确定性编排 + 主力 Agent

- 日期：2026-10-18
- 状态：定稿（用户已认可 A 方案并授权连续执行）
- 关联：`dev-notes/ch05.md`、`docs/superpowers/plans/2026-10-18-ch05-workflow-agent-plan.md`

---

## 1. 目标与非目标

### 1.1 目标

把客服主链路从「模型自己决定调什么」升级为**代码可判定的骨架 + 节点内的 Agent**：

1. 先手写一个最裸的 Agent 循环（调模型 → 有工具调用就执行并回灌 → 没有就收敛），
   看清它就是「带工具的循环」；再用 LangGraph 重构为正式图。
2. 图按固定顺序搭建：指代消解 → 意图识别 → 按意图分流 → 知识检索 → 置信度闸 →
   主力 Agent → 日志记录。
3. 七类意图**写死**映射到四个出口，分流不交给模型。
4. 知识类强制先检索、过闸，证据够才进 Agent；业务数据类不预检索，直接进 Agent。
5. 投诉不进 Agent；转人工与建工单是两个独立的前端按钮，后端不自动执行。
6. 闲聊固定话术，不调用主力 Agent。
7. 会话状态用 LangGraph State 贯穿；checkpointer 落 SQLite；MySQL 仍是聊天记录
   与页面回读的权威副本。
8. 每轮写结构化日志，能看到走了哪些节点、检索/闸/Agent 的关键数字。

### 1.2 明确不做

- 指代消解的正式版：本章节点只做**原样透传**。
- 意图识别的正式版：本章只做「一个简单 prompt + 严格 JSON + 确定性兜底」。
- 上下文管理策略升级：沿用 ch01 的 `build_context` 裁剪，不做摘要/长期记忆。
- MCP 接入、Langfuse、飞轮流水线、新的知识入库链路。
- 新的业务工具：Agent 复用第 2 章工具，本章只加一个**控制面**工具
  `suggest_actions`（只记录建议，不执行动作）。

---

## 2. 流程分类与关键决策

| # | 决策 | 选择 | 理由 / 代价 |
|---|---|---|---|
| D1 | 图的形态 | **薄图 + 厚节点**（A 方案） | 外层确定性分流，ReAct 循环封装在 `agent` 节点；节点可单测、停止条件集中、一轮一个 checkpoint |
| D2 | checkpointer | **官方 `AsyncSqliteSaver`**，`thread_id = session_id` | 重启后图状态仍在；新增官方依赖与一个 SQLite 文件，属于「双存储」，但 MySQL 仍只做审计/页面回读 |
| D3 | 热身代码 | 独立脚本 `scripts/bare_agent_loop.py` + 离线测试 | 不进线上链路，避免两套编排漂移 |
| D4 | Agent 建议动作 | 控制面工具 `suggest_actions` | 结构确定、流式无标记泄漏；代价是模型工具列表多一个非业务工具，前端轨迹里单独标注 |
| D5 | 置信度闸 | 给 `RagPipeline` 加 `retrieve_only()`，跳过 ch04 的模型自评 | 闸的位置在 Agent 前；不额外花一次自评模型调用；`answer_context` 行为不变 |
| D6 | 按钮持久化 | `messages.actions` JSON 列 + `sql/ch05-ddl.sql` | 刷新后按钮还在，与 ch04 引用体验一致；需要一次真实库迁移 |
| D7 | `create_ticket` | **不进 Agent 工具集** | 需求 8 要求用户自选；留在 Agent 手里会被模型直接执行，违背「后端不自动动作」 |
| D8 | 投诉/知识类的工具集 | 知识类去掉 `query_faq`；业务类保留 | 知识类已强制检索，避免二次检索与编号冲突；业务类可能顺带问政策，允许自己检索 |
| D9 | ReAct 步数口径 | **一次模型步只执行第一个工具调用** | 与 ch02 的「单轮只取第一个」一致；复杂问题必然多轮，验收 5 可观测；避免多个 `query_faq` 造成引用编号冲突 |
| D10 | 闲聊零模型调用 | 明显寒暄走**确定性快速通道**；其余先意图分类、再固定话术 | 「知道是闲聊」本身需要判断；对「你好/在吗/谢谢/再见」这类逐字命中不花模型调用，其余只花一次轻量分类调用、不调用主力 Agent |

### 2.1 需求冲突的裁决（已写进设计，不是暗改）

1. **需求 7「记下来留给数据飞轮」 vs 本章不做「飞轮入库」**
   本章只做两件事：① `log_turn` 写 `gate_failed` 事件；② 沿用 ch04 已有的
   `low_confidence_questions` 池（`source=retrieval_low_conf`）落一行。
   **不建新表、不建新流水线**。若后续要求「只写日志」，删掉池子写入即可。
2. **需求 3「闲聊不花模型调用」 vs 需求 6「意图识别用 prompt 判七类」**
   采用 D10 的快速通道 + 意图分类兜底：明显寒暄零调用；其余闲聊付一次分类调用，
   但**绝不调用主力 Agent**。这是在不放弃七类意图识别的前提下能做到的最强保证。
3. **需求 4「Agent 复用第 2 章工具」 vs 需求 8「建工单必须用户点按钮」**
   Agent 可复用查询类工具与 `query_faq`，但 `create_ticket` 只由
   `POST /api/actions/create-ticket` 调用。控制面 `suggest_actions` 只记录建议。

---

## 3. 现状勘查（读代码 + 实测）

### 3.1 现有链路

```
POST /api/chat/completions
  → api.py::chat_completions
  → chat_tools.py::astream_chat_with_tools   （ch02 单轮：第一次非流式拿 tool_calls，
                                              只执行第一个 → 第二次流式）
  → tools/executor.py::execute_tool          （校验/超时/重试/脱敏，永不抛）
  → tools/registry.py::get_tools             （query_order/query_product/
                                              query_logistics/query_faq/create_ticket）
  → knowledge/pipeline.py::RagPipeline       （理解→双路召回→RRF→重排→两道闸→证据）
```

本章替换的是 `chat_tools` 这一层编排；`chat_tools.py` 保留（历史测试与回退参考），
但 HTTP 聊天入口切到新图。

### 3.2 环境实测

| 项 | 实测值 |
|---|---|
| 分支 / HEAD | `main` / `8006c37`（工作区有 ch04 评估看板未提交改动） |
| `langgraph` | 1.2.15（随现有依赖安装） |
| `langgraph-checkpoint` | 4.3.0，含 `InMemorySaver` |
| `langgraph.checkpoint.sqlite` | **不存在**，需新增 `langgraph-checkpoint-sqlite` |
| 上游 | `.env` 已有可用 DeepSeek 兼容端点，`deepseek-flash` |
| 工具 | 五个工具已注册，`execute_tool` 契约稳定 |
| MySQL | 现有 `messages` 表有 `tool_calls/tool_traces/citations`，无 `actions` |

### 3.3 Context7 核对（先查后写）

1. 带 checkpointer 的图必须传 `config={"configurable": {"thread_id": ...}}`。
2. 节点内 `get_stream_writer()` 发自定义事件，消费端 `stream_mode="custom"` 接收；
   Python ≥3.11 的异步节点可直接用。
3. `SqliteSaver` / `AsyncSqliteSaver` 属于 `langgraph.checkpoint.sqlite`，
   需要单独安装官方包；`InMemorySaver` 在 `langgraph-checkpoint` 内。
4. `add_messages` reducer 用于 `messages` 累积；`StateGraph` 条件边按 state 字段分流。

> 仍有三处文档没写死、必须 Task 0 真机探针：`AsyncSqliteSaver.from_conn_string` 的
> 上下文管理形态与 `setup()`、`get_stream_writer` 在异步节点的注入方式、
> `AIMessageChunk` 的 tool_call 合并与 `usage_metadata` 在流式块上的位置。

---

## 4. 总体架构

```
                        ┌────────────────────────────────────────────┐
                        │              LangGraph State               │
                        │  messages / intent / route / evidence /    │
                        │  gate / answer / actions / trace / usage   │
                        └────────────────────────────────────────────┘
START
  → load_history        仅首轮：checkpoint 无 messages 时从 MySQL 回填
  → begin_turn          追加本轮 HumanMessage（turn_id 幂等），重置 trace
  → resolve_coref       原样透传（本章）
  → classify_intent     简单 prompt → 七类 JSON（含确定性兜底/寒暄快速通道）
  → route_intent        纯代码映射，写死
       ├─ knowledge  → knowledge_retrieve → confidence_gate
       │                                    ├─ passed → agent
       │                                    └─ failed → fallback
       ├─ business   → agent
       ├─ complaint  → complaint
       └─ chitchat   → chitchat
  → log_turn            所有出口汇合：落 MySQL + 写 JSONL
  → END
```

### 4.1 七类意图 → 四出口（代码里的唯一映射表）

| 意图 | 中文 | 出口 | 工具集 / 行为 |
|---|---|---|---|
| `product_consult` | 商品咨询 | knowledge | 强制检索 → 闸 → Agent；工具：order/product/logistics + suggest_actions（无 `query_faq`） |
| `refund_return` | 退款退货 | knowledge | 先检索政策 → 闸 → Agent；工具：order/product/logistics + suggest_actions（无 `query_faq`） |
| `logistics` | 物流 | business | 不预检索，直接 Agent；工具：查询四件 + suggest_actions（无 `create_ticket`） |
| `order` | 订单 | business | 同上 |
| `after_sales` | 售后 | business | 同上 |
| `complaint` | 投诉 | complaint | 不进 Agent；固定安抚 + `["transfer_human","create_ticket"]` |
| `chitchat` | 闲聊 | chitchat | 固定话术；明显寒暄零模型调用，其余只付一次分类调用 |

---

## 5. State 契约

```python
Intent = Literal[
    "logistics", "order", "product_consult",
    "refund_return", "after_sales", "complaint", "chitchat",
]
Route = Literal["knowledge", "business", "complaint", "chitchat"]


class AgentState(TypedDict, total=False):
    # 输入 / 会话
    session_id: str
    turn_id: str                    # 本轮幂等键
    user_message: str
    messages: Annotated[list[AnyMessage], add_messages]
    last_user_turn_id: str
    last_committed_turn_id: str

    # 指代消解 / 意图
    resolved_message: str
    coref_changed: bool
    intent: Intent
    intent_confidence: float
    intent_raw: str
    intent_source: Literal["llm", "heuristic", "rule"]
    route: Route

    # 知识类
    retrieval_attempted: bool
    retrieval_score: float
    retrieval_degraded: bool
    evidence: list[dict]            # Evidence 快照（n/chunk_id/section_path/category/quote）
    gate_passed: bool
    gate_reason: str

    # Agent
    answer: str
    tool_traces: list[dict]         # 给前端的徽章（不含 suggest_actions）
    tool_records: list[dict]        # 落库用的完整工具行
    pending_messages: list[AnyMessage]  # 本轮待提交的消息（工具调用/结果/最终回答）
    citations: list[dict]
    suggested_actions: list[str]    # transfer_human | create_ticket
    agent_steps: int                # 已执行的工具轮数（不是 LLM 调用数）
    llm_calls: int
    token_usage: dict               # prompt / completion / total
    stop_reason: str                # final | max_steps | token_budget |
                                    # empty_response | gate_failed | fixed
    context: dict                   # kept / trimmed / budget / used

    # 可观测
    trace: list[str]                # 本轮节点名，按顺序覆盖写
    turn_started_at: float
    turn_log: dict
    error: str | None
```

**为什么 messages 在 state 里**：checkpointer 要能跨轮恢复模型上下文；MySQL 仍是审计与
页面回读的权威副本。`load_history` 只在 checkpoint 没有消息时从 MySQL 回填，避免
「SQLite 文件被删」导致模型突然失忆。

**为什么配置不在 state 里**：`repositories` / `retriever` / `model` 是每请求依赖，
放 `config["configurable"]`，不进 checkpoint（Session 不可跨请求共享）。

---

## 6. 节点行为

### 6.1 `load_history`
- 若 `state.messages` 非空：直接返回 `{}`（checkpoint 已是上下文）。
- 否则用 `repositories.messages.history_as_messages(session_id)` 回填。
- 不做裁剪；裁剪在 Agent 调用前用 `build_context`。

### 6.2 `begin_turn`
- 若 `state.last_user_turn_id == turn_id`：不重复追加（重试幂等）。
- 否则 `messages += [HumanMessage(user_message)]`，写 `last_user_turn_id`。
- 重置 `trace=["begin_turn"]`、`turn_started_at=time.time()`、清空本轮派生字段。

### 6.3 `resolve_coref`
- 本章：`resolved_message = user_message`，`coref_changed=False`。
- 节点单独存在，后续替换实现时不动图结构。

### 6.4 `classify_intent`
- **快速通道**：`is_trivial_chitchat(message)` 逐字/短句命中（你好、在吗、谢谢、再见、
  早上好、晚上好…）→ `intent=chitchat, intent_source="rule", confidence=1.0`，
  不调用模型。
- 其余：`model.ainvoke([SystemMessage(INTENT_PROMPT), HumanMessage(message)])`，
  严格 JSON `{"intent": "...", "confidence": 0..1}`；容忍 ```json 代码块。
- 解析失败 / 枚举非法：`heuristic_intent(message)`（投诉/退款退货/物流/订单/售后/
  商品咨询 的关键词优先级表），`intent_source="heuristic"`，
  并在 `intent_raw` 保留模型原文。**模型调用本身异常/超时也走这条兜底**，
  意图识别的故障不能把聊天变成 500。
- 启发式兜底对「认不出来」的默认值是 `after_sales`（business 出口）—— 把真实问题
  误当闲聊的代价，比多走一次 Agent 大。
- 永远不抛：最差落到 business（Agent 有工具，比把真实问题当闲聊安全）。

### 6.5 `route_intent`
- 纯代码 `INTENT_TO_ROUTE` 映射；写 `route` 与 `trace`。
- 条件边按 `route` 分流；未知 route 兜底 `business`（并记 warning）。

### 6.6 `knowledge_retrieve`
- 调 `RagPipeline.retrieve_only(resolved_message)`：
  Query 理解 → 双路召回 → RRF → 重排 → **只做 0 命中/分数阈值**，不调自评模型。
- 写 `retrieval_attempted=True`、`retrieval_score`（最高重排分；无重排分则 0）、
  `retrieval_degraded`、`evidence=evidence_payload(...)`。
- 检索异常原样冒给 API（故障要可见，不伪装成「没查到」）。

### 6.7 `confidence_gate`
- 通过条件：证据非空，且（有重排分时）最高分 ≥ `rag_min_rerank_score`；
  重排降级时只看「0 命中」。
- 失败：`gate_passed=False`，`gate_reason` 写明原因；尝试把原话写入
  `low_confidence_questions`（旁路，失败只记日志）。
- 通过：`gate_passed=True`。
- 条件边：通过 → `agent`；失败 → `fallback`。

### 6.8 `agent`（ReAct 循环，详见 §7）
- 输入：裁剪后的历史 + 本轮问题 +（知识类）编号证据。
- 工具集：查询类 + `suggest_actions`；业务类额外 `query_faq`；**永不含 `create_ticket`**。
- 输出：`answer`、`tool_traces`、`tool_records`、`citations`、`suggested_actions`、
  `agent_steps`、`llm_calls`、`token_usage`、`stop_reason`。

### 6.9 `complaint`
- 不调 Agent。固定安抚话术（先共情，再给两个选项），
  `suggested_actions=["transfer_human","create_ticket"]`，`stop_reason="fixed"`。
- 话术只**建议**，不写「已转接」「已建单」。

### 6.10 `chitchat`
- 固定话术，`suggested_actions=[]`，`stop_reason="fixed"`。

### 6.11 `fallback`
- 复用 `prompts.REFUSAL_TEXT`；`stop_reason="gate_failed"`；不调 Agent。
- 不附按钮（需求只要求投诉/Agent 合适时给建议）。

### 6.12 `log_turn`
- 若 `last_committed_turn_id == turn_id`：跳过重复落库（重试幂等），只补日志。
- 否则先落 MySQL（`user` + `pending_messages` + `actions`），成功后把
  `pending_messages` 追加进 state.messages 并写 `last_committed_turn_id`；
  再写 JSONL。
- 写 `turn_log` 进 state；字段见 §10。
- 失败：MySQL 写入失败向上抛，由 API 映射成 503；JSONL 写入失败只记 warning
  （日志是旁路，不能因为磁盘满把回答吞掉）。
- **为什么先落库再提交 messages**：agent 节点把「工具调用、工具结果、最终回答」
  放在 `pending_messages`，不直接进 `messages`。MySQL 失败时 state 不会留下
  一段「页面/审计里没有、模型上下文里却有」的回答。

---

## 7. ReAct 循环（agent 节点内部）

### 7.1 伪代码

```python
loop_messages = build_context(history, question, system, budget).messages
answer_parts, steps, llm_calls = [], 0, 0

for llm_calls in range(1, max_steps + 2):          # 最多 max_steps 轮工具 + 1 次收敛
    if usage_total >= token_budget:
        stop_reason = "token_budget"; break

    full = None
    async for chunk in model.bind_tools(tools).astream(loop_messages):
        full = chunk if full is None else full + chunk
        delta = text_of(chunk)
        if delta:
            answer_parts.append(delta)
            writer({"type": "token", "delta": delta})

    ai = to_ai_message(full)                        # 保留 additional_kwargs / usage
    loop_messages.append(ai)
    usage_total += usage_of(full)

    if not ai.tool_calls:
        stop_reason = "final" if answer_parts else "empty_response"
        break

    chosen = ai.tool_calls[0]                       # 一次只执行第一个
    loop_messages[-1] = first_call_only(ai, chosen) # 裁掉多余 tool_calls
    writer({"type": "status", "tool": ..., "label": ...})
    result = execute_tool(...) 或 suggest_actions(...)
    loop_messages.append(ToolMessage(content=..., tool_call_id=...))
    steps += 1

    if steps >= max_steps:
        stop_reason = "max_steps"; break
```

### 7.2 停止条件（全部可配置）

| 条件 | 默认 | 行为 |
|---|---|---|
| 最大工具轮数 | `AGENT_MAX_STEPS=6` | 停止，保留已有文本；没有文本则用兜底话术；附 `transfer_human` + `create_ticket` 建议 |
| token 预算 | `AGENT_TOKEN_BUDGET=6000` | 每步后累计 `usage_metadata.total_tokens`；超预算停止，行为同上 |
| 模型不再调工具 | — | `stop_reason="final"`，收敛 |
| 模型既无文本也无工具调用 | — | `empty_response`，用最后一条工具结果拼兜底话术 |
| 上游异常 | — | 原样冒给 API，转成 SSE `error`；只落用户轮（沿用 ch01/ch02） |

**缺信息追问**：Prompt 明确「订单号缺失时先问，不要猜」；没有工具调用的文本回答即
`final`，不需要额外节点。

### 7.3 流式

- Agent 每一步都用 `astream`，逐块发 `token`；工具调用前的过渡文本也发给用户。
- `status` 在工具执行**之前**发（沿用 ch02 的语义）。
- 不在节点内直接写 SSE：节点只发自定义事件，API 负责编码。
- 客户端断开/取消时，`astream` 的生成器被关闭；节点不再追加消息，避免半截落库。

### 7.4 工具执行

- 查询类：`execute_tool(name, args, repositories, retriever, timeout, max_retries)`；
  成功/失败都包成 `ToolMessage`（`to_tool_message_content`），失败是业务信息，不抛。
- `suggest_actions`：控制面工具，`@tool` + Pydantic schema；执行时只把动作写入
  本轮列表，返回 `{"ok":true,"recorded":[...]}`。**不调用任何业务写操作**。
- 每个工具执行收集一条 `tool_record`（name/args/id/content/traces）用于落库；
  `suggest_actions` 不进 `tool_traces`（前端不显示徽章），但进日志。
- Agent 的工具集由 `get_tools()` 过滤得到：去掉 `create_ticket`（业务写操作），
  知识类再去掉 `query_faq`（已预检索）；`suggest_actions` 单独构造并追加。
  这样复用第 2 章工具，不复制工具实现。

### 7.5 引用

- 知识类：直接使用预取证据；正文 `[n]` 复用 `parse_citations`，只把被引用的证据
  写进 `citations`。
- 业务类：Agent 可调 `query_faq`；**一次循环最多执行一次** `query_faq`，
  避免多组 `[n]` 编号冲突。之后从工具集移除 `query_faq`。
- 工具出参里 `truncated=true` 的证据，落库前回库取全文（复用 ch02 的做法）。

---

## 8. 流式协议与前端

### 8.1 SSE 事件

保持现有事件，新增/扩展：

| 事件 | 变化 |
|---|---|
| `meta` | 不变：`{session_id, model}` |
| `status` | Agent 每次工具执行前发；`{tool, args, label}` |
| `token` | 不变：`{delta}` |
| `tools` | 保持：本轮工具轨迹（不含 `suggest_actions`） |
| `citations` | 保持：有引用才发 |
| `done` | 新增 `intent` / `route` / `actions` / `agent_steps` / `stop_reason` / `gate`；保留 `tools` / `citations` / `kept` / `trimmed` |
| `error` | 不变 |

### 8.2 按钮交互

- `done.actions` 是权威来源；历史回读 `/messages` 的 `actions` 字段同形。
- 前端在 bot 气泡下方渲染两个**互相独立**的按钮：
  - 「转人工」：纯前端。点击后在本轮追加一条系统提示「已转接人工客服」，
    再追加一条 bot 气泡「您好，我是客服小猫，请问有什么可以帮您的」。
    **不发任何后端请求**。按钮锁定，另一按钮不受影响。
  - 「建工单」：`POST /api/actions/create-ticket`，成功后显示「已创建工单 T…」，
    按钮锁定；失败显示错误并允许重试。
- 两个按钮的点击状态用 `localStorage` 按 `session + bot 序号 + action` 锁定，
  刷新后还原；按钮本身从 `messages.actions` 还原。
- 用户不点、继续发消息：不做任何动作，正常走下一轮。

### 8.3 工单端点

```
POST /api/actions/create-ticket
{
  "session_id": "abc",
  "description": "我要投诉…",       # 可选；缺省回退到该会话最后一条用户消息
  "ticket_type": "投诉" | "售后" | "咨询"
}
→ 200 {"ok": true, "ticket_no": "T20261018001", "ticket_type": "投诉",
       "status": "待处理", "message": "已为你创建工单 …"}
```

- 走 `execute_tool("create_ticket", ...)`，即第 2 章的工单工具；写 `tickets` 表。
- `ticket_type` 用 `Literal` 校验；非法 422。
- 会话不存在/描述为空：422 或 404，错误体沿用 `{"error": {...}}`。
- 不自动调用：只有用户点按钮才走这个端点。

---

## 9. 持久化

### 9.1 LangGraph checkpointer

- 生产：`AsyncSqliteSaver`，路径 `CHECKPOINT_DB_PATH`（默认
  `MemHelp-code/data/checkpoints.sqlite`）。lifespan 打开 → `setup()` → 编译图 →
  退出时关闭。
- 测试：注入 `InMemorySaver`，不落盘。
- `thread_id = session_id`；同一会话多轮共享 state。
- `data/` 进 `.gitignore`。

### 9.2 MySQL

- `messages` 表新增 `actions JSON NULL`（`sql/ch05-ddl.sql`）。
- ORM `Message.actions`、`MessageRow.actions`、`MessageRepository.append(actions=...)`。
- `/messages` 在 assistant 行上返回 `actions`（仅非空时出现，与 `tools/citations` 同风格）。
- 落库顺序：`user` →（有工具时）`assistant(tool_calls)` → `tool` → … → `assistant(answer, citations, actions)`。
  多轮工具产生多组 tool_calls/tool 行，按执行顺序写。
- 失败路径：只落用户轮（沿用 ch01/ch02）。

### 9.3 两套存储的分工

| 存储 | 内容 | 权威性 |
|---|---|---|
| SQLite checkpoint | 图 state（messages、intent、evidence、actions…） | 运行时上下文；丢失可由 MySQL 回填 messages |
| MySQL | 聊天流水、工具轨迹、引用、建议动作、工单 | 审计、页面回读、跨重启的最终记录 |

---

## 10. 日志与可观测

- `log_turn` 每轮写一行 JSONL 到 `TURN_LOG_PATH`（默认
  `MemHelp-code/logs/ch05-turns.jsonl`），同时 `logger.info("ch05_turn ...")`。
- 字段：`ts / session_id / turn_id / intent / intent_source / route / trace /
  retrieval_attempted / retrieval_score / retrieval_degraded / evidence_count /
  gate_passed / gate_reason / agent_steps / llm_calls / tools / token_usage /
  suggested_actions / stop_reason / answer_chars / latency_ms`。
- 日志目录由应用启动时创建；测试把 `TURN_LOG_PATH=""` 关掉文件写入，只留 logger。
- 验收 1 的判据：日志行里 `trace` 含 `knowledge_retrieve` 且
  `retrieval_attempted=true`；不需要进数据库查。

---

## 11. 配置（`.env` 新增）

```dotenv
# ch05：LangGraph checkpointer
CHECKPOINT_DB_PATH=data/checkpoints.sqlite
# ch05：Agent 循环
AGENT_MAX_STEPS=6
AGENT_TOKEN_BUDGET=6000
# ch05：每轮结构化日志；留空则不写文件
TURN_LOG_PATH=logs/ch05-turns.jsonl
```

`Settings` 同步新增四个字段；`.env.example` 补注释；`tests/conftest.py` 的
`test_settings` 设 `turn_log_path=""` 并把 checkpointer 注入成 `InMemorySaver`。

---

## 12. Prompt 设计

### 12.1 意图分类 Prompt

- 角色：客服路由分类器。
- 七类定义 + 边界（订单 vs 物流、投诉 vs 售后、退款退货 vs 商品咨询）。
- 只输出 JSON：`{"intent": "<enum>", "confidence": 0.0-1.0}`。
- 禁止解释、禁止 markdown；不确定时给低 confidence，但仍必须选一个枚举。
- 示例覆盖：`订单 1001 的物流到哪了` → logistics；`订单 1001 多少钱` → order；
  `我要投诉` → complaint；`退货政策是什么` → refund_return；`你好` → chitchat。

### 12.2 Agent System Prompt

- 角色沿用「小美」，保留 8 条行为约束（先查再答、不编造、不承诺时间、不索要验证码…）。
- ReAct 规则：
  1. 一次只调一个工具；拿到结果再决定下一步。
  2. 简单问题一次工具调用就收敛；复杂问题可以多步。
  3. 缺订单号先问，不要猜。
  4. 工具失败如实说，给下一步。
  5. 需要人工/建单时调用 `suggest_actions`，**不要**自己执行任何写操作。
  6. 知识类证据已给出，正文用 `[n]` 标注；不要自创编号。
- 注入本轮证据（知识类）：`[n] 章节路径\n原文`，按相关度序。
- 不把 `create_ticket` 描述写进 Agent 工具说明。

### 12.3 固定话术

- `CHITCHAT_REPLY`：一句欢迎 + 能力范围。
- `COMPLAINT_SOOTHE`：先共情 + 说明可转人工或建工单，**不**说已执行。
- `AGENT_HANDOFF_TEXT`：步数/预算耗尽时的兜底，附两个建议动作。
- `REFUSAL_TEXT`：沿用 ch04。

---

## 13. 评估集与验证

### 13.1 意图标注集

`evals/ch05_intent_cases.jsonl`，约 60 条，七类各 6–10 条，覆盖：

- 订单 vs 物流的边界（「订单 1001 的物流到哪了」→ logistics）；
- 投诉 vs 售后的边界（「收到是坏的，我要投诉」→ complaint）；
- 退款退货 vs 商品咨询（「这个能退吗」→ refund_return）；
- 闲聊（「你好」「今天天气不错」）；
- 指代/省略（「它到哪了」→ 本章透传，仍按字面判，不要求消解）。

字段：`id / message / expected_intent / expected_route / bucket / note`。

### 13.2 意图评估脚本

`evals/run_ch05_intent_evals.py`：

- 真实模型跑每条 → 记录 predicted intent / route / source / raw / latency；
- 指标：整体 accuracy、每类 recall、混淆矩阵、route 准确率；
- 退出码：整体 ≥ 0.85 且每类 ≥ 0.60 为 0，否则 1；配置/上游故障 2；
- `--json` / `--markdown` / `--limit` / `--self-test`（离线验证聚合逻辑）。

**纯 Prompt 任务的 TDD 替代**：先写期望标签，再跑真实模型；不达标先改 Prompt /
示例，不改标签（沿用 ch02/ch04 的规则）。

### 13.3 代码测试

- **TDD**：每个模块先写失败测试，再实现。
- 离线假模型：`tests/fakes_agent.py` 的 `ScriptedChatModel`，按调用顺序返回
  意图 JSON、工具调用块、最终文本；支持多步和 `usage_metadata`。
- 图测试：`InMemorySaver` + SQLite 仓库 + 假检索器；断言路由、节点 trace、
  gate 短路、actions、引用、日志。
- HTTP 测试：SSE 事件、`done.actions`、`/messages` 的 actions、
  `/api/actions/create-ticket` 落库。
- 真实集成：`MEMHELP_TEST_MYSQL=1` 与真 Milvus 的用例沿用 skip 策略。

### 13.4 验收演示

`scripts/demo-ch05.py`（UTF-8，避免 PowerShell 中文编码坑）逐条打本地服务：

1. 政策问题 → 日志 `trace` 含 `knowledge_retrieve`；
2. `订单 1001 的物流到哪了` → `done.tools` 含 `query_logistics`；
3. `我要投诉` → `done.actions` 两个；调 create-ticket → 查 `tickets` 表；
4. `你好` → 固定话术，且本轮无 Agent 调用（日志 `intent_source=rule`）；
5. 复杂问题 → `agent_steps >= 2` / 工具轨迹 ≥2 次。

输出写 `dev-notes/ch05-acceptance-output.txt`。

---

## 14. 验收标准对照

| # | 标准 | 设计落点 | 证据 |
|---|---|---|---|
| 1 | 政策问题日志能看到强制检索 | `route_intent=knowledge` → `knowledge_retrieve` → `confidence_gate` | JSONL `trace` / `retrieval_attempted` |
| 2 | 订单 1001 物流，Agent 自己调工具 | 意图 `logistics` → `business` → `agent` → `query_logistics` | `done.tools` + `tool_records` |
| 3 | 投诉出现两个独立按钮；转人工前端模拟；建工单才写 tickets；不点正常对话 | `complaint` 固定话术 + `actions`；前端两按钮；`POST /api/actions/create-ticket` | 页面 + `tickets` 表 + `/messages.actions` |
| 4 | 闲聊固定话术 | 快速通道/意图分类 → `chitchat` | `intent_source=rule` / `CHITCHAT_REPLY` |
| 5 | 复杂问题 ReAct 多步 | 一次一个工具 + `agent_steps` | 日志 `agent_steps>=2` / 工具轨迹 |

---

## 15. 交付物

### 15.1 新增

| 路径 | 作用 |
|---|---|
| `app/agent/__init__.py` | 包 |
| `app/agent/state.py` | `AgentState` / `Intent` / `Route` / 常量 |
| `app/agent/prompts.py` | 意图、Agent、固定话术 |
| `app/agent/intent.py` | `IntentClassifier` + 启发式兜底 + 寒暄快速通道 |
| `app/agent/coref.py` | 本章透传 |
| `app/agent/gate.py` | 置信度闸 |
| `app/agent/react.py` | ReAct 循环 + 流式事件 + 停止条件 |
| `app/agent/tools.py` | `suggest_actions` 控制面工具 + 工具集过滤 |
| `app/agent/evidence.py` | 工具出参 → Evidence、引用反解 |
| `app/agent/graph.py` | `StateGraph` 构建 + 条件边 |
| `app/agent/checkpoint.py` | `AsyncSqliteSaver` / `InMemorySaver` 工厂 |
| `app/agent/persist.py` | 多工具轮落库 |
| `app/agent/logging.py` | JSONL turn log |
| `app/agent/runner.py` | 图执行器（API 与脚本共用） |
| `scripts/bare_agent_loop.py` | 祛魅热身 |
| `scripts/demo-ch05.py` | 验收演示 |
| `evals/ch05_intent_cases.jsonl` | 意图标注集 |
| `evals/run_ch05_intent_evals.py` | 意图评估 |
| `sql/ch05-ddl.sql` | `messages.actions` 迁移 |
| `tests/test_ch05_*.py` | 各模块测试 |

### 15.2 修改

| 路径 | 改动 |
|---|---|
| `app/api.py` | 聊天入口切图；新增 create-ticket 端点；lifespan 管理 checkpointer；`done` 扩展 |
| `app/config.py` | 四个新配置 |
| `app/schemas.py` | `CreateTicketRequest/Response`；`ChatResponse` 扩展 |
| `app/db/models.py` | `Message.actions` |
| `app/db/repositories.py` | `actions` 读写；`last_user_content(session_id)`（建单端点缺省描述用） |
| `app/knowledge/pipeline.py` | `retrieve_only()` |
| `app/prompts.py` | 可能只加常量（Agent prompt 在 `app/agent/prompts.py`） |
| `web/app.js` / `style.css` / `index.html` | 两个按钮 + 交互 + 历史还原 |
| `.env.example` / `.gitignore` / `README.md` | 配置、data/logs 忽略、运行说明 |
| `dev-notes/ch05.md` | 阶段留痕 |

---

## 16. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| `AsyncSqliteSaver` API 与文档不一致 | 启动失败 | Task 0 真机探针；Tests 注入 `InMemorySaver` |
| 流式 `AIMessageChunk` 合并 tool_calls 在网关上分片异常 | 工具没被执行 | Task 0 用真模型探针；失败则退回「非流式拿 tool_calls + 最终流式」的混合模式（不改图契约） |
| 意图分类被寒暄/客套话带偏 | 错路由 | 标注集 + 启发式兜底；不达标先改 Prompt |
| ReAct 死循环 / 烧 token | 成本 | 最大 6 轮 + 6000 token，双重刹车；一次只执行一个工具 |
| SQLite 与 MySQL 不一致 | 刷新后上下文差异 | MySQL 为审计权威；checkpoint 丢失时 `load_history` 回填 |
| 真实库未迁移 `actions` | 服务启动/查询 500 | 提供 `sql/ch05-ddl.sql` 并在交付前对真库执行一次；测试用 SQLite metadata 覆盖 |
| 投诉按钮被误当成后端动作 | 需求违背 | 转人工零请求；建单只在点击时调端点；Agent 不含 `create_ticket` |

---

## 17. 与既有章节的兼容

- `chat_tools.py` 保留，不被删除；HTTP 入口切图后，旧的 `chat_tools` 单测仍可运行，
  作为回退参考。**HTTP 层测试会改成图路径**，不再断言「chat_tools 的调用次数」，
  而是断言图的事件契约（`done.actions` / `done.tools` / 日志 trace）。
- `query_faq` 的出参契约不变；`RagPipeline.answer_context` 行为不变，只加
  `retrieve_only()`。
- SSE 既有事件名不变；`done` 只增字段。前端对缺失字段保持兼容。
- 现有 `messages` 表只加可空列；老行 `actions=NULL` 不影响回读。
