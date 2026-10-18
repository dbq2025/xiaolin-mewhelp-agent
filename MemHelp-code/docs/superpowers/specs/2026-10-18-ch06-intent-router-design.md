# 第六章设计 · 正式版分流器：指代消解、意图识别、退款子流程

- 日期：2026-10-18
- 状态：设计已获用户认可，spec 待 review
- 关联：`dev-notes/ch06.md`、`docs/superpowers/plans/2026-10-18-ch06-intent-router-plan.md`
- 前置：ch05 的图骨架、checkpointer、ReAct Agent、置信度闸与工具层

---

## 1. 目标与非目标

### 1.1 目标

把 ch05 的**占位分流器**（`coref.py` 原样透传、意图只做简单 prompt、退款与售后仍走普通出口）升级为正式版：

1. **指代消解 + Query 改写**：用 LLM 结合多轮历史，把「它能退吗」补成不依赖上下文也能看懂的完整问题；顺带把口语模糊问法归一成标准问法。问题已完整、指代已明确时原样透传，不强行改写。
2. **意图识别正式版**：七类业务意图 + `other` 兜底，共八选一；强制 JSON `{intent, confidence}`；枚举定义 + 边界 few-shot + `other` 规则；默认质量档，配置开关可切成本档（小模型先判、低置信度升级大模型重判）。
3. **分流精化**：`refund_return` 与 `after_sales` 走同一条确定性子流程：订单槽位 → 确定性取订单数据 → Query 扩写 → 强制检索政策条款并去重合并 → 只把「这一单能不能退/能不能售后」交给主力 Agent。
4. **槽位处理**：缺订单号不让模型猜；`interrupt()` 暂停并向前端发订单选择器，用户点选后从 checkpoint 续跑。退款原因不追问，提交退款单时从固定类目下拉选。需求澄清留在主力 Agent 里做，意图识别只判「要干什么」。
5. **Query 扩写**：只对退款退货、售后这类高频、容错低的核心场景做；强制 JSON，字段只有 `queries` 数组；几条一起检索再去重合并；只在检索侧现查现用，入库侧不拆存多份。
6. **前端配套**：订单选择器是聊天流里的可点选订单卡片，点选后自动回填并继续流程；退款单是简单表单，原因从固定类目下拉选。

### 1.2 明确不做

- 微调小模型 / BERT / 分类器做意图分类；本章只走 LLM prompt 路线。
- 跨会话记忆；指代消解只看当前会话的 `messages`。
- 真实订单表、真实退款单表、真实电商/物流接口；订单选择器使用按 `session_id` 派生的演示订单，退款单复用 `tickets`。
- 入库侧拆分多份 query 或知识副本；知识库只保留一份。
- 新增 Langfuse、MCP、飞轮流水线等组件。
- 刷新页面后恢复未完成的订单选择（本轮不做；checkpoint 仍在，但没有对应的页面恢复接口）。

---

## 2. 关键决策

| # | 决策 | 选择 | 理由 / 代价 |
|---|---|---|---|
| D1 | 订单槽位的中断方式 | **LangGraph `interrupt()` + `Command(resume=...)`** | 官方 HITL 机制；checkpointer 保存现场；不自己维护 pending 状态机。代价是 runner/SSE 要识别暂停态。 |
| D2 | 订单选择器的数据源 | 按 `session_id` 确定性派生 3 张演示订单 | 现有仓库没有 orders 表；复用 `derive_rng` + `query_order` + `query_product`，结果可复现。不新增表/迁移。 |
| D3 | 退款单的落库 | 复用 `tickets`，`ticket_type="售后"`，描述含订单号与原因 | 不新增表；与 ch05 的建单动作同一条写入路径；退款原因用后端固定枚举校验。 |
| D4 | 退款/售后路由 | 两意图统一走 `refund` 出口 | 用户明确要求同一条确定性子流程；不按「退款 vs 换货/发票」再拆分支。 |
| D5 | `other` 的出口 | `business` → 主力 Agent 澄清 | 「拿不准就归它，别硬塞业务意图」；澄清属于主力 Agent 的职责。 |
| D6 | 意图模型档位 | 默认质量档（单模型）；`INTENT_COST_MODE=true` 才启用小模型→大模型升级 | 先把准确率做上去；成本档是配置能力，不改变默认验收口径。 |
| D7 | 扩写触发范围 | 仅 `route=refund`；`product_consult` 等简单 FAQ 不扩 | 控制成本与误扩；高频、容错低的退款/售后才值得多查询。 |
| D8 | 扩写 JSON 契约 | `{"queries": ["...", ...]}`，无其它字段 | 用户明确字段只有 `queries` 一个数组；解析器拒绝缺字段/空数组。 |
| D9 | 多查询检索 | 每条查询走现有「理解→混合召回→重排」，按 `chunk_id` 去重、保留最高分、截断 top_n | 复用 ch03/ch04 的检索实现，不新建检索器；入库侧不动。 |
| D10 | 退款子流程的闸 | **不因低分短路**；政策证据为空也进 Agent，但 Prompt 要求「证据不足就说无法确认」 | 用户要求「最后只把这一单能不能退交给主力 Agent 判断」；若低分直接拒答，判断权就不在 Agent。 |
| D11 | 指代消解的调用时机 | 有指代/省略/口语标记，或首轮口语问法时才调模型；完整问句直接透传 | 满足「已完整原样透传」；避免每轮多一次模型调用与测试噪声。 |
| D12 | 停用半截落库 | 暂停轮不进 `log_turn`；续跑完成后一次落库 | 避免 MySQL 里出现「用户问了、客服没答」的半截轮次；checkpoint 仍保存现场。 |
| D13 | 新意图枚举 | 七类业务意图 + `other` = 8 个合法输出；`ALL_INTENTS` 8 项 | 用户明确留「其他」兜底；未知/冲突/怪问题不硬塞。 |

---

## 3. 现状勘查

### 3.1 现有链路（读代码所得）

```
POST /api/chat/completions
  → api.py::chat_completions
  → runner.astream_turn / complete_turn
  → graph.build_agent_graph()
       load_history → begin_turn → resolve_coref（原样透传）
       → classify_intent（简单 prompt + 严格 JSON + 启发式兜底）
       → route_intent（七类 → 四出口，写死）
       → knowledge_retrieve → confidence_gate → agent
         或 business → agent
         或 complaint / chitchat → 固定节点
       → log_turn
```

关键文件：

| 文件 | 现状 | 本章动作 |
|---|---|---|
| `app/agent/coref.py` | 只返回 `(message, False)` | 换成 `CorefResolver`：LLM + 历史 + 透传/防幻觉 |
| `app/agent/intent.py` | 七类、简单 prompt、启发式默认 `after_sales` | 八类、few-shot、质量/成本档、默认 `other` 兜底 |
| `app/agent/prompts.py` | `INTENT_PROMPT` 只有规则，无 few-shot | 重写意图 Prompt；新增 coref/扩写/退款判断 Prompt |
| `app/agent/state.py` | 七类→四出口 | 八类→五出口；新增退款槽位/扩写/订单字段 |
| `app/agent/graph.py` | 退款走 knowledge、售后走 business | 新增 `refund` 子图与 interrupt/resume |
| `app/agent/runner.py` | 只处理自定义流；无暂停态 | 识别 interrupt，发 `awaiting_input`；新增 resume 入口 |
| `app/knowledge/pipeline.py` | `retrieve_only(query)` | 新增 `retrieve_many(queries)` |
| `app/knowledge/query.py` | 单查询理解/改写 | 复用；扩写放在新模块，不塞进入库侧 |
| `app/api.py` | SSE 只转发固定事件 | 新增 `coref/intent/order_picker/awaiting_input` 帧；新增选单/退款单端点 |
| `web/app.js` | 无订单卡片/退款表单 | 聊天流订单卡片、内联退款表单、续跑流 |

### 3.2 数据现状

- 业务工具 `query_order` / `query_product` / `query_logistics` 是**确定性演示数据**：种子由「工具名 + 入参」派生，同一个订单号永远同结果；不查真实电商/物流系统。
- 没有 `orders` 表，也没有「用户 → 订单列表」的归属关系。
- 退款单没有独立表；`tickets` 已有 `ticket_type / description / idempotency_key`，可承载「退款申请」。
- 已有 `knowledge/退货政策.md`、`knowledge/售后手册.md`，检索链路 ch04 已具备混合召回 + 重排。

### 3.3 Context7 与真机探针（先查后写）

1. Context7 `/websites/langchain_oss_python_langgraph`：HITL 用 `interrupt(value)` 暂停，`Command(resume=...)` 恢复；带 checkpointer 时同一 `thread_id` 必须复用；流式消费看到 `stream.interrupted` / `stream.interrupts`。
2. 本机 `langgraph 1.2.15`：
   - `from langgraph.types import Command, interrupt` 可用；
   - `Command(*, graph=None, update=None, resume=None, goto=())`；
   - `interrupt(value, *, response_schema=None)`。
3. 真机探针（`astream(..., stream_mode="custom")`）验证：
   - 节点内 `get_stream_writer()` 发 `order_picker` 后调用 `interrupt()`；
   - `astream` 结束后 `snapshot.next == ("await_order",)`，`snapshot.tasks[0].interrupts[0].value` 是中断载荷；
   - `graph.astream(Command(resume={"order_id": "A1"}), config, stream_mode="custom")` 能从暂停点继续，节点内 token 事件正常发出；
   - 结论：可以采用 interrupt/resume，不需要自建 pending 状态机。

### 3.4 需求冲突的裁决（写进设计，不暗改）

1. **「七类意图」vs「其他兜底类」**：业务语义仍是七类，但合法输出有 8 个（含 `other`）。`ALL_INTENTS` 与 Prompt 都列 8 个；`other` 不计入业务意图统计，但计入总准确率。
2. **「售后走退款子流程」vs「发票/换货不一定需要订单」**：按用户明确要求统一走；订单数据是判断售后资格的前置条件。若后续要按子类型拆分，作为新需求处理。
3. **「退款原因不追问」vs「退款单要原因」**：对话阶段不追问；只在用户提交退款单时由固定下拉选择，后端用枚举校验。
4. **「强制检索政策」vs「检索失败也要回答」**：强制的是**发起检索并带上证据**，不是强制拒绝；检索失败时 Agent 必须明说无法确认，不编造规则。
5. **「默认大模型」vs「成本降级」**：默认只配主模型；成本档是配置开关。主模型名不写死在代码里，`INTENT_MODEL` 为空时回落 `OPENAI_MODEL`。

---

## 4. 总体架构

```
START
  → load_history                 仅首轮从 MySQL 回填
  → begin_turn                   追加 HumanMessage（turn_id 幂等），重置本轮字段
  → resolve_coref                LLM/透传：补全指代 + 口语归一
  → classify_intent              八选一 JSON；质量档或成本档
  → route_intent                 纯代码映射
       ├─ knowledge  product_consult
       │     → knowledge_retrieve → confidence_gate
       │          ├─ passed → agent
       │          └─ failed → fallback
       ├─ business   logistics / order / other
       │     → agent
       ├─ refund     refund_return / after_sales
       │     → refund_prepare_order
       │     → refund_await_order        （缺订单号时 interrupt）
       │     → refund_expand_queries     （LLM 强制 JSON queries）
       │     → refund_retrieve_policies  （多查询检索 + 去重合并）
       │     → refund_agent              （只判断这一单能不能退/售后）
       ├─ complaint  → complaint
       └─ chitchat   → chitchat
  → log_turn                     所有完成出口汇合
  → END
```

**暂停路径**：`refund_await_order` 调 `interrupt()` 后，`astream` 结束；runner 检测到 pending interrupt，发 `awaiting_input` 并**不**发 `done`。前端选单后调 `POST /api/chat/refund/select`，runner 用 `Command(resume={"order_id": ...})` 从同一 `thread_id` 续跑，继续 `refund_expand_queries` → ... → `log_turn`。

---

## 5. State 契约

```python
Intent = Literal[
    "logistics", "order", "product_consult",
    "refund_return", "after_sales", "complaint", "chitchat",
    "other",
]
Route = Literal["knowledge", "business", "refund", "complaint", "chitchat"]

class AgentState(TypedDict, total=False):
    # 输入 / 会话
    session_id: str
    turn_id: str
    user_message: str
    messages: Annotated[list[AnyMessage], add_messages]
    last_user_turn_id: str
    last_committed_turn_id: str

    # 指代消解 / 意图
    resolved_message: str
    coref_changed: bool
    coref_raw: str
    coref_source: Literal["llm", "passthrough", "fallback"]
    coref_reason: str
    coref_llm_calls: int
    intent: Intent
    intent_confidence: float
    intent_raw: str
    intent_source: Literal["llm", "llm_cheap", "llm_escalated", "heuristic", "rule"]
    intent_llm_calls: int
    route: Route

    # 知识类（沿用 ch05）
    retrieval_attempted: bool
    retrieval_score: float
    retrieval_degraded: bool
    retrieval_refused: bool
    evidence: list[dict]
    gate_passed: bool
    gate_reason: str

    # 退款/售后子流程
    refund_order_id: str
    refund_order_candidates: list[dict]   # 选择器卡片
    refund_order: dict                    # 确定性查回的订单快照
    refund_order_error: str
    expanded_queries: list[str]
    query_expansion_raw: str
    query_expansion_degraded: bool
    query_expansion_llm_calls: int
    refund_form: dict                     # {order_id, order_label, reasons, submit_url}

    # Agent（沿用 ch05）
    answer: str
    tool_traces: list[dict]
    tool_records: list[dict]
    pending_messages: list[AnyMessage]
    citations: list[dict]
    suggested_actions: list[SuggestedAction]
    ticket_type: str
    agent_steps: int
    llm_calls: int
    token_usage: dict[str, int]
    stop_reason: str
    context: dict

    # 可观测
    trace: list[str]
    turn_started_at: float
    turn_log: dict
    error: str | None
```

**为什么 `refund_order` 放 state**：续跑时 `refund_await_order` 重新执行，订单查询是确定性的；把快照放进 state 后，Agent 节点不需要再调 `query_order`，也避免「选择器选的是 A、Agent 查的是 B」。
**为什么 `refund_order_candidates` 放 state**：中断恢复时要校验用户点选的订单确实来自本次候选；不能信任前端传任意订单号。

---

## 6. 节点行为

### 6.1 `resolve_coref`

**触发条件**（满足任一才调模型，否则原样透传）：

- 当前消息含指代/省略标记：`它 / 他 / 她 / 这个 / 那个 / 这单 / 那单 / 刚才 / 上面 / 前面 / 那个订单 / 这款 / 这台 / 还有 / 再 / 呢` 等；
- 当前消息像口语问法（复用 `app.knowledge.query` 的口语标记思路：`啥 / 咋 / 多久 / 多少 / 怎么 / 能不能 / 可以不可以` 等）；
- 首轮消息很短且没有明确业务名词（长度 ≤ 12 且不含订单号/型号/品类词），但**整句寒暄除外**。

**寒暄不触发**：`is_trivial_chitchat(message)` 为真时直接透传，保证「你好 / 在吗 / 谢谢」仍然是零模型调用的快速通道；coref 不能把 ch05 的这个保证吃掉。

不满足则 `resolved_message = user_message`，`coref_changed=False`，`coref_source="passthrough"`，零模型调用。

**Prompt 契约**：

```
你是客服系统的问句补全器。根据对话历史，把用户当前这句话补成
不依赖上下文也能看懂的完整问题，并顺带把口语模糊问法归一成标准问法。

规则：
1. 只补全指代、省略和口语表达；不回答用户的问题。
2. 不添加历史中没有的事实；不猜测订单号、型号、金额、时间。
3. 当前问题已经完整、指代明确时，原样返回，不要强行改写。
4. 无法唯一确定指代时，保留原话，不要猜。
5. 输出 JSON：{"resolved": "...", "changed": true/false}
```

**解析与防幻觉**：

- 容忍 ```json 代码块与前后解释；取第一个 `{` 到最后一个 `}`。
- `resolved` 必须是非空字符串；超长截断到 512 字符。
- 抽取 `resolved` 里的订单号（8–20 位数字）与型号 token（如 `XH-3000A`）；若某个 token 不在「原话 + 历史文本」中出现过，判定为模型编造，整句回退原话，`coref_source="fallback"`。
- 模型超时/异常/坏 JSON：回退原话，`coref_source="fallback"`，`coref_reason` 记原因。
- `changed` 以「resolved 去空白后是否等于原话」为准，不盲信模型布尔值。
- 历史只取最近 8 条 Human/AI 文本；忽略 ToolMessage 与带 tool_calls 的中间消息；每条截断到 200 字符。

**可观测**：节点发 `coref` 事件 `{original, resolved, changed}`；state 写 `coref_*`；日志写 `resolved_message` 与 `coref_source`。

### 6.2 `classify_intent`

**输入**：`resolved_message`（不是原始消息）。**输出**：`intent / confidence / raw / source`。

**八类定义**：沿用七类业务定义，新增：

- `other`：无法归入七类、多个意图互相冲突、问题过于模糊、与售后/购物无关的怪问题。拿不准时选它，**不要硬塞**进业务意图。

**Prompt 四件套**：

1. 枚举成选择题：八个选项逐一给定义与典型问法；
2. 强制 JSON：`{"intent": "<八选一>", "confidence": 0.0-1.0}`，不要解释、不要 markdown；
3. 边界 few-shot：至少覆盖
   - 物流 vs 订单：「订单 1001 的物流到哪了」→ logistics；「订单 1001 多少钱」→ order；
   - 投诉优先：「收到的杯子是碎的，我要投诉并退款」→ complaint；
   - 退款 vs 商品：「这个能退吗」→ refund_return；「XH-3000A 续航多久」→ product_consult；
   - 售后：「发票怎么开」「要换货」→ after_sales；
   - 其他：「你们公司股票能买吗」「帮我看看那个」→ other；
4. `other` 兜底规则：低置信度、冲突、怪问题都归 other；不确定时给低 confidence，但仍必须选一个枚举。

**质量档 / 成本档**：

```
质量档（默认）：
  primary_model → 解析成功 → 返回 source="llm"
  失败 → heuristic_intent → source="heuristic"

成本档（INTENT_COST_MODE=true 且配置了 INTENT_CHEAP_MODEL）：
  cheap_model → 解析成功且 confidence >= INTENT_CONFIDENCE_THRESHOLD
      → 返回 source="llm_cheap"
  否则（低置信度 / 坏 JSON / 异常）
      → primary_model 重判一次 → source="llm_escalated"
  主模型仍失败 → heuristic_intent → source="heuristic"
```

**确定性兜底**：

- 明显寒暄整句命中 `TRIVIAL_CHITCHAT` → `chitchat`，零模型调用，`source="rule"`。
- `heuristic_intent` 的关键词优先级保持：投诉 > 退款退货 > 物流 > 订单 > 售后 > 商品咨询；**默认返回值从 `after_sales` 改为 `other`**。
- `parse_intent_reply` 的代码块/前后缀容忍、中文别名、confidence 截断保留。
- 未知枚举/坏 JSON 不抛异常，走兜底。

**可观测**：节点发 `intent` 事件 `{intent, confidence, source}`；state 写 `intent_llm_calls`。

### 6.3 `route_intent`

```python
INTENT_TO_ROUTE = {
    "logistics": "business",
    "order": "business",
    "product_consult": "knowledge",
    "refund_return": "refund",
    "after_sales": "refund",
    "complaint": "complaint",
    "chitchat": "chitchat",
    "other": "business",
}
```

未知值仍回落 `business`。`ticket_type`：投诉出口为「投诉」，退款子流程为「售后」，其他保持「售后」。

`other` 的 Agent Prompt 追加一句：**本轮意图未确定，先用一句话澄清用户要办什么；不要猜订单/政策，不要硬套售后流程。** 这条避免「other → business → Agent」又变成另一条隐式分流。

### 6.4 `refund_prepare_order`

确定性节点，不调模型。

1. 若 `state["refund_order_id"]` 已有（续跑或显式传入），直接返回 trace。
2. 从 `resolved_message` 抽订单号：
   - 先匹配 `(?:订单(?:号)?|单号)[：:\s]*([A-Za-z0-9-]{4,32})`；
   - 再匹配独立的 8–20 位数字 `(?<!\d)(\d{8,20})(?!\d)`；
   - 多个命中取第一个；纯空白/纯标点不抽。
3. 命中 → 写 `refund_order_id`，返回 trace。
4. 未命中 → `list_demo_orders(session_id, count=REFUND_ORDER_COUNT)` 生成候选；写 `refund_order_candidates`；发 `order_picker` 事件：
   ```json
   {"type": "order_picker", "turn_id": "...", "orders": [
     {"order_id": "20240115001", "title": "商品名 型号",
      "amount_yuan": "299.00", "status": "已发货", "created_at": "..."}
   ]}
   ```
5. 候选生成失败/为空 → `refund_order_error`，发空列表并**不**中断；`refund_await_order` 直接返回错误，后续 `refund_agent` 明说「订单列表暂时不可用，请稍后重试或转人工」，不让模型猜订单号。`REFUND_ORDER_COUNT` 校验 `ge=1`，正常路径不会为空。

**演示订单生成**：

- `derive_rng("demo-orders", session_id)` 派生；每单生成 11 位稳定订单号（`2024` + 7 位数字），保证同一 session 同一列表。
- 对每个订单调 `query_order(order_id)` 取状态/金额/商品编号，再调 `query_product(product_id)` 取商品名/型号；组合成卡片字段。
- 不查真实接口、不建表；与 `app/tools/business.py` 的确定性演示数据一致。

### 6.5 `refund_await_order`

1. `order_id = state.get("refund_order_id")`。
2. 若为空且 `refund_order_candidates` 为空（候选生成失败）：
   - 不 `interrupt`，直接返回 `refund_order_error`，由 `refund_agent` 明说无法确认；不让用户面对一个没有卡片的死局。
3. 若为空且候选非空：
   - `payload = interrupt({"kind": "order_picker", "orders": state["refund_order_candidates"]})`；
   - 兼容 `payload` 为 `{"order_id": "..."}` 或直接字符串；
   - 校验选中的 `order_id` 在候选列表里；不在 → 带着 `{"error": "请从列表里选择订单"}` 再次 `interrupt`，不猜、不查。
4. 确定性查订单：
   - `query_order(order_id)` → 解析 JSON；
   - `query_product(product_id)` → 解析 JSON；
   - 组装 `refund_order`：
     ```json
     {"order_id": "...", "status": "已发货", "amount_yuan": "299.00",
      "product_id": "P1234", "product_name": "无线蓝牙耳机",
      "product_model": "A42", "created_at": "2026-..."}
     ```
5. 查询结果坏 JSON/缺字段 → `refund_order_error`，不编造；继续后续节点，由 Agent 明说无法确认订单。
6. 返回 `refund_order`、`refund_order_id`、trace。

**为什么中断节点不自己发 `order_picker`**：LangGraph 在 resume 时会重新执行中断节点；如果把发事件放在中断节点内，续跑时会重复发一次卡片。把「发卡片」放在前一个节点，保证只发一次。

### 6.6 `refund_expand_queries`

1. 若 `state["expanded_queries"]` 已有（重试/重放），直接返回。
2. 调 `QueryExpander.expand(resolved_message, order=refund_order)`。
3. 输出 `{"queries": [...]}`；解析、去空、去重、截断到 `QUERY_EXPANSION_MAX_QUERIES`（默认 4）。
4. 至少保留原问题语义：若模型返回的列表里没有原问题（归一后比较），把原问题插到第一条。
5. 失败/超时/坏 JSON/空数组 → `[resolved_message]`，`query_expansion_degraded=True`。
6. 写 `expanded_queries`、`query_expansion_raw`、`query_expansion_degraded`，`llm_calls += 1`（仅实际调用时）。

### 6.7 `refund_retrieve_policies`

1. `queries = state["expanded_queries"] or [resolved_message]`。
2. 优先调 `retriever.retrieve_many(queries)`；测试/旧 fake 没有该方法时，逐条调 `retrieve_only` 并合并。
3. 合并规则：按 `chunk_id` 去重，保留最高 `score`；按分数降序；截断 `top_n`。
4. 写 `retrieval_attempted=True`、`retrieval_score`、`retrieval_degraded`、`retrieval_refused`、`evidence`、`gate_reason`。
5. **不经过 `confidence_gate`**：本节点结果直接进 `refund_agent`。`retrieval_refused` 只用于日志与 Agent Prompt 提示。
6. 单条查询抛异常：记录 warning，其他查询继续；全部失败 → 空证据，`retrieval_refused=True`，不把聊天变成 500。

### 6.8 `refund_agent`

复用 `_agent`，但传 `refund_mode=True`：

- 消息里的最后一轮用户问题替换为 `resolved_message`（模型看补全后的问题；落库仍用原话）。
- System Prompt 追加：
  1. 本轮是**退款/售后资格判断**；
  2. 订单数据（系统已查，不要再调 `query_order`）；
  3. 用户问题（已补全指代）；
  4. 政策证据（若有）；
  5. 只判断「这一单能不能退/能不能售后」，说明依据与下一步；
  6. 证据不足时明确说无法确认，不编造政策数字，不承诺到账时间；
  7. 订单数据缺失（`refund_order_error`）时，先说明「订单信息没取到」，给重试/转人工建议，不猜订单。
- 工具集沿用 refund 出口：`query_order` / `query_product` / `query_logistics` / `query_faq` / `suggest_actions`（无 `create_ticket`）；Prompt 要求不要重复查订单与政策，但保留工具作为兜底。
- 不调用 `confidence_gate`；`gate_passed` 在退款子流程里可以不设置（`total=False`），done 里按 `gate.passed=None` 呈现。
- 输出 `refund_form`：
  ```json
  {"order_id": "...", "order_label": "商品名 型号 · ¥299.00",
   "reasons": ["商品质量问题", "尺寸/颜色/型号不合适", "与描述不符",
               "不想要了（七天无理由）", "物流破损/少件", "其他"],
   "submit_url": "/api/actions/create-refund"}
  ```
- `ticket_type="售后"`。

### 6.9 `log_turn`

- 完成轮：沿用 ch05，额外写 `resolved_message / coref_changed / coref_source / expanded_queries / query_expansion_degraded / refund_order_id / evidence_count / intent_source`。
- 暂停轮：**不执行** `log_turn`，因此不写 MySQL、不写 JSONL 完成日志。`begin_turn` 已把用户消息放进 checkpoint；续跑完成后一次落库。
- `last_committed_turn_id` 幂等保留；续跑重试不会重复落库。

---

## 7. Prompt 设计

### 7.1 Coref Prompt（新增）

见 §6.1。补充 few-shot：

```
历史：
用户：订单 20240115001 的物流到哪了
客服：这个订单目前运输中，预计明天送达。
当前：那它能退吗
输出：{"resolved": "订单 20240115001 能退吗", "changed": true}

历史：（空）
当前：钱啥时候回来
输出：{"resolved": "退款到账时间", "changed": true}

历史：（空）
当前：退货政策是什么
输出：{"resolved": "退货政策是什么", "changed": false}
```

### 7.2 Intent Prompt（重写）

- 八个选项逐一枚举，每个给 1 句定义 + 2–3 个典型问法。
- 边界 few-shot 用「用户：... → JSON」的形式，至少 8 组，覆盖 §6.2 的边界。
- 明确「只输出 JSON，不要 markdown，不要解释」。
- `other` 规则放在最后，强调「拿不准就归 other，不要硬塞」。

### 7.3 Query Expansion Prompt（新增）

```
你是客服知识库的检索查询扩写器。只处理退款退货、售后场景。
把用户问题扩成 2–4 条侧重点不同的检索查询，用于检索政策条款。

要求：
1. 只输出 JSON，字段只有 queries：{"queries": ["...", "..."]}
2. 每条都是完整、可检索的问句，不是关键词堆砌。
3. 侧重点可覆盖：适用条件/时间窗口、运费与责任、特殊品类/例外、所需凭证/流程。
4. 保留订单号、商品型号、金额等原样。
5. 不回答用户问题，不添加用户没说的事实。
6. 至少包含原问题的语义；不要重复同一条。
```

输入附订单快照（订单号、商品、状态、金额），帮助模型写出与「这一单」相关的查询；但 Prompt 要求不得把订单号写进每一条 query（检索政策不需要订单号），只在需要时保留。

### 7.4 Refund Agent Prompt（追加段落）

见 §6.8。新增 `REFUND_POLICY_MISSING_TEXT`：

> 本轮没有检索到可用的政策条款。你只能基于订单数据说明已知事实，明确告诉用户「这项政策我暂时无法确认」，不要编造退货窗口、运费承担或到账时间；可以建议转人工或建工单。

---

## 8. 多查询检索合并

### 8.1 `RagPipeline.retrieve_many`

```python
async def retrieve_many(
    self, queries: Sequence[str], *, category: str | None = None
) -> RetrievalOutcome:
    """逐条检索，按 chunk_id 去重合并；单条失败不拖垮整轮。"""
```

行为：

1. 对每条 query 调 `_retrieve`（`understand` + `retriever.retrieve(ONLINE_STRATEGY, ...)`）。
2. 收集所有 `RetrievedChunk`；以 `chunk_id` 为键，保留 `score` 更高的那条；同分保留先出现的。
3. 按分数降序排序；截断 `top_n`。
4. `top_score = max(score)`；`rerank_degraded` 任一为真则为真。
5. 0 命中或（未降级且 top_score < min_score）→ `refused=True`，`refusal_source="retrieval_low_conf"`；但 refund 子流程**不使用**这个布尔短路。
6. 单条查询异常：`logger.warning`，继续其它查询；全部异常 → 空证据 + `refused=True`，`refusal_reason` 记「多查询检索全部失败」。
7. 返回的 `RetrievalOutcome.query` 取第一条理解结果；新增可选字段 `queries: tuple[UnderstoodQuery, ...] = ()` 记录全部理解结果，供日志与测试。

### 8.2 为什么不在入库侧拆多份

- 知识库只保留原始文档切块；扩写是**查询时**的临时视图。
- 入库侧拆多份会让同一知识重复计分、更新时多份漂移；本章明确不做。

---

## 9. 订单选择器、退款单与前端

### 9.1 SSE 事件契约

在现有 `meta / status / token / tools / citations / done / error` 之外新增：

| 事件 | 载荷 | 时机 |
|---|---|---|
| `coref` | `{original, resolved, changed}` | `resolve_coref` 之后 |
| `intent` | `{intent, confidence, source}` | `classify_intent` 之后 |
| `order_picker` | `{turn_id, orders: [...]}` | 缺订单号，`refund_prepare_order` |
| `awaiting_input` | `{kind:"order_picker", turn_id, orders, message}` | runner 检测到暂停 |

`done` 新增字段：`resolved_message`、`coref_changed`、`query_expansion: {queries, degraded}`、`refund_form`（仅退款子流程完成轮）。

**暂停轮不发送 `done`**：避免前端把「等待选择」误判成「回答结束」。前端收到 `awaiting_input` 后保留气泡、渲染卡片；续跑流的 `done` 到达时才结束这一轮。

### 9.2 续跑接口

`POST /api/chat/refund/select`

```json
{"session_id": "s-20261018", "order_id": "20240115001"}
```

- 响应：SSE（与聊天同帧格式）；先 `meta`，再 `token` / `tools` / `citations` / `done`。
- 校验：
  - `session_id` 允许字符与 `ChatRequest` 一致；
  - `order_id` `^[A-Za-z0-9-]{4,32}$`；
  - `graph.aget_state(config).next` 必须包含 `refund_await_order` 且存在 interrupt；否则返回 409 `{"error":{"type":"no_pending_refund","message":"没有待选择的订单。"}}`。
- 续跑失败：按现有错误外壳返回；不重复落用户轮（用户消息已在暂停轮的 checkpoint 里，续跑成功后由 `log_turn` 一次落库）。

### 9.3 非流式路径（`stream=false`）

`complete_turn` 遇到暂停时不能抛「没有 done」，也不能伪造一条完整回答。约定：

- `ChatResponse` 增加 `awaiting_input: dict | None`；
- 暂停时 `content = "请选择要办理的订单。"`，`awaiting_input` 填中断载荷，`stop_reason = "awaiting_input"`；
- `finish_reason` 用 `"stop"`（不新增 OpenAI 之外的枚举），`awaiting_input` 非空即表示未完成；
- 非流式调用方若要继续，仍需走 `POST /api/chat/refund/select`（SSE）。本章浏览器验收走流式。

### 9.4 退款单接口

`POST /api/actions/create-refund`

```json
{"session_id": "s-20261018", "order_id": "20240115001",
 "reason": "商品质量问题", "idempotency_key": "refund-20261018-001"}
```

- `reason` 用 `Literal` 固定六类，非法值 422。
- 复用 `execute_tool("create_ticket", {...})`：
  - `description = "退款申请｜订单 {order_id}｜原因：{reason}"`
  - `ticket_type = "售后"`
  - `conversation_id = session_id`
  - `idempotency_key` 透传
- 响应：
  ```json
  {"ok": true, "refund_no": "T20261018001", "ticket_no": "T20261018001",
   "order_id": "...", "reason": "...", "status": "待处理",
   "message": "已提交退款申请，工单号 ..."}
  ```
- 不新增退款表；退款单以售后工单形式落 `tickets`。

### 9.5 前端（Vibe Coding 例外）

- `web/app.js` 增加：
  - `handleFrame` 处理 `coref / intent / order_picker / awaiting_input`；
  - 订单卡片：`.order-picker` 内每张卡显示商品名、订单号、金额、状态、下单时间；点击后禁用整组卡片、显示「已选择订单 …」，调续跑接口；
  - 续跑复用同一个 `applyFrame`/`consumeStream`，token 接在**同一张 bot 气泡**里；
  - 退款表单：`.refund-form`，订单号只读、原因 `<select>`（来自 `done.refund_form.reasons`）、提交按钮；成功显示 `refund_no`；
  - 暂停态：`awaiting_input` 后不显示「（客服没有返回内容）」，状态栏提示「等待选择订单」。
- `web/style.css` 增加卡片/表单样式，沿用现有配色与圆角。
- 前端不套 TDD/code review；以浏览器实际效果为准。

### 9.6 验收 3 的可视证据

浏览器里问「这个能退吗」时：

1. `coref` 事件先显示「已补全：订单 ... 能退吗」；
2. 若消息里没有订单号，`order_picker` 卡片出现；
3. 选单后 `token` 流继续，`done.refund_form` 出现；
4. `done.trace` / 日志里能看到 `refund_prepare_order → refund_await_order → refund_expand_queries → refund_retrieve_policies → refund_agent`。

---

## 10. 配置

新增到 `app/config.py` 与 `.env.example`（均有默认值，现有 `.env` 不改也能跑）：

| 配置 | 默认 | 说明 |
|---|---|---|
| `INTENT_MODEL` | `""` | 空 = 用 `OPENAI_MODEL`；质量档主模型 |
| `INTENT_CHEAP_MODEL` | `""` | 成本档小模型；空 = 成本档不启用 |
| `INTENT_COST_MODE` | `false` | true 才走「小模型先判 → 低置信度升级」 |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.6` | 成本档升级阈值；`0–1` |
| `INTENT_TIMEOUT` | `5.0` | 意图调用超时（秒） |
| `COREF_ENABLED` | `true` | 关掉只走确定性透传（排障） |
| `COREF_TIMEOUT` | `5.0` | 指代消解超时（秒） |
| `COREF_HISTORY_MESSAGES` | `8` | 送入 coref 的最近消息条数（Human/AI 文本） |
| `QUERY_EXPANSION_ENABLED` | `true` | 关掉退款子流程扩写，直接单查询 |
| `QUERY_EXPANSION_TIMEOUT` | `5.0` | 扩写超时（秒） |
| `QUERY_EXPANSION_MAX_QUERIES` | `4` | 扩写最多几条；`2–6` |
| `REFUND_ORDER_COUNT` | `3` | 订单选择器候选数；`ge=1` |

`app/llm.py` 增加 `get_chat_model(settings, model_name: str | None = None)`；`INTENT_MODEL` 为空时回落 `openai_model`。不新增组件、不换上游协议。

---

## 11. 失败路径与幂等

| 场景 | 行为 |
|---|---|
| coref 超时/坏 JSON/编造 token | 回退原话；`coref_source=fallback`；不阻断 |
| 意图模型超时/坏 JSON | 成本档先尝试主模型；仍失败走启发式（默认 other） |
| 扩写超时/坏 JSON | `[resolved_message]`；`query_expansion_degraded=true` |
| 多查询某条检索失败 | 跳过该条，继续；全失败 → 空证据 + refused；不 500 |
| 订单候选生成失败 | `refund_order_error`；中断载荷空列表；Agent 明说无法确认 |
| 用户点选非法订单 | 重新 interrupt；不查、不猜 |
| 暂停轮 | 不写 MySQL、不写完成日志；checkpoint 保存现场 |
| 续跑重试 | `last_committed_turn_id` 幂等；不会重复落库 |
| 用户没点卡片又发新消息 | runner 检测 pending interrupt，重新发 `awaiting_input`，不把新消息塞进暂停图 |
| 退款单非法原因 | 422；不写 tickets |
| 退款单重复提交 | `idempotency_key` 复用 `create_ticket` 的幂等语义 |

**暂停轮的可见性**：用户消息已进入 checkpoint，但未进 MySQL；刷新页面看不到这条 pending 轮。本轮明确不做「刷新后恢复选择器」（见 §1.2），但同一页面内续跑不受影响。

---

## 12. 日志与可观测

`build_turn_log` 新增字段：

- `resolved_message`
- `coref_changed` / `coref_source`
- `intent_source`
- `expanded_queries` / `query_expansion_degraded`
- `refund_order_id`
- `refund_policy_evidence_count`（= evidence_count，保留原名亦可）
- `interrupted`（完成轮恒为 false；暂停轮不写完成日志，由 runner 的 `awaiting_input` 事件体现）

`done_payload` 新增：

- `resolved_message`
- `coref_changed`
- `intent` / `route`（已有）
- `query_expansion`
- `refund_form`
- `trace`（本轮节点序列，便于验收）

---

## 13. 测试与评估策略

### 13.1 纯 Prompt / 数据类任务（TDD 例外）

按用户要求，Prompt/数据类不先写单测再实现，而是用**标注样例或评估集**验证：

- 新增 `evals/ch06_multiturn_cases.jsonl`：多轮会话，每轮标 `expected_intent`、`coref_must_contain`、`coref_must_not_contain`（代词）。
- 新增 `evals/run_ch06_evals.py`：
  - 真实模型跑 coref + intent；
  - 统计 intent accuracy、other 落点准确率、coref 通过率、JSON 解析失败数；
  - 输出 `dev-notes/ch06-eval-report.md` 与 `dev-notes/ch06-evals.json`。
- 覆盖：
  - 物流 → 退款 → 物流 三轮切换；
  - 「这个能退吗」缺订单号；
  - 怪问题 → other；
  - 投诉优先；
  - 商品咨询 vs 退款边界；
  - 寒暄快速通道。
- 评估集必须与 Prompt few-shot **不逐字重合**，减少提示记忆；报告里注明这一点。

### 13.2 代码类任务（照常 TDD）

- `CorefResolver`：触发/透传、解析、防幻觉、超时回退、历史截断。
- `IntentClassifier`：八类、other、成本档升级、质量档、坏 JSON、寒暄快速通道、默认 other。
- `QueryExpander`：强制 JSON、条数/去重/截断、失败回退、保留原问题。
- `RagPipeline.retrieve_many`：多查询合并、去重取最高分、单条失败、全失败。
- 退款图：缺订单 → `order_picker` + interrupt；resume → 订单快照 → 扩写 → 多查询检索 → Agent → done；非法选择重新中断。
- API：续跑 409、SSE 新事件、退款单校验/幂等。
- 回归更新：ch05 的 state/intent/graph 测试按新契约调整；确保 ch01–ch05 其余测试仍绿。

### 13.3 前端

Vibe Coding：先在浏览器跑通四条验收，再按用户描述调整卡片/表单样式与交互；不要求前端单测。

---

## 14. 验收对照

| 验收标准 | 设计对应 | 证据 |
|---|---|---|
| 1. 多轮对话从物流切退款再切回物流，每轮意图对、指代补全对 | `CorefResolver` + `IntentClassifier` + 评估集 M01/M02 | `dev-notes/ch06-eval-report.md` 逐轮明细 |
| 2. 意图 JSON 稳定可解析，怪问题落 other | 强制 JSON + few-shot + `other` 规则 + 解析器 | 评估集 JSON 失败数 0；other 用例全中 |
| 3. 问「这个能退吗」先补全指代、再走退款子流程拿订单和政策 | coref 事件 + refund 子图 + 强制多查询检索 | 浏览器截图 + `done.trace` + 日志 |
| 4. 不带订单号问退款，弹出订单选择器，点选后子流程走完 | `refund_prepare_order` → `interrupt` → `/api/chat/refund/select` → resume | 浏览器截图 + SSE 事件 + 续跑 done |

---

## 15. 交付物

- 代码：`app/agent/coref.py`、`intent.py`、`query_expansion.py`、`refund_orders.py`、`prompts.py`、`state.py`、`graph.py`、`runner.py`、`logging.py`；`app/knowledge/pipeline.py`；`app/api.py`、`app/schemas.py`、`app/llm.py`、`app/config.py`；`web/app.js`、`web/style.css`。
- 测试：`tests/test_ch06_*.py` + 更新的 ch05 测试。
- 评估：`evals/ch06_multiturn_cases.jsonl`、`evals/run_ch06_evals.py`、报告。
- 文档：本 spec、plan、`dev-notes/ch06.md`、README/.env.example 更新。
- 演示：`scripts/demo-ch06.py`（或复用 `scripts/demo-ch05.py` 改造）输出四条验收的命令与结果。

---

## 16. 风险与回滚

| 风险 | 影响 | 应对 |
|---|---|---|
| `deepseek-flash` 对 coref/意图/few-shot 跟随不佳 | 评估准确率不达标 | 先报告评估结果；由用户决定是否配置更强的 `INTENT_MODEL`，不擅自换模型 |
| 多查询检索成本上升 | 退款/售后每轮多 1–3 次嵌入/重排 | 仅 refund 路由扩写；默认最多 4 条；配置可关 |
| interrupt 在 SQLite saver 下的行为与 InMemory 不同 | 续跑失败 | Task 0 用真实 `AsyncSqliteSaver` 做探针；测试同时覆盖 InMemory |
| 暂停轮不进 MySQL，刷新丢 pending 轮 | 页面刷新后看不到选择器 | 本章明确不做；同一页面内续跑正常；checkpoint 仍在，后续可加恢复接口 |
| 前端续跑流与首次流共用状态出错 | 卡片点选后气泡错位 | Vibe Coding 以浏览器实际效果为准；复用同一 `applyFrame`/ctx |
| 旧 ch05 测试按 7 类/4 出口写死 | 回归大量失败 | 计划里单列「契约更新」任务，逐条改断言，不删测试 |

---

## 17. 明确不做的细节

- 不做意图分类模型训练/微调；不做 BERT。
- 不做跨会话记忆；coref 只看当前会话。
- 不做真实订单归属、真实退款单表、真实支付/物流接口。
- 不做入库侧 query 拆存；知识库仍一份。
- 不做刷新后 pending 选择器恢复。
- 不做新的 Langfuse/MCP/飞轮组件。
- 不把 `create_ticket` 放回 Agent 工具集；退款单仍由用户在前端提交。
