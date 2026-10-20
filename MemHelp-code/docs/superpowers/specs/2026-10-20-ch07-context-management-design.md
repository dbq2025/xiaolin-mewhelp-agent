# ch07 设计：会话上下文管理（三层历史 + 异步摘要 + 预算倒推）

- 日期：2026-10-20
- 状态：待评审
- 范围：**只服务当前会话**。不跨会话、不存用户画像、不做语义检索捞历史、不做主题重要度留事实。
- 定死的选型（用户 m00001 点名，不得替换）：LangGraph `State.messages` + `add_messages`
  reducer + checkpoint；LangChain `trim_messages`；MySQL `conversations` / `messages`。

---

## 1. 要解决的问题

ch01–ch06 的上下文只有一层：`app/context.py` 用 `trim_messages` 按固定
`CONTEXT_TOKEN_BUDGET=2000` 从尾部截历史。两个缺陷：

1. **截掉就没了**。超过预算的早期轮次直接消失，用户问「最开始那个订单后来怎么说」时
   模型已经不知道订单号。
2. **预算与窗口脱钩**。2000 是写死的常量，不是从模型窗口倒推的；调大窗口不生效，
   调小窗口则会溢出。

ch07 把它换成三层：**近处留原文、中间按规则截短、远处压成梗概**。压缩有成本，
所以「装得下就一个字都不压」。

## 2. 三层与两个锚点

```
messages 表（按 id 升序，全部原文）
─────────────────────────────────────────────────────────────►
[ 已摘要区间 ]        [ 层 2：规则截短 ]        [ 层 1：原文滑窗 ]
 id ≤ summary_upto      (summary_upto, layer1_from)      id ≥ layer1_from
        │                      │                          │
        │                      │                          └─ 一个字节都不改
        │                      └─ user 原话不动；assistant 只留开头 N 字；tool 换成一行标识
        └─ 由 conversation_summaries 的若干段落代表（不读 messages）
```

两个锚点都在 `conversations` 表上，**降级只改这两个 int，不搬任何数据**：

| 列 | 含义 |
|---|---|
| `summary_upto_msg_id` | 摘要已覆盖到哪条消息（含）。`NULL` = 还没有摘要 |
| `layer1_from_msg_id` | 层 1 从哪条消息起（含）。`NULL` = 全部历史都算层 1 |

判定一个消息 id `m` 落在哪层：

- `m <= summary_upto_msg_id` → 层 3（不进上下文，只由摘要代表）
- `summary_upto_msg_id < m < layer1_from_msg_id` → 层 2
- `m >= layer1_from_msg_id` → 层 1

### 2.1 规则截短（层 2）

`app/context/layers.py:rule_trim_row(row, *, assistant_preview_chars=60, tool_marker_chars=40)`

| 角色 | 处理 |
|---|---|
| `user` | **原文不动**（用户原话是事实源，改写会引入编造风险） |
| `assistant` | 保留开头 `assistant_preview_chars` 个字符，超出则补 `…`。空正文（纯工具调用行）保留为空 |
| `tool` | 换成一行标识：`[工具结果 · {name 或 tool_call_id} · {原字数} 字已省略]` |

层 2 只做**表达层**压缩，不改行数：一条消息仍对应一条消息，方便日志里逐条对照。

### 2.2 层 1 降级（挪 `layer1_from_msg_id`）

条件：层 1 原文按校准 token 计 > `layer1_budget`。

做法：用 `trim_messages(strategy="last", start_on="human", allow_partial=False,
token_counter=校准计数器)` 找出「从尾部起能装进 `layer1_budget` 的连续消息段」，
把 `layer1_from_msg_id` 前移到该段第一条消息的 id。**按整轮切**（`start_on="human"`），
不会出现半轮。

不变量：**用户当前这句话永远保留**（它不在历史里，是拼装时最后追加的）。
若一整轮都装不下，保留该轮并标记 `degraded=True`——宁可超一点，也不能让模型看不到
刚刚发生的事。

### 2.3 层 2 触发摘要（挪 `summary_upto_msg_id`）

条件：层 2 经过规则截短后的 token 数 > `layer2_budget`。

做法：后台摘要任务把层 2 这一段压成**新的一段**追加进 `conversation_summaries`，
成功后把 `summary_upto_msg_id` 推到「层 2 最后一条消息的 id」（即层 1 起点之前一条）。
于是层 2 清空，层 1 不动。

**不回头重写**：只 append，不 update 旧段、不把旧段合并进新段。旧梗概只作为
背景文本喂给摘要模型（让它知道「这些已经记过了，别重复」），不参与本次合并——
否则同一个事实每压一次就被有损压一次，五轮之后订单号会变形。

## 3. 数据模型

### 3.1 `conversations` 加两列（`sql/ch07-ddl.sql`）

```sql
ALTER TABLE conversations
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息(含)',
  ADD COLUMN layer1_from_msg_id  BIGINT UNSIGNED NULL COMMENT '层1原文滑窗从哪条消息起(含)';
```

> 旧版 `sql/ch07-ddl.sql` 是「两列 + 摘要塞 conversations.summary」的双层方案，
> 与本设计冲突，**整份重写**。`summary` 列不再需要。

### 3.2 新表 `conversation_summaries`（一段一行，只 append）

| 列 | 类型 | 说明 |
|---|---|---|
| `id` | BIGINT UNSIGNED PK AI | |
| `conversation_id` | BIGINT UNSIGNED NOT NULL | 外键 → `conversations.id` |
| `segment_no` | INT NOT NULL | 第 N 段（从 1 起，`UNIQUE(conversation_id, segment_no)`） |
| `from_msg_id` | BIGINT UNSIGNED NOT NULL | 本段覆盖起始消息 id（含） |
| `upto_msg_id` | BIGINT UNSIGNED NOT NULL | 本段覆盖结束消息 id（含） |
| `content` | TEXT NOT NULL | 梗概正文（几十到一两百字） |
| `msg_count` | INT NOT NULL | 本段压了多少条消息 |
| `source_tokens` | INT NOT NULL | 压缩前层 2 的校准 token 数（留痕用） |
| `model` | VARCHAR(64) NULL | 产出本段的模型名 |
| `elapsed_ms` | INT NULL | 摘要耗时 |
| `created_at` | DATETIME NOT NULL | |

索引：`idx_conversation_id (conversation_id, segment_no)`。

**为什么单开表**：`messages` 表是「逐条消息」的语义，摘要不是消息；塞进去会污染
`history_as_messages`、前端回载、以及所有按条数/角色统计的地方。

## 4. token 预算：从窗口倒推

`app/context/budget.py:compute_budget(settings) -> BudgetReport`（纯函数，无 IO）。

```
W                = MODEL_CONTEXT_WINDOW
output_reserve   = MAX_OUTPUT_TOKENS
react_peak       = MAX_USER_INPUT_TOKENS + MAX_AGENT_STEPS × TOOL_RESULT_MAX_TOKENS
safety_margin    = ceil(W × SAFETY_MARGIN_RATIO)          # 0.10
system_reserve   = SYSTEM_PROMPT_RESERVE_TOKENS           # 1200
evidence_reserve = RERANK_TOP_K × EVIDENCE_TOKENS_PER_ITEM # 250
summary_reserve  = SUMMARY_RESERVE_TOKENS                 # 500

overhead         = output_reserve + react_peak + safety_margin
                   + system_reserve + evidence_reserve + summary_reserve
window_available = W − overhead
desired_history  = HISTORY_DESIRED_TURNS × HISTORY_TOKENS_PER_TURN
history_budget   = min(window_available, desired_history)
layer2_budget    = round(history_budget × LAYER2_RATIO)   # 0.3
layer1_budget    = history_budget − layer2_budget − 1     # 那 1 个 token 是层 2 边界标记
```

### 4.1 演示配置的精确复现（验收第 2 条）

`MODEL_CONTEXT_WINDOW=18000 MAX_OUTPUT_TOKENS=2000 MAX_USER_INPUT_TOKENS=2000
MAX_AGENT_STEPS=3 TOOL_RESULT_MAX_TOKENS=1200 RERANK_TOP_K=5`：

| 项 | 值 |
|---|---|
| 输出预留 | 2000 |
| 单轮 ReAct 峰值 | 2000 + 3×1200 = **5600** |
| 安全余量 | ceil(18000×0.10) = **1800** |
| 系统提示 | 1200 |
| 检索证据 | 5×250 = 1250 |
| 注入梗概 | 500 |
| 合计固定开销 | **12350** |
| 窗口匀得出来 | 18000 − 12350 = **5650** |
| 想留住的轮数 | 20×400 = 8000（≥ 5650，取小的那个） |
| **历史预算** | **5650** ✓ |
| **层 1 预算** | 5650 − 1695 − 1 = **3954** ✓ |
| **层 2 预算** | round(5650×0.3) = **1695** ✓ |

口径全部由本设计的公式决定（用户 m00105：「你来决定」）。**只调窗口不调固定开销**
（例如只把窗口设成 8000），`window_available` 会变负 → `ok=False`，
`reason` 以「上下文预算不足」开头，这正对应验收第 2 条的括号说明。

### 4.2 启动自检

`BudgetReport.ok=False` 的两种情形：

1. `window_available <= 0`：固定开销已经把窗口吃光；
2. `history_budget < HISTORY_TOKENS_PER_TURN`：连一轮都装不下。

`api.py` 的 lifespan 里启动时算一次：`ok=False` 打 `ERROR` 级
`上下文预算不足：…`（含各项开销明细），`ok=True` 打 `INFO` 一行摘要（窗口/历史/层1/层2）。
不阻断启动——预算再紧也要能回答「你好」。

### 4.3 中文 token 口径必须与预算同源

实测：`count_tokens_approximately` 默认 `chars_per_token=4.0` 在中文上严重低估
（28 个汉字计 17 tokens，真实分词约 40–50）。校准计数器
`app/context/tokens.py:count_tokens`：

```
每条消息 = ceil(汉字及全角标点数 × CJK_CHARS_PER_TOKEN(1.0)
              + 其余字符数 ÷ OTHER_CHARS_PER_TOKEN(3.5))
          + MESSAGE_OVERHEAD_TOKENS(3)
```

固定开销里的 `SYSTEM_PROMPT_RESERVE_TOKENS` 等常量就是按这个口径标定的
（实测真实 system prompt 720 字符 = 583 校准 token，预留 1200 是含证据余量的保守值）。
**只改预算不改计数器净效果是反的**：预算按 4 字符/token 算得很宽松，计数器按 1 算得很紧，
结果是每轮都「超预算」，压缩永远触发。

## 5. 上下文拼装（固定顺序）

`app/context/assemble.py:assemble_context(...) -> ContextResult`

```
[0] SystemMessage(人设 + 红线 + 工具定义)        ← 每轮完全一致，前缀缓存锚点
[1..] 层 2 规则截短后的消息（旧 → 新）
[..] 层 1 原文消息（旧 → 新）
[..] HumanMessage(用户当前这句话)                ← resolved / 改写后的版本
[..] HumanMessage("[会话梗概]… [检索证据]…")     ← 梗概与证据合成一条，挂在这句之后
```

三条硬约束：

1. **`SystemMessage` 只有第 0 条，且每轮字节一致**。梗概与证据**不许**进 system：
   上游 chat template 会把所有 system 消息上提到最前面合并渲染，工具定义会被挤到
   可变内容之后，前缀缓存整段作废。
2. **梗概与证据合成一条**（不是两条），放在用户这句之后——它们都是「本轮参考」，
   语义上属于同一次输入。
3. **当前这句永远在梗概/证据之前**：模型读到最后一条 human 是「参考资料」时，
   仍能从前一条 human 看到真正要回答的问题。

`build_context(history, question, system_message, budget_tokens)` 保留为**兼容外壳**
（`app/chat.py` 与 `tests/test_context.py` 在用）：内部转成单层规划，行为与 ch01–ch06
逐字节一致，不触发摘要、不改锚点。

## 6. 后台异步摘要

`app/agent/summarizer.py:ContextSummarizer`

```python
class ContextSummarizer:
    def __init__(self, *, model, repositories_factory, settings, logger=None)
    def schedule(self, *, session_id, conversation_id, plan, background=None) -> bool
    async def run_once(self, job: SummaryJob) -> SummaryOutcome   # 可被测试直接 await
```

- `schedule()` 只在 `plan.summary_due` 时动作。**用量驱动，不数条数**：判据是层 2
  规则截短后的校准 token 数，不看消息条数——工具结果的体量在行数上根本看不出来。
- `asyncio.create_task(...)` 起任务，**立刻返回**，不 `await`；当前轮的回复照常流式发出。
- 会话级 `_inflight: set[str]` 去重：同一会话已有摘要任务在跑时，本轮 `skip` 并留痕。
- 任务自己开 Session（请求级 Session 在响应结束后就关了），自己 commit / rollback / close。
- `run_once` 全程 try/except：异常只落日志（`summary failed`），不冒泡到请求。
- 落库用**乐观并发**：写段之前重新读一次 `layer1_from_msg_id`，与规划时不一致就放弃
  （`summary skipped：anchor moved`），避免把已经被降级覆盖的区间再压一遍。

摘要提示词 `SUMMARY_PROMPT`（`app/agent/prompts.py`）只提炼：

- 问过哪款商品 / 哪个订单号 / 报过的手机号
- 明确诉求（退货、换货、催发货、退款…）
- 还没解决的问题
- 对话里没出现的内容**一个字不许编**；寒暄闲聊不留；长度几十到一两百字

旧段正文以「已记录背景」形式传给模型，但在提示词里明确「本次只总结【待压缩对话】」。

落成日志（`log/app.log`）：

```
summary trigger session=… conv=… layer2_tokens=1832 > budget=1695 msgs=14
summary start   session=… seg=3 from_msg=41 upto_msg=54
summary done    session=… seg=3 chars=118 elapsed_ms=742 upto_msg_id=54
summary skip    session=… reason=inflight
summary failed  session=… error=…
```

## 7. LangGraph State 与完整历史

- `AgentState.messages` 继续挂 `add_messages`；各节点只 `return {"messages": [...]}`，
  框架按顺序并入。完整历史由 checkpoint（SQLite saver）落盘。
- **完整历史与精简版各走各的**：`_run_agent_node` 只用 `assemble_context()` 的返回值
  去调模型，**不回写** `state["messages"]`；`state["context"]` 只存元数据
  （计数、预算、锚点），不存裁过的消息列表（避免 checkpoint 里存两份历史）。
- 历史的数据源：`repositories.messages.list_for_conversation(conversation_id)` 返回带
  **数据库 id** 的 `MessageRow`。锚点是数据库 id，所以必须从库里读；`state.messages`
  里的 LangChain 消息 id 是 UUID，对不上。
- 库为空（首轮 / 测试未落库）时退化为 `state.messages`，此时无 id → 全算层 1，
  走 legacy 裁剪路径，不触发摘要。

## 8. 可观测性

新增 `app/agent/ctxlog.py`，写 `log/app.log`（`APP_LOG_PATH`，相对项目根解析）：

- `log_model_ctx(...)`：**每轮调模型前**，主 Agent 实际发出的上下文原样落盘——
  摘要全文（逐段）、滑窗逐条消息（角色 + 正文）、窗口条数、校准 token 估算、
  两个锚点、层 1 / 层 2 预算与实占。
- `log_history_ctx(...)`：指代消解 / 意图识别共用的历史——摘要（若有）+ 滑窗。
  **每轮必打**，闲聊兜底轮也打（那轮同样把历史喂给了 coref / intent）。
- 摘要任务五种状态全留痕，带覆盖边界（from/upto msg id）与耗时。

日志格式：一条记录以 `model_ctx session=… turn=… …` 开头，随后是 `\n` 分隔的正文，
`grep model_ctx` / `grep history_ctx` 一行就能命中，`tail -n` 能读全。

## 9. HTTP 接口与前端

### 9.1 两个只读接口

```
GET /api/conversations?user_id=anonymous&limit=50
→ {"conversations": [
     {"id": 12, "session_id": "s-…", "user_id": "anonymous",
      "preview": "我上周买的耳机还没发货", "created_at": "...", "updated_at": "...",
      "message_count": 24, "summarized": true, "summary_segments": 2}
   ]}

GET /api/conversations/{conversation_id}/messages
→ {"session_id": "s-…", "messages": [{"role": "human"|"ai"|"tool", "content": "...",
                                      "tools": [...], "citations": [...]}]}
```

- 二者只读，不改任何状态。
- `{id}` 是 `conversations.id`（不是 session_id）——侧栏按 id 定位，避免 session_id
  在前端被当作主键传来传去。
- 详情复用 `_db_session_messages_payload` 的既有形状（`human` / `ai` / `tool`、
  `tools` / `citations` 条件出现），前端回载逻辑不用改。
- 列表「已摘要」标记来自 `conversation_summaries` 是否有行，不是看 `summary` 字段。

### 9.2 前端（`web/`，无构建步骤）

- 聊天页左侧加会话侧栏：新建在前，显示首问预览 + 已摘要标记（`✂`），当前会话高亮。
- 点击某会话：`GET /api/conversations/{id}/messages` 回载历史原文，把 `session_id`
  换成该会话的，继续聊（后续消息落到同一个 `conversations` 行）。
- 「新对话」只清空当前视图并生成新 session_id，旧会话仍在侧栏。
- **侧栏加载失败静默降级**：`try/catch` 包住，失败只是侧栏空着，聊天功能完全不受影响。
- 移动端（<720px）侧栏折叠为顶部抽屉，默认收起。

## 10. 配置项（`app/config.py` 新增）

| 键 | 默认 | 说明 |
|---|---|---|
| `MODEL_CONTEXT_WINDOW` | 32000 | 模型窗口；默认值要能装下 20 轮（验收第 3 条） |
| `MAX_OUTPUT_TOKENS` | 4000 | 输出预留 |
| `MAX_USER_INPUT_TOKENS` | 2000 | 单轮用户输入峰值 |
| `TOOL_RESULT_MAX_TOKENS` | 1200 | 单次工具结果峰值 |
| `MAX_AGENT_STEPS` | 3 | ReAct 步数上限（新键，同时兼容旧名 `AGENT_MAX_STEPS`） |
| `RERANK_TOP_K` | 10 | 最终证据条数（同时兼容旧名 `RAG_RERANK_TOP_N`，一个字段两个环境变量名） |
| `SAFETY_MARGIN_RATIO` | 0.10 | 安全余量占窗口比例 |
| `SYSTEM_PROMPT_RESERVE_TOKENS` | 1200 | 系统提示固定估算 |
| `EVIDENCE_TOKENS_PER_ITEM` | 250 | 每条证据估算 |
| `SUMMARY_RESERVE_TOKENS` | 500 | 注入梗概估算 |
| `HISTORY_DESIRED_TURNS` | 20 | 想留住的轮数 |
| `HISTORY_TOKENS_PER_TURN` | 400 | 每轮稳态占用 |
| `LAYER1_RATIO` / `LAYER2_RATIO` | 0.7 / 0.3 | 层 1 / 层 2 切分 |
| `ASSISTANT_PREVIEW_CHARS` | 60 | 层 2 客服答复保留字数 |
| `SUMMARY_ENABLED` | true | 摘要总开关（关掉只降级不摘要，方便对照实验） |
| `SUMMARY_MODEL` | "" | 空 = 用主模型 |
| `SUMMARY_TIMEOUT_SECONDS` | 20 | 单次摘要超时 |
| `SUMMARY_MAX_TOKENS` | 300 | 摘要输出上限 |
| `CJK_CHARS_PER_TOKEN` / `OTHER_CHARS_PER_TOKEN` / `MESSAGE_OVERHEAD_TOKENS` | 1.0 / 3.5 / 3 | 校准口径 |
| `APP_LOG_PATH` | `log/app.log` | 上下文日志 |

`agent_max_steps` 保留原字段与默认值（6），用 `AliasChoices("AGENT_MAX_STEPS",
"MAX_AGENT_STEPS")` 接收两个环境变量名，避免改默认值把 ch05/ch06 的既有测试打翻。

## 11. 不做

- 语义检索捞历史（向量检索早期轮次）
- 主题重要度 / 关键事实固化
- 跨会话长期记忆、用户画像
- 摘要的增量重写 / 旧段合并

## 12. 验收映射

| 验收 | 靠什么保证 |
|---|---|
| 20+ 轮不爆不崩 | 预算倒推 + 三层降级；`tests/test_ch07_budget.py` 与端到端脚本 |
| 演示配置 5650/3954/1695 + 完整级联日志 | `compute_budget` 单测钉死数字；`scripts/demo_ch07.py` 跑真配置并 grep 日志 |
| 默认窗口 20 轮零降级零摘要 | `tests/test_ch07_no_compression.py`：默认配置下 `degraded=False`、`summary_due=False` |
| 摘要不阻塞 + grep 可见 | `tests/test_ch07_summarizer.py` 断言 `schedule()` 立即返回；日志断言 |
| 侧栏多会话 / 回载 / 续聊 | `tests/test_ch07_api.py` + `scripts/ch07_ui_check.py` |
