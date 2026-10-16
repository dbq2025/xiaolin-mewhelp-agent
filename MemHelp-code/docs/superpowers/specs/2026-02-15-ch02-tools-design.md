# MewHelp 第二章「Function Calling 工具链」设计文档

- 日期：2026-02-15
- 状态：待用户评审
- 上游：沿用 ch01 单端点（实测 `deepseek-flash`，已确认返回真实 `tool_calls`）
- 前置：ch01「纯对话」已验收（分支 `feat/ch01-pure-chat`，HEAD `061dd68`）

## 0. 流程分类与范围

**分类：architectural**（新增持久化子系统、工具子系统，并扩展 SSE 事件契约与
前后端接口）。已在对话中向用户宣布，允许推翻。

**本章流程变体（用户指定）**

| 对象 | 走的流程 |
|---|---|
| 数据库层、工具层、工具编排、SSE 契约 | brainstorm → spec → plan → 执行；有可单测代码的走 TDD |
| 测试数据（FAQ 语料）、Prompt 类产出 | 不走 TDD，改用**标注样例 / 评估集跑一遍**验证 |
| `web/` 聊天页（徽章 UI） | **Vibe Coding**：用户描述效果、我直接改，不套 brainstorm / TDD / code review |

**本章不做（用户明确）**

- 多轮自动循环的 Agent Loop（**只做单轮**：模型调一次工具就收敛）
- 向量检索 / RAG
- 真实电商系统与物流系统 API 接入（演示起见在工具内部随机生成数据，**不接真实接口、不建表**）

### 验收标准

| # | 验收标准 | 判定方式 | 预期 |
|---|---|---|---|
| 1 | 浏览器问「订单 1001 的物流到哪了」 | 聊天页气泡显示工具徽章 + 按工具返回结果作答 | 通过 |
| 2 | 问「退货政策是什么」 | `query_faq` 关键词命中 `faq` 表并作答 | 通过 |
| 3 | 换个说法问「邮费是多少」 | 关键词 `LIKE` **查不出来** | **故意失败**：漏召回是预期结果，记入 `dev-notes/ch02.md` 留给 ch03 升级 |

> 验收 3 是本章**刻意保留的能力缺口**，不是 bug。它的价值在于为 ch03 的向量检索
> 提供可复现的「改造前基线」，所以必须能稳定复现，并且评估集里要有一条专门钉住它。

## 1. 已确认的决策（对话中逐个敲定）

| 决策点 | 结论 | 理由 |
|---|---|---|
| 会话身份 | `conversations` 表**加 `session_id VARCHAR(64) NOT NULL UNIQUE`** | 前后端契约零改动：聊天页 `sessionStorage` 与 `POST /api/chat/sessions/{session_id}/messages` 直接就是库里的会话键 |
| 建表灌数 | `docker-compose.yml` 挂 `sql/` 到 `/docker-entrypoint-initdb.d` | 一条 `docker compose up -d` 把表与测试数据灌好，演示可一次跑通 |
| 无关改动 | `app/prompts.py` 的「小美→小强」未提交改动**还原** | 不属于 ch02；ch01 的 Prompt 已随评估集验收，改名会让两者不符 |
| 工具实现 | 五个 `@tool`；订单/商品/物流**工具内随机生成**，FAQ/工单**真连库** | 用户明确要求；随机数据用固定种子派生，保证可复现 |
| 工具注册 | 显式 `TOOL_REGISTRY` 字典 + 启动时一致性自检 | 名字是字符串（SSE 帧、落库都要用），显式注册表比反射更可测、更少惊喜 |
| 会话存储 | ch01 的内存存储**被数据库取代** | 需求要求「聊天记录含工具调用与结果落 conversations/messages 表」；两套并存的真相分裂是 ch01 `memory.py` 注释里明确预告要避免的 |

## 2. 数据层

### 2.1 DDL 改动（`sql/ch02-ddl.sql`，仅一处）

```sql
CREATE TABLE conversations (
  id          BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '会话主键',
  session_id  VARCHAR(64)     NOT NULL                COMMENT '对外会话标识,聊天页与接口用它',  -- 新增
  user_id     VARCHAR(64)     NOT NULL                COMMENT '用户标识',
  ...
  PRIMARY KEY (id),
  UNIQUE KEY uk_session_id (session_id),   -- 新增
  KEY idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='客服会话';
```

其余三张表（`messages` / `faq` / `tickets`）保持原样。`messages` 的
`tool_calls JSON` / `tool_call_id` 两列正是本章需要的，无需改动。

### 2.2 `sql/ch02-seed.sql`（新增，与 DDL 同目录，被 compose 自动执行）

| 表 | 内容 | 说明 |
|---|---|---|
| `faq` | 10 条，覆盖退换货 / 退款 / 物流 / 发票 / 质量 5 个分类 | 「退货政策」「运费」是验收 1/3 的关键条目 |
| `conversations` + `messages` | 1 个演示会话（`session_id='demo-ch01-0001'`），含 1 组工具调用轨迹 | 给 `messages` 表灌真实形状的数据，`/messages` 接口可立刻看到 |
| `tickets` | 2 条示例工单 | 便于核对 `create_ticket` 写入结果与已有数据不冲突 |

**验收 3 的语料设计（刻意为之）**：`faq` 中相关条目的问法是
「退货运费谁承担？」「满多少包邮？」，**不含「邮费」二字**。所以「邮费是多少」
用 `LIKE '%邮费%'` 命中 0 条 → 漏召回稳定复现。同时 `faq` 里**另有**一条含
「邮费」的无关条目（如「发票邮费说明」）用于确认工具不会假命中 —— 这样验收 3
钉住的是「关键词对不上」，而不是「表里没数据」这种弱结论。

### 2.3 分层与职责

```
MemHelp-code/app/
├─ config.py           # + MySQL 连接配置（唯一配置来源，仍然只有这里读 .env）
├─ db/
│  ├─ base.py          # DeclarativeBase + 通用约定
│  ├─ models.py        # Conversation / Message / Faq / Ticket 四个 ORM 类
│  ├─ session.py       # engine / sessionmaker / get_db 依赖（测试可注入）
│  └─ repositories.py  # 数据访问：会话、消息流水、FAQ 查询、工单写入
├─ tools/
│  ├─ registry.py      # 注册表 + 一致性自检
│  ├─ schemas.py       # 五个工具的 Pydantic 参数模型（参数 Schema 校验）
│  ├─ business.py      # query_order / query_product / query_logistics（工具内随机数据）
│  ├─ knowledge.py     # query_faq（SQL LIKE 查 faq 表）
│  ├─ ticketing.py     # create_ticket（写 tickets 表）
│  └─ executor.py      # 超时、重试、错误分类、结果序列化
└─ chat_tools.py       # 单轮工具编排：模型定工具 → 执行 → 回灌 → 流式收敛
```

| 单元 | 做什么 | 对外接口 | 依赖 |
|---|---|---|---|
| `db/models.py` | 四张表的 ORM 映射，与 DDL 逐列对齐 | 四个 ORM 类 | SQLAlchemy |
| `db/session.py` | engine/sessionmaker，按请求给会话 | `get_db()`, `create_engine_from(settings)` | config |
| `db/repositories.py` | 把 SQL 收敛成业务语义方法 | `ConversationRepository` / `MessageRepository` / `FaqRepository` / `TicketRepository` | models |
| `tools/registry.py` | 名字 → 工具对象；启动自检「注册表 ⊆ 实现」 | `TOOL_REGISTRY`, `get_tools()`, `get_tool(name)` | 五个工具模块 |
| `tools/schemas.py` | 五个参数模型，字段描述直接喂给模型 | `QueryOrderArgs` 等 | pydantic |
| `tools/executor.py` | 参数校验 → 超时 → 重试 → 结构化结果 | `execute_tool(name, args) -> ToolResult` | registry, schemas |
| `chat_tools.py` | 一轮内完成「定工具 → 执行 → 回灌 → 流式」 | `astream_chat_with_tools(...)` | 全部 |

**隔离设计要点**：`tools/business.py` 的三个随机数据工具**不 import 数据库**，
所以它们能脱离 MySQL 单独测；`knowledge.py` / `ticketing.py` 通过仓库协议拿库，
测试用内存假仓库即可离线跑。这条边界保证「没有 MySQL 也能跑完单测」。

### 2.4 配置新增（`app/config.py` + `.env.example`）

```
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=mewhelp
MYSQL_PASSWORD=mewhelp
MYSQL_DATABASE=mewhelp
TOOL_TIMEOUT_SECONDS=3.0
TOOL_MAX_RETRIES=1
```

`Settings` 上提供只读派生属性 `database_url`，拼成
`mysql+pymysql://user:pass@host:port/db?charset=utf8mb4`。
**密码不进日志**：`database_url` 只用于建立连接，`healthz` 不回显它。

## 3. 工具层

### 3.1 五个工具

| 工具名 | 参数（Pydantic） | 实现 | 数据来源 |
|---|---|---|---|
| `query_order` | `order_id: str` | 工具内生成 | 随机（按 `order_id` 派生，固定种子 → 同号同结果） |
| `query_product` | `product_id: str` | 工具内生成 | 随机（同上） |
| `query_logistics` | `order_id: str` | 工具内生成 | 随机（同上，含运单号/承运商/轨迹节点/预计到达） |
| `query_faq` | `keyword: str` | `SELECT ... WHERE question LIKE %kw% OR answer LIKE %kw%` | `faq` 表 |
| `create_ticket` | `description: str`, `ticket_type: Literal['售后','投诉','咨询']`, `conversation_id: str` | `INSERT INTO tickets` | `tickets` 表 |

**随机数据为什么要「按入参派生」**：如果每次调用都 `random.random()`，
验收 1 里模型拿到的结果与我在浏览器看到的徽章结果无法对账，评估集也不可复现。
改用 `random.Random(f"{tool}:{id}")` 派生后，同一个订单号永远得到同一份数据 ——
既是「随机生成」，又可复现、可断言。

**工单号规则**：`T` + `%Y%m%d` + 3 位序号（当日已有工单数 +1），
与 DDL 注释里的 `T20260701008` 一致。并发下靠主键冲突兜底重试。

### 3.2 工具基础设施

**注册管理**：`TOOL_REGISTRY: dict[str, BaseTool]` 显式列出五个工具。
`registry.py` 的自检断言「注册表的键 == 每个工具的 `.name`」，
以及「每个工具的 `.args_schema` 是 Pydantic 模型」——防止改名后静默失联
（SSE 帧与落库都用字符串工具名，改名不报错反而最难查）。

**参数 Schema 校验**：校验由两层完成 —— 模型侧靠 `@tool(args_schema=...)`
把 JSON Schema 发给上游；我们侧靠 `args_schema.model_validate(tool_call['args'])`
在**执行之前**再校验一次。模型偶尔会漏参数或给错类型，让它在自己的工具调用上
翻车比让 `KeyError` 冒到 SSE 连接上要好。

**执行错误处理**：`execute_tool` **从不抛异常**，一律返回结构化结果：

```python
@dataclass(frozen=True)
class ToolResult:
    tool: str
    args: dict[str, Any]
    ok: bool
    data: Any | None       # 成功时的业务数据
    error: str | None      # 失败时的**已脱敏**原因
    attempts: int          # 实际尝试次数（供留痕/调试）
    duration_ms: int
```

失败也回灌给模型（`ToolMessage` 里是 `{"ok": false, "error": ...}`），
让模型能对用户说「订单系统暂时查不到」。**错误文本必须过 `app.errors.redact()`**
—— 数据库异常的消息里可能带连接串。

**超时重试**：`asyncio.wait_for(tool.ainvoke(args), timeout=TOOL_TIMEOUT_SECONDS)`；
仅对**瞬时错误**重试（`asyncio.TimeoutError`、`ConnectionError`、`OperationalError`），
参数校验失败与业务性「查不到」**不重试**（重试也是同样结果，白等 2 倍超时）。
重试总次数 `TOOL_MAX_RETRIES + 1`。超时与重试都可注入，测试用 `0.0` / 假时钟跑得快。

## 4. 聊天链路改造（单轮工具调用）

### 4.1 单轮编排精确语义

```
构建消息(history + System + 本轮)          ← 沿用 ch01 的 build_context
  ↓
model.bind_tools(五工具).ainvoke(messages)  ← 第 1 次模型调用（非流式：只为拿 tool_calls）
  ↓
有 tool_calls？
├─ 无 → 拿到的 content 就是最终答案，逐字流式吐出（本轮流无工具徽章）
└─ 有 → 只取 tool_calls[0]                  ← 单轮：多余的调用记录进日志但不执行
        yield status 帧 {tool, args}         ← 先推状态帧
        ToolResult = execute_tool(...)       ← 参数校验 + 超时 + 重试
        messages += [AIMessage(tool_calls), ToolMessage(result, tool_call_id)]
        model.ainvoke/astream(messages)      ← 第 2 次模型调用（流式）
        ├─ 有文本 → 逐 token yield token 帧
        └─ 空文本 → 模型还想调工具，但本章不循环
                    → yield token 帧 output_text（由工具结果渲染的兜底话术）
  ↓
落库 → yield done 帧
```

**为什么第 1 次模型调用不用流式**：工具调用块的增量合并是 LangChain 侧的行为，
把它和我们自己的编排混在一起，会让「工具到底选择了没有」变得难测。用一次非流式
调用换取**确定的 tool_calls**，第 2 次调用再用流式（用户要的逐 token 就在这一段）。
代价是首个 token 的延迟多一个 RTT —— 本章可接受，并在留痕里记明。

**「只调一次就收敛」的硬保证**：第 2 次调用后**不再检查 `tool_calls`**。
代码里没有循环结构，所以「模型想连调两次」在结构上不可能发生 ——
这比用 `max_iterations=1` 这种运行时刹车更明确。

### 4.2 SSE 契约扩展（向后兼容）

| 事件 | 时机 | data |
|---|---|---|
| `meta` | 连接建立 | `{session_id, model}`（沿用） |
| **`status`** | **工具执行之前** | `{tool, args, label}` —— `label` 是「正在查询物流…」这类中文短语 |
| `token` | 第 2 次调用逐字 | `{delta}`（沿用） |
| `done` | 结束 | `{finish_reason, session_id, trimmed, kept, tools: [{name, args, ok, duration_ms}]}` ← 新增 `tools` |
| `error` | 失败 | `{type, message, session_id}`（沿用） |

`tools` 数组是给前端徽章用的**权威来源**；`status` 帧只负责让用户立刻看到
「它在查数据」，不做持久化职责。两者都发，是因为徽章需要「本轮调了什么」的完整
结果（含成败），而 `status` 需要在结果产生**之前**就到达。

### 4.3 落库

每轮按顺序写：

| 顺序 | role | content | tool_calls | tool_call_id |
|---|---|---|---|---|
| 1 | `user` | 用户原话 | — | — |
| 2 | `assistant` | `""` | `[{name, args, id}]` | — |
| 3 | `tool` | 工具结果 JSON | — | 工具调用 id |
| 4 | `assistant` | 最终答案 | — | — |

`conversations` 行在该 `session_id` 首次出现时创建（`user_id` 暂固定
`"anonymous"`，用户体系不在本章范围）。

**失败语义**（沿用 ch01 的原则）：上游出错时只写第 1 行（用户那轮），
不写半截回答。工具执行失败**不算上游出错** —— 那是正常业务流程，四条都写，
第 3 行的 `content` 是 `{"ok": false, "error": ...}`。

**历史回读**：`GET /api/chat/sessions/{session_id}/messages` 改为从
`messages` 表按 `id` 升序回读并映射成 `{role, content, tools}`，
`tools` 字段来自 assistant 行的 `tool_calls` JSON —— 这是徽章**刷新后仍在**的
保证。响应体保持「消息列表」语义，只新增字段，不破坏 ch01 的调用方。

## 5. 前后端契约与聊天页（Vibe Coding 部分）

聊天页改造**不走评审流程**，但改造所依赖的后端契约必须先把上面 4.2 / 4.3 写死，
否则会出现「页面按某个形状解析、后端发另一个形状」的静默错位。

- 新增 `event: status` 处理：气泡上方显示一行「正在查询物流…」，收到 `token` 后隐去
- 新增 `done.tools` 处理：在气泡内渲染**工具轨迹小徽章**（如 `🔧 query_logistics · 成功 · 12ms`），
  并写入气泡的 `data-tools` 属性
- 历史回读渲染：用 `/messages` 返回的 `tools` 字段复用同一个徽章渲染函数

## 6. 错误处理汇总

| 场景 | 行为 |
|---|---|
| 模型不调工具 | 直接流式回答；无徽章（这是正常路径，不报错） |
| 模型调了不存在的工具名 | `ToolResult(ok=false, error="未知工具 ...")`，回灌让模型说明；徽章显示失败 |
| 工具参数缺字段/类型错 | 执行前 `model_validate` 拦下，同上；**不重试** |
| 工具超时 | 重试至 `TOOL_MAX_RETRIES` 次，仍失败则 `ok=false`；徽章显示 `超时` |
| 工具内部异常 | 捕获、`redact()`、`ok=false`；**绝不让异常冒到 SSE 连接** |
| MySQL 不可用 | `/healthz` 报 `database: "unavailable"` 且整体 503；聊天接口返回规范 503 外壳 |
| 上游（模型）故障 | 沿用 ch01：`error` 帧 / 502-503，只落库用户那轮 |

## 7. 测试策略

| 层 | 方式 | 钉住什么 |
|---|---|---|
| `tools/business.py` | TDD，纯离线 | 同入参同结果（可复现）、字段齐全、无数据库依赖 |
| `tools/executor.py` | TDD，假工具 + 假仓库 | 参数校验拦截、超时触发、仅瞬时错误重试、结果结构化、错误脱敏 |
| `tools/registry.py` | TDD | 注册表与工具 `.name` 一致、五个工具都在 |
| `db/repositories.py` | TDD，**SQLite 内存库**（ENUM→VARCHAR 的等价映射） | 会话 upsert 幂等、消息顺序、FAQ LIKE 命中与不命中、工单号格式 |
| `chat_tools.py` | TDD，假模型 + 假仓库 | 单轮收敛（第 2 次调用后不再执行工具）、status 帧顺序、落库四行、失败只落一行 |
| `api.py` | 扩展 ch01 的 HTTP 测试 | SSE 新事件契约、`done.tools`、503 外壳 |
| **真实 MySQL** | 一条集成测试（默认 skip，`MYSQL_HOST` 可达时跑） | DDL 与 ORM 映射逐列对齐 —— 防止「单测全绿、真库报 Unknown column」 |
| **评估集** | `evals/tool_cases.jsonl` + `evals/run_tool_evals.py` | 三条验收标准逐条跑**真实上游**；验收 3 断言「查不到」 |
| `web/` | Vibe Coding，浏览器实测 | 徽章显示、刷新后仍在 |

**评估集为什么必须存在**：三条验收标准全是「模型自己决定调什么工具」，
这是**模型的判断行为**，不是代码逻辑 —— 无法用断言替代。所以要拿标注样例跑
真实上游，记录命中率。评估集按 ch01 的既定规则：**未达标先改 Prompt / 工具
描述，不许改评估标准**。

## 8. 交付物

| 交付物 | 路径 |
|---|---|
| 建表 DDL | `sql/ch02-ddl.sql`（+`session_id`） |
| 测试数据 | `sql/ch02-seed.sql` |
| 容器编排 | `docker-compose.yml` |
| 后端 | `MemHelp-code/app/{db,tools}/**`, `app/chat_tools.py` |
| 前端 | `MemHelp-code/web/{app.js,style.css}`（Vibe Coding） |
| 评估集 | `MemHelp-code/evals/tool_cases.jsonl`, `evals/run_tool_evals.py` |
| 开发留痕 | `MemHelp-code/dev-notes/ch02.md` |
| 演示脚本 | `MemHelp-code/scripts/demo-ch02.ps1` |

## 9. 风险与已知限制

| 风险 | 应对 |
|---|---|
| 沙箱打不开 Docker 的命名管道（已实测 `Access is denied`） | **根因是沙箱策略**（`workspace-write` 不允许进程打开命名管道），不是 daemon 未启动。起 MySQL 那步用一次 `danger-full-access` 提权重试同一命令，并请用户批准；留痕记明 |
| `docker compose down -v` 才能让 initdb 脚本重跑 | 演示脚本里显式提供 `-Reset` 开关并在留痕写明 |
| 第 1 次非流式调用让首 token 变慢 | 明确记入留痕，作为 ch03 可优化项 |
| 模型可能对「邮费是多少」强行调 `query_faq` 并收到 0 条 | 这是验收 3 的预期形态：徽章显示 `query_faq · 成功`，回答如实说没查到 —— 漏召回表现在**数据层**而非**工具选择层**，留痕里要区分清楚 |
| 随机数据可复现依赖 `order_id` 字符串 | 评估集固定用 `1001`，与验收标准一致 |
