# 第八章设计 · 即插即用工具系统（注册中心 / 校验 / 权限 / 执行引擎 / 审计 / MCP）

- 日期：2026-10-24
- 状态：定稿（用户确认三项选择：每轮刷新 MCP 清单、建单成功用工具 message 作回复、整体方案照此推进）
- 关联：`dev-notes/ch08.md`、`docs/superpowers/plans/2026-10-24-ch08-tool-ecosystem-plan.md`、`sql/ch08-ddl.sql`

---

## 1. 目标与非目标

### 1.1 目标

1. **一个注册中心**：内置工具与 MCP 工具统一登记为 `ToolSpec`（名字 / 用途描述 / JSON Schema 参数 / 来源 / 只读标记 / 执行体）；内置在启动时登记，MCP 在**每轮对话前**动态发现；新工具进表即可被主力 Agent 绑定，不改核心代码、不重启服务。
2. **统一参数校验**：执行前按 JSON Schema 校验；类型错、必填缺失、越界一律拦下，把中文错误说明包成 `ToolResult`（`status=校验拦下`）回灌模型，不抛异常。
3. **权限闸**：工具分只读/写两类；写操作（本章只有 `create_ticket`）必须携带用户确认；模型侧调用默认 `confirmed=False` → 引擎直接拒绝并落审计；外部 MCP 工具一律按**我方**规则判为只读，忽略 Server 的 annotations。
4. **统一执行引擎**：一处管超时、重试、错误分诊、结果格式化；暂时性故障才重试、写操作永不自动重试；结果只留回答用得上的字段、内部枚举码翻人话、`ensure_ascii=False`。
5. **审计**：`tool_audit_logs` 每次调用一条，含校验拦下/权限拒绝；写审计失败只告警，绝不反拦执行。
6. **两个业务 MCP Server**：物流、售后，各自独立进程、Streamable HTTP、进程内随机 mock；客服侧用 `MultiServerMCPClient` 接入，与内置清单合成一份；内置 `query_logistics` 下线。
7. **建工单确认流**：`create_ticket` 进 Agent 工具集，但 ReAct 循环只**收集**调用、不执行；图节点 `ticket_await_confirm` 用 LangGraph `interrupt()` 推预览，前端卡片「确认提交 / 取消」；确认 → 落库并把工单号写进回复；取消 → 不执行、审计记权限拒绝。ch05 的 `/api/actions/create-ticket` 按钮路径保持不动（按钮即确认）。

### 1.2 明确不做

- Skill 机制（生态概念，不落代码）。
- 仓储等更多外部系统；只接两个业务 MCP Server。
- MCP 工具的写操作；本章外部工具全部只读。
- 审计表不挂外键、不做归档/分区。

---

## 2. 关键决策

| # | 决策 | 选择 | 理由 / 代价 |
|---|---|---|---|
| D1 | 注册中心形态 | `ToolRegistry` **每轮构建**（内置闭包要捕获当轮 `repositories`）；MCP 发现结果由进程级 `McpToolProvider` 缓存；`ToolSpec` 不可变 | 一表统管来源/权限/校验/格式化；避免进程级单例抓住请求级 DB Session（ch02 的闭包工厂就是这个约束） |
| D2 | MCP 刷新时机 | **每轮 `agent` 节点前 `refresh_mcp()`**，`asyncio.Lock` 串行 | 验收 3 要求「只重启 Server、下一句就能用」，TTL 缓存会留空窗；代价是每轮两次 `list_tools`（本机 HTTP，毫秒级） |
| D3 | 校验库 | `jsonschema`（新增显式依赖） | MCP 的 `inputSchema` 本来就是 JSON Schema；内置用 Pydantic 模型 `model_json_schema()` 导出，两边同一套校验 |
| D4 | 权限判定 | 只认 `ToolSpec.read_only` + `confirmed` 标志 | Server 的 `annotations` 只存进 `metadata` 参考，**不参与判定**；模型无法伪造 `confirmed`（由节点/路由传） |
| D5 | 写操作拦截点 | ReAct 循环拦截 `create_ticket`，不交给 executor | 需要图节点调 `interrupt()`；executor 里没有图上下文 |
| D6 | 缺参处理 | 拦截时先按 schema 校验；不合法 → 回灌「参数不合法」ToolMessage，让模型追问 | 满足「缺问题描述先追问、不许瞎编」，且不产生空预览卡片 |
| D7 | 建单成功回复 | 直接用工具返回的 `message` 字段 | 确定性强、不会让模型改错工单号；省一次 LLM 调用 |
| D8 | 审计写入位置 | executor `finally` 统一写；节点取消路径单独补一条 | 成功/失败/超时/校验/权限全在一处；取消发生在 executor 之外，必须显式补 |
| D9 | 审计失败 | `try/except` + `logger.warning`，结果照常返回 | 需求 5 明确「写审计失败不许反拦工具执行」 |
| D10 | 重试边界 | 仅 `read_only=True` 且 `is_transient` 才重试；写操作 `max_retries` 强制 0 | 写超时可能已执行，重复执行比失败更糟 |
| D11 | MCP 名冲突 | 与内置同名 → 丢弃 MCP 那条并 `logger.warning`，内置优先 | 静默覆盖会让「下线 query_logistics」变成隐式行为；冲突必须显形 |
| D12 | 内置 `query_logistics` | 从 `schemas.py` / `registry.py` / `TOOL_NAMES` / `TOOL_LABELS` 全量移除 | 需求 6「别留重名工具」；物流查询由 MCP 接管 |

---

## 3. 架构

```
                    ┌──────────────────────────────┐
                    │  ToolRegistry（每轮一个）     │
                    │  name → ToolSpec(不可变)      │
                    │  builtin: 构造时 register     │
                    │  mcp: refresh_mcp() 从进程级  │
                    │       McpToolProvider 取快照  │
                    └──────────────┬───────────────┘
                                   │ snapshot()
        ┌──────────────────────────┼───────────────────────────┐
        │                          │                           │
┌───────▼────────┐      ┌──────────▼──────────┐      ┌─────────▼─────────┐
│ agent_tools()  │      │  execute_tool()     │      │  /api/actions/*   │
│ bind_tools     │      │ 校验→权限→超时/重试  │      │  ch05 按钮路径     │
│ (ReAct 绑定)    │      │ →格式化→审计         │      │  confirmed=True   │
└───────┬────────┘      └──────────┬──────────┘      └─────────┬─────────┘
        │                          │                           │
        │ create_ticket 被拦截      │ ToolResult                │
        ▼                          ▼                           ▼
┌───────────────────┐      ┌──────────────┐            ┌──────────────┐
│ ticket_await_     │      │ tool_audit_  │            │ tickets 表    │
│ confirm (interrupt)│      │ logs         │            └──────────────┘
└─────────┬─────────┘      └──────────────┘
          │ resume {confirmed}
          ▼
   确认 → execute_tool(confirmed=True) → 回复带工单号
   取消 → 审计「权限拒绝」 → 回复已取消
```

### 3.1 模块划分（新增/改动）

| 文件 | 动作 | 职责 |
|---|---|---|
| `app/tools/contract.py` | 新增 | `ToolSpec` / `ToolSource` / `ToolStatus` 常量 |
| `app/tools/registry.py` | 重写 | `ToolRegistry`（每轮构建；register/refresh/snapshot/get），内置工具登记；保留 `TOOL_NAMES` 兼容导出 |
| `app/tools/builtin.py` | 新增 | 五个内置工具的 `@tool` 工厂（从 registry 迁出，去掉 `query_logistics`） |
| `app/tools/mcp_client.py` | 新增 | `McpToolProvider`（进程级）：`MultiServerMCPClient` + `discover()`，把 adapter 的 `StructuredTool` 转 `ToolSpec` |
| `app/tools/validation.py` | 新增 | `validate_against_schema(schema, args) -> str | None`，jsonschema 错误压成中文「字段: 原因」 |
| `app/tools/permissions.py` | 新增 | `PermissionPolicy.check(spec, confirmed) -> str | None`（写且未确认 → 拒绝原因） |
| `app/tools/presenter.py` | 新增 | 字段白名单 + 枚举翻译（订单状态、物流状态、退货状态、工单状态） |
| `app/tools/audit.py` | 新增 | `AuditEntry` + `write_audit()`，吞异常 |
| `app/tools/executor.py` | 重构 | 接 `ToolRegistry`/`ToolSpec`；新增 `status/source/mcp_server/tool_call_id`；保持 `ToolResult` 旧字段与 `summary()` 兼容 |
| `app/tools/schemas.py` | 改动 | 删 `QueryLogisticsArgs`；其余保留 |
| `app/db/models.py` | 改动 | 新增 `ToolAuditLog`（对应 `ch08-ddl.sql`） |
| `app/db/repositories.py` | 改动 | `ToolAuditRepository.create(...)` |
| `app/config.py` | 改动 | `mcp_enabled`、`mcp_logistics_url`、`mcp_aftersales_url`、`mcp_readonly_servers`、`mcp_refresh_timeout_seconds` |
| `app/agent/react.py` | 改动 | 工具来自 registry 快照；拦截 `create_ticket`；返回 `pending_ticket_call` |
| `app/agent/graph.py` | 改动 | `agent` 前刷新 MCP；`ticket_await_confirm` 节点与条件边 |
| `app/agent/state.py` | 改动 | `pending_ticket_call`、`ticket_preview` |
| `app/agent/runner.py` | 改动 | `awaiting_input` 支持 `kind=ticket_preview`；通用 resume 值 |
| `app/api.py` | 改动 | 启动注册内置 + 首次发现 MCP；`POST /api/chat/ticket/confirm`；`create-ticket` 传 `confirmed=True` |
| `mcp_servers/logistics_server.py` | 新增 | 物流 MCP（track_shipment） |
| `mcp_servers/aftersales_server.py` | 新增 | 售后 MCP（check_warranty / check_return_progress） |
| `web/app.js` / `style.css` | 改动 | 工单预览卡片 + 确认/取消 |
| `scripts/run_mcp_servers.py` / `scripts/demo_ch08.py` | 新增 | 一键起两个 Server；演示命令 |
| `sql/ch08-ddl.sql` | 已存在 | 不改 |

---

## 4. 数据与契约

### 4.1 ToolSpec

```python
@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str                      # 模型看到的工具名
    description: str               # 用途描述（内置手写；MCP 用 Server 的 description）
    args_schema: dict[str, Any]    # JSON Schema（object）
    source: Literal["builtin", "mcp"]
    mcp_server: str | None
    read_only: bool
    tool: BaseTool                 # LangChain 可绑定工具
    presenter: Callable[[Any], Any] | None = None
```

- `ToolRegistry.snapshot() -> tuple[ToolSpec, ...]`（按名字排序，稳定绑定顺序）。
- `ToolRegistry.register(spec)`：重名抛 `ValueError`（内置之间）；`refresh_mcp()` 遇到与内置重名 → 跳过 + warning。
- `refresh_mcp()` 用 `asyncio.Lock`；失败（Server 未起）只 warning，保留上一轮 MCP 快照，**绝不清空内置**。

### 4.2 ToolResult（兼容扩展）

沿用现有字段，新增：

```python
status: Literal["成功", "失败", "超时", "校验拦下", "权限拒绝"]
source: Literal["builtin", "mcp"] = "builtin"
mcp_server: str | None = None
tool_call_id: str | None = None
```

`summary()` 继续返回旧键（name/args/ok/duration_ms/attempts/started_at/result/error），供 SSE/前端不回退。

### 4.3 审计

- `conversation_id`：调用方能给就给（`create_ticket` 参数里有）；否则 `repositories.conversations.find_id(session_id)` 尽力解析，失败置 `NULL`。
- `result_summary`：presenter 后的 JSON 截断 500 字符；错误走 `redact()`。
- `duration_ms`：`int((time.monotonic() - started) * 1000)`。
- 取消建单的审计：`status=权限拒绝`，`error_message="用户取消建单，未执行"`，`retry_count=0`，`duration_ms` 为预览等待耗时的近似值（节点内计时）。

---

## 5. 执行引擎流程

```
execute_tool(name, args, *, registry, repositories, retriever,
             session_id=None, conversation_id=None, tool_call_id=None,
             timeout, max_retries, confirmed=False)
  1. registry.get(name)  → 无 → status=失败, error="未知工具：{name}"
  2. validation.validate_against_schema(spec.args_schema, args)
        → 有错 → status=校验拦下, error="参数不合法 —— {字段}: {原因}"（attempts=0）
  3. permissions.check(spec, confirmed)
        → 拒 → status=权限拒绝（写操作未确认）
  4. attempts=0; while:
        attempts += 1
        await asyncio.wait_for(spec.tool.ainvoke(args), timeout)
        - 成功 → status=成功, data=present(raw)
        - TimeoutError → status=超时；仅 read_only 且 attempts<=max_retries 才重试
        - 业务空结果（ok 但空）→ 不重试
        - 其他异常 → is_transient 且 read_only 才重试；否则 status=失败
  5. write_audit(...)  # 任何分支都写；异常吞掉
  6. return ToolResult
```

**错误分诊文案**（回灌模型）：

| 类别 | status | error 文本 |
|---|---|---|
| 参数不合法 | 校验拦下 | `参数不合法 —— description: 不能为空`（多条用「；」连接） |
| 查询落空 | 成功（`data` 为空 + `empty=True`） | `没有查到相关记录`，提示模型换条件或如实告知 |
| 真故障 | 失败 / 超时 | `工具 X 执行失败：Type: msg` / `工具 X 执行超时（超过 N 秒）` |

**结果格式化**：`presenter` 输出只含回答字段；枚举翻译表：

- 订单 `PAID/SHIPPED/DELIVERED/...` → 待发货/已发货/已签收
- 物流 `CREATED/IN_TRANSIT/OUT_FOR_DELIVERY/DELIVERED/EXCEPTION` → 已揽收/运输中/派送中/已签收/异常
- 退货 `APPLIED/REVIEWING/APPROVED/REFUNDING/REFUNDED/REJECTED` → 已申请/审核中/已通过/退款中/已退款/已驳回
- 工单 `待处理/处理中/已解决` 原样
- `json.dumps(..., ensure_ascii=False)`

---

## 6. MCP 接入

### 6.1 Server 侧（`mcp_servers/`）

- `FastMCP("memhelp-logistics", host="127.0.0.1", port=8101)`，`mcp.run(transport="streamable-http")`。
- 物流工具：
  - `track_shipment(order_id: str)` → `{order_id, carrier, tracking_no, status, status_text, nodes:[{time, location, description}], eta}`
- 售后工具：
  - `check_warranty(order_id: str)` → `{order_id, product_name, in_warranty, warranty_until, policy}`
  - `check_return_progress(order_id: str)` → `{order_id, return_no, status, status_text, steps:[...], refund_amount_yuan}`
- Mock：`random.Random(hash(order_id) & 0xFFFFFFFF)` 保证同一订单多次查询一致；不建表、不连外部系统。
- 依赖钉版本：`mcp>=1.24,<2`（`langchain-mcp-adapters 0.3.2` 要求 `<2.0`），`langchain-mcp-adapters>=0.3.2,<0.4`，`jsonschema>=4.21`。

### 6.2 Client 侧（`app/tools/mcp_client.py`）

```python
MultiServerMCPClient({
  "logistics": {"transport": "streamable_http", "url": settings.mcp_logistics_url, "timeout": ...},
  "aftersales": {"transport": "streamable_http", "url": settings.mcp_aftersales_url, "timeout": ...},
})
```

- `await client.get_tools(server_name=...)` 逐 Server 拉，便于标注 `mcp_server` 与失败隔离。
- `StructuredTool.args_schema` 可能是 dict（MCP inputSchema）或 Pydantic 模型 → 统一转 dict。
- `metadata`（含 annotations）只保留给日志，不参与权限。
- 结果 `content_and_artifact`：取 `content`，能 `json.loads` 就转 dict，否则原样字符串。
- `mcp_enabled=False` 时 `refresh_mcp()` 直接清空 MCP 部分（测试/离线用）。

---

## 7. 建工单确认流

### 7.1 图

```
agent ──(pending_ticket_call 非空)──► ticket_await_confirm ──► log_turn ──► END
  └────(空)──────────────────────────────────────────────────► log_turn
```

### 7.2 ReAct 拦截（`react.py`）

- `create_ticket` 出现在 `tools_by_name` 中（ch05 的剔除规则对本章失效，改成「拦截而非剔除」）。
- 模型发出 `create_ticket` 调用时：
  1. `validate_against_schema(spec.args_schema, args)`；不合法 → 追加 `ToolMessage`（`status=校验拦下` 文案）+ 继续循环，让模型追问用户。
  2. 合法 → 不执行，记录 `pending_ticket_call={args, call_id}`，`stop_reason="ticket_pending"`，跳出循环。
- 不产生 `tool_result` 事件（还没执行）。

### 7.3 节点 `ticket_await_confirm`（graph.py）

```python
pending = state["pending_ticket_call"]
payload = {"kind": "ticket_preview", "turn_id": state["turn_id"],
           "call_id": pending["call_id"],
           "preview": {"ticket_type": args["ticket_type"], "description": args["description"]}}
decision = interrupt(payload)          # 恢复后节点从头重跑，preview 构造是纯函数
if not decision.get("confirmed"):
    write_audit(status="权限拒绝", error="用户取消建单，未执行", ...)
    return {"answer": "已取消建单，没有创建工单。", ...}
result = await execute_tool("create_ticket", args, registry=..., confirmed=True, ...)
return {"answer": result.data["message"], "tool_records": [...], "pending_messages": [...]}
```

- `pending_messages` 追加 `AIMessage(tool_calls=[...])` + `ToolMessage`，保证历史回读看到「调用过 create_ticket」。
- 工单号来自 `result.data["message"]`（`ticketing.py` 已含 `ticket_no`）。

### 7.4 API / 前端

- `POST /api/chat/ticket/confirm`：body `{session_id, confirmed: bool}`；校验 `snapshot.next` 含 `ticket_await_confirm` 且 `pending_interrupt` 的 `kind == "ticket_preview"`，否则 409 `no_pending_ticket`；占用 `_TICKET_RESUME_INFLIGHT` 防并发；`Command(resume={"confirmed": confirmed})` 走 SSE。
- 前端 `applyFrame` 新增 `ticket_preview`：渲染 `.ticket-preview` 卡片（标题「工单预览」、类型、描述、「确认提交」「取消」）；点击后禁用按钮 → POST confirm → 消费 SSE；`awaiting_input` 的 `kind=ticket_preview` 用于刷新后恢复卡片。
- 视觉延续 ch06 订单选择器（同色系、同圆角、按钮禁用态）。

---

## 8. 配置

| 键 | 默认 | 说明 |
|---|---|---|
| `mcp_enabled` | `True` | 关掉则只用内置工具（测试默认 False） |
| `mcp_logistics_url` | `http://127.0.0.1:8101/mcp` | 物流 Server |
| `mcp_aftersales_url` | `http://127.0.0.1:8102/mcp` | 售后 Server |
| `mcp_readonly_servers` | `logistics,aftersales` | **我方**只读白名单；不在此列的 Server 工具一律拒绝注册 |
| `mcp_timeout_seconds` | `3.0` | 单次 MCP 调用超时（与工具超时取小） |
| `tool_timeout_seconds` | `3.0`（沿用） | 执行引擎超时 |
| `tool_max_retries` | `1`（沿用） | 只读工具重试次数 |

---

## 9. 验收映射

| 验收 | 实现 | 自动化/演示 |
|---|---|---|
| 1 新工具只注册就能用 | `ToolRegistry.register(ToolSpec(...))` + `agent_tools` 读快照 | 单测：注册一个 `echo` 工具 → `agent_tools` 含它；演示脚本 |
| 2 两个 MCP Server 独立进程 | `scripts/run_mcp_servers.py` 起 8101/8102 | `scripts/demo_ch08.py` 问物流/在保 |
| 3 加工具只重启 Server | `refresh_mcp()` 每轮 | 演示：Server 加 `query_carrier` → 重启 → 同一客服进程再问 |
| 4 缺参追问 + 确认建单 | 拦截校验 + `ticket_await_confirm` | HTTP 测试：预览帧 → confirm → tickets 落库 + 工单号 |
| 5 取消落权限拒绝 | 节点 cancel 分支审计 | HTTP 测试查 `tool_audit_logs.status` |
| 6 超时重试/写不重试 | executor 重试策略 | 单测：注入 sleep 工具，读审计 `retry_count/duration_ms/status` |

---

## 10. 测试策略（TDD）

- 每个任务先写失败测试，再实现。
- 纯 mock 测试不启动进程：registry / validation / permissions / presenter / audit / executor。
- 图与 HTTP：`create_app` 注入 `fake_repositories`（SQLite）+ `InMemorySaver` + 假 registry（含假的 `create_ticket`）。
- MCP Server 单测：直接调工具函数（不经过 HTTP）；另加一个可选集成测试 `tests/test_mcp_live.py`，标记 `skip` 除非 `MCP_LIVE=1`。
- 前端：手工验收 + 截图（Vibe Coding，不写前端单测）。

---

## 11. 风险与回退

| 风险 | 处置 |
|---|---|
| `mcp` 2.x 与 adapter 不兼容 | 依赖钉 `<2`；`uv.lock` 锁定 |
| 每轮刷新拖慢首 token | 只对已连接的 Server 拉清单；单 Server 超时 0.5s，失败沿用上一轮快照 |
| interrupt 节点重跑导致重复审计 | 取消审计写在 `interrupt()` 之后；preview 构造无副作用 |
| 模型绕过确认 | executor 的 `confirmed` 只能由路由/节点传；写操作未确认直接拒绝并审计 |
| 审计写库失败拖垮工具 | 全程 try/except，仅 warning |
| 工单号被模型改写 | 成功回复直接用工具 `message`，不经模型 |
