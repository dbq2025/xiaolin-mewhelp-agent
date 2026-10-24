# 第八章 即插即用工具系统 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把写死的 5 个内置工具升级为「注册中心 + JSON Schema 校验 + 读写权限闸 + 统一执行/审计 + 两个 MCP Server + 建工单前端确认」的即插即用工具系统。

**Architecture:** 每轮构建一个 `ToolRegistry`（内置工具闭包捕获当轮 `repositories`），MCP 发现结果由进程级 `McpToolProvider` 缓存、每轮 `agent` 节点前刷新；所有调用统一走 `execute_tool`（校验 → 权限 → 超时/重试 → 呈现 → 审计）；`create_ticket` 在 ReAct 循环里只收集不执行，由图节点 `ticket_await_confirm` 调 `interrupt()` 等前端确认。

**Tech Stack:** Python 3.12、FastAPI、SQLAlchemy、LangGraph 1.2、LangChain 1.x、`mcp` 1.x（FastMCP，Streamable HTTP）、`langchain-mcp-adapters` 0.3.x、`jsonschema`、原生前端（web/app.js）。

**Spec:** `docs/superpowers/specs/2026-10-24-ch08-tool-ecosystem-design.md`

## Global Constraints

- 依赖钉版本：`mcp>=1.24,<2.0`、`langchain-mcp-adapters>=0.3.2,<0.4`、`jsonschema>=4.21`。
- 只有 `app/config.py` 读配置；只有 `app/llm.py` import `langchain_openai`；错误脱敏只用 `app/errors.py::redact`。
- `execute_tool` 永不抛异常；审计写失败只 `logger.warning`，不反拦执行。
- 写操作（`read_only=False`）永不自动重试；只有 `is_transient` 且只读才重试。
- 审计状态枚举固定为：`成功/失败/超时/校验拦下/权限拒绝`（与 `sql/ch08-ddl.sql` 一致）。
- 所有 JSON 序列化 `ensure_ascii=False`。
- 测试不用 `tmp_path`；pytest 已配 `-p no:cacheprovider`。
- 运行命令统一先设：`$env:UV_CACHE_DIR='F:\DeepSeek-Harness\memhelp-code-myself\.uv-cache'`，在 `MemHelp-code/` 下执行 `& C:\Users\dbq\.local\bin\uv.exe run pytest ...`。
- 每个阶段完成立刻往 `dev-notes/ch08.md` 追记四样，不许收尾补记。

## Review Focus

- MCP Server 未启动/半启动：聊天必须继续可用（内置工具照常），不能 500。
- MCP 返回非 JSON 文本 / 空 content / 超大结果：呈现器不能崩，模型要能读到可判读的说明。
- 模型伪造 `create_ticket` 参数（缺 description、ticket_type 非法）：不能出预览卡片，必须先回灌校验错误让它追问。
- 用户刷新页面后确认卡片消失但图仍挂起：`awaiting_input` 必须能按 `kind=ticket_preview` 恢复卡片。
- 审计写库抛异常（表不存在/连接断）：工具结果照常返回，只落 warning。

---

### Task 1: 依赖、ToolSpec 契约、配置项

**Files:**
- Modify: `pyproject.toml`
- Create: `app/tools/contract.py`
- Modify: `app/config.py`
- Test: `tests/test_tool_contract.py`

**Interfaces:**
- Produces:
  - `ToolSource = Literal["builtin", "mcp"]`；`ToolStatus = Literal["成功","失败","超时","校验拦下","权限拒绝"]`
  - `@dataclass(frozen=True, slots=True) ToolSpec(name: str, description: str, args_schema: dict[str, Any], source: ToolSource, mcp_server: str | None, read_only: bool, tool: BaseTool, presenter: Callable[[Any], Any] | None = None)`
  - `Settings.mcp_enabled: bool = True`、`mcp_logistics_url: str = "http://127.0.0.1:8101/mcp"`、`mcp_aftersales_url: str = "http://127.0.0.1:8102/mcp"`、`mcp_readonly_servers: str = "logistics,aftersales"`、`mcp_timeout_seconds: float = 3.0`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_contract.py
import dataclasses
import pytest
from langchain_core.tools import tool
from app.tools.contract import ToolSpec

def test_spec_is_frozen_and_keeps_schema():
    @tool
    def echo(text: str) -> str:
        """回显。"""
        return text
    spec = ToolSpec(name="echo", description="回显", args_schema={"type": "object"},
                    source="builtin", mcp_server=None, read_only=True, tool=echo)
    assert spec.args_schema["type"] == "object"
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]

def test_settings_mcp_defaults(test_settings):
    assert test_settings.mcp_enabled is True
    assert test_settings.mcp_logistics_url.endswith(":8101/mcp")
    assert test_settings.mcp_aftersales_url.endswith(":8102/mcp")
    assert "logistics" in test_settings.mcp_readonly_servers
```

- [ ] **Step 2: 跑测试确认失败**（`ModuleNotFoundError: app.tools.contract`）
- [ ] **Step 3: `uv add "mcp>=1.24,<2.0" "langchain-mcp-adapters>=0.3.2,<0.4" "jsonschema>=4.21"`，实现 `contract.py` 与 `config.py` 字段**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add tool spec contract and MCP settings`

---

### Task 2: JSON Schema 校验层

**Files:**
- Create: `app/tools/validation.py`
- Test: `tests/test_tool_validation.py`

**Interfaces:**
- Produces: `validate_against_schema(schema: dict[str, Any], args: Mapping[str, Any]) -> str | None`（`None` 表示通过；错误文本形如 `"description: 'description' 是必填项"`，多条用 `"；"` 连接）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_validation.py
from app.tools.validation import validate_against_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string", "minLength": 1},
        "ticket_type": {"type": "string", "enum": ["售后", "投诉", "咨询"]},
        "count": {"type": "integer", "minimum": 1, "maximum": 9},
    },
    "required": ["description", "ticket_type"],
    "additionalProperties": False,
}

def test_valid_args_pass():
    assert validate_against_schema(SCHEMA, {"description": "x", "ticket_type": "售后"}) is None

def test_missing_required_is_chinese_and_named():
    err = validate_against_schema(SCHEMA, {"ticket_type": "售后"})
    assert "description" in err and "必填" in err

def test_enum_and_range_errors():
    err = validate_against_schema(SCHEMA, {"description": "x", "ticket_type": "退货", "count": 0})
    assert "ticket_type" in err and "售后" in err
    assert "count" in err

def test_additional_property_rejected():
    err = validate_against_schema(SCHEMA, {"description": "x", "ticket_type": "售后", "oops": 1})
    assert "oops" in err
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**：`Draft202012Validator` 逐条 `iter_errors`；路径 `"/properties/..."` → 取最后一段；`required` 错误从 `error.message` 提取字段名；`enum` 把候选值拼进中文文案；任何校验器自身异常也转成 `"参数不合法 —— {exc}"`，绝不抛。
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add JSON Schema validation with Chinese errors`

---

### Task 3: 内置工具迁移 + ToolRegistry（删除 query_logistics）

**Files:**
- Create: `app/tools/builtin.py`
- Rewrite: `app/tools/registry.py`
- Modify: `app/tools/schemas.py`（删 `QueryLogisticsArgs`）
- Modify: `app/agent/tools.py`（`agent_tools` 接 `ToolRegistry`）
- Test: `tests/test_tool_registry.py`；更新引用 `query_logistics` 的既有测试

**Interfaces:**
- `builtin.py`：`build_builtin_tools(repositories, retriever=None) -> list[ToolSpec]`，返回 `query_order / query_product / query_faq / create_ticket` 四个（`query_logistics` 删除）。
- `registry.py`：
  - `ToolRegistry(repositories, retriever=None, *, mcp_provider=None)`
  - `register(spec) -> None`（重名 `ValueError`）
  - `get(name) -> ToolSpec | None`
  - `snapshot() -> tuple[ToolSpec, ...]`（按 name 排序）
  - `async refresh_mcp() -> None`
  - `build_registry(repositories, retriever=None, *, mcp_provider=None, enable_mcp=False) -> ToolRegistry`（`enable_mcp=True` 才在构造后把 provider 缓存合并进来）
  - 保留 `TOOL_NAMES: tuple[str, ...]`、`TOOL_LABELS: dict[str,str]`（不含物流）
- `agent_tools(registry, *, include_faq, include_order=True, record_actions) -> list[BaseTool]`：`create_ticket` **保留在列表中**（由 ReAct 拦截），`suggest_actions` 仍追加。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_tool_registry.py
import pytest
from app.tools.registry import ToolRegistry, build_registry, TOOL_NAMES

def test_registry_has_no_logistics_and_keeps_four(fake_repositories):
    reg = build_registry(fake_repositories, None)
    names = [s.name for s in reg.snapshot()]
    assert "query_logistics" not in names
    assert set(names) == {"query_order", "query_product", "query_faq", "create_ticket"}
    assert "query_logistics" not in TOOL_NAMES

def test_duplicate_builtin_name_rejected(fake_repositories):
    reg = build_registry(fake_repositories, None)
    spec = reg.get("query_order")
    with pytest.raises(ValueError):
        reg.register(spec)

def test_snapshot_is_stable_sorted(fake_repositories):
    names = [s.name for s in build_registry(fake_repositories, None).snapshot()]
    assert names == sorted(names)

def test_agent_tools_keeps_create_ticket_for_gated_flow(fake_repositories):
    from app.agent.tools import agent_tools
    names = {t.name for t in agent_tools(build_registry(fake_repositories, None),
                                         include_faq=True, record_actions=lambda *a, **k: None)}
    assert "create_ticket" in names and "query_logistics" not in names
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 迁移闭包工厂到 `builtin.py`；`ToolSpec.args_schema = Args.model_json_schema()`；`read_only = name != "create_ticket"`；presenter 暂为 `None`（Task 5 补）**
- [ ] **Step 4: 跑测试 + `tests/test_agent_tools.py`、`tests/test_tools*.py`，修掉物流下线导致的失败**
- [ ] **Step 5: Commit** `feat(ch08): migrate builtin tools into registry, drop query_logistics`

---

### Task 4: 权限策略

**Files:**
- Create: `app/tools/permissions.py`
- Test: `tests/test_tool_permissions.py`

**Interfaces:**
- `check_permission(spec: ToolSpec, *, confirmed: bool) -> str | None`
  - `spec.read_only=True` → `None`
  - 写且 `confirmed=False` → `"写操作 create_ticket 需要用户确认后才能执行"`
  - 写且 `confirmed=True` → `None`

- [ ] **Step 1: 写失败测试**（只读放行；写未确认拒绝且文案含工具名；写已确认放行）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现（纯函数，不读配置、不看 MCP annotations）**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add read/write permission policy`

---

### Task 5: 结果呈现与枚举翻译

**Files:**
- Create: `app/tools/presenter.py`
- Test: `tests/test_tool_presenter.py`

**Interfaces:**
- `ORDER_STATUS_TEXT: dict[str,str]`、`LOGISTICS_STATUS_TEXT`、`RETURN_STATUS_TEXT`
- `present_result(spec: ToolSpec, raw: Any) -> Any`：能 `json.loads` 的字符串转 dict；`dict` 去掉 `None` 与 `_` 前缀键；按 `spec.name` 选用字段白名单；枚举值翻译。
- `summarize_result(spec: ToolSpec, data: Any, *, limit: int = 500) -> str`

白名单：`query_order → {order_id,status,status_text,amount_yuan,product_name,created_at}`；`track_shipment → {order_id,carrier,tracking_no,status,status_text,nodes,eta}`；`check_warranty → {order_id,product_name,in_warranty,warranty_until,policy}`；`check_return_progress → {order_id,return_no,status,status_text,steps,refund_amount_yuan}`；未列出的工具保留全部键。

- [ ] **Step 1: 写失败测试**（枚举码翻译；未知键被剔除；`ensure_ascii=False` 的中文原文；非 JSON 字符串原样返回）
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现；`present_result` 永不抛（异常返回 `raw`）**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add result presenter with enum translation`

---

### Task 6: 审计表模型、仓库、写入器

**Files:**
- Modify: `app/db/models.py`
- Modify: `app/db/repositories.py`
- Create: `app/tools/audit.py`
- Test: `tests/test_tool_audit.py`

**Interfaces:**
- `class ToolAuditLog(Base)`：表 `tool_audit_logs`，列与 `sql/ch08-ddl.sql` 一致，`conversation_id` 不挂外键。
- `ToolAuditRepository.create(*, tool_name, tool_source, mcp_server=None, tool_call_id=None, conversation_id=None, arguments=None, result_summary=None, status, error_message=None, retry_count=0, duration_ms=None) -> None`
- `Repositories.audit` 属性
- `write_audit(repositories, entry: AuditEntry) -> None`：`try/except Exception: logger.warning(...)`，不抛。

- [ ] **Step 1: 写失败测试**

```python
def test_audit_row_persists_all_fields(fake_repositories):
    from app.tools.audit import AuditEntry, write_audit
    write_audit(fake_repositories, AuditEntry(tool_name="query_order", tool_source="builtin",
        mcp_server=None, tool_call_id="call_1", conversation_id=None,
        args={"order_id": "A1"}, result_summary='{"ok":true}', status="成功",
        error=None, retry_count=2, duration_ms=37))
    rows = fake_repositories.audit.list_recent(limit=1)
    assert rows[0].tool_name == "query_order" and rows[0].retry_count == 2

def test_write_audit_swallows_failure(monkeypatch):
    from app.tools import audit
    class Boom:
        def create(self, **kwargs): raise RuntimeError("db down")
    audit.write_audit(Boom(), audit.AuditEntry(...))  # 不抛
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 加模型/仓库/写入器；`arguments` 用 `JSON` 列**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add tool_audit_logs model, repository and writer`

---

### Task 7: 执行引擎重构

**Files:**
- Modify: `app/tools/executor.py`
- Test: `tests/test_tool_executor_ch08.py`

**Interfaces:**
- `ToolResult` 新增 `status: ToolStatus`、`source: ToolSource`、`mcp_server: str | None`、`tool_call_id: str | None`（保留旧字段与 `summary()` 键）。
- `async execute_tool(name, args, *, registry: ToolRegistry, repositories, retriever=None, session_id=None, conversation_id=None, tool_call_id=None, timeout=3.0, max_retries=1, confirmed=False) -> ToolResult`
- 流程：未知工具 → 校验 → 权限 → 调用（只读且瞬时故障重试）→ 呈现 → 审计 → 返回。任何分支都写审计。

- [ ] **Step 1: 写失败测试**

```python
async def test_invalid_args_are_validation_status_and_audited(fake_repositories):
    reg = build_registry(fake_repositories, None)
    res = await execute_tool("query_order", {}, registry=reg, repositories=fake_repositories)
    assert res.status == "校验拦下" and res.attempts == 0 and res.ok is False
    assert fake_repositories.audit.list_recent(limit=1)[0].status == "校验拦下"

async def test_write_without_confirmation_denied(fake_repositories):
    reg = build_registry(fake_repositories, None)
    res = await execute_tool("create_ticket", {"description": "x", "ticket_type": "售后",
        "conversation_id": "s1"}, registry=reg, repositories=fake_repositories)
    assert res.status == "权限拒绝" and res.error

async def test_write_timeout_not_retried(fake_repositories):
    # 注册一个 read_only=False 的慢工具，max_retries=3
    res = await execute_tool("slow_write", {}, registry=reg, repositories=fake_repositories,
                             timeout=0.05, max_retries=3)
    assert res.status == "超时" and res.attempts == 1
    assert fake_repositories.audit.list_recent(limit=1)[0].retry_count == 0

async def test_read_timeout_retried_once(fake_repositories):
    ...  # read_only=True, max_retries=1 → attempts == 2, 审计 retry_count == 1
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现；`confirmed` 默认 `False`，只有路由/节点可传 `True`；超时文案与旧版一致**
- [ ] **Step 4: 跑测试 + 既有 `tests/test_tools_executor.py`；旧调用点补 `registry=build_registry(...)`**
- [ ] **Step 5: Commit** `feat(ch08): unify tool execution with validation, permissions and audit`

---

### Task 8: 两个业务 MCP Server

**Files:**
- Create: `mcp_servers/__init__.py`、`mcp_servers/mock_data.py`、`mcp_servers/logistics_server.py`、`mcp_servers/aftersales_server.py`
- Test: `tests/test_mcp_servers.py`

**Interfaces:**
- `logistics_server.py`：`mcp = FastMCP("memhelp-logistics", host="127.0.0.1", port=8101)`；`@mcp.tool() def track_shipment(order_id: str) -> dict`；`main()` 里 `mcp.run(transport="streamable-http")`。
- `aftersales_server.py`：端口 8102；`check_warranty(order_id)`、`check_return_progress(order_id)`。
- `mock_data.py`：`seed_for(order_id) -> random.Random`（`hashlib.sha256` 派生，跨进程稳定）。

- [ ] **Step 1: 写失败测试**：直接 import 工具函数（FastMCP 的 `@mcp.tool` 返回原函数或可 `.fn` 调用；若被包装则从 `mcp._tool_manager` 取，测试里两种都试）。断言同一 order_id 两次结果一致、字段齐全、状态码在枚举内。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现三个工具 + 两个 `main()`；不建表、不连外部系统**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: 启动探针**：`uv run python -m mcp_servers.logistics_server`，另开终端 `curl -s http://127.0.0.1:8101/mcp` 确认进程在监听（Streamable HTTP 对 GET 返回 4xx 属正常），日志无异常后停止。
- [ ] **Step 6: Commit** `feat(ch08): add logistics and aftersales MCP servers`

---

### Task 9: MCP Client Provider（发现 + 缓存）

**Files:**
- Create: `app/tools/mcp_client.py`
- Test: `tests/test_mcp_client.py`（用假 `MultiServerMCPClient` 注入，不连网）

**Interfaces:**
- `McpToolProvider(connections: dict[str, dict], *, read_only_servers: set[str], timeout: float)`；`connections` 由 `Settings` 构造。
- `async discover() -> list[ToolSpec]`：
  - 逐 server `await client.get_tools(server_name=name)`（一个 Server 失败不影响另一个）
  - `args_schema`：`tool.args_schema` 是 dict 直接用，是 Pydantic 模型用 `.model_json_schema()`
  - `read_only = name in read_only_servers`；不在白名单的 Server **整站拒绝**并 warning
  - `description = tool.description or ""`；`presenter` 按工具名取 Task 5 的表
  - 名字与内置冲突在此不处理（registry 负责跳过）
- `cached_specs() -> tuple[ToolSpec, ...]`；`discover()` 成功后整体替换缓存，失败保留旧缓存。

- [ ] **Step 1: 写失败测试**：假 client 返回两个 `StructuredTool`（`args_schema` 一 dict 一 Pydantic），断言来源/服务器名/只读标记正确；一个 server 抛异常时另一个仍返回。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现；`MultiServerMCPClient` 懒构造，`transport="streamable_http"`**
- [ ] **Step 4: 跑测试通过**
- [ ] **Step 5: Commit** `feat(ch08): add MCP tool provider with per-server discovery`

---

### Task 10: 每轮刷新接线（registry / graph / api / lifespan）

**Files:**
- Modify: `app/tools/registry.py`（`refresh_mcp` 用 provider 缓存 + 锁 + 冲突跳过）
- Modify: `app/agent/graph.py`（`_run_agent_node` 前 `await registry.refresh_mcp()`；`GraphDeps` 加 `mcp_provider`/`tool_registry_factory`）
- Modify: `app/agent/react.py`（签名收 `registry`，`agent_tools(registry, ...)`）
- Modify: `app/api.py`（lifespan 建 `app.state.mcp_provider`；`create_app` 接受注入）
- Test: `tests/test_registry_mcp_refresh.py`、更新 `tests/test_agent_react.py`

**Interfaces:**
- `ToolRegistry.refresh_mcp() -> None`：`async with self._lock:`；`await provider.discover()`；把 `provider.cached_specs()` 合并进 `self._mcp`；与内置重名 → `logger.warning("ch08 MCP 工具与内置重名，已跳过：%s", name)`。
- `build_registry(..., mcp_provider=None, enable_mcp=False)`：`enable_mcp` 才拉 MCP。
- `run_react_agent(..., registry: ToolRegistry, ...)`（必填；旧调用点全改）。

- [ ] **Step 1: 写失败测试**：假 provider 首次返回 1 个工具、第二次返回 2 个 → 两次 `refresh_mcp()` 后 snapshot 从 5 个变 6 个；与内置同名时被跳过；provider 抛异常时内置仍在。
- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现并接线**
- [ ] **Step 4: 跑 `tests/test_agent_graph*.py`、`tests/test_agent_react*.py`、`tests/test_api_chat*.py`**
- [ ] **Step 5: Commit** `feat(ch08): refresh MCP tools every turn and bind into agent`

---

### Task 11: 建工单确认流（拦截 + 图节点 + resume API）

**Files:**
- Modify: `app/agent/react.py`（拦截 `create_ticket`；`AgentOutcome.pending_ticket_call`）
- Modify: `app/agent/state.py`（`pending_ticket_call`、`ticket_preview`）
- Modify: `app/agent/graph.py`（`ticket_await_confirm` 节点 + 条件边）
- Modify: `app/agent/runner.py`（`awaiting_input` 支持 `ticket_preview`；`astream_resume` 接受任意 resume 值）
- Modify: `app/api.py`（`POST /api/chat/ticket/confirm`；`/api/actions/create-ticket` 传 `confirmed=True`）
- Test: `tests/test_ticket_confirm_flow.py`

**Interfaces:**
- `AgentOutcome` 新字段 `pending_ticket_call: dict[str, Any] | None`（`{"args": {...}, "call_id": "..."}`）。
- 拦截逻辑：`spec = registry.get("create_ticket")`；`validate_against_schema(spec.args_schema, args)` 失败 → 追加 `ToolMessage`（校验文案）继续循环；通过 → 不执行、记录、`stop_reason="ticket_pending"`、break。
- `ticket_await_confirm(state, config)`：
  - `decision = interrupt({"kind": "ticket_preview", "turn_id": ..., "call_id": ..., "preview": {"ticket_type": ..., "description": ...}})`
  - `decision.get("confirmed")` 为假 → `write_audit(status="权限拒绝", error_message="用户取消建单，未执行")`，`answer="已取消建单，没有创建工单。"`
  - 为真 → `execute_tool(..., confirmed=True, tool_call_id=call_id)`；`answer = json.loads(data)["message"]`（失败则兜底文案）；`pending_messages` 追加 `AIMessage(tool_calls=[...])` + `ToolMessage`。
- `run_agent` 条件边：`pending_ticket_call` 非空 → `ticket_await_confirm`，否则 `log_turn`。
- `POST /api/chat/ticket/confirm`：body `{session_id: str, confirmed: bool}`；409 `no_pending_ticket` / `ticket_resume_in_progress`；SSE resume `{"confirmed": ...}`。

- [ ] **Step 1: 写失败测试**

```python
def test_missing_description_returns_validation_message_to_model(...)
def test_valid_call_pauses_with_ticket_preview_frame(...)   # SSE 首帧 kind=ticket_preview
def test_confirm_creates_ticket_and_reply_has_ticket_no(...) # tickets 落库 + "TK" in answer
def test_cancel_audits_permission_denied_and_no_ticket(...)
```

- [ ] **Step 2: 跑测试确认失败**
- [ ] **Step 3: 实现**
- [ ] **Step 4: 跑测试 + 既有 ch05/ch06 图测试**
- [ ] **Step 5: Commit** `feat(ch08): gate create_ticket behind LangGraph interrupt confirmation`

---

### Task 12: 前端工单预览卡片（Vibe Coding）

**Files:**
- Modify: `web/app.js`、`web/style.css`、`web/index.html`（如需）

- [ ] **Step 1: `applyFrame` 增加 `ticket_preview` 分支**：清空 `bot.body`，`renderTicketPreview(target, data, ctx)`；`ctx.awaiting=true`、禁用输入、状态「等待确认建单」。
- [ ] **Step 2: `renderTicketPreview`**：`.ticket-preview` 卡片，标题「工单预览」，两行「类型 / 问题描述」，底部「确认提交」（主按钮）与「取消」（次按钮）。
- [ ] **Step 3: `confirmTicket(ctx, confirmed, target)`**：禁用两按钮 → `fetch('/api/chat/ticket/confirm', {method:'POST', body: JSON.stringify({session_id: ctx.sessionId, confirmed})})` → `consumeStream`。
- [ ] **Step 4: `awaiting_input` 的 `kind=ticket_preview` 调同一渲染（刷新恢复）。**
- [ ] **Step 5: 手工验收 + 截图存 `dev-notes/ch08-ticket-preview.png`。**
- [ ] **Step 6: Commit** `feat(ch08): render ticket preview card with confirm/cancel`

---

### Task 13: 演示脚本、验收、文档

**Files:**
- Create: `scripts/run_mcp_servers.py`、`scripts/demo_ch08.py`（或 `scripts/demo_ch08.ps1`）
- Create: `dev-notes/ch08-acceptance-output.txt`
- Modify: `dev-notes/ch08.md`、`README.md`（运行说明）
- Test: 全量 `pytest`

- [ ] **Step 1: `run_mcp_servers.py`**：用 `multiprocessing` 起两个 `mcp.run()` 子进程，Ctrl+C 一起退出；端口 8101/8102。
- [ ] **Step 2: `demo_ch08.py`**：起 FastAPI（TestClient 或 uvicorn）→ 问「查一下订单 A1001 的物流」→ 打印 SSE 中的工具结果；再演示确认/取消两条流（用 mock 模型或真实模型，真实模型时打印 `tool_audit_logs`）。
- [ ] **Step 3: 跑全量测试**，把输出存 `dev-notes/ch08-acceptance-output.txt`。
- [ ] **Step 4: 逐条走验收 1–6**（含人为超时：临时注册 `slow_tool` 的演示入口），记录命令与结果。
- [ ] **Step 5: 追记 `dev-notes/ch08.md` 的「计划评审 / 任务完成 / code review / finish」各段（四样齐全）。**
- [ ] **Step 6: Commit** `docs(ch08): add demo scripts and acceptance notes`
