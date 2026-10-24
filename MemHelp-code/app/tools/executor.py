"""工具执行器 —— 参数校验、超时、重试、错误脱敏、结果序列化。

**本模块的契约：``execute_tool`` 永不抛异常。** 工具失败是业务信息
（「这个订单查不到」「数据库暂时连不上」），它要回灌给模型让模型组织一句人话；
让它冒到 HTTP 层，SSE 连接会被打断，用户看到的是断流而不是回答。

## 为什么要自己的 ``is_transient`` 而不复用 ``app/errors.is_infrastructure_error``

``app/errors.py`` 是全项目错误分类的单一来源，但它的分类器是为**上游 LLM 错误**
写的：标记表里是 ``apiconnectionerror``、``apitimeouterror``、``invalidrequesterror``
这些 OpenAI SDK 的异常名。拿它判数据库错误会得到错误答案 ——
SQLAlchemy 的 ``OperationalError``（「server has gone away」，典型可重试）
不匹配任何上游标记，会被归成兜底的 ``upstream_error``，从而**不重试**。

而且复用它会污染上游分类语义：上游分类决定的是给客户端什么错误码（502/503），
工具分类决定的是「要不要再试一次」。两者的判据不同，混用会让改动一处、
影响另一处。所以这里独立实现，并在文档里写清边界。

## 重试策略

只有**瞬时错误**重试：超时、连接断开、数据库连接抖动 —— 它们下一次可能成功。
参数校验失败与业务性失败（「订单不存在」）**不重试**：重试也是同样结果，
白等 N 倍超时，而用户感知到的是「客服卡住了」。
"""

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy.exc import InterfaceError, OperationalError, PendingRollbackError

from app.db.repositories import Repositories
from app.errors import redact
from app.tools.audit import AuditEntry, write_audit
from app.tools.contract import ToolSource, ToolStatus, ToolSpec
from app.tools.mcp_client import unwrap_mcp_result
from app.tools.permissions import check_permission
from app.tools.presenter import present_result, summarize_result
from app.tools.registry import TOOL_NAMES, ToolRegistry, build_registry
from app.tools.validation import validate_against_schema

#: 回灌给模型的单个工具结果上限（字符）。
#:
#: 没有上限时，一个返回上万行 FAQ 的查询会把上下文预算一次吃光，
#: 于是**这一轮的最终回答**再也放不进去 —— 表现为「模型答非所问」。
#: 4000 字符约等于 1000~1500 token，对客服场景的工具体量足够。
MAX_TOOL_RESULT_CHARS = 4000

#: 工具记录详情里展示的结果预览上限（字符）。完整结果仍按原契约回灌模型，
#: 这里只截给界面用，避免一条上万字的 FAQ 把气泡撑爆。
TOOL_TRACE_PREVIEW_CHARS = 600


@dataclass(frozen=True)
class ToolResult:
    """一次工具执行的完整结果。**纯数据**，可直接落库、打日志、序列化。"""

    tool: str
    args: dict[str, Any]
    ok: bool
    #: 成功时工具返回的原始载荷（通常是 JSON 字符串）。
    data: Any | None
    #: 失败原因，**已脱敏**。
    error: str | None
    #: 实际尝试次数（含首次）。未知工具为 0。
    attempts: int
    duration_ms: int = 0
    #: 工具开始执行的 UTC 时间。给界面展示「调用时间」，不参与模型历史。
    started_at: str | None = None
    #: 是否因为瞬时错误重试过（供留痕与调试）。
    retried: bool = False
    #: 失败归类：``tool``（参数/业务/未知工具）或 ``database``（库不可用）。
    #: HTTP 层据此把建单失败映射成 502 或 503，而不是一律 502。
    fault: str | None = None
    # -- ch08 新增：审计与来源 -------------------------------------------
    #: 本次调用的结论，与审计表 ``status`` 同一套枚举。
    status: ToolStatus = "成功"
    #: 工具来源；未知工具按内置处理（它根本没登记）。
    source: ToolSource = "builtin"
    #: MCP 工具所属 Server 名；内置为 ``None``。
    mcp_server: str | None = None
    #: 模型给的 tool_call_id；直接调用（按钮/节点）时为 ``None``。
    tool_call_id: str | None = None

    def summary(self) -> dict[str, Any]:
        """给 SSE 工具记录与落库用的精简描述。

        在 ch02 的五个字段之外补充 ``started_at`` / ``result`` / ``error``：
        前端详情面板要展示调用时间与结果，刷新回读也靠同一份摘要。
        ``result`` 只放截断后的预览，完整数据仍走 ``ToolMessage``。
        """
        return {
            "name": self.tool,
            "args": self.args,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "result": _preview_value(self.data) if self.ok else None,
            "error": _preview_text(self.error) if not self.ok else None,
        }


def is_transient(exc: BaseException) -> bool:
    """这个异常值得重试吗。

    沿 ``__cause__`` / ``__context__`` 链遍历：真实故障常被包一层
    （``ToolException`` 里套 ``OperationalError``），只看最外层会把可重试的
    判成不可重试。
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(
            current,
            (
                asyncio.TimeoutError,
                TimeoutError,
                ConnectionError,
                OperationalError,
                InterfaceError,
            ),
        ):
            return True

        current = current.__cause__ or current.__context__

    return False


def is_database_fault(exc: BaseException) -> bool:
    """这个工具失败是不是数据库故障（用于把 502 tool_error 映射成 503）。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (OperationalError, InterfaceError, PendingRollbackError)):
            return True
        current = current.__cause__ or current.__context__
    return False


def _validation_error_text(exc: ValidationError) -> str:
    """把 pydantic 的报错压成**给模型看**的一行。

    整段 pydantic 输出含文档 URL 与输入回显，既占 token，模型也看不出该改哪。
    这里只取「字段名：原因」。
    """
    parts: list[str] = []
    for item in exc.errors():
        location = ".".join(str(piece) for piece in item.get("loc", ())) or "(根)"
        parts.append(f"{location}: {item.get('msg', '不合法')}")
    return "参数不合法 —— " + "；".join(parts)


def _serialise(data: Any) -> Any:
    """尽量把工具返回值转成可 JSON 序列化的形式；做不到就转字符串。

    这一层是「回灌给模型」的最后一道，**不能抛**。
    """
    if data is None or isinstance(data, (str, int, float, bool, list, dict)):
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (ValueError, TypeError):
                return data
        return data
    try:
        json.dumps(data)
    except (TypeError, ValueError):
        return repr(data)
    return data


def _preview_text(value: Any, *, limit: int = TOOL_TRACE_PREVIEW_CHARS) -> str | None:
    """截断给界面看的文本；空值保持 None，前端据此显示「无」。"""
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _preview_value(value: Any, *, limit: int = TOOL_TRACE_PREVIEW_CHARS) -> str | None:
    """把工具返回值压成可读的 JSON 预览，失败时退回普通字符串。"""
    if value is None:
        return None
    serialised = _serialise(value)
    try:
        text = json.dumps(serialised, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(serialised)
    return _preview_text(text, limit=limit)


def to_tool_message_content(result: ToolResult) -> str:
    """把结果编码成 ``ToolMessage`` 的正文。

    形状固定为 ``{"ok": ..., "tool": ..., "data"|"error": ...}``，
    让模型一眼能分辨成功与失败，不必去猜工具各自的返回格式。

    **错误文本在这里再过一次 ``redact()``**：``execute_tool`` 已经脱敏过一次，
    但 ``ToolResult`` 是公开的纯数据类型，可能由别的路径构造出来
    （测试、将来的批处理、其它调用点）。这一层是「回灌给模型」的最后一道，
    漏一次就等于把库密码或密钥送到聊天窗口 —— 二次脱敏是幂等的，成本为零。
    """
    payload: dict[str, Any] = {"ok": result.ok, "tool": result.tool}

    if result.ok:
        payload["data"] = _serialise(result.data)
    else:
        payload["error"] = redact(result.error or "未知错误")

    text = json.dumps(payload, ensure_ascii=False)

    if len(text) > MAX_TOOL_RESULT_CHARS:
        # 按「编码后的总长度」截断，而不是按 data 的长度 ——
        # 否则一个短 data 配上长 error 照样能超。
        payload = {"ok": result.ok, "tool": result.tool, "truncated": True}
        if result.ok:
            payload["data"] = str(result.data)[:MAX_TOOL_RESULT_CHARS]
        else:
            payload["error"] = redact(result.error or "")[:MAX_TOOL_RESULT_CHARS]
        text = json.dumps(payload, ensure_ascii=False)

    return text


async def _invoke(
    tool: Any, args: dict[str, Any], timeout: float
) -> Any:
    """跑一次工具，带超时。"""
    return await asyncio.wait_for(tool.ainvoke(args), timeout=timeout)


async def execute_tool(
    name: str,
    args: dict[str, Any],
    *,
    registry: ToolRegistry | None = None,
    repositories: Repositories,
    retriever: Any = None,
    session_id: str | None = None,
    conversation_id: int | None = None,
    tool_call_id: str | None = None,
    timeout: float = 3.0,
    max_retries: int = 1,
    confirmed: bool = False,
    resolve: Callable[[str, Repositories, Any], Any] | None = None,
) -> ToolResult:
    """执行一个工具，**任何失败都转成 ``ToolResult(ok=False)``**。

    顺序固定：**未知工具 → 参数校验 → 权限 → 带超时执行 → 呈现 → 审计**。
    校验必须在执行之前：不能让工具收到半个参数再炸，那时错误信息来自工具内部，
    对模型毫无可操作性。

    :param registry: 本轮工具表。``None`` 时按 ``repositories`` 现造一份（只含内置）。
    :param repositories: 当次请求的仓库。工具依赖由它按名字构造出来。
    :param retriever: 当次请求的向量检索器（ch03 起 ``query_faq`` 依赖它）。
    :param session_id: 会话 id，仅用于调用方留痕（审计存的是数字 conversation_id）。
    :param conversation_id: 数字会话 id，直接落审计表的 ``conversation_id``。
    :param tool_call_id: 模型给的调用 id，落审计表便于把一轮对话和调用对上。
    :param timeout: 单次执行的超时（秒）。
    :param max_retries: 仅只读工具 + 瞬时错误的重试次数，不含首次。
        **写操作永不自动重试**：超时未必没执行，重复执行比失败更糟。
    :param confirmed: 写操作是否已获用户确认。默认 ``False`` —— 只有前端确认
        回来的路由（``/api/chat/ticket/confirm``）与 ch05 的按钮路由能传 ``True``。
    :param resolve: **兼容 ch02 旧测试**的按名字取工具钩子；新代码一律用
        ``registry``。传了它时按只读工具处理（旧测试都在测只读语义）。

        **为什么要有这个参数**（踩过的坑）：测试若用
        ``monkeypatch.setattr(executor, "get_tool", fake)`` 注入，在**全量测试**下会失效 ——
        测试隔离会把这个模块加载第二次，于是 ``execute_tool.__globals__`` 里的
        ``get_tool`` 与 ``sys.modules["app.tools.executor"].get_tool`` **不是同一个对象**：
        测试 patch 的是后者，而函数用的是前者。单独跑该文件时不触发，因此这个坑
        只在全量运行时表现为「注入毫无效果」。显式参数不经过模块全局名，
        任何加载方式下都稳定。
    """
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def _elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    table = registry if registry is not None else build_registry(repositories, retriever)
    spec: ToolSpec | None = None if resolve is not None else table.get(name)
    legacy = False
    if resolve is not None:
        # ch02 兼容路径优先：显式注入的假工具没有 ToolSpec，按只读内置工具包一层。
        # 必须先于 registry 查表 —— 否则同名的真内置工具会抢走调用，
        # 旧测试就会在真库上跑（表现为注入静默失效）。
        try:
            tool = resolve(name, repositories, retriever)
        except BaseException as exc:  # noqa: BLE001
            return await _finish(
                repositories,
                ToolResult(
                    tool=name,
                    args=dict(args),
                    ok=False,
                    data=None,
                    error=redact(f"工具 {name} 无法加载：{type(exc).__name__}: {exc}"),
                    attempts=0,
                    duration_ms=_elapsed_ms(),
                    started_at=started_at,
                    fault="tool",
                    status="失败",
                ),
                spec=None,
                timeout=timeout,
                session_id=session_id,
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                status="失败",
                error=None,
            )
        if tool is not None:
            schema = getattr(tool, "args_schema", None)
            if schema is not None and not isinstance(schema, dict):
                schema = schema.model_json_schema()
            spec = ToolSpec(
                name=name,
                description=getattr(tool, "description", "") or "",
                args_schema=schema or {"type": "object", "properties": {}},
                source="builtin",
                mcp_server=None,
                read_only=True,
                tool=tool,
            )
            legacy = True

    # -- 1. 未知工具 ------------------------------------------------------
    # 模型自己编的工具名是**正常输入**（它可能记混），不是程序缺陷：
    # 转成结构化失败让它自我纠正，并在错误里列出可用工具。
    if spec is None:
        return await _finish(
            repositories,
            ToolResult(
                tool=name,
                args=dict(args),
                ok=False,
                data=None,
                error=redact(f"未知工具 {name!r}。可用工具：{', '.join(TOOL_NAMES)}"),
                attempts=0,
                duration_ms=_elapsed_ms(),
                started_at=started_at,
                fault="tool",
                status="失败",
            ),
            spec=None,
            timeout=timeout,
            session_id=session_id,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="失败",
            error=None,
        )

    source: ToolSource = spec.source
    server = spec.mcp_server

    # -- 2. JSON Schema 校验（不重试，回灌可操作的错误文本）----------------
    error = validate_against_schema(spec.args_schema, args)
    if error is not None:
        return await _finish(
            repositories,
            ToolResult(
                tool=name,
                args=dict(args),
                ok=False,
                data=None,
                error=redact(error),
                attempts=0,
                duration_ms=_elapsed_ms(),
                started_at=started_at,
                fault="tool",
                status="校验拦下",
                source=source,
                mcp_server=server,
                tool_call_id=tool_call_id,
            ),
            spec=spec,
            timeout=timeout,
            session_id=session_id,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="校验拦下",
            error=error,
        )

    # -- 3. 权限闸（只认本侧 read_only + confirmed）------------------------
    denied = check_permission(spec, confirmed=confirmed)
    if denied is not None:
        return await _finish(
            repositories,
            ToolResult(
                tool=name,
                args=dict(args),
                ok=False,
                data=None,
                error=redact(denied),
                attempts=0,
                duration_ms=_elapsed_ms(),
                started_at=started_at,
                fault="tool",
                status="权限拒绝",
                source=source,
                mcp_server=server,
                tool_call_id=tool_call_id,
            ),
            spec=spec,
            timeout=timeout,
            session_id=session_id,
            conversation_id=conversation_id,
            tool_call_id=tool_call_id,
            status="权限拒绝",
            error=denied,
        )

    # -- 4. 执行 + 仅只读工具的瞬时错误重试 -------------------------------
    attempts = 0
    last_error: BaseException | None = None
    retried = False
    # 写操作永远只有一次机会；旧兼容路径按只读处理。
    allowed_retries = max_retries if (spec.read_only or legacy) else 0

    for attempt_index in range(allowed_retries + 1):
        attempts += 1
        if attempt_index > 0:
            retried = True
        try:
            raw = await _invoke(spec.tool, dict(args), timeout)
        except (Exception, asyncio.CancelledError, SystemExit) as exc:  # noqa: BLE001
            last_error = exc
            if is_transient(exc) and attempt_index < allowed_retries:
                continue
            break
        else:
            if source == "mcp":
                # MCP 返回 content blocks；先摊平成文本，呈现层才解析得了 JSON。
                raw = unwrap_mcp_result(raw)
            if legacy:
                # 旧测试注入的假工具没有业务字段，呈现层的白名单会把它削空；
                # 兼容路径保持 ch02 的「原样回灌」语义。
                data = raw
            else:
                presented = (
                    spec.presenter(raw) if spec.presenter else present_result(spec, raw)
                )
                data = (
                    presented
                    if isinstance(presented, str)
                    else json.dumps(presented, ensure_ascii=False, default=str)
                )
            return await _finish(
                repositories,
                ToolResult(
                    tool=name,
                    args=dict(args),
                    ok=True,
                    data=data,
                    error=None,
                    attempts=attempts,
                    duration_ms=_elapsed_ms(),
                    started_at=started_at,
                    retried=retried,
                    status="成功",
                    source=source,
                    mcp_server=server,
                    tool_call_id=tool_call_id,
                ),
                spec=spec,
                timeout=timeout,
                session_id=session_id,
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                status="成功",
                error=None,
            )

    assert last_error is not None  # 循环只在异常或成功时退出
    timed_out = isinstance(last_error, (asyncio.TimeoutError, TimeoutError))
    if timed_out:
        reason = f"工具 {name} 执行超时（超过 {timeout} 秒）"
    elif isinstance(last_error, asyncio.CancelledError):
        # 上游/客户端断开把这次执行取消了。它不是工具的问题，但也不能装作成功。
        reason = f"工具 {name} 的执行被取消（连接可能已断开）"
    else:
        reason = f"工具 {name} 执行失败：{type(last_error).__name__}: {last_error}"

    status: ToolStatus = "超时" if timed_out else "失败"
    return await _finish(
        repositories,
        ToolResult(
            tool=name,
            args=dict(args),
            ok=False,
            data=None,
            error=redact(reason),
            attempts=attempts,
            duration_ms=_elapsed_ms(),
            started_at=started_at,
            retried=retried,
            fault="database" if is_database_fault(last_error) else "tool",
            status=status,
            source=source,
            mcp_server=server,
            tool_call_id=tool_call_id,
        ),
        spec=spec,
        timeout=timeout,
        session_id=session_id,
        conversation_id=conversation_id,
        tool_call_id=tool_call_id,
        status=status,
        error=reason,
    )


async def _finish(
    repositories: Repositories,
    result: ToolResult,
    *,
    spec: ToolSpec | None,
    timeout: float,
    session_id: str | None,
    conversation_id: int | None,
    tool_call_id: str | None,
    status: ToolStatus,
    error: str | None,
) -> ToolResult:
    """写一条审计后把结果原样交回。**审计失败只 warning，绝不改结果。**"""
    conversation = conversation_id
    if conversation is None and session_id:
        # Agent 只在首轮拿不到 conversation_id（会话还没落库）；这里按
        # session_id 反查一次，让审计能挂上会话。查不到就算了，不升级错误。
        try:
            conversation = await repositories.conversations.find_id(session_id)
        except Exception:  # noqa: BLE001 - 审计关联失败不影响结果
            conversation = None
    summary = None
    if result.ok:
        try:
            summary = summarize_result(spec, result.data) if spec is not None else None
        except Exception:  # noqa: BLE001 - 摘要失败不影响结果
            summary = None
    entry = AuditEntry(
        tool_name=result.tool,
        tool_source=result.source,
        status=status,
        mcp_server=result.mcp_server,
        tool_call_id=tool_call_id or result.tool_call_id,
        conversation_id=conversation,
        args=result.args,
        result_summary=summary,
        error=(redact(error)[:512] if error else None),
        retry_count=max(0, result.attempts - 1) if result.attempts else 0,
        duration_ms=result.duration_ms,
    )
    await write_audit(repositories, entry)
    return result
