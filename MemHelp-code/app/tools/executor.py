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
from app.tools.registry import TOOL_NAMES, get_tool

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
    repositories: Repositories,
    retriever: Any = None,
    timeout: float = 3.0,
    max_retries: int = 1,
    resolve: Callable[[str, Repositories, Any], Any] | None = None,
) -> ToolResult:
    """执行一个工具，**任何失败都转成 ``ToolResult(ok=False)``**。

    顺序固定：**未知工具 → 参数校验 → 带超时执行 → （必要时）重试**。
    校验必须在执行之前：不能让工具收到半个参数再炸，那时错误信息来自工具内部，
    对模型毫无可操作性。

    :param repositories: 当次请求的仓库。工具依赖由它按名字构造出来。
    :param retriever: 当次请求的向量检索器（ch03 起 ``query_faq`` 依赖它）。
        不关心检索的工具会忽略它；``None`` 时 ``query_faq`` 会在被调用时抛错，
        由本函数分类成 ``ok=false``。
    :param timeout: 单次执行的超时（秒）。
    :param max_retries: 仅瞬时错误的重试次数，不含首次。总尝试数为 ``max_retries + 1``。
    :param resolve: 按名字取工具的可调用对象。默认走注册表 ``get_tool``；
        测试显式注入假工具，从而**不依赖 monkeypatch**。

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

    resolver = resolve if resolve is not None else get_tool

    # -- 1. 未知工具 ------------------------------------------------------
    # 模型自己编的工具名是**正常输入**（它可能记混），不是程序缺陷：
    # 转成结构化失败让它自我纠正，并在错误里列出可用工具。
    #
    # resolver 本身也被包住：构造工具时可能抛异常（某个工具模块 import 期出问题），
    # 而「永不抛异常」是本函数对外的契约 —— 不能因为异常发生在校验之前就破例。
    try:
        tool = resolver(name, repositories, retriever)
    except BaseException as exc:  # noqa: BLE001
        return ToolResult(
            tool=name,
            args=dict(args),
            ok=False,
            data=None,
            error=redact(f"工具 {name} 无法加载：{type(exc).__name__}: {exc}"),
            attempts=0,
            duration_ms=_elapsed_ms(),
            started_at=started_at,
            fault="tool",
        )

    if tool is None:
        return ToolResult(
            tool=name,
            args=dict(args),
            ok=False,
            data=None,
            error=redact(f"未知工具 {name!r}。可用工具：{', '.join(TOOL_NAMES)}"),
            attempts=0,
            duration_ms=_elapsed_ms(),
            started_at=started_at,
            fault="tool",
        )

    # -- 2. 参数校验（不重试）---------------------------------------------
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            validated = schema.model_validate(args)
        except ValidationError as exc:
            return ToolResult(
                tool=name,
                args=dict(args),
                ok=False,
                data=None,
                error=redact(_validation_error_text(exc)),
                attempts=1,
                duration_ms=_elapsed_ms(),
                started_at=started_at,
                fault="tool",
            )
        call_args = validated.model_dump()
    else:
        call_args = dict(args)

    # -- 3. 执行 + 仅瞬时错误重试 -----------------------------------------
    attempts = 0
    last_error: BaseException | None = None
    retried = False

    for attempt_index in range(max_retries + 1):
        attempts += 1
        if attempt_index > 0:
            retried = True
        try:
            data = await _invoke(tool, call_args, timeout)
        except (Exception, asyncio.CancelledError, SystemExit) as exc:  # noqa: BLE001
            last_error = exc
            if is_transient(exc) and attempt_index < max_retries:
                continue
            break
        else:
            return ToolResult(
                tool=name,
                args=call_args,
                ok=True,
                data=data,
                error=None,
                attempts=attempts,
                duration_ms=_elapsed_ms(),
                started_at=started_at,
                retried=retried,
            )

    assert last_error is not None  # 循环只在异常或成功时退出
    if isinstance(last_error, (asyncio.TimeoutError, TimeoutError)):
        reason = f"工具 {name} 执行超时（超过 {timeout} 秒）"
    elif isinstance(last_error, asyncio.CancelledError):
        # 上游/客户端断开把这次执行取消了。它不是工具的问题，但也不能装作成功。
        reason = f"工具 {name} 的执行被取消（连接可能已断开）"
    else:
        reason = f"工具 {name} 执行失败：{type(last_error).__name__}: {last_error}"

    return ToolResult(
        tool=name,
        args=call_args,
        ok=False,
        data=None,
        error=redact(reason),
        attempts=attempts,
        duration_ms=_elapsed_ms(),
        started_at=started_at,
        retried=retried,
        fault="database" if is_database_fault(last_error) else "tool",
    )
