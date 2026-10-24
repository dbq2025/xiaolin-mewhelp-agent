"""工具执行器的测试。

执行器是工具链里唯一「出错了也绝不抛异常」的地方。理由：工具失败是**业务信息**
（「这个订单查不到」「数据库暂时连不上」），它要回灌给模型，让模型组织一句人话；
若让它冒到 HTTP 层，SSE 连接会被打断，用户看到的是断流而不是回答。

所以这里的用例大部分在钉「失败被转成了什么形状的结果」，而不是「抛出了什么异常」。

## 关于重试

只有**瞬时错误**（超时、连接断开）才重试 —— 它们下一次可能成功。
参数校验失败与业务性「查不到」**不重试**：重试也是同样结果，白等 N 倍超时，
而用户感知到的是「客服卡住了」。

## 为什么注入假工具用的是 ``resolve=`` 参数而不是 ``monkeypatch``

最初这里用 ``monkeypatch.setattr(executor, "get_tool", fake)``，**单独跑这个文件时全绿，
全量跑时 6 条红**。根因：测试隔离会把这个模块加载第二次，于是
``execute_tool.__globals__["get_tool"]`` 与 ``sys.modules["app.tools.executor"].get_tool``
是两个不同对象 —— 测试 patch 的是后者，函数用的是前者。

所以 ``execute_tool`` 接受一个显式的 ``resolve`` 可调用对象。它不经过模块全局名，
任何模块加载方式下都稳定。**不要**把这些改回 monkeypatch。
"""

import asyncio
import json

import pytest

from app.tools.executor import (
    MAX_TOOL_RESULT_CHARS,
    ToolResult,
    execute_tool,
    is_transient,
    to_tool_message_content,
)

TIMEOUT = 1.0


@pytest.fixture
def repos(sqlite_session):
    """执行器要求一个真实的 ``Repositories``。

    这里用接到 SQLite 内存库上的真仓库，而不是假对象 —— 未知工具那条路径
    真的会去注册表里查名字，用真仓库才跑得到。
    """
    from app.db.repositories import Repositories

    return Repositories.from_session(sqlite_session)


def _fake_tool(name="query_order", *, result='{"ok":true}', delay=0.0, error=None):
    """造一个可注入的假工具。

    刻意**不**用注册表里的真工具：这些用例要精确控制「工具内部发生了什么」
    （快返回 / 慢返回 / 抛特定异常），真工具做不到也不该做到。

    实例本身可调用，所以能直接当 ``resolve=`` 用（它忽略 name/repos 两个入参）。
    """

    class Fake:
        def __init__(self):
            self.name = name
            self.calls = 0
            from app.tools.schemas import QueryOrderArgs

            self.args_schema = QueryOrderArgs

        def __call__(self, _name, _repos, _retriever=None):
            """当作 ``resolve`` 用：无论要哪个名字都返回自己。"""
            return self

        async def ainvoke(self, args):
            self.calls += 1
            if delay:
                await asyncio.sleep(delay)
            if error is not None:
                raise error
            return result

    return Fake()


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


async def test_successful_execution_returns_ok_result(repos):
    fake = _fake_tool(result='{"ok": true, "order_id": "1001"}')

    result = await execute_tool(
        "query_order",
        {"order_id": "1001"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=fake,
    )

    assert result.ok is True
    assert result.error is None
    assert result.attempts == 1
    assert json.loads(result.data)["order_id"] == "1001"
    assert result.tool == "query_order"
    assert result.args == {"order_id": "1001"}
    assert result.duration_ms >= 0


async def test_successful_execution_reports_duration(repos):
    fake = _fake_tool(delay=0.02)

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=fake,
    )

    assert result.duration_ms >= 15, "耗时要真实反映等待时长"


async def test_result_summary_is_serialisable_for_the_done_frame(repos):
    """``done.tools`` 与落库都用这个摘要，它必须能直接进 JSON。"""
    fake = _fake_tool()
    result = await execute_tool(
        "query_order", {"order_id": "1"}, repositories=repos, timeout=TIMEOUT,
        max_retries=0, resolve=fake,
    )

    summary = result.summary()
    assert summary["name"] == "query_order"
    assert summary["ok"] is True
    assert set(summary) == {
        "name",
        "args",
        "ok",
        "duration_ms",
        "attempts",
        "started_at",
        "result",
        "error",
    }
    assert summary["started_at"].endswith("+00:00")
    assert '"ok": true' in summary["result"]
    assert summary["error"] is None
    json.dumps(summary)  # 不可序列化会抛


# ---------------------------------------------------------------------------
# 未知工具
# ---------------------------------------------------------------------------


async def test_unknown_tool_name_is_a_structured_failure(repos):
    """模型编一个不存在的工具名是**正常输入**（它可能记混），不是程序缺陷。

    所以返回结构化失败让它自己纠正，而不是抛 KeyError 打断整轮对话。
    这条**不注入 resolve** —— 走真实注册表才验得到「未知」这个判断。
    """
    result = await execute_tool(
        "query_nothing", {}, repositories=repos, timeout=TIMEOUT, max_retries=0
    )

    assert result.ok is False
    assert "未知工具" in result.error
    assert result.attempts == 0, "工具都不存在，没有可尝试的执行"


async def test_unknown_tool_error_lists_available_names(repos):
    """错误信息要能让模型自我纠正 —— 告诉它有哪些工具可用。"""
    result = await execute_tool(
        "query_nothing", {}, repositories=repos, timeout=TIMEOUT, max_retries=0
    )

    assert "query_order" in result.error
    assert "query_faq" in result.error


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


async def test_missing_required_argument_is_rejected_before_execution(repos):
    """校验必须发生在**调用工具之前** —— 不能让工具收到半个参数再炸。"""
    fake = _fake_tool()

    result = await execute_tool(
        "query_order", {}, repositories=repos, timeout=TIMEOUT, max_retries=3, resolve=fake
    )

    assert result.ok is False
    assert "order_id" in result.error
    assert fake.calls == 0, "参数不合法就不该执行工具"


async def test_validation_failure_is_not_retried(repos):
    """参数错重试也是同样结果，只是让用户多等三个超时。"""
    fake = _fake_tool()

    result = await execute_tool(
        "query_order", {}, repositories=repos, timeout=TIMEOUT, max_retries=3, resolve=fake
    )

    assert result.attempts == 1


async def test_wrong_argument_type_is_rejected(repos):
    fake = _fake_tool()

    result = await execute_tool(
        "query_order",
        {"order_id": ["1", "2"]},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=fake,
    )

    assert result.ok is False
    assert fake.calls == 0


async def test_extra_argument_is_rejected(repos):
    """模型偶尔会自作主张加参数。静默忽略会让它以为参数生效了。"""
    fake = _fake_tool()

    result = await execute_tool(
        "query_order",
        {"order_id": "1", "user_id": "u"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=fake,
    )

    assert result.ok is False
    assert fake.calls == 0


async def test_validation_error_message_names_the_field_without_pydantic_noise(repos):
    """错误信息是给**模型**看的。

    整段 pydantic 报错（含文档 URL、输入回显）既占 token 又难读，
    模型看不出该改哪个字段。
    """
    fake = _fake_tool()

    result = await execute_tool(
        "query_order", {}, repositories=repos, timeout=TIMEOUT, max_retries=0, resolve=fake
    )

    assert "order_id" in result.error
    assert "pydantic.dev" not in result.error
    assert "Traceback" not in result.error


# ---------------------------------------------------------------------------
# 超时与重试
# ---------------------------------------------------------------------------


async def test_timeout_retries_then_gives_up(repos):
    fake = _fake_tool(delay=5.0)

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=0.01,
        max_retries=1,
        resolve=fake,
    )

    assert result.ok is False
    assert result.attempts == 2, "首次 + 1 次重试"
    assert "超时" in result.error
    assert result.retried is True
    assert fake.calls == 2


async def test_timeout_with_no_retries_attempts_once(repos):
    fake = _fake_tool(delay=5.0)

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=0.01,
        max_retries=0,
        resolve=fake,
    )

    assert result.attempts == 1
    assert result.retried is False
    assert fake.calls == 1


async def test_transient_failure_succeeds_on_retry(repos):
    """重试的意义：第一次连接断了，第二次成功。"""

    class Flaky:
        name = "query_order"

        def __init__(self):
            from app.tools.schemas import QueryOrderArgs

            self.args_schema = QueryOrderArgs
            self.calls = 0

        def __call__(self, _name, _repos, _retriever=None):
            return self

        async def ainvoke(self, args):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("connection reset by peer")
            return '{"ok": true}'

    flaky = Flaky()

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=2,
        resolve=flaky,
    )

    assert result.ok is True
    assert result.attempts == 2
    assert flaky.calls == 2


async def test_non_transient_error_is_not_retried(repos):
    """业务性失败重试也是同样结果 —— 白等，且把「查不到」拖成「超时」。"""
    fake = _fake_tool(error=ValueError("该订单号不存在"))

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=3,
        resolve=fake,
    )

    assert result.ok is False
    assert result.attempts == 1
    assert "该订单号不存在" in result.error


async def test_execute_tool_never_raises_when_the_resolver_itself_explodes(repos):
    """最后一道保险：连「按名字取工具」这一步炸了也要转成失败结果。

    真实场景：注册表构造工具时抛异常（比如某个工具模块 import 期出问题）。
    这时还没有 tool 可校验，但不能让异常冒到 SSE 连接上。
    """

    def exploding_resolver(_name, _repos, _retriever=None):
        raise SystemExit(1)

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=exploding_resolver,
    )

    assert result.ok is False
    assert result.attempts == 0


# ---------------------------------------------------------------------------
# 瞬时错误分类
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        asyncio.TimeoutError(),
        ConnectionError("reset"),
        ConnectionResetError("reset"),
        TimeoutError("timed out"),
    ],
)
def test_transient_errors_are_classified_as_retryable(exc):
    assert is_transient(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("该订单号不存在"),
        KeyError("missing"),
        TypeError("bad type"),
        RuntimeError("随便什么业务错误"),
    ],
)
def test_business_errors_are_not_retryable(exc):
    """``RuntimeError`` 也算不可重试：工具与仓库用它表示确定性失败
    （比如「会话缺少 tool_call_id」），重试不会改变结果。"""
    assert is_transient(exc) is False


def test_sqlalchemy_operational_error_is_transient():
    """数据库连接抖动是典型的瞬时错误，值得重试一次。"""
    from sqlalchemy.exc import OperationalError

    assert is_transient(OperationalError("SELECT 1", {}, Exception("server has gone away")))


def test_sqlalchemy_interface_error_is_transient():
    from sqlalchemy.exc import InterfaceError

    assert is_transient(InterfaceError("connection already closed", {}, Exception()))


def test_transient_classification_follows_the_exception_chain():
    """真实故障常被包一层（``ToolException`` 里套 ``OperationalError``）。
    只看最外层会把可重试的判成不可重试。"""
    from sqlalchemy.exc import OperationalError

    inner = OperationalError("SELECT 1", {}, Exception("gone away"))
    try:
        try:
            raise inner
        except OperationalError as exc:
            raise RuntimeError("工具包装后的错误") from exc
    except RuntimeError as wrapped:
        assert is_transient(wrapped) is True


def test_is_transient_survives_a_self_referencing_chain():
    """异常链可能有环（自己 cause 自己）。遍历必须去重，否则死循环。"""
    exc = RuntimeError("loop")
    exc.__cause__ = exc
    assert is_transient(exc) is False


# ---------------------------------------------------------------------------
# 结果序列化与脱敏
# ---------------------------------------------------------------------------


def test_error_text_is_redacted_on_the_way_to_the_model():
    """数据库异常的消息里可能带连接串（含明文密码），回灌给模型前必须脱敏。

    模型会把工具结果原样复述给用户 —— 没脱敏等于把库密码泄露到聊天窗口。
    """
    result = ToolResult(
        tool="query_faq",
        args={},
        ok=False,
        data=None,
        error="conn failed api_key=sk-abcdef123456",
        attempts=1,
        duration_ms=1,
    )

    assert "sk-abcdef123456" not in to_tool_message_content(result)


async def test_execute_tool_redacts_the_error_it_stores(repos):
    """脱敏要在**执行器内部**做，不能只靠序列化那一刻。

    ``ToolResult.error`` 还会被送去落库与打日志，那里同样不该出现密钥。
    """
    fake = _fake_tool(error=RuntimeError("auth failed token=sk-live-abcdef123456"))

    result = await execute_tool(
        "query_order",
        {"order_id": "1"},
        repositories=repos,
        timeout=TIMEOUT,
        max_retries=0,
        resolve=fake,
    )

    assert "sk-live-abcdef123456" not in result.error


def test_tool_message_content_is_valid_json_with_ok_flag():
    result = ToolResult(
        tool="query_order",
        args={"order_id": "1001"},
        ok=True,
        data='{"order_id": "1001"}',
        error=None,
        attempts=1,
        duration_ms=3,
    )

    payload = json.loads(to_tool_message_content(result))

    assert payload["ok"] is True
    assert payload["tool"] == "query_order"
    assert payload["data"] == {"order_id": "1001"}


def test_tool_message_content_keeps_data_as_a_string_when_not_json():
    """工具返回的不是 JSON 时不能丢内容，也不能抛 —— 原样转成字符串。"""
    result = ToolResult(
        tool="query_x", args={}, ok=True, data="纯文本结果", error=None, attempts=1, duration_ms=1
    )

    payload = json.loads(to_tool_message_content(result))
    assert payload["data"] == "纯文本结果"


def test_oversized_tool_result_is_truncated_and_flagged():
    """超大工具结果会把上下文预算一次吃光，必须截断并标记。

    断言的是「载荷被限到上限内 + 有 ``truncated`` 标记」，而不是整个 JSON 串的长度：
    外壳字段（``ok`` / ``tool`` / ``truncated``）本来就要占几十个字符，
    把外壳也算进上限会让上限的含义变得含糊。
    """
    result = ToolResult(
        tool="query_faq",
        args={},
        ok=True,
        data="x" * (MAX_TOOL_RESULT_CHARS * 3),
        error=None,
        attempts=1,
        duration_ms=1,
    )

    payload = json.loads(to_tool_message_content(result))

    assert payload["truncated"] is True
    assert len(payload["data"]) <= MAX_TOOL_RESULT_CHARS


def test_truncation_flag_is_absent_when_nothing_was_cut():
    result = ToolResult(
        tool="query_faq", args={}, ok=True, data="短", error=None, attempts=1, duration_ms=1
    )

    assert "truncated" not in json.loads(to_tool_message_content(result))


def test_to_tool_message_content_never_raises_on_unserialisable_data():
    """工具返回了不可序列化的对象时不能炸 —— 兜底成字符串。

    这一层是「回灌给模型」的最后一道，它抛异常就等于把 SSE 连接打断。
    """

    class Weird:
        def __repr__(self):
            return "<weird>"

    result = ToolResult(
        tool="query_x", args={}, ok=True, data=Weird(), error=None, attempts=1, duration_ms=1
    )

    assert "weird" in to_tool_message_content(result)


def test_failed_result_content_carries_the_error_not_data():
    result = ToolResult(
        tool="query_x", args={}, ok=False, data=None, error="查不到", attempts=1, duration_ms=1
    )

    payload = json.loads(to_tool_message_content(result))
    assert payload["ok"] is False
    assert payload["error"] == "查不到"
    assert "data" not in payload
