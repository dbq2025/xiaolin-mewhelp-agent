"""ch02 的 HTTP 层测试：SSE 新契约、``/messages`` 读库、``/healthz`` 探库。

全部离线：注入假模型 + 接到 SQLite 内存库上的真仓库，不发网络请求、不连 MySQL。

## 这一层要钉住的三件事

1. **SSE 契约**：``status`` 帧在 ``token`` 之前到达；``done.tools`` 带完整成败信息。
   前端徽章完全依赖它 —— 契约漂移的表现是「徽章不显示」，而页面看起来正常。
2. **``/messages`` 读库且带 ``tools``**：徽章**刷新后仍在**的唯一保证。
3. **数据库故障的降级**：``/healthz`` 必须 503 且带可判读的字段，不能裸 500。
"""

import json
from typing import Any

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app import main
from tests.fakes import FakeChatModel


def _tool_call(name="query_logistics", args=None, call_id="call_1"):
    return {
        "name": name,
        "args": args if args is not None else {"order_id": "1001"},
        "id": call_id,
        "type": "tool_call",
    }


def _client(test_settings, fake_repositories, model=None, db_ping=None) -> TestClient:
    # 必须写 ``main.create_app``（不能 ``from app.main import create_app``）：
    # 后者在 import 时就把名字绑死了，conftest 的包装器（注入 SQLite 仓库、
    # 空检索器、rag_warmup=False）对它无效 —— 这正是 agent.md 里写过的坑。
    return TestClient(
        main.create_app(
            model=model or FakeChatModel(chunks=["好"]),
            settings=test_settings,
            repositories=fake_repositories,
            db_ping=db_ping,
        )
    )


def _stream(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, dict | None]]:
    with client.stream("POST", "/api/chat/completions", json=payload) as resp:
        text = "".join(resp.iter_text())
        assert resp.status_code == 200
    return _events(text)


def _events(text: str) -> list[tuple[str, dict | None]]:
    events: list[tuple[str, dict | None]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data: str | None = None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((name or "", json.loads(data) if data else None))
    return events


def _tool_model(**overrides) -> FakeChatModel:
    kwargs: dict[str, Any] = {
        "chunks": ["已", "发货"],
        "tool_calls_then_stream": [_tool_call()],
        "reply": json.dumps(
            {"status": "运输中", "carrier": "顺丰速运"}, ensure_ascii=False
        ),
    }
    kwargs.update(overrides)
    return FakeChatModel(**kwargs)


# ---------------------------------------------------------------------------
# SSE 契约：status 帧
# ---------------------------------------------------------------------------


def test_status_frame_arrives_before_the_first_token(test_settings, fake_repositories):
    """用户在工具执行那几秒里要先看到「正在查物流…」，而不是一片空白。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    names = [n for n, _ in events]

    assert "status" in names
    assert names.index("status") < names.index("token")
    assert names[0] == "meta"
    assert names[-1] == "done"


def test_status_frame_payload_is_complete(test_settings, fake_repositories):
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    status = next(d for n, d in events if n == "status") or {}

    assert status["tool"] == "query_logistics"
    assert status["args"] == {"order_id": "1001"}
    assert status["label"].endswith("…")


def test_no_status_frame_when_the_model_does_not_call_a_tool(test_settings, fake_repositories):
    """不调工具是正常路径，不该有 status 帧（否则界面会闪一下假的「正在查询」）。"""
    events = _stream(
        _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"])),
        {"session_id": "s1", "message": "你好"},
    )
    assert "status" not in [n for n, _ in events]


# ---------------------------------------------------------------------------
# SSE 契约：done.tools
# ---------------------------------------------------------------------------


def test_done_frame_carries_the_tools_array(test_settings, fake_repositories):
    """``done.tools`` 是徽章的权威来源。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    done = next(d for n, d in events if n == "done") or {}

    assert isinstance(done["tools"], list)
    assert done["tools"][0]["name"] == "query_logistics"
    assert done["tools"][0]["ok"] is True
    assert "duration_ms" in done["tools"][0]


def test_done_frame_has_an_empty_tools_array_without_a_tool_call(test_settings, fake_repositories):
    events = _stream(
        _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"])),
        {"session_id": "s1", "message": "你好"},
    )
    done = next(d for n, d in events if n == "done") or {}
    assert done["tools"] == []


def test_done_frame_still_reports_context_stats(test_settings, fake_repositories):
    """ch01 的 ``trimmed`` / ``kept`` 不能被 ch02 挤掉 —— 前端统计开关还在用。"""
    events = _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "订单 1001 的物流到哪了"},
    )
    done = next(d for n, d in events if n == "done") or {}

    assert done["finish_reason"] == "stop"
    assert done["trimmed"] == 0
    assert done["kept"] == 0
    assert done["session_id"] == "s1"


def test_every_ch02_sse_data_field_is_valid_json(test_settings, fake_repositories):
    for name, data in _stream(
        _client(test_settings, fake_repositories, _tool_model()),
        {"session_id": "s1", "message": "查物流"},
    ):
        assert isinstance(data, dict), f"{name} 的 data 不是 JSON 对象"


# ---------------------------------------------------------------------------
# 落库与 /messages 读库
# ---------------------------------------------------------------------------


def test_tool_turn_is_persisted_and_readable_from_messages(test_settings, fake_repositories):
    """四行落库 + ``/messages`` 能读回来（含工具轨迹）。"""
    client = _client(test_settings, fake_repositories, _tool_model())
    _stream(client, {"session_id": "s1", "message": "订单 1001 的物流到哪了"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]

    assert [r["role"] for r in rows] == ["human", "ai", "tool", "ai"]
    assert rows[0]["content"] == "订单 1001 的物流到哪了"
    assert rows[3]["content"] == "已发货"


def test_messages_exposes_tools_for_each_assistant_row(test_settings, fake_repositories):
    """徽章**刷新后仍在**的唯一保证：``/messages`` 要回传每轮调了什么工具。"""
    client = _client(test_settings, fake_repositories, _tool_model())
    _stream(client, {"session_id": "s1", "message": "订单 1001 的物流到哪了"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    tool_turn = next(r for r in rows if r.get("tools"))

    assert tool_turn["tools"][0]["name"] == "query_logistics"
    assert tool_turn["tools"][0]["ok"] is True


def test_messages_for_a_plain_turn_has_no_tools(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    _stream(client, {"session_id": "s1", "message": "你好"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert all(not r.get("tools") for r in rows)


def test_messages_is_empty_for_an_unknown_session(test_settings, fake_repositories):
    """没聊过的会话返回空列表，而不是 404 —— 聊天页启动时会探一次。"""
    client = _client(test_settings, fake_repositories)
    resp = client.post("/api/chat/sessions/never-seen/messages")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []


def test_multi_turn_history_is_visible_on_the_second_turn(test_settings, fake_repositories):
    """第二轮必须能看到第一轮 —— 历史来自库而不是进程内存。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    _stream(client, {"session_id": "s1", "message": "第一句"})
    _stream(client, {"session_id": "s1", "message": "第二句"})

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == ["第一句", "你好", "第二句", "你好"]


# ---------------------------------------------------------------------------
# 非流式路径
# ---------------------------------------------------------------------------


def test_non_streaming_chat_still_works(test_settings, fake_repositories):
    """ch01 的 ``stream=false`` 契约不能破。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=["你好"]))
    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "你好"
    assert body["context"]["budget_tokens"] == 2000


def test_non_streaming_empty_response_is_still_an_error(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, FakeChatModel(chunks=[], reply=""))
    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "upstream_empty_response"


def test_non_streaming_turn_is_also_persisted(test_settings, fake_repositories):
    """非流式也必须落库。

    **为什么这条必须有**：ch02 的两条路（流式 / 非流式）如果历史来源不一致
    （一个读库、一个读内存），同一个 ``session_id`` 在两条路上会看到不同的历史 ——
    表现为「偶发答非所问」，而且极难复现。所以两条路共用同一个编排。
    """
    client = _client(test_settings, fake_repositories, FakeChatModel(reply="好的"))
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == ["在吗", "好的"]


def test_streaming_and_non_streaming_share_the_same_history(test_settings, fake_repositories):
    """两条路混用同一个会话时，历史必须是同一条。"""
    client = _client(test_settings, fake_repositories, FakeChatModel(reply="好的"))
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "流式那句"},
    )
    client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "非流式那句", "stream": False},
    )

    rows = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [r["content"] for r in rows] == [
        "流式那句",
        "好的",
        "非流式那句",
        "好的",
    ]


# ---------------------------------------------------------------------------
# /healthz 探数据库（Review Focus 4）
# ---------------------------------------------------------------------------


def test_healthz_reports_database_ok_when_ping_succeeds(test_settings, fake_repositories):
    client = _client(test_settings, fake_repositories, db_ping=lambda: True)
    body = client.get("/healthz").json()

    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_healthz_returns_503_when_the_database_is_down(test_settings, fake_repositories):
    """Review Focus 4：库挂了必须 503，且字段可判读 —— 不能裸 500。

    200 会让编排器继续把流量打进来，而每个请求都会失败。
    """
    client = _client(test_settings, fake_repositories, db_ping=lambda: False)
    resp = client.get("/healthz")

    assert resp.status_code == 503
    body = resp.json()
    assert body["database"] == "unavailable"
    assert body["status"] != "ok"


def test_healthz_database_failure_does_not_leak_the_dsn(test_settings, fake_repositories):
    """响应里不能出现连接串 —— 它含明文密码。"""
    client = _client(test_settings, fake_repositories, db_ping=lambda: False)
    raw = client.get("/healthz").text

    assert "mysql+pymysql" not in raw
    assert "mewhelp:mewhelp" not in raw


def test_healthz_without_injected_ping_still_answers(test_settings, fake_repositories):
    """没注入 ``db_ping`` 时走真实探测。

    这里用真仓库（SQLite）—— 探测会打到 SQLite 上，所以本用例不该依赖 MySQL。
    但 ``create_app`` 在没注入 repositories 时会去连 MySQL，所以这里仍注入仓库，
    只让 ``db_ping`` 缺省，验证「缺省路径不会崩」。
    """
    client = _client(test_settings, fake_repositories)
    resp = client.get("/healthz")
    assert resp.status_code in (200, 503)
    assert "database" in resp.json()


# ---------------------------------------------------------------------------
# 未注入仓库时的 Session 生命周期
# ---------------------------------------------------------------------------


def test_real_dependency_path_closes_its_session(monkeypatch, test_settings):
    """走**真实** ``get_repositories`` 时，请求结束后 Session 必须被关闭。

    **为什么必须有这条**：``create_app`` 不注入仓库时，``get_repositories`` 每请求
    新建一个 ``Session``。若从不关闭它，连接要等这个 Session 被 GC 才回到池里 ——
    在持续流量下表现为「MySQL 连接数缓慢涨到上限」，排查起来很痛苦
    （看起来像连接池配置问题，实际是生命周期问题）。

    **为什么之前没测出来**：所有 HTTP 用例都**注入**了仓库（``conftest`` 的包装器
    默认注入 SQLite），于是真实那一条路径从未被跑到。这里用一个假的
    ``session_factory`` 把它跑到，并断言 ``close()`` 被调用。
    """
    from app import api as api_module

    closed: list[bool] = []

    class FakeSession:
        def close(self):
            closed.append(True)

    class FakeFactory:
        def __call__(self):
            return FakeSession()

    monkeypatch.setattr(api_module, "session_factory", lambda _engine: FakeFactory())
    monkeypatch.setattr(api_module, "create_engine_from", lambda _settings: object())

    from app import main as app_module

    app = app_module.create_app(model=FakeChatModel(chunks=["好"]), settings=test_settings)
    # 显式绕过 conftest 的默认注入：这里要测的正是「没注入」那条路
    app.state.repositories = None

    with TestClient(app) as client:
        client.post(
            "/api/chat/completions",
            json={"session_id": "s1", "message": "在吗"},
        )

    assert closed, (
        "请求结束后 Session 没有被 close() —— 连接要等 GC 才回池，"
        "持续流量下会把 MySQL 连接数顶到上限"
    )


# ---------------------------------------------------------------------------
# 数据库故障的分类（Review I2）
# ---------------------------------------------------------------------------


def test_healthz_reuses_one_engine_across_calls(monkeypatch, test_settings):
    """**Review I3**：``/healthz`` 不能每次都新建引擎（也就新建一个连接池）。

    ``/healthz`` 不是只在启动时跑 —— 聊天页**每次加载**都会探它。
    每次新建引擎 = 每次泄漏一个连接池 + 一次完整的 MySQL 连接握手，
    而且连不上时要等驱动的连接超时（PyMySQL 默认 10 秒）。
    在持续流量下表现为「MySQL 连接数缓慢涨到上限 + 健康检查偶发很慢」。

    这条断言「建引擎只发生一次」—— 与 ``get_repositories`` 共用同一个缓存的工厂。
    """
    from app import api as api_module
    from app import main as app_module

    created: list[object] = []

    class FakeEngine:
        def connect(self):
            raise AssertionError("不应真的连接（用了假引擎）")

        def dispose(self):
            pass

    class FakeFactory:
        def __call__(self):
            raise AssertionError("探测不该新建 Session")

    def counting_create_engine(_settings):
        created.append(object())
        return FakeEngine()

    monkeypatch.setattr(api_module, "create_engine_from", counting_create_engine)
    monkeypatch.setattr(api_module, "session_factory", lambda _engine: FakeFactory())
    # 让真实探测走假引擎：这里只关心「建了几次引擎」
    async def fake_ping(_engine):
        return True

    monkeypatch.setattr(api_module, "ping", fake_ping)

    app = app_module.create_app(model=FakeChatModel(chunks=["好"]), settings=test_settings)
    app.state.repositories = None
    app.state.db_ping = None

    with TestClient(app) as client:
        for _ in range(3):
            assert client.get("/healthz").status_code == 200

    assert len(created) == 1, (
        f"3 次 /healthz 建了 {len(created)} 个引擎 —— 应复用同一个（每次新建会泄漏连接池）"
    )


def _db_down_repositories(monkeypatch):
    """让仓库层像「MySQL 挂了」那样抛 SQLAlchemy ``OperationalError``。"""

    class ExplodingMessages:
        async def history_as_messages(self, session_id):
            raise OperationalError(
                "SELECT 1",
                {},
                Exception("(2003, \"Can't connect to MySQL server on '127.0.0.1'\")"),
            )

    class ExplodingConversations:
        async def upsert_by_session(self, session_id, user_id):
            raise OperationalError("INSERT", {}, Exception("server has gone away"))

    class Repos:
        messages = ExplodingMessages()
        conversations = ExplodingConversations()
        faq = None
        tickets = None

    return Repos()


def test_database_fault_is_reported_as_database_error_not_upstream(
    test_settings, fake_repositories, monkeypatch
):
    """**Review I2**：数据库故障必须是 ``database_error``，不能被归成上游错误。

    不改的话：``OperationalError(2003, "Can't connect...")`` 会被
    ``classify_upstream_error_type`` 归成 ``upstream_unavailable``，于是错误信息提示
    运维「请检查 OPENAI_BASE_URL 和网络」—— 把排查方向指向一个**没坏**的依赖。
    数据库是本服务自己的依赖，与「模型服务怎么了」是两类问题。
    """
    client = _client(test_settings, _db_down_repositories(monkeypatch))

    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )

    assert resp.status_code == 503, f"数据库故障应 503（服务未就绪），实际 {resp.status_code}"
    body = resp.json()
    assert body["error"]["type"] == "database_error"


def test_database_fault_message_does_not_leak_driver_internals(
    test_settings, monkeypatch
):
    """响应里不能出现驱动原文 —— 它含库地址、端口、账号，甚至 SQL 语句。

    ``/healthz`` 早就为这个原因把细节收进日志了，聊天接口必须一致。
    """
    client = _client(test_settings, _db_down_repositories(monkeypatch))

    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    raw = resp.text

    assert "127.0.0.1" not in raw
    assert "SQL: SELECT 1" not in raw
    assert "2003" not in raw
    assert "SELECT" not in raw


def test_streaming_database_fault_is_an_sse_error_event_not_a_crash(
    test_settings, monkeypatch
):
    """流式路径：数据库故障要变成 ``error`` 帧，而不是把连接打断。"""
    client = _client(test_settings, _db_down_repositories(monkeypatch))

    events = _stream(client, {"session_id": "s1", "message": "在吗"})
    names = [n for n, _ in events]

    assert names[0] == "meta"
    assert names[-1] == "error"
    err = events[-1][1] or {}
    assert err["type"] == "database_error"


def test_failed_persist_does_not_change_the_reported_fault_type(
    test_settings, fake_repositories, monkeypatch
):
    """上游挂了 + 落库也挂了：对外暴露的必须是**上游**那个故障。

    这条与 ``tests/test_chat_tools.py`` 的同名用例是两层保险：
    那里测用例层不盖掉原始异常，这里测 HTTP 层的错误类型没被落库故障改写。
    """
    from tests.fakes import FakeChatModel

    class ExplodingFirstCall(FakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
            raise RuntimeError("upstream model exploded")

    class FailingMessages:
        async def history_as_messages(self, session_id):
            return []

        async def append(self, *args, **kwargs):
            raise OperationalError("INSERT", {}, Exception("db gone"))

    class Repos:
        messages = FailingMessages()
        conversations = fake_repositories.conversations
        faq = None
        tickets = None

    client = _client(test_settings, Repos(), ExplodingFirstCall(chunks=["好"]))

    resp = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )

    assert resp.status_code == 502, "上游故障应 502，不能被落库故障改写成 503"
    assert resp.json()["error"]["type"] not in ("database_error", "configuration_error")
