"""非流式接口测试。

验收标准 2（第二轮接住上下文）在 HTTP 层的证据：两轮之后用
`POST /api/chat/sessions/{id}/messages` 能看到完整历史。
验收标准 3（拿到结构化 JSON）在 HTTP 层的证据：`POST /api/extract`。

`create_app(model=..., store=..., settings=...)` 支持注入假模型、独立会话存储和
写死的配置，所以这些测试既不联网、不读 .env，也互不干扰。

**ch02 起**：聊天记录改走数据库，所以 ``conftest.py`` 的 ``_offline_database``
自动注入一个接到 SQLite 内存库上的仓库。这里的用例因此不需要知道数据库存在。

**注意本文件用 ``app_main.create_app`` 而不是直接 import 函数** —— 后者在 import
时就把名字绑定了，conftest 的包装器对它无效（见 ``_offline_database`` 的说明）。
"""

from fastapi.testclient import TestClient

from app import main as app_main
from app.config import Settings
from app.memory import InMemorySessionStore
from tests.fakes import FakeChatModel


def _client(test_settings, model=None, store=None) -> TestClient:
    return TestClient(
        app_main.create_app(
            model=model or FakeChatModel(reply="好的"),
            store=store,
            settings=test_settings,
        )
    )


def test_healthz_does_not_leak_api_key(test_settings):
    """密钥不得出现在任何响应里。"""
    body = _client(test_settings).get("/healthz").json()
    assert body["status"] == "ok"
    assert body["base_url_configured"] is True
    assert "api_key" not in str(body).lower()
    assert "sk-" not in str(body)


def test_lifespan_honours_rag_warmup_from_settings_when_not_given(test_settings):
    """``create_app(rag_warmup=None)`` 必须读配置里的 ``RAG_WARMUP``。

    否则 ``.env`` 里的 ``RAG_WARMUP=false`` 是个死开关：运维关掉预热后
    启动时仍然会去连一次嵌入服务。
    """
    calls: list[str] = []

    class _Retriever:
        async def search(self, query, top_k=None):
            return []

        async def warmup(self) -> None:
            calls.append("warmup")

    off = test_settings.model_copy(update={"rag_warmup": False})
    with TestClient(
        app_main.create_app(
            model=FakeChatModel(reply="好"), settings=off, retriever=_Retriever(), rag_warmup=None
        )
    ) as client:
        client.get("/healthz")
    assert calls == []

    on = test_settings.model_copy(update={"rag_warmup": True})
    with TestClient(
        app_main.create_app(
            model=FakeChatModel(reply="好"), settings=on, retriever=_Retriever(), rag_warmup=None
        )
    ) as client:
        client.get("/healthz")
    assert calls == ["warmup"]


def test_lifespan_warms_up_the_retriever_only_when_enabled(test_settings):
    """``rag_warmup`` 是生产与测试的分界线。

    生产默认开启：BGE-M3 首载 20-40s，不预热的话第一个问句必然打穿 3s 工具超时。
    测试必须关掉：否则每次跑套件都要加载 2.3GB 模型。
    这里用 ``TestClient`` 的上下文管理器触发 lifespan，两个方向都钉住。
    """
    calls: list[str] = []

    class _Retriever:
        async def search(self, query, top_k=5):
            return []

        async def warmup(self) -> None:
            calls.append("warmup")

    with TestClient(
        app_main.create_app(
            model=FakeChatModel(reply="好"),
            settings=test_settings,
            retriever=_Retriever(),
            rag_warmup=False,
        )
    ) as client:
        client.get("/healthz")
    assert calls == []

    with TestClient(
        app_main.create_app(
            model=FakeChatModel(reply="好"),
            settings=test_settings,
            retriever=_Retriever(),
            rag_warmup=True,
        )
    ) as client:
        client.get("/healthz")
    assert calls == ["warmup"]


def test_healthz_reports_model_and_budget(test_settings):
    body = _client(test_settings).get("/healthz").json()
    assert body["model"]
    assert body["context_token_budget"] > 0


def test_healthz_misconfigured_when_env_missing(monkeypatch):
    """没有 .env 时 /healthz 要报 misconfigured，且**状态码是 503**。

    返回 200 会让编排器以为实例健康、持续打流量进来，而每个请求都会 503。
    """
    from app.config import get_settings

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    # 把 .env 指到一个不存在的路径，模拟「干净克隆、还没配 .env」
    monkeypatch.setattr(
        "app.config.Settings.model_config",
        {**Settings.model_config, "env_file": "/nonexistent/.env"},
    )
    get_settings.cache_clear()

    resp = TestClient(app_main.create_app()).get("/healthz")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "misconfigured"
    assert body["base_url_configured"] is False
    assert body["model"] is None
    assert "hint" in body


def test_healthz_misconfigured_body_does_not_leak_partial_key(monkeypatch):
    """配置错误里 pydantic 会带出 input_value（含密钥前几位），必须脱敏。"""
    from app.config import get_settings

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-partial")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(
        "app.config.Settings.model_config",
        {**Settings.model_config, "env_file": "/nonexistent/.env"},
    )
    get_settings.cache_clear()

    body = TestClient(app_main.create_app()).get("/healthz").text
    assert "sk-super-secret" not in body
    assert "sk-replace" not in body


def test_chat_returns_503_configuration_error_when_env_missing(monkeypatch):
    """配置缺失时对话接口要给明确的 503 + configuration_error，而不是 500。

    这是对计划 Task 9 里「配置缺失在启动时就报错退出」的**有意偏离**：
    改成延迟到请求时返回 503 —— 这样 `import app.main` 在没有 .env 的环境下
    也能成功（CI、文档生成、单元测试都需要），而使用者拿到的仍是可诊断的错误。
    """
    from app.config import get_settings

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr("app.config.Settings.model_config", {**Settings.model_config, "env_file": None})
    get_settings.cache_clear()

    resp = TestClient(app_main.create_app()).post(
        "/api/chat/completions", json={"message": "在吗", "stream": False}
    )
    assert resp.status_code == 503
    body = resp.json()
    # 必须和本项目其他错误一样走 {"error": {...}} 外壳。
    # 早期实现用 HTTPException，会得到 FastAPI 默认的 {"detail": {...}}，
    # 同一个接口在配置缺失时错误体形状不一致 —— 这条断言就是防它回退的。
    assert "error" in body
    assert body["error"]["type"] == "configuration_error"


def test_non_streaming_chat_returns_full_json(test_settings):
    resp = _client(test_settings, FakeChatModel(reply="您好，我帮您查一下")).post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "在吗", "stream": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "s1"
    assert body["content"] == "您好，我帮您查一下"
    assert body["context"]["budget_tokens"] > 0
    assert body["context"]["kept_messages"] == 0
    assert body["context"]["trimmed_messages"] == 0


def test_non_streaming_chat_generates_session_id_when_absent(test_settings):
    body = _client(test_settings).post(
        "/api/chat/completions", json={"message": "在吗", "stream": False}
    ).json()
    assert len(body["session_id"]) == 12


def test_second_turn_history_is_visible_via_sessions_endpoint(test_settings):
    """验收标准 2 的 HTTP 层证据。

    走流式路径（不传 ``stream``，默认 true）。**注意：两条路径都会落库** ——
    它们共用同一套单轮编排（``app/api.py`` 的 ``astream_chat_with_tools`` 与
    ``complete_chat_with_tools``），所以历史来源唯一。
    非流式的落库由 ``tests/test_api_tools.py::test_non_streaming_turn_is_also_persisted``
    单独钉住。

    （早期这里曾写「只有流式路径会落库」——那是错的，而且会让后来者按它去删掉
    共享编排这个不变量。已改正。）
    """
    client = _client(test_settings, FakeChatModel(reply="答"))
    for turn in ("第一轮", "第二轮"):
        client.post(
            "/api/chat/completions",
            json={"session_id": "s1", "message": turn},
        )
    msgs = client.post("/api/chat/sessions/s1/messages").json()["messages"]
    assert [m["content"] for m in msgs] == ["第一轮", "答", "第二轮", "答"]
    assert [m["role"] for m in msgs] == ["human", "ai", "human", "ai"]


def test_unknown_session_returns_empty_history(test_settings):
    body = _client(test_settings).post("/api/chat/sessions/none/messages").json()
    assert body == {"session_id": "none", "messages": []}


def test_context_stats_reflect_trimming(test_settings):
    """历史塞满小预算后，响应里的统计要如实反映裁剪。

    用一个很小的预算（而不是默认 2000）：`count_tokens_approximately` 对重复
    中文估得偏低，用默认预算根本撑不满，测不出裁剪。
    """
    store = InMemorySessionStore()
    small_budget = test_settings.model_copy(update={"context_token_budget": 40})
    client = _client(small_budget, FakeChatModel(reply="答" * 20), store=store)
    for i in range(5):
        client.post(
            "/api/chat/completions",
            json={"session_id": "s1", "message": f"第{i}轮问题", "stream": False},
        )
    body = client.post(
        "/api/chat/completions",
        json={"session_id": "s1", "message": "继续", "stream": False},
    ).json()
    assert body["context"]["trimmed_messages"] > 0
    assert body["context"]["budget_tokens"] == 40


def test_extract_endpoint_returns_structured_json(test_settings):
    """验收标准 3 的 HTTP 层证据。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": "20240115001",
            "request_type": "logistics",
            "expected_solution": "refund",
            "detail": "订单未按时送达",
        }
    )
    resp = _client(test_settings, model).post(
        "/api/extract", json={"text": "订单 20240115001 没按时到，我要退款"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["order_no"] == "20240115001"
    assert body["request_type"] == "logistics"
    assert body["expected_solution"] == "refund"
    assert body["detail"] == "订单未按时送达"


def test_extract_endpoint_returns_null_order_no_when_absent(test_settings):
    """Review Focus 4 在 HTTP 层的证据：不编造订单号。"""
    model = FakeChatModel(
        structured_payload={
            "order_no": None,
            "request_type": "logistics",
            "expected_solution": "unknown",
            "detail": "快递没动",
        }
    )
    body = _client(test_settings, model).post(
        "/api/extract", json={"text": "我的快递一直没动"}
    ).json()
    assert body["order_no"] is None


def test_extract_endpoint_422_when_model_output_unparsable(test_settings):
    model = FakeChatModel(reply="我不太明白")
    resp = _client(test_settings, model).post("/api/extract", json={"text": "x"})
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "extraction_failed"


def test_empty_message_is_rejected(test_settings):
    resp = _client(test_settings).post(
        "/api/chat/completions", json={"message": "", "stream": False}
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_unknown_session_id_is_rejected(test_settings):
    resp = _client(test_settings).post(
        "/api/chat/completions",
        json={"session_id": "with spaces!", "message": "hi", "stream": False},
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["type"] == "validation_error"


def test_sessions_are_isolated_over_http(test_settings):
    """Review Focus 2 在 HTTP 层：甲的历史不能出现在乙的会话里。

    走流式路径（ch02 起只有它落库，见上一条的说明）。会话隔离现在由
    ``messages.conversation_id`` 保证 —— 比内存字典更硬：跨进程、重启后都成立。
    """
    client = _client(test_settings, FakeChatModel(reply="答"))
    client.post(
        "/api/chat/completions",
        json={"session_id": "a", "message": "甲的问题"},
    )
    client.post(
        "/api/chat/completions",
        json={"session_id": "b", "message": "乙的问题"},
    )
    a = [m["content"] for m in client.post("/api/chat/sessions/a/messages").json()["messages"]]
    b = [m["content"] for m in client.post("/api/chat/sessions/b/messages").json()["messages"]]
    assert a == ["甲的问题", "答"]
    assert b == ["乙的问题", "答"]
