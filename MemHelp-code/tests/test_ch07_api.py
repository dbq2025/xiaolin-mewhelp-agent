"""Task 11：两个只读会话接口 + 启动预算自检。

这组用例守三件事：

1. 侧栏要的形状（首问预览、已摘要标记、新在前）一次查询就能给全，
   不让前端为每个会话再打一发请求；
2. ``{conversation_id}`` 是 ``conversations.id`` 而不是 ``session_id`` ——
   写错了接口看着能通，但前端一定会传成 session_id；
3. 只读接口真的只读，且预算连一轮都装不下时打 ERROR 却**不阻断启动**。
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app import main
from app.context.budget import INSUFFICIENT_BUDGET_PREFIX, compute_budget
from tests.fakes_agent import ScriptedChatModel


def _client(test_settings, *, model=None):
    return TestClient(
        main.create_app(
            model=model or ScriptedChatModel(script=[]), settings=test_settings
        )
    )


def _seed(repositories, *, session_id: str, user_id: str = "anonymous", first: str = "耳机什么时候到"):
    """建一个会话、塞两条消息，返回 conversation_id。"""
    import asyncio

    async def _go():
        conversation_id = await repositories.conversations.upsert_by_session(
            session_id, user_id
        )
        await repositories.messages.append(
            conversation_id, role="user", content=first
        )
        await repositories.messages.append(
            conversation_id, role="assistant", content="我帮您查一下。"
        )
        return conversation_id

    return asyncio.run(_go())


def _sync(afn):
    """在同步测试里跑协程。

    ``pytest-asyncio`` 只在 ``async def`` 用例里给事件循环，而这些用例全走
    ``TestClient``（同步）。这里要的只是「把协程跑完」，用独立循环即可。
    """
    import asyncio

    return asyncio.run(afn())


@pytest.fixture()
def seeded(sqlite_session, fake_repositories):
    """三个会话：两个属于 anonymous，一个新的在前。"""
    first = _seed(fake_repositories, session_id="s-old", first="订单 1001 到哪了")
    second = _seed(fake_repositories, session_id="s-new", first="耳机怎么退货")
    return {"ids": [second, first], "repositories": fake_repositories}


def test_list_conversations_returns_newest_first(test_settings, seeded, sqlite_session):
    client = _client(test_settings)
    resp = client.get("/api/conversations", params={"user_id": "anonymous"})
    assert resp.status_code == 200
    items = resp.json()["conversations"]
    assert [item["session_id"] for item in items] == ["s-new", "s-old"]
    # 首问预览：拿第一条 user 消息，不是最新一条
    assert items[0]["preview"] == "耳机怎么退货"
    assert items[1]["preview"] == "订单 1001 到哪了"
    assert items[0]["message_count"] == 2
    assert items[0]["summarized"] is False
    assert items[0]["summary_segments"] == 0
    # id 是 conversations.id，不是 session_id —— 前端拿它拼详情接口
    assert items[0]["id"] == seeded["ids"][0]
    assert items[0]["id"] != items[0]["session_id"]


def test_list_conversations_marks_summarized(test_settings, seeded):
    repositories = seeded["repositories"]
    conversation_id = seeded["ids"][1]

    async def _add():
        await repositories.conversations.set_anchors(
            conversation_id=conversation_id,
            summary_upto_msg_id=1,
            layer1_from_msg_id=2,
        )
        await repositories.summaries.append(
            conversation_id=conversation_id,
            segment_no=1,
            from_msg_id=1,
            upto_msg_id=1,
            content="用户问过订单 1001 的到货时间。",
            msg_count=1,
            source_tokens=30,
        )

    _sync(_add)
    resp = _client(test_settings).get("/api/conversations")
    items = resp.json()["conversations"]
    marked = [item for item in items if item["session_id"] == "s-old"][0]
    assert marked["summarized"] is True
    assert marked["summary_segments"] == 1
    assert marked["summary_upto_msg_id"] == 1
    plain = [item for item in items if item["session_id"] == "s-new"][0]
    assert plain["summarized"] is False


def test_list_conversations_limit_is_applied(test_settings, seeded):
    for index in range(5):
        _seed(seeded["repositories"], session_id=f"s-extra-{index}", first=f"第 {index} 个")
    resp = _client(test_settings).get("/api/conversations", params={"limit": 3})
    assert resp.status_code == 200
    assert len(resp.json()["conversations"]) == 3


def test_list_conversations_isolates_users(test_settings, seeded):
    _seed(seeded["repositories"], session_id="s-other", user_id="someone-else")
    resp = _client(test_settings).get("/api/conversations", params={"user_id": "anonymous"})
    session_ids = [item["session_id"] for item in resp.json()["conversations"]]
    assert "s-other" not in session_ids


def test_conversation_messages_round_trips_the_ui_shape(test_settings, seeded):
    conversation_id = seeded["ids"][1]
    resp = _client(test_settings).get(f"/api/conversations/{conversation_id}/messages")
    assert resp.status_code == 200
    body = resp.json()
    # 与 POST /api/chat/sessions/{session_id}/messages 同形状：human / ai，不是 user / assistant
    assert [m["role"] for m in body["messages"]] == ["human", "ai"]
    assert body["messages"][0]["content"] == "订单 1001 到哪了"
    assert body["session_id"] == "s-old"


def test_unknown_conversation_returns_404(test_settings, seeded):
    resp = _client(test_settings).get("/api/conversations/999999/messages")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found"


def test_read_only_endpoints_do_not_mutate_anything(test_settings, seeded, sqlite_session):
    repositories = seeded["repositories"]

    async def _counts():
        return (
            len(await repositories.messages.list_for_conversation(seeded["ids"][0])),
            len(await repositories.messages.list_for_conversation(seeded["ids"][1])),
            await repositories.conversations.get_anchors(seeded["ids"][0]),
        )

    before = _sync(_counts)
    client = _client(test_settings)
    client.get("/api/conversations")
    client.get(f"/api/conversations/{seeded['ids'][0]}/messages")
    after = _sync(_counts)
    assert before == after


def test_startup_budget_check_logs_error_for_tiny_window(caplog):
    """spec §4.2：连一轮都装不下要打 ERROR，但**不阻断启动**。"""
    from app.config import Settings

    tight = Settings(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="fake-model",
        model_context_window=12000,
        max_output_tokens=2000,
        max_user_input_tokens=2000,
        agent_max_steps=3,
        tool_result_max_tokens=1200,
        rag_rerank_top_n=5,
        turn_log_path="",
        app_log_path="",
    )
    report = compute_budget(tight)
    assert report.ok is False
    with caplog.at_level(logging.INFO):
        with TestClient(
            main.create_app(model=ScriptedChatModel(script=[]), settings=tight)
        ) as client:
            # 启动没被打断
            assert client.get("/healthz").status_code == 200
    assert any(
        INSUFFICIENT_BUDGET_PREFIX in record.getMessage() for record in caplog.records
    )
