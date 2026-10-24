"""ch08 Task 11：建工单确认流（图内 interrupt + resume）。

规则（与 ch05 的按钮建单共存）：
- 只有客户明确要求建单，模型才会发起 ``create_ticket``；
- 参数不全**不执行**，把校验错误回灌给模型让它追问；
- 参数齐全也不执行，图暂停并向用户出示预览；确认才落库，取消按
  「权限拒绝」记审计。
"""

from __future__ import annotations

import json

from langgraph.types import Command
from sqlalchemy import select

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.db.models import Ticket
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn
from tests.test_ch05_graph import _run


def _intent(intent: str = "logistics") -> object:
    return text_turn(json.dumps({"intent": intent, "confidence": 0.9}, ensure_ascii=False))


def _call(args: dict, *, call_id: str = "call_ticket"):
    return tool_turn("create_ticket", args, call_id=call_id)


async def _pause(graph, model, repositories, settings, session: str, args: dict):
    model.script.extend([_intent(), _call(args)])
    values, events, config = await _run(
        graph,
        message="帮我建个工单",
        model=model,
        repositories=repositories,
        retriever=None,
        settings=settings,
        session=session,
    )
    return values, events, config


async def test_missing_description_goes_back_to_model(
    test_settings, fake_repositories, sqlite_session
) -> None:
    """缺必填项：不暂停、不建单，模型收到「description 必填」后追问。"""
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel(
        script=[
            _intent(),
            _call({"ticket_type": "投诉", "conversation_id": "s-missing"}),
            text_turn("请问具体是什么问题？我帮你登记。"),
        ]
    )

    values, _, config = await _run(
        graph,
        message="帮我建个工单",
        model=model,
        repositories=fake_repositories,
        retriever=None,
        settings=test_settings,
        session="s-missing",
    )

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ()
    assert values["answer"] == "请问具体是什么问题？我帮你登记。"
    assert values.get("pending_ticket_call") is None
    assert sqlite_session.scalars(select(Ticket)).all() == []
    traces = values["tool_traces"]
    assert traces and "description" in json.dumps(traces, ensure_ascii=False)


async def test_valid_call_pauses_with_preview(
    test_settings, fake_repositories, sqlite_session
) -> None:
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel()
    args = {
        "description": "耳机左声道没有声音",
        "ticket_type": "售后",
        "conversation_id": "s-preview",
    }

    values, events, config = await _pause(
        graph, model, fake_repositories, test_settings, "s-preview", args
    )

    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("ticket_await_confirm",)
    interrupt_value = snapshot.tasks[0].interrupts[0].value
    assert interrupt_value["kind"] == "ticket_preview"
    assert interrupt_value["preview"] == {
        "ticket_type": "售后",
        "description": "耳机左声道没有声音",
    }
    assert interrupt_value["call_id"] == "call_ticket"
    assert values["pending_ticket_call"]["args"]["description"] == "耳机左声道没有声音"
    assert sqlite_session.scalars(select(Ticket)).all() == []


async def test_confirm_creates_ticket_with_number(
    test_settings, fake_repositories, sqlite_session
) -> None:
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel()
    args = {
        "description": "耳机左声道没有声音",
        "ticket_type": "售后",
        "conversation_id": "s-confirm",
    }
    _, _, config = await _pause(
        graph, model, fake_repositories, test_settings, "s-confirm", args
    )

    async for _ in graph.astream(
        Command(resume={"confirmed": True}), config, stream_mode="custom"
    ):
        pass

    snapshot = await graph.aget_state(config)
    tickets = sqlite_session.scalars(select(Ticket)).all()
    assert len(tickets) == 1
    assert snapshot.values["answer"].find(tickets[0].ticket_no) >= 0
    rows = await fake_repositories.audit.list_recent(limit=5)
    created = [row for row in rows if row.tool_name == "create_ticket"]
    assert created and created[0].status == "成功"
    assert snapshot.next == ()


async def test_cancel_audits_permission_denied(
    test_settings, fake_repositories, sqlite_session
) -> None:
    graph = build_agent_graph(checkpointer=memory_checkpointer())
    model = ScriptedChatModel()
    args = {
        "description": "耳机左声道没有声音",
        "ticket_type": "售后",
        "conversation_id": "s-cancel",
    }
    _, _, config = await _pause(
        graph, model, fake_repositories, test_settings, "s-cancel", args
    )

    async for _ in graph.astream(
        Command(resume={"confirmed": False}), config, stream_mode="custom"
    ):
        pass

    snapshot = await graph.aget_state(config)
    assert sqlite_session.scalars(select(Ticket)).all() == []
    assert "取消" in snapshot.values["answer"]
    rows = await fake_repositories.audit.list_recent(limit=5)
    cancelled = [
        row
        for row in rows
        if row.tool_name == "create_ticket" and row.status == "权限拒绝"
    ]
    assert cancelled
    assert "取消" in (cancelled[0].error_message or "")
    assert snapshot.next == ()
