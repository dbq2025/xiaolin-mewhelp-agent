"""ch04 编排接线：证据捕获、拒答短路、低置信度入池、引用落库与传输。

## 为什么拒答要测「没调第二次模型」

如果拒答只是「工具告诉模型请拒答」，那验收就取决于模型当天的心情。
真正的实现是**短路**：证据不足时不发第二次模型调用，直接吐固定话术。
这里的断言方式很直接 —— 让假模型在第二次调用时说一句独特的话，然后确认它
**没有出现在答案里**。
"""

from __future__ import annotations

import json

import pytest

from app.chat_tools import astream_chat_with_tools
from app.knowledge.citations import Evidence, build_evidence
from app.knowledge.hybrid import RetrievedChunk
from app.prompts import REFUSAL_TEXT
from app.tools.executor import ToolResult
from tests.fakes import FakeChatModel

MESSAGE = "你们支持货到付款吗"


def _tool_call(keyword: str = MESSAGE, call_id: str = "call_1") -> dict:
    return {
        "name": "query_faq",
        "args": {"keyword": keyword},
        "id": call_id,
        "type": "tool_call",
    }


def _executor(payload: dict):
    calls: list[tuple[str, dict]] = []

    async def _run(name, args, **_kwargs):
        calls.append((name, args))
        return ToolResult(
            tool=name,
            args=args,
            ok=True,
            data=payload,
            error=None,
            attempts=1,
            duration_ms=3,
        )

    _run.calls = calls  # type: ignore[attr-defined]
    return _run


def _items(*specs: tuple[int, int, str]) -> list[dict]:
    return [
        {
            "n": n,
            "chunk_id": chunk_id,
            "section_path": f"售后手册 / 第{chunk_id}节",
            "category": "售后政策",
            "question": f"问法{chunk_id}",
            "answer": quote,
            "quote": quote,
        }
        for n, chunk_id, quote in specs
    ]


def _answer_payload(*specs: tuple[int, int, str]) -> dict:
    return {
        "ok": True,
        "keyword": MESSAGE,
        "count": len(specs),
        "items": _items(*specs),
        "refused": False,
        "refusal_source": None,
        "refusal_reason": None,
    }


def _refusal_payload(source: str = "self_check", reason: str = "证据只讲流程") -> dict:
    return {
        "ok": True,
        "keyword": MESSAGE,
        "count": 0,
        "items": [],
        "refused": True,
        "refusal_source": source,
        "refusal_reason": reason,
    }


async def _run_turn(repositories, *, payload: dict, chunks: list[str], message: str = MESSAGE):
    model = FakeChatModel(chunks=chunks, tool_calls_then_stream=[_tool_call(message)])
    executor = _executor(payload)
    events = [
        event
        async for event in astream_chat_with_tools(
            session_id="sess-cite",
            message=message,
            model=model,
            repositories=repositories,
            retriever=None,
            budget_tokens=2000,
            model_name="fake-model",
            timeout=1.0,
            max_retries=0,
            executor=executor,
        )
    ]
    answer = "".join(e.payload.get("delta", "") for e in events if e.kind == "token")
    return events, answer


# ---------------------------------------------------------------------------
# 拒答短路
# ---------------------------------------------------------------------------


async def test_refusal_short_circuits_the_second_model_call(fake_repositories):
    """第二次调用的文本绝不能出现在答案里 —— 那证明它压根没被调用。"""
    events, answer = await _run_turn(
        fake_repositories,
        payload=_refusal_payload(),
        chunks=["这句来自第二次模型调用，拒答时不该出现"],
    )

    assert answer == REFUSAL_TEXT
    assert "这句来自第二次模型调用" not in answer
    assert not any(e.kind == "citations" for e in events)  # 拒答没有引用


async def test_refusal_writes_a_low_confidence_row(fake_repositories):
    await _run_turn(fake_repositories, payload=_refusal_payload("self_check", "证据不足"), chunks=["x"])

    rows = await fake_repositories.low_confidence.list_recent(5)
    assert rows[0].raw_question == MESSAGE
    assert rows[0].source == "self_check"
    assert rows[0].reason == "证据不足"
    assert rows[0].conversation_id is not None  # 外键必须真的指到一个会话


async def test_retrieval_low_conf_source_is_recorded_verbatim(fake_repositories):
    await _run_turn(fake_repositories, payload=_refusal_payload("retrieval_low_conf", "0 命中"), chunks=["x"])

    rows = await fake_repositories.low_confidence.list_recent(1)
    assert rows[0].source == "retrieval_low_conf"


async def test_pool_write_failure_does_not_break_the_refusal(fake_repositories):
    """入池是旁路：池子写不进去也要把拒答话术给到用户。

    反过来（先写池、失败就抛）会让「数据库抖一下」变成 500，
    而验收 4 要求的是「明确拒答」这件用户可见的事。
    """
    async def _boom(*_args, **_kwargs):
        raise RuntimeError("池子挂了")

    fake_repositories.low_confidence.add = _boom  # type: ignore[assignment]

    events, answer = await _run_turn(
        fake_repositories, payload=_refusal_payload(), chunks=["x"]
    )

    assert answer == REFUSAL_TEXT
    assert any(e.kind == "tools" for e in events)


# ---------------------------------------------------------------------------
# 引用捕获与落库
# ---------------------------------------------------------------------------


async def test_citations_are_captured_and_persisted(fake_repositories):
    events, answer = await _run_turn(
        fake_repositories,
        payload=_answer_payload((1, 3, "单笔订单满 99 元包邮")),
        chunks=["满 99 元包邮[1]。"],
    )

    assert answer == "满 99 元包邮[1]。"
    event = next(e for e in events if e.kind == "citations")
    assert event.payload["citations"][0]["n"] == 1

    conversation_id = await fake_repositories.messages.find_conversation_id("sess-cite")
    rows = await fake_repositories.messages.list_for_conversation(conversation_id)
    answer_row = [row for row in rows if row.role == "assistant" and row.content][-1]
    assert answer_row.citations[0]["chunk_id"] == 3
    assert answer_row.citations[0]["section_path"] == "售后手册 / 第3节"


async def test_only_cited_evidence_is_persisted(fake_repositories):
    """快照回答「这段话依据了哪几条」，不是「当时手里有什么」。"""
    _, answer = await _run_turn(
        fake_repositories,
        payload=_answer_payload((1, 3, "满 99 包邮"), (2, 4, "赔付 500")),
        chunks=["满 99 包邮[1]。"],
    )

    assert "满 99 包邮" in answer
    conversation_id = await fake_repositories.messages.find_conversation_id("sess-cite")
    rows = await fake_repositories.messages.list_for_conversation(conversation_id)
    answer_row = [row for row in rows if row.role == "assistant" and row.content][-1]
    assert [item["n"] for item in answer_row.citations] == [1]


async def test_out_of_range_citation_does_not_destroy_the_answer(fake_repositories):
    """模型多写一个角标不该让整段回答消失 —— 只丢映射。"""
    _, answer = await _run_turn(
        fake_repositories,
        payload=_answer_payload((1, 3, "满 99 包邮")),
        chunks=["满 99 包邮[1]，破损赔 500[9]。"],
    )

    assert "破损赔 500" in answer
    conversation_id = await fake_repositories.messages.find_conversation_id("sess-cite")
    rows = await fake_repositories.messages.list_for_conversation(conversation_id)
    answer_row = [row for row in rows if row.role == "assistant" and row.content][-1]
    assert [item["n"] for item in answer_row.citations] == [1]


async def test_no_citation_event_when_the_model_writes_no_brackets(fake_repositories):
    events, answer = await _run_turn(
        fake_repositories,
        payload=_answer_payload((1, 3, "满 99 包邮")),
        chunks=["满 99 元包邮。"],
    )

    assert answer == "满 99 元包邮。"
    assert not any(e.kind == "citations" for e in events)


# ---------------------------------------------------------------------------
# 历史回读
# ---------------------------------------------------------------------------


async def test_history_payload_includes_citations(fake_repositories):
    from app.api import _db_session_messages_payload

    await _run_turn(
        fake_repositories,
        payload=_answer_payload((1, 3, "满 99 包邮")),
        chunks=["满 99 包邮[1]。"],
    )

    payload = await _db_session_messages_payload("sess-cite", fake_repositories)
    ai_messages = [m for m in payload.messages if m["role"] == "ai" and m.get("citations")]
    assert ai_messages, "刷新后引用丢了 —— 角标会变成死链"
    assert ai_messages[-1]["citations"][0]["chunk_id"] == 3


def test_evidence_roundtrip_from_tool_items():
    """工具出参 → Evidence 的回环（编排层靠它还原编号映射）。"""
    from app.knowledge.citations import evidence_from_payload

    restored = evidence_from_payload(_items((1, 3, "满 99 包邮")))
    assert isinstance(restored[0], Evidence)
    assert restored[0].chunk_id == 3
    assert restored[0].quote == "满 99 包邮"


async def test_truncated_evidence_is_refilled_from_the_database(fake_repositories):
    """给模型看的是节选，但用户点开引用必须看到原文。"""
    full_text = "满 99 元包邮。" * 60

    async def _seed_chunk():
        await fake_repositories.knowledge.upsert_chunk(
            {
                "category": "售后政策",
                "questions": "运费怎么算",
                "answer": full_text,
                "section_path": "售后政策 / 运费",
                "content_type": "faq",
                "is_key_clause": False,
                "ordinal": 0,
            }
        )
        rows = await fake_repositories.knowledge.list_all()
        return rows[0].id

    chunk_id = await _seed_chunk()

    payload = {
        "ok": True,
        "keyword": MESSAGE,
        "count": 1,
        "items": [
            {
                "n": 1,
                "chunk_id": chunk_id,
                "section_path": "售后政策 / 运费",
                "category": "售后政策",
                "question": "运费怎么算",
                "answer": full_text[:320],
                "truncated": True,
            }
        ],
        "refused": False,
        "refusal_source": None,
        "refusal_reason": None,
    }

    _, answer = await _run_turn(
        fake_repositories, payload=payload, chunks=["满 99 元包邮[1]。"]
    )

    conversation_id = await fake_repositories.messages.find_conversation_id("sess-cite")
    rows = await fake_repositories.messages.list_for_conversation(conversation_id)
    answer_row = [row for row in rows if row.role == "assistant" and row.content][-1]
    assert answer_row.citations[0]["quote"] == full_text   # 全文，不是节选
