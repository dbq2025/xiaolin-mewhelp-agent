"""ch05 Task 6：Agent 工具集过滤、suggest_actions 控制面工具、证据与引用助手。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.evidence import citations_for_answer, evidence_from_tool_result
from app.agent.tools import agent_tools, make_suggest_actions_tool
from app.tools.executor import ToolResult
from app.tools.registry import build_registry


class _KnowledgeRepo:
    def __init__(self, rows=None) -> None:
        self.rows = rows or []

    async def get_many(self, ids):
        return [row for row in self.rows if row.id in ids]


class _Repos:
    def __init__(self, rows=None) -> None:
        self.knowledge = _KnowledgeRepo(rows)


def _tool_result(items, *, ok=True) -> ToolResult:
    return ToolResult(
        tool="query_faq",
        args={"keyword": "退货"},
        ok=ok,
        data=json.dumps({"ok": True, "count": len(items), "items": items}, ensure_ascii=False),
        error=None,
        attempts=1,
        duration_ms=1,
    )


def test_agent_tools_include_create_ticket_for_confirmation_flow(
    fake_repositories, empty_retriever
):
    """ch08：create_ticket 回到工具集，由 ReAct 拦截后转前端确认，不再直接执行。"""
    names = [
        tool.name
        for tool in agent_tools(
            build_registry(fake_repositories, empty_retriever),
            include_faq=True,
            record_actions=[],
        )
    ]
    assert "create_ticket" in names
    assert "suggest_actions" in names


def test_knowledge_tools_drop_query_faq(fake_repositories, empty_retriever):
    names = [
        tool.name
        for tool in agent_tools(
            build_registry(fake_repositories, empty_retriever),
            include_faq=False,
            record_actions=[],
        )
    ]
    assert "query_faq" not in names
    assert {"query_order", "query_product", "suggest_actions"} <= set(names)


def test_refund_tools_lock_the_order(fake_repositories, empty_retriever):
    names = [
        tool.name
        for tool in agent_tools(
            build_registry(fake_repositories, empty_retriever),
            include_faq=False,
            include_order=False,
            record_actions=[],
        )
    ]
    assert "query_order" not in names
    assert "track_shipment" not in names
    assert "query_product" in names
    assert "suggest_actions" in names


def test_business_tools_keep_query_faq(fake_repositories, empty_retriever):
    names = [
        tool.name
        for tool in agent_tools(
            build_registry(fake_repositories, empty_retriever),
            include_faq=True,
            record_actions=[],
        )
    ]
    assert "query_faq" in names


async def test_suggest_actions_records_without_writing():
    record: list[str] = []
    tool = make_suggest_actions_tool(record)

    payload = json.loads(
        await tool.ainvoke(
            {"actions": ["transfer_human", "create_ticket"], "reason": "用户要求人工"}
        )
    )
    assert payload["ok"] is True
    assert record == ["transfer_human", "create_ticket"]
    assert "已记录" in payload["message"]


async def test_suggest_actions_rejects_unknown_action():
    tool = make_suggest_actions_tool([])
    with pytest.raises(Exception):
        await tool.ainvoke({"actions": ["delete_order"]})


async def test_evidence_falls_back_to_answer_field():
    """真实 query_faq 出参没有 quote，只有 answer；引用快照不能因此空掉。"""
    result = _tool_result(
        [
            {
                "n": 1,
                "chunk_id": 7,
                "section_path": "售后政策 / 退货",
                "category": "售后政策",
                "question": "退货政策",
                "answer": "7 天无理由退货",
            }
        ]
    )
    evidence = await evidence_from_tool_result(result, _Repos())

    assert len(evidence) == 1
    assert evidence[0].quote == "7 天无理由退货"


async def test_evidence_fetches_full_text_for_truncated_items():
    result = _tool_result(
        [
            {
                "n": 1,
                "chunk_id": 9,
                "section_path": "售后政策 / 退款",
                "category": "售后政策",
                "question": "退款",
                "answer": "前半段…",
                "truncated": True,
            }
        ]
    )
    repos = _Repos(rows=[SimpleNamespace(id=9, answer="完整的退款政策原文")])
    evidence = await evidence_from_tool_result(result, repos)

    assert evidence[0].quote == "完整的退款政策原文"


def test_citations_only_include_cited_numbers():
    evidence = (
        SimpleNamespace(
            n=1, chunk_id=11, section_path="a", category="c", quote="q1"
        ),
        SimpleNamespace(
            n=2, chunk_id=12, section_path="b", category="c", quote="q2"
        ),
    )
    citations = citations_for_answer("按政策可以退 [2]，但要保留包装。", evidence)

    assert [item["n"] for item in citations] == [2]
    assert citations[0]["quote"] == "q2"
