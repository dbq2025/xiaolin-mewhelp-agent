"""Task 10：``history_ctx`` 每轮必打。

``model_ctx`` 只在真正调主力 Agent 的轮次才有（闲聊、投诉、兜底这些固定回复
轮根本不进 ``agent`` 节点）。但指代消解与意图识别是**每一轮都跑**的前置节点，
它们看到的历史同样要留痕，否则「模型这轮看到了什么」在闲聊轮就成了黑洞。

三条断言：

1. 闲聊 / 投诉 / 兜底轮都要有 ``history_ctx``；
2. 有梗概时，``history_ctx`` 里要能看到梗概行；
3. 一轮只写一条 —— coref 与 intent 共用同一份历史，写两遍等于把日志读花。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from tests.fakes_agent import ScriptedChatModel, text_turn

#: ``tmp_path`` 在本沙箱不可用（见 ``tests/conftest.py`` 顶部），日志落在工作区。
LOG_FIXTURE_DIR = Path(".uv-cache") / "history-ctx-fixtures"


class _Retriever:
    def __init__(self, *, refused: bool = False) -> None:
        self.refused = refused
        self.calls: list[str] = []

    async def retrieve_only(self, query: str, *, category=None):  # noqa: ANN001
        from app.knowledge.pipeline import RetrievalOutcome
        from app.knowledge.query import UnderstoodQuery

        return RetrievalOutcome(
            query=UnderstoodQuery(original=query, standard=query, bm25_text=query),
            evidence=(),
            refused=self.refused,
            refusal_source="retrieval_low_conf" if self.refused else None,
            refusal_reason="测试里判定证据不足" if self.refused else None,
            rerank_degraded=False,
        )

    async def search(self, query, top_k=None):  # noqa: ANN001
        raise AssertionError("这些用例都走不过检索的路由")


def _intent_turn(intent: str, confidence: float = 0.9):
    return text_turn(
        json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False)
    )


def _log_dir() -> Path:
    target = LOG_FIXTURE_DIR / uuid.uuid4().hex
    target.mkdir(parents=True, exist_ok=True)
    return target


def _read(log_dir: Path) -> str:
    path = log_dir / "app.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def _run(graph, *, message, model, repositories, settings, session, retriever=None):
    config = {
        "configurable": {
            "thread_id": session,
            "model": model,
            "repositories": repositories,
            "retriever": retriever or _Retriever(),
            "settings": settings,
        }
    }
    payload = {
        "session_id": session,
        "turn_id": uuid.uuid4().hex,
        "user_message": message,
    }
    async for _chunk in graph.astream(payload, config, stream_mode="custom"):
        pass
    snapshot = await graph.aget_state(config)
    return snapshot.values


def _graph():
    return build_agent_graph(checkpointer=memory_checkpointer())


def _settings(settings, log_dir: Path, **over):
    """核心解消在这里一律关掉：本文件验的是历史日志，不是指代词补全本身。"""
    return settings.model_copy(
        update={
            "coref_enabled": False,
            "app_log_path": str(log_dir / "app.log"),
            **over,
        }
    )


@pytest.mark.parametrize(
    "text,intent,reply",
    [
        ("你好呀", "chitchat", "您好，很高兴为您服务。"),
        ("我要投诉", "complaint", "非常抱歉给您带来不好的体验。"),
    ],
)
async def test_history_ctx_is_logged_on_every_route(
    text, intent, reply, test_settings, fake_repositories
):
    """闲聊与投诉都不进 agent 节点，但历史照样要看 —— 所以照样要留痕。"""
    log_dir = _log_dir()
    try:
        model = ScriptedChatModel(script=[_intent_turn(intent)])
        settings = _settings(test_settings, log_dir)
        values = await _run(
            _graph(),
            message=text,
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session=f"s-history-{intent}",
        )
        assert values["intent"] == intent
        # 这两条路由都不调主力模型
        assert values.get("context") == {}
        body = _read(log_dir)
        assert "history_ctx" in body
        assert f"session=s-history-{intent}" in body
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


async def test_fallback_route_also_logs_history_ctx(
    test_settings, fake_repositories
):
    """证据不足落到固定兜底回复，历史同样要留痕。"""
    log_dir = _log_dir()
    try:
        # knowledge 路由 + 检索判定证据不足 → 置信门失败 → fallback 固定回复，
        # 这条路径不调主力 Agent，最容易漏日志。
        model = ScriptedChatModel(script=[_intent_turn("product_consult")])
        settings = _settings(test_settings, log_dir)
        values = await _run(
            _graph(),
            message="这个耳机支持降噪吗",
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session="s-history-fallback",
            retriever=_Retriever(refused=True),
        )
        assert values["route"] == "knowledge"
        assert values["gate_passed"] is False
        assert values.get("context") == {}
        body = _read(log_dir)
        assert "history_ctx" in body
        assert "session=s-history-fallback" in body
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


async def test_history_ctx_contains_summary_when_present(
    test_settings, fake_repositories
):
    """有梗概时，前置节点看到的历史里也要有梗概行，别只给滑窗。"""
    log_dir = _log_dir()
    try:
        session = "s-history-summary"
        conversation_id = await fake_repositories.conversations.upsert_by_session(
            session, "anonymous"
        )
        await fake_repositories.messages.append(
            conversation_id, role="user", content="我上周买的耳机什么时候到"
        )
        await fake_repositories.conversations.set_anchors(
            conversation_id=conversation_id,
            summary_upto_msg_id=1,
            layer1_from_msg_id=2,
        )
        await fake_repositories.summaries.append(
            conversation_id=conversation_id,
            segment_no=1,
            from_msg_id=1,
            upto_msg_id=1,
            content="用户问过耳机的到货时间。",
            msg_count=1,
            source_tokens=20,
        )
        model = ScriptedChatModel(script=[_intent_turn("chitchat")])
        settings = _settings(test_settings, log_dir)
        await _run(
            _graph(),
            message="那算了",
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session=session,
        )
        body = _read(log_dir)
        assert "history_ctx" in body
        assert "用户问过耳机的到货时间。" in body
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


async def test_history_ctx_is_only_written_once_per_turn(
    test_settings, fake_repositories
):
    """coref 与 intent 共用同一份历史：一轮只能写一条，写两遍等于日志读花。"""
    log_dir = _log_dir()
    try:
        model = ScriptedChatModel(script=[_intent_turn("chitchat")])
        settings = _settings(test_settings, log_dir)
        await _run(
            _graph(),
            message="你好",
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session="s-history-once",
        )
        body = _read(log_dir)
        assert body.count("history_ctx session=s-history-once") == 1
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)
