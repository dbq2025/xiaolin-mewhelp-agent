"""Task 9：把三层结构接进图。

这一层的四条断言覆盖三种「接错就会静默坏掉」的情形：

1. 装配顺序错 → System 不在第 0 条、或出现第二条 System（前缀缓存整段作废）；
2. 摘要调度被 await → 本轮用户回复被上游摘要拖住；
3. 库里没有历史时**不该**报错，也不能凭空造 id（走 legacy 裁剪）。

日志断言走 ``log/app.log`` 的真实文件，不 mock ``AppLogger``：本章验收第 4 条
要求 ``grep model_ctx`` 就能看到每轮内容，只有真写盘才算验到。
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.checkpoint import memory_checkpointer
from app.agent.graph import build_agent_graph
from app.context.budget import compute_budget
from tests.fakes_agent import ScriptedChatModel, text_turn

#: ``tmp_path`` 在本沙箱不可用（见 ``tests/conftest.py`` 顶部的说明），
#: 所以日志夹具落在工作区内，每个用例一个 uuid 子目录、用完即删。
LOG_FIXTURE_DIR = Path(".uv-cache") / "graph-fixtures"


class _Retriever:
    """最小检索桩：本文件的用例都走 business 路由，检索器只被构造不被调用。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def retrieve_only(self, query: str, *, category=None):  # noqa: ANN001
        raise AssertionError("business 路由不该检索")

    async def search(self, query, top_k=None):  # noqa: ANN001
        raise AssertionError("business 路由不该检索")


class _FakeSummarizer:
    """记录调度请求的摘要器桩。

    ``schedule`` 必须是**同步**方法：真实实现里它只 ``create_task`` 就返回，
    调用方不 await。这里同步记录，任何 ``await summarizer.schedule(...)`` 的写法
    都会在语法上就落空（await None → TypeError），从而被这个桩抓住。
    """

    def __init__(self) -> None:
        self.scheduled: list[dict] = []
        self.aclose_calls = 0

    def schedule(self, **kwargs):  # noqa: ANN003
        self.scheduled.append(kwargs)
        return True

    async def aclose(self) -> None:
        self.aclose_calls += 1


def _intent_turn(intent: str, confidence: float = 0.9):
    return text_turn(json.dumps({"intent": intent, "confidence": confidence}, ensure_ascii=False))


def _log_dir() -> Path:
    target = LOG_FIXTURE_DIR / uuid.uuid4().hex
    target.mkdir(parents=True, exist_ok=True)
    return target


async def _run(
    graph,
    *,
    message: str,
    model,
    repositories,
    settings,
    session: str,
    retriever=None,
    summarizer=None,
    initial: dict | None = None,
):
    config = {
        "configurable": {
            "thread_id": session,
            "model": model,
            "repositories": repositories,
            "retriever": retriever or _Retriever(),
            "settings": settings,
            "summarizer": summarizer,
        }
    }
    events: list[dict] = []
    payload = {
        "session_id": session,
        "turn_id": uuid.uuid4().hex,
        "user_message": message,
    }
    if initial:
        payload.update(initial)
    async for chunk in graph.astream(payload, config, stream_mode="custom"):
        if isinstance(chunk, dict):
            events.append(chunk)
    snapshot = await graph.aget_state(config)
    return snapshot.values, events


def _graph():
    return build_agent_graph(checkpointer=memory_checkpointer())


def _settings(settings, **over):
    """图用例一律关掉指代消解。

    理由：``coref_enabled`` 默认 True，带指代的问句会额外调一次模型，
    ``ScriptedChatModel`` 的脚本步数就被打乱（表现为「多调用了一次模型」）。
    指代消解本身在 ch04 有自己的用例，这里只验上下文装配。
    """
    return settings.model_copy(update={"coref_enabled": False, **over})


async def test_agent_node_uses_three_layers_and_logs_model_ctx(
    test_settings, fake_repositories
):
    """端到端最薄一条：两轮之后日志里能看到 model_ctx，且 System 在第 0 条。"""
    log_dir = _log_dir()
    try:
        settings = _settings(test_settings, app_log_path=str(log_dir / "app.log"))
        graph = _graph()
        model = ScriptedChatModel(
            script=[
                _intent_turn("logistics"),
                text_turn("我帮您查。"),
                _intent_turn("logistics"),
                text_turn("还是那单，在派送中。"),
            ]
        )

        await _run(
            graph,
            message="订单 1001 到哪了",
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session="s-layers",
        )
        values, _ = await _run(
            graph,
            message="那什么时候到",
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session="s-layers",
        )

        text = (log_dir / "app.log").read_text(encoding="utf-8")
        assert "model_ctx" in text
        assert "session=s-layers" in text
        # 第二轮的历史里应当已经能看见第一轮的问答
        assert "订单 1001 到哪了" in text
        assert "我帮您查。" in text

        # 统计字段要透出来（前端统计开关与验收都看它）
        assert values["context"]["layer1_messages"] >= 1
        assert values["context"]["budget_tokens"] > 0
        assert values["context"]["summary_messages"] == 0
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


async def test_context_never_contains_a_second_system_message(
    test_settings, fake_repositories
):
    """返回给模型的消息里，System 有且只有第 0 条。

    第二条 System 的后果不是「多一段字」：上游模板会把所有 system 上提合并，
    工具定义被挤到可变内容之后，前缀缓存从此整段作废。
    """
    graph = _graph()
    model = ScriptedChatModel(
        script=[
            _intent_turn("logistics"),
            text_turn("查到了。"),
            _intent_turn("logistics"),
            text_turn("还在路上。"),
        ]
    )
    settings = _settings(test_settings)
    await _run(
        graph,
        message="订单 1001 到哪了",
        model=model,
        repositories=fake_repositories,
        settings=settings,
        session="s-system",
    )
    await _run(
        graph,
        message="还要几天",
        model=model,
        repositories=fake_repositories,
        settings=settings,
        session="s-system",
    )

    sent = model.state.calls[-1]
    assert isinstance(sent[0], SystemMessage)
    assert sum(1 for message in sent if isinstance(message, SystemMessage)) == 1
    # System 之后不许再出现「像 system 的内容」
    assert all(not isinstance(message, SystemMessage) for message in sent[1:])


async def test_summary_is_scheduled_but_not_awaited(test_settings, fake_repositories):
    """层 2 攒到超预算 → 调度后台摘要，但本轮回复照常先返回。

    预算压到极小：``history_tokens_per_turn=40`` 让滑窗只有 40 token，
    两三轮之后层 1 必然降级、层 2 必然超预算。
    """
    log_dir = _log_dir()
    try:
        settings = _settings(
            test_settings,
            history_desired_turns=1,
            history_tokens_per_turn=40,
            app_log_path=str(log_dir / "app.log"),
        )
        summarizer = _FakeSummarizer()
        graph = _graph()
        model = ScriptedChatModel(
            script=[
                _intent_turn("logistics"),
                text_turn("第一轮答复，这里故意写长一点，" * 3),
                _intent_turn("logistics"),
                text_turn("第二轮答复，也要长一些，" * 3),
                _intent_turn("logistics"),
                text_turn("第三轮答复，" * 5),
                _intent_turn("logistics"),
                text_turn("第四轮答复，继续拉长内容。" * 4),
            ]
        )

        for index, message in enumerate(
            ("订单 1001 到哪了", "什么时候到", "能不能改地址", "那算了帮我退款吧"),
            1,
        ):
            values, _ = await _run(
                graph,
                message=message,
                model=model,
                repositories=fake_repositories,
                settings=settings,
                session="s-summary",
                summarizer=summarizer,
            )
            # 关键是这一句：每轮都必须拿到答复，摘要在后台跑
            assert values["answer"] or values["stop_reason"]

            if summarizer.scheduled:
                break

        assert summarizer.scheduled, "层 2 超预算后必须调度摘要"
        job = summarizer.scheduled[0]
        assert job["session_id"] == "s-summary"
        assert job["conversation_id"] > 0
        assert job["layer2_budget"] == compute_budget(settings).layer2_budget
        assert job["plan"].summary_due is True
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


async def test_library_empty_falls_back_to_state_messages(test_settings, fake_repositories):
    """库里还没有历史时用 State 里的消息兜底 —— 不炸、也不触发摘要。"""
    graph = _graph()
    model = ScriptedChatModel(script=[_intent_turn("logistics"), text_turn("好的。")])

    values, _ = await _run(
        graph,
        message="订单 1001 到哪了",
        model=model,
        repositories=fake_repositories,
        settings=_settings(test_settings),
        session="s-empty",
        initial={
            "messages": [
                HumanMessage(content="早先的问题"),
                AIMessage(content="早先的回答"),
                HumanMessage(content="订单 1001 到哪了"),
            ]
        },
    )

    assert values["answer"] == "好的。"
    # 无数据库 id → 全算层 1，不会凭空造出层 2
    assert values["context"]["layer2_messages"] == 0
    assert values["context"]["summary_upto_msg_id"] is None


async def test_degrade_moves_the_anchor_forward_and_keeps_it_monotonic(
    test_settings, fake_repositories
):
    """层 1 超预算 → 降级把 ``layer1_from_msg_id`` 往前推，且落库可复用。"""
    settings = _settings(
        test_settings, history_desired_turns=1, history_tokens_per_turn=40
    )
    graph = _graph()
    model = ScriptedChatModel(
        script=[
            _intent_turn("logistics"),
            text_turn("一" * 60),
            _intent_turn("logistics"),
            text_turn("二" * 60),
            _intent_turn("logistics"),
            text_turn("三" * 60),
        ]
    )
    session = "s-degrade"
    for message in ("订单 1001 到哪了", "什么时候到", "还能改地址吗"):
        await _run(
            graph,
            message=message,
            model=model,
            repositories=fake_repositories,
            settings=settings,
            session=session,
        )

    conversation_id = await fake_repositories.conversations.find_id(session)
    anchors = await fake_repositories.conversations.get_anchors(conversation_id)
    assert anchors.layer1_from_msg_id is not None
    assert anchors.layer1_from_msg_id > 1
    if anchors.summary_upto_msg_id is not None:
        assert anchors.layer1_from_msg_id >= anchors.summary_upto_msg_id + 1
