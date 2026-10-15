"""对话用例测试。

验收标准 2（第二轮接住第一轮上下文）的**离线证据**在这里：用 `Recorder`
拿到模型实际收到的消息，断言第二轮请求里确实带着第一轮的问与答。

同时钉住 Review Focus 2：不同 session_id 之间不能串味。
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.chat import ChatMeta, astream_chat, complete_chat
from app.memory import InMemorySessionStore
from tests.fakes import FakeChatModel, Recorder


async def _collect(agen):
    return [chunk async for chunk in agen]


async def test_astream_yields_only_non_empty_deltas():
    """astream 的尾块是空串（Task 3 实测），不能当成一个 token 事件发出去。"""
    store = InMemorySessionStore()
    model = FakeChatModel(chunks=["您", "好", "！"])
    deltas = await _collect(
        astream_chat("s1", "在吗", store, model, budget_tokens=2000, model_name="fake")
    )
    assert deltas == ["您", "好", "！"]


async def test_astream_persists_question_and_full_answer():
    store = InMemorySessionStore()
    model = FakeChatModel(chunks=["好", "的"])
    await _collect(
        astream_chat("s1", "在吗", store, model, budget_tokens=2000, model_name="fake")
    )
    history = store.get("s1")
    assert isinstance(history[0], HumanMessage)
    assert history[0].content == "在吗"
    assert isinstance(history[1], AIMessage)
    assert history[1].content == "好的"


async def test_second_turn_sees_first_turn_context():
    """核心：第二轮发给模型的消息里必须带第一轮的问与答。"""
    store = InMemorySessionStore()
    await complete_chat(
        "s1", "第一轮问题", store, FakeChatModel(reply="第一轮回答"),
        budget_tokens=2000, model_name="fake",
    )

    rec = Recorder()
    await complete_chat(
        "s1", "第二轮问题", store, FakeChatModel(reply="二", recorder=rec),
        budget_tokens=2000, model_name="fake",
    )

    contents = [m.content for m in rec.messages]
    assert "第一轮问题" in contents
    assert "第一轮回答" in contents
    assert contents[-1] == "第二轮问题"
    # 第一条必须是 System Prompt
    assert contents[0].startswith("你是电商平台")


async def test_system_prompt_is_always_sent():
    store = InMemorySessionStore()
    rec = Recorder()
    await complete_chat(
        "s1", "问", store, FakeChatModel(reply="答", recorder=rec),
        budget_tokens=2000, model_name="fake",
    )
    assert len(rec.messages) == 2
    assert rec.messages[0].type == "system"
    assert rec.messages[1].content == "问"


async def test_complete_chat_returns_text_and_meta():
    store = InMemorySessionStore()
    text, meta = await complete_chat(
        "s1", "问", store, FakeChatModel(reply="完整回答"),
        budget_tokens=2000, model_name="fake",
    )
    assert text == "完整回答"
    assert isinstance(meta, ChatMeta)
    assert meta.session_id == "s1"
    assert meta.model == "fake"
    assert meta.context.budget_tokens == 2000


async def test_complete_chat_persists_history():
    store = InMemorySessionStore()
    await complete_chat(
        "s1", "问", store, FakeChatModel(reply="答"),
        budget_tokens=2000, model_name="fake",
    )
    assert [m.content for m in store.get("s1")] == ["问", "答"]


async def test_sessions_do_not_leak_across_ids():
    """Review Focus 2。"""
    store = InMemorySessionStore()
    await complete_chat(
        "a", "甲的问题", store, FakeChatModel(reply="甲的回答"),
        budget_tokens=2000, model_name="f",
    )
    await complete_chat(
        "b", "乙的问题", store, FakeChatModel(reply="乙的回答"),
        budget_tokens=2000, model_name="f",
    )
    assert [m.content for m in store.get("a")] == ["甲的问题", "甲的回答"]
    assert [m.content for m in store.get("b")] == ["乙的问题", "乙的回答"]


async def test_other_sessions_history_never_reaches_the_model():
    """更强的串味断言：问 b 会话时，模型不该看到 a 会话的任何内容。"""
    store = InMemorySessionStore()
    await complete_chat(
        "a", "甲的秘密订单号 X1", store, FakeChatModel(reply="收到"),
        budget_tokens=2000, model_name="f",
    )
    rec = Recorder()
    await complete_chat(
        "b", "我是乙", store, FakeChatModel(reply="您好", recorder=rec),
        budget_tokens=2000, model_name="f",
    )
    assert all("甲的秘密订单号" not in m.content for m in rec.messages)


async def test_zero_chunk_upstream_raises_but_question_is_still_persisted():
    """上游一块都没吐时，langchain 的 astream 会抛 ValueError（实测行为）。

    关键是 `finally` 保住了落库：用户的问题不能因为上游抽风就从历史里消失，
    否则下一轮模型会以为用户没说过话。
    """
    store = InMemorySessionStore()
    with pytest.raises(ValueError, match="No generation chunks"):
        await _collect(
            astream_chat("s1", "问", store, FakeChatModel(chunks=[]),
                         budget_tokens=2000, model_name="f")
        )
    assert store.get("s1")[0].content == "问"


async def test_midstream_upstream_failure_still_persists_question_and_partial():
    """上游流到一半挂了：已收到的部分与问题都要落库，异常继续向上抛给 HTTP 层。"""
    store = InMemorySessionStore()
    model = FakeChatModel(chunks=["已收到的一半", "再也发不出来"], fail_after=1)

    with pytest.raises(RuntimeError, match="fake upstream failure"):
        await _collect(
            astream_chat("s1", "问", store, model, budget_tokens=2000, model_name="f")
        )

    assert [m.content for m in store.get("s1")] == ["问", "已收到的一半"]


async def test_meta_out_is_filled_before_streaming():
    """HTTP 层靠 meta_out 拿到裁剪统计来填 done 事件。"""
    store = InMemorySessionStore()
    meta_out: dict = {}
    await _collect(
        astream_chat("s1", "问", store, FakeChatModel(chunks=["答"]),
                     budget_tokens=2000, model_name="fake", meta_out=meta_out)
    )
    assert meta_out["meta"].session_id == "s1"
    assert meta_out["meta"].context.budget_tokens == 2000
