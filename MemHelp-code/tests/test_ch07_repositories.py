"""锚点与摘要段的仓库层行为。

这一层要钉住三件事：

1. **锚点单调**：降级只前进不后退（Review Focus 1）；
2. **分段唯一**：多进程同时压同一段时数据库收口（Review Focus 2）；
3. **侧栏数据**：一次查询算出预览、条数、已摘要段数，不建 ORM relationship。
"""

import pytest

from app.db.repositories import Repositories, SummarySegmentConflict


async def _conv(repos, session_id="s1"):
    return await repos.conversations.upsert_by_session(session_id, "anonymous")


async def _msg(repos, cid, role="user", content="你好"):
    return await repos.messages.append(
        conversation_id=cid,
        role=role,
        content=content,
        tool_calls=None,
        tool_call_id=None,
        tool_traces=[],
    )


async def test_anchors_are_none_on_a_fresh_conversation(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    got = await repos.conversations.get_anchors(cid)
    assert got is not None
    assert got.summary_upto_msg_id is None
    assert got.layer1_from_msg_id is None


async def test_set_anchors_round_trips(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    assert await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=10, layer1_from_msg_id=11
    )
    got = await repos.conversations.get_anchors(cid)
    assert (got.summary_upto_msg_id, got.layer1_from_msg_id) == (10, 11)


async def test_set_anchors_never_lets_layer1_fall_behind_summary(sqlite_session):
    """Review Focus 1：层 1 起点不允许落回摘要区间里。

    落回去的后果：同一条消息既被梗概覆盖、又被当成原文喂给模型 —— 模型会看到
    「梗概说用户报过订单号 A1」紧接着「用户原文：我报过订单号 A1」，重复且浪费。
    """
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=20, layer1_from_msg_id=21
    )
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=20, layer1_from_msg_id=15
    )
    got = await repos.conversations.get_anchors(cid)
    assert got.layer1_from_msg_id == 21


async def test_set_anchors_is_monotonic_for_summary_upto(sqlite_session):
    """摘要是「只追加」的：覆盖边界不许往回退。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(conversation_id=cid, summary_upto_msg_id=30)
    await repos.conversations.set_anchors(conversation_id=cid, summary_upto_msg_id=10)
    assert (await repos.conversations.get_anchors(cid)).summary_upto_msg_id == 30


async def test_set_anchors_clamps_layer1_to_summary_written_in_the_same_call(sqlite_session):
    """同一次调用里 summary 追上来了，layer1 也要被顶上去。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=50, layer1_from_msg_id=51
    )
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=60, layer1_from_msg_id=51
    )
    got = await repos.conversations.get_anchors(cid)
    assert got.summary_upto_msg_id == 60
    assert got.layer1_from_msg_id == 61


async def test_optimistic_guard_refuses_when_anchor_moved(sqlite_session):
    """后台摘要任务落库前必须复核层 1 起点 —— 期间用户又聊了几轮就放弃。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=11)
    ok = await repos.conversations.set_anchors(
        conversation_id=cid,
        summary_upto_msg_id=10,
        layer1_from_msg_id=11,
        only_if_layer1_from=99,
    )
    assert ok is False
    assert (await repos.conversations.get_anchors(cid)).summary_upto_msg_id is None


async def test_optimistic_guard_passes_when_anchor_matches(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=11)
    ok = await repos.conversations.set_anchors(
        conversation_id=cid,
        summary_upto_msg_id=10,
        layer1_from_msg_id=11,
        only_if_layer1_from=11,
    )
    assert ok is True
    assert (await repos.conversations.get_anchors(cid)).summary_upto_msg_id == 10


async def test_get_anchors_returns_none_for_unknown_conversation(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    assert await repos.conversations.get_anchors(9999) is None


async def test_summary_append_and_list(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=4,
        content="用户问过耳机",
        msg_count=4,
        source_tokens=120,
    )
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=2,
        from_msg_id=5,
        upto_msg_id=8,
        content="用户报了订单号 A1",
        msg_count=4,
        source_tokens=130,
    )
    rows = await repos.summaries.list_for_conversation(cid)
    assert [r.segment_no for r in rows] == [1, 2]
    assert rows[1].content.startswith("用户报了订单号")


async def test_summary_segment_no_is_unique(sqlite_session):
    """Review Focus 2：多进程重复压同一段时数据库收口。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=4,
        content="a",
        msg_count=4,
        source_tokens=10,
    )
    with pytest.raises(SummarySegmentConflict):
        await repos.summaries.append(
            conversation_id=cid,
            segment_no=1,
            from_msg_id=1,
            upto_msg_id=4,
            content="b",
            msg_count=4,
            source_tokens=10,
        )


async def test_next_segment_no(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    assert await repos.summaries.next_segment_no(cid) == 1
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=2,
        content="a",
        msg_count=2,
        source_tokens=5,
    )
    assert await repos.summaries.next_segment_no(cid) == 2


async def test_summary_rows_are_ordered_by_segment_not_by_insert_order(sqlite_session):
    """段号是权威顺序：先插第 2 段再插第 1 段，读出来仍是 [1, 2]。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid, segment_no=2, from_msg_id=5, upto_msg_id=8,
        content="第二段", msg_count=4, source_tokens=10,
    )
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=4,
        content="第一段", msg_count=4, source_tokens=10,
    )
    rows = await repos.summaries.list_for_conversation(cid)
    assert [r.segment_no for r in rows] == [1, 2]


async def test_summary_optional_fields_round_trip(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=2,
        content="a",
        msg_count=2,
        source_tokens=5,
        model="deepseek-flash",
        elapsed_ms=742,
    )
    row = (await repos.summaries.list_for_conversation(cid))[0]
    assert row.model == "deepseek-flash"
    assert row.elapsed_ms == 742


async def test_list_for_user_returns_newest_first_with_preview(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    old = await _conv(repos, "s-old")
    await _msg(repos, old, content="我上周买的耳机还没发货")
    new = await _conv(repos, "s-new")
    await _msg(repos, new, content="我想退货")

    rows = await repos.conversations.list_for_user(user_id="anonymous", limit=50)
    assert [r.session_id for r in rows] == ["s-new", "s-old"]
    assert rows[0].preview == "我想退货"
    assert rows[0].message_count == 1
    assert rows[0].summary_segments == 0


async def test_list_for_user_marks_summarized_conversations(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await _msg(repos, cid)
    await repos.summaries.append(
        conversation_id=cid, segment_no=1, from_msg_id=1, upto_msg_id=1,
        content="a", msg_count=1, source_tokens=5,
    )
    rows = await repos.conversations.list_for_user(user_id="anonymous", limit=50)
    assert rows[0].summary_segments == 1


async def test_list_for_user_respects_limit(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    for i in range(5):
        await _conv(repos, f"s{i}")
    assert len(await repos.conversations.list_for_user(user_id="anonymous", limit=2)) == 2


async def test_list_for_user_ignores_other_users(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    await repos.conversations.upsert_by_session("mine", "me")
    await repos.conversations.upsert_by_session("theirs", "someone-else")
    rows = await repos.conversations.list_for_user(user_id="me", limit=50)
    assert [r.session_id for r in rows] == ["mine"]


async def test_preview_uses_the_first_user_message_not_the_latest(sqlite_session):
    """侧栏预览是「这个会话在聊什么」，用首问；用最新一句会变成「好的」。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await _msg(repos, cid, content="我买的耳机什么时候发货")
    await _msg(repos, cid, role="assistant", content="我帮您查一下")
    await _msg(repos, cid, content="好的")

    row = (await repos.conversations.list_for_user(user_id="anonymous", limit=50))[0]
    assert row.preview == "我买的耳机什么时候发货"
    assert row.message_count == 3


async def test_preview_is_truncated(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await _msg(repos, cid, content="我" * 200)
    row = (await repos.conversations.list_for_user(user_id="anonymous", limit=50))[0]
    assert len(row.preview) < 200


async def test_preview_is_empty_when_no_user_message_yet(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    await _conv(repos)
    row = (await repos.conversations.list_for_user(user_id="anonymous", limit=50))[0]
    assert row.preview == ""
    assert row.message_count == 0


async def test_list_for_user_reports_anchors_and_updated_at(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    await _msg(repos, cid)
    await repos.conversations.set_anchors(
        conversation_id=cid, summary_upto_msg_id=1, layer1_from_msg_id=2
    )
    row = (await repos.conversations.list_for_user(user_id="anonymous", limit=50))[0]
    assert row.id == cid
    assert row.created_at is not None
    assert row.updated_at is not None


async def test_repositories_container_exposes_summaries(sqlite_session):
    repos = Repositories.from_session(sqlite_session)
    assert repos.summaries is not None

async def test_latest_summary_texts_returns_oldest_first(sqlite_session):
    """梗概按段号升序给模型看 —— 顺序颠倒会让「先问耳机、后报订单号」读反。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    for seg, text in enumerate(["第一段", "第二段", "第三段"], start=1):
        await repos.summaries.append(
            conversation_id=cid, segment_no=seg, from_msg_id=seg, upto_msg_id=seg,
            content=text, msg_count=1, source_tokens=5,
        )
    texts = await repos.summaries.latest_texts(cid)
    assert texts == ["第一段", "第二段", "第三段"]


async def test_latest_summary_texts_can_be_capped(sqlite_session):
    """段数很多时只喂最近的若干段 —— 更早的已经被后面的梗概隐含覆盖了。"""
    repos = Repositories.from_session(sqlite_session)
    cid = await _conv(repos)
    for seg in range(1, 6):
        await repos.summaries.append(
            conversation_id=cid, segment_no=seg, from_msg_id=seg, upto_msg_id=seg,
            content=f"第{seg}段", msg_count=1, source_tokens=5,
        )
    assert await repos.summaries.latest_texts(cid, limit=2) == ["第4段", "第5段"]
