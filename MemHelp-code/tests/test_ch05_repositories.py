"""ch05 Task 2：actions 列、单事务多行写入、最后一条用户消息。"""

import pytest

from tests.fakes import retrieval_outcome  # noqa: F401  (保持与其它测试相同的导入习惯)


async def test_actions_roundtrip(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-actions", "anonymous")
    await fake_repositories.messages.append(
        cid,
        role="assistant",
        content="可以帮你转人工或建工单。",
        actions=[{"action": "transfer_human", "ticket_type": "投诉"}],
    )
    rows = await fake_repositories.messages.list_for_conversation(cid)
    assert rows[-1].actions == [{"action": "transfer_human", "ticket_type": "投诉"}]


async def test_append_many_writes_all_rows(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-many", "anonymous")
    await fake_repositories.messages.append_many(
        cid,
        [
            {"role": "user", "content": "我要投诉"},
            {"role": "assistant", "content": "抱歉…"},
        ],
    )
    rows = await fake_repositories.messages.list_for_conversation(cid)
    assert [row.role for row in rows] == ["user", "assistant"]


async def test_append_many_rejects_invalid_role_without_partial_write(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-bad", "anonymous")
    with pytest.raises(ValueError):
        await fake_repositories.messages.append_many(
            cid,
            [
                {"role": "user", "content": "第一行"},
                {"role": "wizard", "content": "非法角色"},
            ],
        )
    assert await fake_repositories.messages.list_for_conversation(cid) == []


async def test_last_user_content_returns_latest_user_turn(fake_repositories):
    cid = await fake_repositories.conversations.upsert_by_session("s-last", "anonymous")
    await fake_repositories.messages.append_many(
        cid,
        [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "回答"},
            {"role": "user", "content": "订单 1001 的物流"},
        ],
    )
    assert await fake_repositories.messages.last_user_content("s-last") == "订单 1001 的物流"


async def test_last_user_content_returns_none_for_unknown_session(fake_repositories):
    assert await fake_repositories.messages.last_user_content("nope") is None
