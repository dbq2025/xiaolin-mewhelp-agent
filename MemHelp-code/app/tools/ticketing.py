"""``create_ticket`` —— 写 ``tickets`` 表。

## 为什么需要两个仓库

工单表的外键是 ``conversation_id``（数字），而模型拿到的是 ``session_id``（字符串，
聊天页给的）。所以这里要先把字符串映射成数字行 —— 让模型去猜数字 id 是不现实的，
它一定会猜错，而猜错的后果是外键指向不存在的行（真库上直接报约束失败）。

用 ``upsert_by_session`` 而不是 ``find_id``：用户第一句话就要求建工单时，
会话可能还不存在，此时必须能建出来。

## 类型与描述在校验层已经拦过一次

``CreateTicketArgs`` 用 ``Literal`` 限定了三个类型，工具实现里再查一遍是因为
**工具可能被直接调用**（测试、脚本、未来的批量导入），不只经由 ``executor``。
两层都便宜，而漏一层会让非法值走到数据库驱动才报错，错误信息对模型毫无可操作性。
"""

import json
from typing import Protocol

from app.db.repositories import TICKET_TYPES


class TicketCreateRepo(Protocol):
    """本工具对工单仓库的最小依赖。"""

    async def create(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        *,
        day=None,
        idempotency_key: str | None = None,
    ) -> str: ...


class ConversationUpsertRepo(Protocol):
    """本工具对会话仓库的最小依赖。"""

    async def upsert_by_session(self, session_id: str, user_id: str) -> int: ...


async def create_ticket(
    description: str,
    ticket_type: str,
    conversation_id: str,
    *,
    idempotency_key: str | None = None,
    ticket_repo: TicketCreateRepo,
    conversation_repo: ConversationUpsertRepo,
) -> str:
    """为当前会话创建一张人工工单。

    :param description: 问题描述。
    :param ticket_type: 工单类型，必须是 ``TICKET_TYPES`` 之一。
    :param conversation_id: 当前的**对外会话标识**（字符串），不是数字外键。
    :param idempotency_key: 可选的一次性键；同一个键重复提交返回同一张工单。
    :param ticket_repo: 工单仓库。
    :param conversation_repo: 会话仓库，用于把 session_id 换成数字 id。
    :return: JSON 字符串，含 ``ok`` / ``ticket_no`` / ``ticket_type`` / ``status``。
        工单号形如 ``T20260701008``，是给用户念的编号。

    :raises ValueError: ``ticket_type`` 不在 DDL 的 ENUM 取值内，或 ``description`` 为空白。
    :raises Exception: 仓库层异常原样冒出去，由 ``executor.py`` 分类。
    """
    description = description.strip()
    if not description:
        raise ValueError("description 不能是空白，工单需要写明用户诉求")

    if ticket_type not in TICKET_TYPES:
        raise ValueError(
            f"ticket_type {ticket_type!r} 不合法，只允许 {TICKET_TYPES}"
        )

    numeric_id = await conversation_repo.upsert_by_session(conversation_id, "anonymous")
    ticket_no = await ticket_repo.create(
        numeric_id, description, ticket_type, idempotency_key=idempotency_key
    )

    return json.dumps(
        {
            "ok": True,
            "ticket_no": ticket_no,
            "ticket_type": ticket_type,
            "status": "待处理",
            "message": f"已为你创建工单 {ticket_no}，人工客服会尽快跟进。",
        },
        ensure_ascii=False,
    )
