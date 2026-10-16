"""数据访问层 —— **全项目唯一写 SQL / 用 SQLAlchemy 查询 API 的地方**。

上层（``app/tools/``、``app/chat_tools.py``）只面对这里的业务语义方法，不碰
``select()`` / ``session.execute()``。这条边界换来两件事：

1. 「SQL 写在哪」只有一个答案，评审时不必满仓库找；
2. 工具层可以注入**内存假仓库**（见 ``tests/test_fakes_repos.py``），
   于是没有 MySQL 也能跑完整个编排链路。

## 为什么每个方法都是 ``async def``，内部却是同步调用

``PyMySQL`` 是**同步**驱动 —— ``session.execute()`` 会阻塞当前线程直到数据库应答。
如果直接在 ``async def`` 里同步调用，阻塞的就是事件循环本身：单请求看不出问题
（反正只有一个请求），并发时所有请求排成一队等同一个数据库往返。

所以每个方法用 ``anyio.to_thread.run_sync`` 把同步调用送进工作线程。
`async def` 不等于不阻塞，这里是 `async def` 必须配 `to_thread` 的典型场景。

**线程安全**：``Session`` 不是线程安全的，但这里所有调用都经 ``run_sync``
串行化（一次只有一个线程在用它），所以安全。不要改成并发 await 同一 session。

## 返回值是纯数据，不是 ORM 实例

各方法把结果**在工作线程内**拆成 ``dict`` / dataclass 再返回。ORM 实例带 lazy-load
和过期属性，跨线程读它会触发新的数据库往返（在新线程里），是最难查的一类竞态。
返回纯数据让「数据在哪个线程被读过」这个问题不再存在。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from typing import Any, TypeVar

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models import (
    MESSAGE_ROLES,
    TICKET_TYPES as _TICKET_TYPES,
    Conversation,
    Faq,
    Message,
    Ticket,
)

T = TypeVar("T")

#: 工单类型，取自 ORM 层的**单一来源**后再导出。
#:
#: 为什么在这里再导出一遍：工具层只允许依赖本模块（见
#: ``tests/test_tool_knowledge.py`` 的分层用例），不该 import ``app.db.models``。
#: 若让工具层自己写一份常量，DDL 改了 ENUM 而常量没跟上，就会变成
#: 「参数校验通过、写库被拒」这种最难查的半对齐状态。
TICKET_TYPES: tuple[str, ...] = _TICKET_TYPES

#: ``LIKE`` 的转义字符。SQL 标准里 ``LIKE ... ESCAPE '\'`` 指定它；
#: MySQL 默认就是 ``\``，但显式写出来才不依赖默认值。
LIKE_ESCAPE_CHAR = "\\"


async def _run(fn: Callable[..., T], *args: object) -> T:
    """把同步的数据库调用送进工作线程。

    单独抽出来是为了让「哪些调用会被送进线程」在代码里一眼可见 ——
    漏包一个就会静默阻塞事件循环，而那种退化在单请求测试里完全没有症状。
    """
    return await anyio.to_thread.run_sync(partial(fn, *args))


def escape_like(keyword: str) -> str:
    r"""把 ``LIKE`` 的通配符转义成字面量。

    不转义的话，用户问「100%纯棉」会命中**全表**：``%`` 通配任意长度。
    更糟的是工具会返回 ``ok=true`` 和一堆不相关的 FAQ，模型拿着它们一本正经地
    回答 —— 故障表现为「答非所问」，而根因在一个通配符上。

    反斜杠必须**最先**替换，否则后两步插入的 ``\`` 会被二次转义。
    """
    return (
        keyword.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _like_pattern(keyword: str) -> str:
    return f"%{escape_like(keyword)}%"


# ---------------------------------------------------------------------------
# 返回用的纯数据类型
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaqHit:
    """一条 FAQ 命中结果。给工具层用，不带 ORM 语义。"""

    question: str
    answer: str
    category: str


@dataclass(frozen=True)
class MessageRow:
    """一条消息流水。"""

    id: int
    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_call_id: str | None
    created_at: datetime


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------


class ConversationRepository:
    """``conversations`` 表的读写。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _upsert(self, session_id: str, user_id: str) -> int:
        existing = self._session.scalar(
            select(Conversation.id).where(Conversation.session_id == session_id)
        )
        if existing is not None:
            return int(existing)

        conversation = Conversation(session_id=session_id, user_id=user_id)
        self._session.add(conversation)
        self._session.commit()
        return int(conversation.id)

    async def upsert_by_session(self, session_id: str, user_id: str) -> int:
        """按 ``session_id`` 取会话，没有就建。**幂等**：已有则复用同一个 id。

        刻意**不用** MySQL 的 ``ON DUPLICATE KEY UPDATE``：那在 SQLite 上跑不了，
        会让这一层无法离线测试，而离线可测是注入假仓库的前提。

        并发下同一个新 ``session_id`` 可能有两个请求同时走到 INSERT，撞唯一键；
        这里让它抛出去由调用方重试 —— 吞掉冲突去 SELECT 会把「两个请求各自建了
        自己的上下文」这种真实竞态藏起来。
        """
        return await _run(self._upsert, session_id, user_id)

    def _find_id(self, session_id: str) -> int | None:
        found = self._session.scalar(
            select(Conversation.id).where(Conversation.session_id == session_id)
        )
        return int(found) if found is not None else None

    async def find_id(self, session_id: str) -> int | None:
        return await _run(self._find_id, session_id)


# ---------------------------------------------------------------------------
# 消息流水
# ---------------------------------------------------------------------------


class MessageRepository:
    """``messages`` 表的读写。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _append(
        self,
        conversation_id: int,
        role: str,
        content: str | None,
        tool_calls: Sequence[dict[str, Any]] | None,
        tool_call_id: str | None,
    ) -> None:
        # 入口校验：非法 role 在这里就炸，错误信息带角色名。
        # 靠 MySQL 的 ENUM 兜会拖到 commit，报的是驱动的
        # "Data truncated for column 'role'" —— 把拼错字符串伪装成数据库故障。
        if role not in MESSAGE_ROLES:
            raise ValueError(
                f"非法消息角色 {role!r}，只允许 {MESSAGE_ROLES}（对应 DDL 的 ENUM）"
            )

        self._session.add(
            Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tool_calls=list(tool_calls) if tool_calls is not None else None,
                tool_call_id=tool_call_id,
            )
        )
        self._session.commit()

    async def append(
        self,
        conversation_id: int,
        *,
        role: str,
        content: str | None,
        tool_calls: Sequence[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """追加一条消息。``role`` 必须是 ``MESSAGE_ROLES`` 之一。

        :raises ValueError: 角色不在 DDL 的 ENUM 取值内。
        """
        await _run(self._append, conversation_id, role, content, tool_calls, tool_call_id)

    def _list(self, conversation_id: int) -> list[MessageRow]:
        rows = self._session.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.id)
        ).all()
        return [
            MessageRow(
                id=row.id,
                role=row.role,
                content=row.content,
                tool_calls=row.tool_calls,
                tool_call_id=row.tool_call_id,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_for_conversation(self, conversation_id: int) -> list[MessageRow]:
        """按 ``id`` 升序取该会话的全部消息。"""
        return await _run(self._list, conversation_id)

    def _find_conversation_id(self, session_id: str) -> int | None:
        found = self._session.scalar(
            select(Conversation.id).where(Conversation.session_id == session_id)
        )
        return int(found) if found is not None else None

    async def find_conversation_id(self, session_id: str) -> int | None:
        return await _run(self._find_conversation_id, session_id)

    def _history(self, session_id: str) -> list[BaseMessage]:
        conversation_id = self._find_conversation_id(session_id)
        if conversation_id is None:
            return []
        return [_row_to_message(row) for row in self._list(conversation_id)]

    async def history_as_messages(self, session_id: str) -> list[BaseMessage]:
        """把库里的流水重建成 LangChain 消息列表，供上下文裁剪与模型调用。

        未知角色抛 ``ValueError`` 而不是静默跳过：静默跳过会让模型看到一个
        **缺了一环**的历史（比如有工具调用申请却没有工具结果），
        而模型的反应是拒绝回答或编造 —— 排查方向会完全跑偏。
        """
        return await _run(self._history, session_id)


def _row_to_message(row: MessageRow) -> BaseMessage:
    """一条流水 → 对应的 LangChain 消息对象。"""
    if row.role == "user":
        return HumanMessage(content=row.content or "")
    if row.role == "assistant":
        if row.tool_calls:
            return AIMessage(content=row.content or "", tool_calls=list(row.tool_calls))
        return AIMessage(content=row.content or "")
    if row.role == "tool":
        if row.tool_call_id is None:
            raise ValueError(
                f"tool 消息（id={row.id}）缺少 tool_call_id，无法对号入座回灌给模型"
            )
        return ToolMessage(content=row.content or "", tool_call_id=row.tool_call_id)
    raise ValueError(f"未知消息角色 {row.role!r}（id={row.id}）")


# ---------------------------------------------------------------------------
# FAQ
# ---------------------------------------------------------------------------


class FaqRepository:
    """``faq`` 表的关键词查询。

    **转义由本类负责**，调用方（``app/tools/knowledge.py``）把关键词原样传进来即可。
    把转义放在这一层是因为它属于「怎么拼 SQL」，而不是「业务上要查什么」；
    让工具层也转一次会造成**双重转义**（``100%`` → ``100\\%`` → ``100\\\\%``），
    结果是本来能命中的词反而查不到。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _search(self, keyword: str, limit: int) -> list[FaqHit]:
        pattern = _like_pattern(keyword)
        rows = self._session.scalars(
            select(Faq)
            .where(
                or_(
                    Faq.question.like(pattern, escape=LIKE_ESCAPE_CHAR),
                    Faq.answer.like(pattern, escape=LIKE_ESCAPE_CHAR),
                )
            )
            .order_by(Faq.id)
            .limit(limit)
        ).all()
        return [FaqHit(question=r.question, answer=r.answer, category=r.category) for r in rows]

    async def search(self, keyword: str, limit: int = 5) -> list[FaqHit]:
        """按关键词做 ``LIKE`` 匹配，问题与答案两处都找。

        返回空列表表示**没查到**（正常业务结果，不是错误）——
        验收标准 3 的「漏召回」正是走到这里。
        """
        return await _run(self._search, keyword, limit)


# ---------------------------------------------------------------------------
# 工单
# ---------------------------------------------------------------------------


class TicketRepository:
    """``tickets`` 表的写入。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _count_on(self, day: date) -> int:
        prefix = f"T{day:%Y%m%d}"
        return int(
            self._session.scalar(
                select(func.count()).select_from(Ticket).where(Ticket.ticket_no.like(f"{prefix}%"))
            )
            or 0
        )

    def _create(
        self, conversation_id: int, description: str, ticket_type: str, day: date
    ) -> str:
        if ticket_type not in _TICKET_TYPES:
            raise ValueError(
                f"非法工单类型 {ticket_type!r}，只允许 {_TICKET_TYPES}（对应 DDL 的 ENUM）"
            )

        ticket_no = f"T{day:%Y%m%d}{self._count_on(day) + 1:03d}"
        self._session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
            )
        )
        self._session.commit()
        return ticket_no

    async def create(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        *,
        day: date | None = None,
    ) -> str:
        """写入工单并返回工单号。

        ``day`` 可注入：测试要断言「同一天序号递增、换一天重新从 1 开始」，
        不能依赖跑测试当天是不是跨了零点。

        :raises ValueError: ``ticket_type`` 不在 DDL 的 ENUM 取值内。
        """
        return await _run(
            self._create, conversation_id, description, ticket_type, day or date.today()
        )

    def _count_created_on(self, day: date) -> int:
        return self._count_on(day)

    async def count_created_on(self, day: date) -> int:
        return await _run(self._count_created_on, day)


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Repositories:
    """四个仓库的装配容器。

    作为**一个**依赖注入点，而不是四个：编排层需要同时拿到会话与消息仓库，
    分四个参数会让每个调用点都要透传四个参数。测试替换整个容器即可离线运行。
    """

    conversations: ConversationRepository
    messages: MessageRepository
    faq: FaqRepository
    tickets: TicketRepository

    @classmethod
    def from_session(cls, session: Session) -> "Repositories":
        """按一个 ``Session`` 装配出四个仓库。

        **同步**方法（不是 ``async``）：它只是构造四个对象，没有任何 IO。
        写成 ``async`` 会逼每个调用方 ``await`` 一个不等待任何东西的调用。
        """
        return cls(
            conversations=ConversationRepository(session),
            messages=MessageRepository(session),
            faq=FaqRepository(session),
            tickets=TicketRepository(session),
        )
