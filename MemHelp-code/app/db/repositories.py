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
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.orm import Session

from app.db.models import (
    MESSAGE_ROLES,
    TICKET_TYPES as _TICKET_TYPES,
    Conversation,
    Faq,
    KnowledgeChunk,
    Message,
    QaStaging,
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
    #: 给界面看的工具轨迹（含耗时/成败）。**不参与模型历史**，见 models.Message。
    tool_traces: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class ChunkRow:
    """一条知识块（纯数据，跨线程安全）。

    进向量的只有 ``category`` / ``questions`` / ``answer``；其余是元数据与双写状态。
    """

    id: int
    category: str
    questions: str
    answer: str
    section_path: str | None
    content_type: str | None
    is_key_clause: bool
    prev_chunk_id: int | None
    next_chunk_id: int | None
    vector_id: str | None
    vectorize_status: str


@dataclass(frozen=True)
class StagingRow:
    """暂存表里的一条候选问答对。"""

    id: int
    batch_no: str
    source_ref: str | None
    question: str
    answer: str
    status: str


@dataclass(frozen=True)
class SessionRow:
    """会话的轻量视图（挖矿时用来枚举「有哪些会话」）。"""

    session_id: str
    status: str


def _field(row: object, name: str) -> Any:
    """从 dict 或任意对象上取一个字段。

    仓库层接受**duck-typed 行**：切分器有 ``Chunk``，挖矿有 ``ExtractedQa``，
    测试有各种小类。让它们各自定义形状，比逼所有调用方 import 仓库层的类型好。
    """
    if isinstance(row, dict):
        return row[name]
    return getattr(row, name)


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

    def _list_sessions(self, limit: int | None) -> list[SessionRow]:
        stmt = select(Conversation.session_id, Conversation.status).order_by(Conversation.id)
        if limit is not None:
            stmt = stmt.limit(limit)
        return [
            SessionRow(session_id=str(session_id), status=str(status))
            for session_id, status in self._session.execute(stmt).all()
        ]

    async def list_sessions(self, limit: int | None = None) -> list[SessionRow]:
        """按创建顺序枚举会话（ch03 挖矿的输入）。

        只返回 ``session_id`` 与 ``status``：挖矿只需要「有哪些会话」以及
        「是否已结束」，不需要把整个会话对象搬出来。
        """
        return await _run(self._list_sessions, limit)


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
        tool_traces: Sequence[dict[str, Any]] | None,
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
                tool_traces=list(tool_traces) if tool_traces is not None else None,
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
        tool_traces: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """追加一条消息。``role`` 必须是 ``MESSAGE_ROLES`` 之一。

        :param tool_traces: 给界面看的工具轨迹（含耗时/成败）。
            **必须与 ``tool_calls`` 分开传** —— 后者会被当模型历史回灌，
            字段是严格的（多一个键就会让 ``AIMessage`` 构造失败）。

        :raises ValueError: 角色不在 DDL 的 ENUM 取值内。
        """
        await _run(
            self._append,
            conversation_id,
            role,
            content,
            tool_calls,
            tool_call_id,
            tool_traces,
        )

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
                tool_traces=row.tool_traces,
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

        注意：ch03 起 ``query_faq`` **不再调用本方法**（已改走向量检索）。
        保留它是为了两件事：ch02 的既有测试，以及评估集里的**关键词基线**
        （同一语料、只换检索方式，才能证明向量升级真的解决了漏召回）。
        """
        return await _run(self._search, keyword, limit)

    def _list_all(self) -> list[FaqHit]:
        rows = self._session.scalars(select(Faq).order_by(Faq.id)).all()
        return [FaqHit(question=r.question, answer=r.answer, category=r.category) for r in rows]

    async def list_all(self) -> list[FaqHit]:
        """把 ``faq`` 表整表读出来 —— ch03 建库时把它灌进 ``knowledge_chunks``。"""
        return await _run(self._list_all)


# ---------------------------------------------------------------------------
# ch03：知识块（原文权威源 + 双写状态机）
# ---------------------------------------------------------------------------


class KnowledgeChunkRepository:
    """``knowledge_chunks`` 的读写。

    ## 自然键幂等

    chunk 的自增主键不能当幂等键（重跑会拿到新 id）。真正稳定的身份是
    ``(content_type, section_path, questions)``：同一篇文档的同一节标题，
    重跑一万次都指向同一行。``answer`` 变了就 UPDATE 并把状态打回 ``pending``，
    让旧向量作废、等着重算。

    ## ``section_path`` 用空串而不是 NULL

    MySQL 里 ``NULL != NULL``，拿它参与自然键比较会让每一行都「不相等」，
    幂等直接失效（表现为重跑后块数翻倍）。所以写入时统一 ``or ""``。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _upsert_chunk(self, chunk: object) -> tuple[int, str]:
        content_type = _field(chunk, "content_type")
        section_path = _field(chunk, "section_path") or ""
        questions = _field(chunk, "questions")
        answer = _field(chunk, "answer")

        existing = self._session.scalars(
            select(KnowledgeChunk).where(
                KnowledgeChunk.content_type == content_type,
                KnowledgeChunk.section_path == section_path,
                KnowledgeChunk.questions == questions,
            )
        ).first()

        if existing is None:
            row = KnowledgeChunk(
                category=_field(chunk, "category"),
                questions=questions,
                answer=answer,
                section_path=section_path,
                content_type=content_type,
                is_key_clause=bool(_field(chunk, "is_key_clause")),
            )
            self._session.add(row)
            self._session.commit()
            return int(row.id), "created"

        if existing.answer != answer or existing.category != _field(chunk, "category"):
            # 原文**或分类**变了 → 旧向量作废。category 也进向量化文本
            # （`render_embed_text` 的第一格），只盯 answer 会留下「答案没变、
            # 分类变了」的陈旧向量，状态却还是 done。
            existing.answer = answer
            existing.category = _field(chunk, "category")
            existing.is_key_clause = bool(_field(chunk, "is_key_clause"))
            existing.vectorize_status = "pending"
            existing.vector_id = None
            self._session.commit()
            return int(existing.id), "updated"
        return int(existing.id), "unchanged"

    async def upsert_chunk(self, chunk: object) -> tuple[int, str]:
        """按自然键写入一块，返回 ``(id, outcome)``。

        :param chunk: duck-typed 行，需有 ``category`` / ``questions`` / ``answer`` /
            ``section_path`` / ``content_type`` / ``is_key_clause``。
        :return: ``outcome`` 是 ``"created"`` / ``"updated"`` / ``"unchanged"`` 之一。
            三值而不是 bool：建库报告要分别统计「新建 / 更新 / 跳过」，
            合成两值就分不出「重跑没事干」与「原文改过了」。
        """
        return await _run(self._upsert_chunk, chunk)

    def _link_neighbours(self, ids: list[int]) -> None:
        if not ids:
            return
        # **先把这批块的指针清空**：否则上次的尾块（现在不再是尾块）会留着
        # 一个指向已删除块的 next_chunk_id，顺着链走会走到不存在的原文。
        self._session.execute(
            update(KnowledgeChunk)
            .where(KnowledgeChunk.id.in_(ids))
            .values(prev_chunk_id=None, next_chunk_id=None)
        )
        for left, right in zip(ids, ids[1:]):
            self._session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id == left)
                .values(next_chunk_id=right)
            )
            self._session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id == right)
                .values(prev_chunk_id=left)
            )
        self._session.commit()

    async def link_neighbours(self, ids_in_order: Sequence[int]) -> None:
        """按给定顺序把同一篇内的块前后连起来。跨文档不要连。"""
        await _run(self._link_neighbours, list(ids_in_order))

    def _list_pending(self, limit: int | None) -> list[ChunkRow]:
        stmt = (
            select(KnowledgeChunk)
            .where(
                or_(
                    KnowledgeChunk.vectorize_status == "pending",
                    KnowledgeChunk.vector_id.is_(None),
                )
            )
            .order_by(KnowledgeChunk.id)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return [_to_chunk_row(r) for r in self._session.scalars(stmt).all()]

    async def list_pending(self, limit: int | None = None) -> list[ChunkRow]:
        """待向量化的块（``pending`` 或 ``vector_id`` 为空）。续跑只拣这些。"""
        return await _run(self._list_pending, limit)

    def _mark_done(self, pairs: list[tuple[int, str]]) -> None:
        for chunk_id, vector_id in pairs:
            self._session.execute(
                update(KnowledgeChunk)
                .where(KnowledgeChunk.id == chunk_id)
                .values(vector_id=vector_id, vectorize_status="done")
            )
        self._session.commit()

    async def mark_done(self, pairs: Sequence[tuple[int, str]]) -> None:
        """回填 ``vector_id`` 并把状态转 ``done``（Milvus 写成功之后才调）。"""
        await _run(self._mark_done, list(pairs))

    def _get_many(self, ids: list[int]) -> list[ChunkRow]:
        if not ids:
            return []
        rows = self._session.scalars(
            select(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))
        ).all()
        return [_to_chunk_row(r) for r in rows]

    async def get_many(self, ids: Sequence[int]) -> list[ChunkRow]:
        """按 id 批量取。顺序由调用方按向量分数重排，这里不保证。"""
        return await _run(self._get_many, list(ids))

    def _count_by_status(self) -> dict[str, int]:
        rows = self._session.execute(
            select(KnowledgeChunk.vectorize_status, func.count()).group_by(
                KnowledgeChunk.vectorize_status
            )
        ).all()
        counts = {"pending": 0, "done": 0}
        for status, n in rows:
            counts[str(status)] = int(n)
        return counts

    async def count_by_status(self) -> dict[str, int]:
        return await _run(self._count_by_status)

    def _truncate(self) -> None:
        # 用 DELETE 而不是 TRUNCATE：MySQL 的 TRUNCATE 会被自引用外键拒绝。
        self._session.execute(delete(KnowledgeChunk))
        self._session.commit()

    async def truncate(self) -> None:
        await _run(self._truncate)

    def _list_by_scope(self, root: str) -> list[ChunkRow]:
        """取某篇文档派生出的全部块（``section_path`` 以该标题为根）。

        一篇文档的块共享同一个一级标题，所以「根 + 分隔符」能圈定它们；
        同节多块带 `` #n`` 后缀，也在这个范围里。
        """
        conditions = [KnowledgeChunk.section_path == root]
        for separator in (" /", " #"):
            conditions.append(
                KnowledgeChunk.section_path.like(f"{root}{separator}%", escape="\\")
            )
        rows = self._session.scalars(select(KnowledgeChunk).where(or_(*conditions))).all()
        return [_to_chunk_row(r) for r in rows]

    async def list_by_scope(self, root: str) -> list[ChunkRow]:
        """按文档根标题取块 —— 增量重建时用来找出「本次已不存在的旧块」。"""
        return await _run(self._list_by_scope, root)

    def _delete_ids(self, ids: list[int]) -> int:
        if not ids:
            return 0
        result = self._session.execute(
            delete(KnowledgeChunk).where(KnowledgeChunk.id.in_(ids))
        )
        self._session.commit()
        return int(result.rowcount or 0)

    async def delete_ids(self, ids: Sequence[int]) -> int:
        """删掉指定块（调用方**必须先删掉 Milvus 里对应的向量**）。"""
        return await _run(self._delete_ids, list(ids))


def _to_chunk_row(row: KnowledgeChunk) -> ChunkRow:
    return ChunkRow(
        id=int(row.id),
        category=row.category,
        questions=row.questions,
        answer=row.answer,
        section_path=row.section_path,
        content_type=row.content_type,
        is_key_clause=bool(row.is_key_clause),
        prev_chunk_id=row.prev_chunk_id,
        next_chunk_id=row.next_chunk_id,
        vector_id=row.vector_id,
        vectorize_status=row.vectorize_status,
    )


# ---------------------------------------------------------------------------
# ch03：问答抽取暂存表
# ---------------------------------------------------------------------------


class QaStagingRepository:
    """``qa_extraction_staging`` 的读写。

    这张表是「抽出来」与「判重入库」之间的缓冲：抽取按批写、去重在全量上做，
    中间还能调阈值重跑（``--dedupe-only``）。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _add_rows(self, rows: list[object]) -> None:
        for row in rows:
            self._session.add(
                QaStaging(
                    batch_no=_field(row, "batch_no"),
                    source_ref=_field(row, "source_ref"),
                    question=_field(row, "question"),
                    answer=_field(row, "answer"),
                )
            )
        self._session.commit()

    async def add_rows(self, rows: Sequence[object]) -> None:
        """一批抽取结果一次性写入（整批提交：失败就整批不落，见 mining）。"""
        await _run(self._add_rows, list(rows))

    def _list_by_status(self, status: str) -> list[StagingRow]:
        rows = self._session.scalars(
            select(QaStaging).where(QaStaging.status == status).order_by(QaStaging.id)
        ).all()
        return [
            StagingRow(
                id=int(r.id),
                batch_no=r.batch_no,
                source_ref=r.source_ref,
                question=r.question,
                answer=r.answer,
                status=r.status,
            )
            for r in rows
        ]

    async def list_by_status(self, status: str) -> list[StagingRow]:
        return await _run(self._list_by_status, status)

    def _mark(self, ids: list[int], status: str) -> None:
        if not ids:
            return
        self._session.execute(
            update(QaStaging).where(QaStaging.id.in_(ids)).values(status=status)
        )
        self._session.commit()

    async def mark(self, ids: Sequence[int], status: str) -> None:
        """把一批行标成 ``kept`` / ``discarded``（丢弃行留档，便于人工复核）。"""
        await _run(self._mark, list(ids), status)

    def _known_sources(self) -> set[str]:
        rows = self._session.scalars(
            select(QaStaging.source_ref).where(QaStaging.source_ref.is_not(None)).distinct()
        ).all()
        return {str(r) for r in rows if r is not None}

    async def known_sources(self) -> set[str]:
        """已经挖过的会话标识 —— 定时任务靠它做幂等（重跑不重复挖）。"""
        return await _run(self._known_sources)

    def _count_by_status(self) -> dict[str, int]:
        rows = self._session.execute(
            select(QaStaging.status, func.count()).group_by(QaStaging.status)
        ).all()
        return {str(status): int(n) for status, n in rows}

    async def count_by_status(self) -> dict[str, int]:
        return await _run(self._count_by_status)

    def _update_questions(self, row_id: int, questions: str) -> None:
        # 暂存表的列叫 question（单数）；合并后的多行问法直接写进这一列，
        # 入 knowledge_chunks 时再原样搬到那边的 questions（复数）。
        self._session.execute(
            update(QaStaging).where(QaStaging.id == row_id).values(question=questions)
        )
        self._session.commit()

    async def update_questions(self, row_id: int, questions: str) -> None:
        """同义问法合并后回写 ``question``（多行）。"""
        await _run(self._update_questions, row_id, questions)


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
    #: ch03：知识块（原文权威源 + 双写状态）
    knowledge: KnowledgeChunkRepository
    #: ch03：对话抽取暂存
    qa_staging: QaStagingRepository

    @classmethod
    def from_session(cls, session: Session) -> "Repositories":
        """按一个 ``Session`` 装配出全部仓库。

        **同步**方法（不是 ``async``）：它只是构造几个对象，没有任何 IO。
        写成 ``async`` 会逼每个调用方 ``await`` 一个不等待任何东西的调用。
        """
        return cls(
            conversations=ConversationRepository(session),
            messages=MessageRepository(session),
            faq=FaqRepository(session),
            tickets=TicketRepository(session),
            knowledge=KnowledgeChunkRepository(session),
            qa_staging=QaStagingRepository(session),
        )
