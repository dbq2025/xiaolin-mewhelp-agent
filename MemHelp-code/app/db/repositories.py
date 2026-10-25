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

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from typing import Any, TypeVar

import anyio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import (
    FAITH_CASE_STATUSES as _FAITH_CASE_STATUSES,
    LOW_CONFIDENCE_SOURCES as _LOW_CONFIDENCE_SOURCES,
    MESSAGE_ROLES,
    REVIEW_STATUSES as _REVIEW_STATUSES,
    TICKET_TYPES as _TICKET_TYPES,
    Conversation,
    ConversationSummary,
    EvalRun,
    FaithCase,
    Faq,
    KnowledgeChunk,
    LowConfidenceQuestion,
    Message,
    QaStaging,
    ReviewQueue,
    Ticket,
    ToolAuditLog,
    TopicClassification,
)

T = TypeVar("T")

#: 工单类型，取自 ORM 层的**单一来源**后再导出。
#:
#: 为什么在这里再导出一遍：工具层只允许依赖本模块（见
#: ``tests/test_tool_knowledge.py`` 的分层用例），不该 import ``app.db.models``。
#: 若让工具层自己写一份常量，DDL 改了 ENUM 而常量没跟上，就会变成
#: 「参数校验通过、写库被拒」这种最难查的半对齐状态。
TICKET_TYPES: tuple[str, ...] = _TICKET_TYPES

#: ch04：低置信度入池入口与台账处置状态，同样取自 ORM 层的单一来源。
LOW_CONFIDENCE_SOURCES: tuple[str, ...] = _LOW_CONFIDENCE_SOURCES
FAITH_CASE_STATUSES: tuple[str, ...] = _FAITH_CASE_STATUSES
REVIEW_STATUSES: tuple[str, ...] = _REVIEW_STATUSES

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
    #: ch04：本轮引用证据快照。刷新页面后角标还能点开，就靠它。
    citations: list[dict[str, Any]] | None = None
    #: ch05：本轮建议动作（转人工 / 建工单）。刷新后按钮还能渲染，就靠它。
    actions: list[dict[str, Any]] | None = None
    #: ch09：👎 反馈要按会话回捞，消息行得能回答「我属于哪个会话」。
    conversation_id: int | None = None


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


class SummarySegmentConflict(Exception):
    """同一个 ``(conversation_id, segment_no)`` 已被别的写入方占了。

    **必须是一个具名异常，不能让 ``IntegrityError`` 冒到调用方**：后台摘要任务
    跑在独立 Task 里，它的失败没人 await，只能靠日志。抛出「数据库完整性错误」
    这句话读起来像数据库故障，而真实含义是「另一个进程已经把这段压完了，放弃即可」。
    把语义翻译在这一层，任务是放弃、不是报警。
    """


@dataclass(frozen=True)
class AnchorRow:
    """``conversations`` 上的两个锚点 + 主键。

    单独一个数据类而不是把整行 ORM 对象递出去：调用方只需要两个整数，
    递 ORM 对象会让它有机会 lazy-load 别的列（跨线程就是一个新的数据库往返）。
    """

    conversation_id: int
    summary_upto_msg_id: int | None
    layer1_from_msg_id: int | None


@dataclass(frozen=True)
class SummaryRow:
    """``conversation_summaries`` 的一段梗概。"""

    id: int
    conversation_id: int
    segment_no: int
    from_msg_id: int
    upto_msg_id: int
    content: str
    msg_count: int
    source_tokens: int
    model: str | None
    elapsed_ms: int | None
    created_at: datetime


@dataclass(frozen=True)
class ConversationListRow:
    """会话侧栏的一行。

    侧栏只要「切换 + 预览 + 是否已压过」，不需要整条会话历史，所以这里一次
    查询就算好 ``message_count`` / ``preview`` / ``summary_segments``：
    逐会话再查一次会变成 N+1 次数据库往返，而侧栏是**每次进聊天页都要加载**的。
    """

    id: int
    session_id: str
    user_id: str
    status: str
    preview: str
    message_count: int
    summary_segments: int
    summary_upto_msg_id: int | None
    layer1_from_msg_id: int | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class LowConfidenceRow:
    """低置信度池里的一条。"""

    id: int
    conversation_id: int | None
    raw_question: str
    source: str
    reason: str | None
    created_at: datetime
    #: ch09：当轮召回片段快照；老数据没有这一列的值。
    retrieved_chunks: list[dict[str, Any]] | None = None
    #: ch09：归并到的待审行；未归并时为 ``None``。
    matched_review_id: int | None = None


def _low_confidence_row(row: LowConfidenceQuestion) -> LowConfidenceRow:
    return LowConfidenceRow(
        id=int(row.id),
        conversation_id=row.conversation_id,
        raw_question=row.raw_question,
        source=row.source,
        reason=row.reason,
        created_at=row.created_at,
        retrieved_chunks=row.retrieved_chunks,
        matched_review_id=row.matched_review_id,
    )


@dataclass(frozen=True)
class ReviewRow:
    """待审队列里的一条。"""

    id: int
    normalized_question: str
    ai_suggested_answer: str | None
    occurrence_count: int
    review_status: str
    approved_answer: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class EvalRunRow:
    """一条评估记录。"""

    id: int
    triggered_by: str
    dataset_size: int
    metrics: dict[str, Any]
    created_at: datetime


def _review_row(row: ReviewQueue) -> ReviewRow:
    return ReviewRow(
        id=int(row.id),
        normalized_question=row.normalized_question,
        ai_suggested_answer=row.ai_suggested_answer,
        occurrence_count=int(row.occurrence_count),
        review_status=row.review_status,
        approved_answer=row.approved_answer,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


@dataclass(frozen=True)
class FaithCaseRow:
    """编造个案台账里的一条。"""

    id: int
    eval_id: str
    bucket: str
    query: str
    strategy: str
    answer: str
    reason: str
    citations: list[dict[str, Any]] | None
    judge_model: str | None
    status: str
    seen_count: int
    first_seen_at: datetime
    last_seen_at: datetime
    resolution: str | None
    resolved_at: datetime | None


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

    def _find_session_id(self, conversation_id: int) -> str | None:
        found = self._session.scalar(
            select(Conversation.session_id).where(Conversation.id == conversation_id)
        )
        return str(found) if found is not None else None

    async def find_session_id(self, conversation_id: int) -> str | None:
        """``find_id`` 的反向：``conversations.id`` → ``session_id``。

        侧栏列的是 ``id``（数字，详情接口按它取数），聊天用的是 ``session_id``。
        回载历史要复用「按 session_id 取消息」那唯一一份实现，就得先转一道。
        """
        return await _run(self._find_session_id, conversation_id)

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

    # -- ch07：两个锚点 ----------------------------------------------------

    def _get_anchors(self, conversation_id: int) -> AnchorRow | None:
        row = self._session.execute(
            select(
                Conversation.id,
                Conversation.summary_upto_msg_id,
                Conversation.layer1_from_msg_id,
            ).where(Conversation.id == conversation_id)
        ).first()
        if row is None:
            return None
        return AnchorRow(
            conversation_id=int(row[0]),
            summary_upto_msg_id=None if row[1] is None else int(row[1]),
            layer1_from_msg_id=None if row[2] is None else int(row[2]),
        )

    async def get_anchors(self, conversation_id: int) -> AnchorRow | None:
        """读两个锚点。会话不存在返回 ``None``。"""
        return await _run(self._get_anchors, conversation_id)

    def _set_anchors(
        self,
        conversation_id: int,
        summary_upto_msg_id: int | None,
        layer1_from_msg_id: int | None,
        only_if_layer1_from: int | None,
    ) -> bool:
        """写两个锚点，带两条守卫：**单调**与**乐观并发**。

        单调：两个锚点都只允许变大。降级是「把边界往前推」，推回去意味着曾经
        被截短的消息又要变回原文、曾经被梗概覆盖的消息又要重进上下文 —— 那会让
        同一事实既在梗概里又在原文里，也破坏「压完不回头重写」的承诺。

        层 1 不得落进摘要区间：``layer1_from >= summary_upto + 1``。同一次调用里
        ``summary_upto`` 追上来了，就把 ``layer1_from`` 一起顶上去，而不是放任
        两个锚点交叉。

        ``only_if_layer1_from``：后台任务读历史 → 调模型 → 落库，中间隔着一次
        网络往返（几秒）。这期间用户可能又聊了几轮、锚点已被本轮拒绝降级推走。
        落库前复核一次，对不上就返回 ``False`` 让任务放弃 —— **不重试**：重试会
        压到一段已经变了的区间，而那段的梗概可能已经由本轮的写入方产出。
        """
        current = self._session.execute(
            select(Conversation.summary_upto_msg_id, Conversation.layer1_from_msg_id).where(
                Conversation.id == conversation_id
            )
        ).first()
        if current is None:
            return False
        current_summary = None if current[0] is None else int(current[0])
        current_layer1 = None if current[1] is None else int(current[1])

        if only_if_layer1_from is not None and current_layer1 != only_if_layer1_from:
            return False

        next_summary = current_summary
        if summary_upto_msg_id is not None:
            next_summary = (
                summary_upto_msg_id
                if current_summary is None
                else max(current_summary, summary_upto_msg_id)
            )

        next_layer1 = current_layer1
        if layer1_from_msg_id is not None:
            next_layer1 = (
                layer1_from_msg_id
                if current_layer1 is None
                else max(current_layer1, layer1_from_msg_id)
            )

        # 层 1 不许落进摘要区间。摘要边界已知而层 1 更小 → 顶到摘要之后。
        if next_summary is not None and (next_layer1 is None or next_layer1 <= next_summary):
            next_layer1 = next_summary + 1

        if next_summary == current_summary and next_layer1 == current_layer1:
            return True

        self._session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(
                summary_upto_msg_id=next_summary,
                layer1_from_msg_id=next_layer1,
            )
        )
        self._session.commit()
        return True

    async def set_anchors(
        self,
        *,
        conversation_id: int,
        summary_upto_msg_id: int | None = None,
        layer1_from_msg_id: int | None = None,
        only_if_layer1_from: int | None = None,
    ) -> bool:
        """推进锚点。返回值表示「写成功且未被并发守卫拦下」。

        传 ``None`` 表示「这一项不动」（不是「置空」）：两个锚点一旦有了值就
        只增不减，没有清空需求 —— 清空等于撤销摘要，而摘要段已经落库了。
        """
        return await _run(
            self._set_anchors,
            conversation_id,
            summary_upto_msg_id,
            layer1_from_msg_id,
            only_if_layer1_from,
        )

    # -- ch07：会话侧栏 ----------------------------------------------------

    def _list_for_user(self, user_id: str, limit: int) -> list[ConversationListRow]:
        """一次查询算齐侧栏要的全部字段。

        三次 ``func.count`` 走**相关子查询**而不是 JOIN + GROUP BY：一个会话可能
        有多条消息、多段梗概，JOIN 会先做笛卡尔积再聚合，条数一多就退化。相关
        子查询各自走 ``idx_conversation_id`` / ``idx_summary_conversation_id``。
        """
        message_count = (
            select(func.count(Message.id))
            .where(Message.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery()
        )
        summary_segments = (
            select(func.count(ConversationSummary.id))
            .where(ConversationSummary.conversation_id == Conversation.id)
            .correlate(Conversation)
            .scalar_subquery()
        )
        # 预览取**首条 user 消息**：侧栏要回答「这个会话在聊什么」，用最新一句
        # 会变成「好的」这种无信息量的东西。取最小 id 而不是最小 created_at ——
        # 同一秒内连发两条时时间戳会并列，id 不会。
        first_user_id = (
            select(func.min(Message.id))
            .where(Message.conversation_id == Conversation.id, Message.role == "user")
            .correlate(Conversation)
            .scalar_subquery()
        )
        preview = (
            select(Message.content)
            .where(Message.id == first_user_id)
            .correlate(Conversation)
            .scalar_subquery()
        )

        stmt = (
            select(
                Conversation.id,
                Conversation.session_id,
                Conversation.user_id,
                Conversation.status,
                preview,
                message_count,
                summary_segments,
                Conversation.summary_upto_msg_id,
                Conversation.layer1_from_msg_id,
                Conversation.created_at,
                Conversation.updated_at,
            )
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.id.desc())
            .limit(limit)
        )

        return [
            ConversationListRow(
                id=int(row[0]),
                session_id=str(row[1]),
                user_id=str(row[2]),
                status=str(row[3]),
                preview=_preview_text(row[4]),
                message_count=int(row[5] or 0),
                summary_segments=int(row[6] or 0),
                summary_upto_msg_id=None if row[7] is None else int(row[7]),
                layer1_from_msg_id=None if row[8] is None else int(row[8]),
                created_at=row[9],
                updated_at=row[10],
            )
            for row in self._session.execute(stmt).all()
        ]

    async def list_for_user(self, *, user_id: str, limit: int = 50) -> list[ConversationListRow]:
        """某个用户的全部会话，**新的在前**（侧栏顺序）。"""
        return await _run(self._list_for_user, user_id, limit)


#: 侧栏预览的最大字数。
#:
#: 60 而不是 20：中文客服场景里「我上周买的那个蓝牙耳机什么时候能到」这种
#: 一句话就 20 字，截太短等于全部显示成同一个开头。60 字能装下两三句。
PREVIEW_MAX_CHARS = 60


def _preview_text(content: str | None) -> str:
    """把首问压成一行预览：折叠空白、截断、补省略号。"""
    if not content:
        return ""
    text = " ".join(str(content).split())
    if len(text) <= PREVIEW_MAX_CHARS:
        return text
    return text[:PREVIEW_MAX_CHARS] + "…"


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
        citations: Sequence[dict[str, Any]] | None = None,
        actions: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self._check_role(role)

        self._session.add(
            Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                tool_calls=list(tool_calls) if tool_calls is not None else None,
                tool_call_id=tool_call_id,
                tool_traces=list(tool_traces) if tool_traces is not None else None,
                citations=list(citations) if citations is not None else None,
                actions=list(actions) if actions is not None else None,
            )
        )
        self._session.commit()

    @staticmethod
    def _check_role(role: str) -> None:
        # 入口校验：非法 role 在这里就炸，错误信息带角色名。
        # 靠 MySQL 的 ENUM 兜会拖到 commit，报的是驱动的
        # "Data truncated for column 'role'" —— 把拼错字符串伪装成数据库故障。
        if role not in MESSAGE_ROLES:
            raise ValueError(
                f"非法消息角色 {role!r}，只允许 {MESSAGE_ROLES}（对应 DDL 的 ENUM）"
            )

    async def append(
        self,
        conversation_id: int,
        *,
        role: str,
        content: str | None,
        tool_calls: Sequence[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
        tool_traces: Sequence[dict[str, Any]] | None = None,
        citations: Sequence[dict[str, Any]] | None = None,
        actions: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """追加一条消息。``role`` 必须是 ``MESSAGE_ROLES`` 之一。

        :param tool_traces: 给界面看的工具轨迹（含耗时/成败）。
            **必须与 ``tool_calls`` 分开传** —— 后者会被当模型历史回灌，
            字段是严格的（多一个键就会让 ``AIMessage`` 构造失败）。

        :param citations: ch04 的引用证据快照。只有**最终答案**那一行需要它；
            带工具调用的「申请单」行不需要。

        :param actions: ch05 的建议动作快照。同样只挂最终答案那一行。

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
            citations,
            actions,
        )

    def _append_many(
        self, conversation_id: int, rows: Sequence[dict[str, Any]]
    ) -> list[int]:
        """一次事务写入多行，返回新行 id（按传入顺序）。

        **先把所有 role 校验完再 add**：否则第二行非法时第一行已经进 Session，
        调用方 catch 了异常也不知道要不要 rollback。这里要么全写、要么一行不写。
        """
        for row in rows:
            self._check_role(str(row["role"]))

        messages: list[Message] = []
        for row in rows:
            message = Message(
                conversation_id=conversation_id,
                role=row["role"],
                content=row.get("content"),
                tool_calls=row.get("tool_calls"),
                tool_call_id=row.get("tool_call_id"),
                tool_traces=row.get("tool_traces"),
                citations=row.get("citations"),
                actions=row.get("actions"),
            )
            self._session.add(message)
            messages.append(message)
        self._session.flush()
        ids = [int(message.id) for message in messages]
        self._session.commit()
        return ids

    async def append_many(
        self, conversation_id: int, rows: Sequence[dict[str, Any]]
    ) -> list[int]:
        """把一轮的多行（user / assistant tool_calls / tool / assistant）单事务写入。

        ch05 的 ReAct 一轮可能产生多组工具调用；逐行 ``append`` 每次 commit，
        中途失败会留下「有工具申请单、没有工具结果」的半截历史，下一轮回灌时
        上游会直接拒绝。单事务是这条链路的结构性保证。

        返回 id 供 ch09 的 👎 反馈精确定位「是哪一条回答被差评」。
        """
        return await _run(self._append_many, conversation_id, rows)

    def _get(self, message_id: int) -> MessageRow | None:
        row = self._session.get(Message, message_id)
        if row is None:
            return None
        return MessageRow(
            id=row.id,
            role=row.role,
            content=row.content,
            tool_calls=row.tool_calls,
            tool_call_id=row.tool_call_id,
            created_at=row.created_at,
            tool_traces=row.tool_traces,
            citations=row.citations,
            actions=row.actions,
            conversation_id=row.conversation_id,
        )

    async def get(self, message_id: int) -> MessageRow | None:
        """按主键取一条消息；不存在返回 ``None``。"""
        return await _run(self._get, message_id)

    def _last_user_before(
        self, conversation_id: int, message_id: int
    ) -> MessageRow | None:
        row = self._session.scalars(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                Message.role == "user",
                Message.id < message_id,
            )
            .order_by(Message.id.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        return MessageRow(
            id=row.id,
            role=row.role,
            content=row.content,
            tool_calls=row.tool_calls,
            tool_call_id=row.tool_call_id,
            created_at=row.created_at,
            tool_traces=row.tool_traces,
            citations=row.citations,
            actions=row.actions,
            conversation_id=row.conversation_id,
        )

    async def last_user_before(
        self, conversation_id: int, message_id: int
    ) -> MessageRow | None:
        """同会话里、这条消息之前最近的一条 user 消息。👎 用它反查问题原话。"""
        return await _run(self._last_user_before, conversation_id, message_id)

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
                citations=row.citations,
                actions=row.actions,
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

    def _last_user_content(self, session_id: str) -> str | None:
        value = self._session.scalar(
            select(Message.content)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.session_id == session_id, Message.role == "user")
            .order_by(Message.id.desc())
            .limit(1)
        )
        return None if value is None else str(value)

    async def last_user_content(self, session_id: str) -> str | None:
        """该会话最后一条用户消息。

        ch05 的「建工单」按钮只带 session_id 时，用这句话当工单描述 ——
        比让前端把整段聊天历史回传更安全，也比让模型再生成一次更省。
        """
        return await _run(self._last_user_content, session_id)


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
# 会话梗概
# ---------------------------------------------------------------------------


class ConversationSummaryRepository:
    """``conversation_summaries`` 表的读写 —— 只追加，不更新、不删除。

    没有任何 ``update`` / ``delete`` 方法不是遗漏：梗概是「历史被压过一次」的
    证据链，改它等于篡改事实。要修正措辞只能**追加新段**（那会让同一区间被压两次，
    由 ``UNIQUE(conversation_id, segment_no)`` 挡住）。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _append(
        self,
        conversation_id: int,
        segment_no: int,
        from_msg_id: int,
        upto_msg_id: int,
        content: str,
        msg_count: int,
        source_tokens: int,
        model: str | None,
        elapsed_ms: int | None,
    ) -> int:
        row = ConversationSummary(
            conversation_id=conversation_id,
            segment_no=segment_no,
            from_msg_id=from_msg_id,
            upto_msg_id=upto_msg_id,
            content=content,
            msg_count=msg_count,
            source_tokens=source_tokens,
            model=model,
            elapsed_ms=elapsed_ms,
        )
        self._session.add(row)
        try:
            self._session.commit()
        except IntegrityError as exc:
            # 唯一键冲突 = 另一个进程已经把这段压完了。翻译成具名异常让任务
            # 「安静放弃」，而不是让驱动错误一路冒到日志里显得像数据库故障。
            self._session.rollback()
            raise SummarySegmentConflict(
                f"会话 {conversation_id} 的第 {segment_no} 段梗概已存在（并发写入）"
            ) from exc
        return int(row.id)

    async def append(
        self,
        *,
        conversation_id: int,
        segment_no: int,
        from_msg_id: int,
        upto_msg_id: int,
        content: str,
        msg_count: int,
        source_tokens: int,
        model: str | None = None,
        elapsed_ms: int | None = None,
    ) -> int:
        """追加一段梗概，返回行 id。段号冲突抛 :class:`SummarySegmentConflict`。"""
        return await _run(
            self._append,
            conversation_id,
            segment_no,
            from_msg_id,
            upto_msg_id,
            content,
            msg_count,
            source_tokens,
            model,
            elapsed_ms,
        )

    def _list_for_conversation(self, conversation_id: int) -> list[SummaryRow]:
        rows = self._session.scalars(
            select(ConversationSummary)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.segment_no)
        ).all()
        return [
            SummaryRow(
                id=row.id,
                conversation_id=row.conversation_id,
                segment_no=row.segment_no,
                from_msg_id=row.from_msg_id,
                upto_msg_id=row.upto_msg_id,
                content=row.content,
                msg_count=row.msg_count,
                source_tokens=row.source_tokens,
                model=row.model,
                elapsed_ms=row.elapsed_ms,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_for_conversation(self, conversation_id: int) -> list[SummaryRow]:
        """按 ``segment_no`` 升序取全部梗概（段号是权威顺序，不是插入顺序）。"""
        return await _run(self._list_for_conversation, conversation_id)

    def _next_segment_no(self, conversation_id: int) -> int:
        current = self._session.scalar(
            select(func.max(ConversationSummary.segment_no)).where(
                ConversationSummary.conversation_id == conversation_id
            )
        )
        return 1 if current is None else int(current) + 1

    async def next_segment_no(self, conversation_id: int) -> int:
        """下一段的段号。**只是建议值** —— 真正的收口是唯一键。"""
        return await _run(self._next_segment_no, conversation_id)

    def _latest_texts(self, conversation_id: int, limit: int | None) -> list[str]:
        stmt = (
            select(ConversationSummary.content)
            .where(ConversationSummary.conversation_id == conversation_id)
            .order_by(ConversationSummary.segment_no.desc())
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        # 先按段号**倒序**取最近 N 段，再翻回正序：直接正序 + LIMIT 拿到的是
        # **最早**的 N 段，与「只喂最近的梗概」正好相反。
        return [str(text) for text in reversed(self._session.scalars(stmt).all())]

    async def latest_texts(self, conversation_id: int, limit: int | None = None) -> list[str]:
        """最近的若干段梗概正文，**按段号升序**返回。

        ``limit=None`` 取全部。有上限时取**最近**的 N 段：更早的段已经被后面的
        段在语义上隐含覆盖（「用户先问耳机、后报订单号」读起来比「只有第一段」更全），
        而把它们全喂进去只会挤占预算。
        """
        return await _run(self._latest_texts, conversation_id, limit)


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

    def _reset_status(self) -> int:
        """把**所有**块打回 ``pending`` 并清掉 ``vector_id``，原文本行不动。

        用途：Milvus 集合因 schema 升级被重建（ch04 的 BM25 字段只能在建集合时声明），
        而 ``knowledge_chunks`` 里的原文与「挖矿」产物要保留。只清表会丢数据；
        只删向量又会让 ``vectorize_status='done'`` 说谎 —— 续跑只拣 ``pending``，
        于是那些块永远回不到索引里，表现为「库里明明有，检索就是查不到」。
        """
        result = self._session.execute(
            update(KnowledgeChunk).values(vectorize_status="pending", vector_id=None)
        )
        self._session.commit()
        return int(result.rowcount or 0)

    async def reset_status(self) -> int:
        """把所有块打回 ``pending``，返回受影响行数。"""
        return await _run(self._reset_status)

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

    def _list_all(self, limit: int | None) -> list[ChunkRow]:
        statement = select(KnowledgeChunk).order_by(KnowledgeChunk.id)
        if limit is not None:
            statement = statement.limit(limit)
        return [_to_chunk_row(row) for row in self._session.scalars(statement).all()]

    async def list_all(self, limit: int | None = None) -> list[ChunkRow]:
        """取全部块的轻量视图 —— 评估集把「金标锚串」解析成 chunk_id 时用。

        为什么不写在评估脚本里：项目约定「查询一律在 repositories」，
        脚本里写裸 SQL 会让「表结构改了但脚本没跟上」变成运行期才发现的错误。
        """
        return await _run(self._list_all, limit)

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
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        day: date,
        idempotency_key: str | None = None,
    ) -> str:
        if ticket_type not in _TICKET_TYPES:
            raise ValueError(
                f"非法工单类型 {ticket_type!r}，只允许 {_TICKET_TYPES}（对应 DDL 的 ENUM）"
            )

        if idempotency_key:
            existing = self._session.scalar(
                select(Ticket.ticket_no).where(Ticket.idempotency_key == idempotency_key)
            )
            if existing is not None:
                return str(existing)

        ticket_no = f"T{day:%Y%m%d}{self._count_on(day) + 1:03d}"
        self._session.add(
            Ticket(
                ticket_no=ticket_no,
                conversation_id=conversation_id,
                description=description,
                ticket_type=ticket_type,
                idempotency_key=idempotency_key,
            )
        )
        self._session.commit()
        return ticket_no

    def _find_by_idempotency_key(
        self, idempotency_key: str
    ) -> tuple[str, str] | None:
        row = self._session.scalar(
            select(Ticket).where(Ticket.idempotency_key == idempotency_key)
        )
        if row is None:
            return None
        return str(row.ticket_no), str(row.description)

    async def find_by_idempotency_key(
        self, idempotency_key: str
    ) -> tuple[str, str] | None:
        """返回 ``(ticket_no, description)``；没有命中返回 None。

        退款端点用它判断「同一个 key 是不是同一笔申请」：同 key 不同描述
        不能把旧工单号当作本次成功返回。
        """
        return await _run(self._find_by_idempotency_key, idempotency_key)

    async def create(
        self,
        conversation_id: int,
        description: str,
        ticket_type: str,
        *,
        day: date | None = None,
        idempotency_key: str | None = None,
    ) -> str:
        """写入工单并返回工单号。

        ``day`` 可注入：测试要断言「同一天序号递增、换一天重新从 1 开始」，
        不能依赖跑测试当天是不是跨了零点。

        ``idempotency_key`` 是 ch05 的防重：同一个键重复提交返回同一张工单，
        不再生成新号。唯一索引是最后一道防线（并发下仍可能有一方失败）。

        :raises ValueError: ``ticket_type`` 不在 DDL 的 ENUM 取值内。
        """
        return await _run(
            self._create,
            conversation_id,
            description,
            ticket_type,
            day or date.today(),
            idempotency_key,
        )

    def _count_created_on(self, day: date) -> int:
        return self._count_on(day)

    async def count_created_on(self, day: date) -> int:
        return await _run(self._count_created_on, day)


# ---------------------------------------------------------------------------
# ch04：低置信度池 / 编造个案台账
# ---------------------------------------------------------------------------


class LowConfidenceQuestionRepository:
    """``low_confidence_questions`` 的写入与查询。

    **不做去重**：同一个问题问三次就是三行。这张表是「什么时候、因为什么答不上来」
    的时间序列，压成一行会把复发频率这条最有用的信号丢掉；归并是后续章节的事。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _add(
        self,
        conversation_id: int | None,
        raw_question: str,
        source: str,
        reason: str | None,
        retrieved_chunks: list[dict[str, Any]] | None,
    ) -> int:
        if source not in LOW_CONFIDENCE_SOURCES:
            raise ValueError(
                f"非法入池来源 {source!r}，只允许 {LOW_CONFIDENCE_SOURCES}"
                "（对应 DDL 的 ENUM）"
            )
        if not raw_question.strip():
            raise ValueError("入池的 raw_question 不能为空 —— 空问题没有复盘价值")

        row = LowConfidenceQuestion(
            conversation_id=conversation_id,
            raw_question=raw_question,
            source=source,
            reason=reason,
            retrieved_chunks=retrieved_chunks or None,
        )
        self._session.add(row)
        self._session.commit()
        return int(row.id)

    async def add(
        self,
        conversation_id: int | None,
        raw_question: str,
        source: str,
        reason: str | None = None,
        *,
        retrieved_chunks: list[dict[str, Any]] | None = None,
    ) -> int:
        """把一条答不上来的原话落进池子，返回主键。

        :param source: ``retrieval_low_conf`` / ``self_check`` / ``user_feedback``。
        :param reason: 判「不能答」的原因，复盘时看的就是它。
        :param retrieved_chunks: 当轮 Top-N 召回片段快照（原文 + 得分），
            给人工审核判断「知识库真缺」还是「有但没检到」。
        """
        return await _run(
            self._add, conversation_id, raw_question, source, reason, retrieved_chunks
        )

    def _get(self, question_id: int) -> LowConfidenceRow | None:
        row = self._session.get(LowConfidenceQuestion, question_id)
        return None if row is None else _low_confidence_row(row)

    async def get(self, question_id: int) -> LowConfidenceRow | None:
        return await _run(self._get, question_id)

    def _attach_review(self, question_id: int, review_id: int) -> None:
        row = self._session.get(LowConfidenceQuestion, question_id)
        if row is None:
            raise ValueError(f"低置信度问题 {question_id} 不存在")
        row.matched_review_id = review_id
        self._session.commit()

    async def attach_review(self, question_id: int, review_id: int) -> None:
        """记下这条原话被归并到哪一行待审问题。"""
        await _run(self._attach_review, question_id, review_id)

    def _list_recent(self, limit: int) -> list[LowConfidenceRow]:
        rows = self._session.scalars(
            select(LowConfidenceQuestion)
            .order_by(LowConfidenceQuestion.id.desc())
            .limit(limit)
        ).all()
        return [_low_confidence_row(row) for row in rows]

    async def list_recent(self, limit: int = 20) -> list[LowConfidenceRow]:
        """最近入池的若干条（新的在前）。"""
        return await _run(self._list_recent, limit)

    def _list_for_review(self, review_id: int) -> list[LowConfidenceRow]:
        rows = self._session.scalars(
            select(LowConfidenceQuestion)
            .where(LowConfidenceQuestion.matched_review_id == review_id)
            .order_by(LowConfidenceQuestion.id.asc())
        ).all()
        return [_low_confidence_row(row) for row in rows]

    async def list_for_review(self, review_id: int) -> list[LowConfidenceRow]:
        """归并到某条待审问题的全部原始提问（审核详情页要逐条展示）。"""
        return await _run(self._list_for_review, review_id)

    def _has_reason_fragment(self, fragment: str) -> bool:
        found = self._session.scalar(
            select(LowConfidenceQuestion.id).where(
                LowConfidenceQuestion.reason.like(f"%{fragment}%")
            )
        )
        return found is not None

    async def has_reason_fragment(self, fragment: str) -> bool:
        """``reason`` 里是否已有某个标记；👎 双击幂等靠它。"""
        return await _run(self._has_reason_fragment, fragment)

    def _count_by_source(self) -> dict[str, int]:
        rows = self._session.execute(
            select(LowConfidenceQuestion.source, func.count()).group_by(
                LowConfidenceQuestion.source
            )
        ).all()
        counts = {source: 0 for source in LOW_CONFIDENCE_SOURCES}
        for source, number in rows:
            counts[str(source)] = int(number)
        return counts

    async def count_by_source(self) -> dict[str, int]:
        return await _run(self._count_by_source)


class FaithCaseRepository:
    """``faith_cases`` 的写入与处置。

    写入是 **upsert**（一题一行）：同一道题再次被判编造只累加 ``seen_count``、
    刷新答案/理由/角标快照，绝不新增第二行。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _upsert_from_judgement(
        self,
        eval_id: str,
        bucket: str,
        query: str,
        strategy: str,
        answer: str,
        reason: str,
        citations: Sequence[dict[str, Any]] | None,
        judge_model: str | None,
    ) -> None:
        now = datetime.now()
        existing = self._session.scalar(
            select(FaithCase).where(FaithCase.eval_id == eval_id)
        )
        if existing is None:
            self._session.add(
                FaithCase(
                    eval_id=eval_id,
                    bucket=bucket,
                    query=query,
                    strategy=strategy,
                    answer=answer,
                    reason=reason,
                    citations=list(citations) if citations is not None else None,
                    judge_model=judge_model,
                    status=FAITH_CASE_STATUSES[0],
                    seen_count=1,
                    first_seen_at=now,
                    last_seen_at=now,
                )
            )
        else:
            existing.bucket = bucket
            existing.query = query
            existing.strategy = strategy
            existing.answer = answer
            existing.reason = reason
            existing.citations = list(citations) if citations is not None else None
            existing.judge_model = judge_model
            existing.seen_count = int(existing.seen_count or 1) + 1
            existing.last_seen_at = now
            # **复发**：已标已解决的题又被判出来，退回未解决。
            # resolved_at 保留 —— 「上次解决过、这次又犯了」正是要看见的信号。
            if existing.status == "已解决":
                existing.status = "未解决"
                existing.resolution = None
        self._session.commit()

    async def upsert_from_judgement(
        self,
        *,
        eval_id: str,
        bucket: str,
        query: str,
        answer: str,
        reason: str,
        strategy: str = "hybrid_rerank",
        citations: Sequence[dict[str, Any]] | None = None,
        judge_model: str | None = None,
    ) -> None:
        """记一条编造个案（同一 ``eval_id`` 累加，不新增行）。"""
        await _run(
            self._upsert_from_judgement,
            eval_id,
            bucket,
            query,
            strategy,
            answer,
            reason,
            citations,
            judge_model,
        )

    def _list_by_status(self, status: str | None) -> list[FaithCaseRow]:
        statement = select(FaithCase).order_by(FaithCase.last_seen_at.desc())
        if status is not None:
            statement = statement.where(FaithCase.status == status)
        return [_to_faith_case_row(row) for row in self._session.scalars(statement).all()]

    async def list_by_status(self, status: str | None = None) -> list[FaithCaseRow]:
        """按处置状态取个案；``None`` 表示全部（最近见过的在前）。"""
        return await _run(self._list_by_status, status)

    def _get(self, eval_id: str) -> FaithCaseRow | None:
        row = self._session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
        return _to_faith_case_row(row) if row is not None else None

    async def get(self, eval_id: str) -> FaithCaseRow | None:
        """按题号取一行；不存在返回 ``None``（由端点决定 404）。"""
        return await _run(self._get, eval_id)

    def _update_status(self, eval_id: str, status: str, resolution: str | None) -> bool:
        if status not in FAITH_CASE_STATUSES:
            raise ValueError(
                f"非法处置状态 {status!r}，只允许 {FAITH_CASE_STATUSES}"
            )
        if status != "未解决" and not (resolution or "").strip():
            raise ValueError(
                f"把状态标成 {status!r} 必须写处置说明（resolution）——"
                "空着的处置在台账上等于没有交代"
            )

        row = self._session.scalar(select(FaithCase).where(FaithCase.eval_id == eval_id))
        if row is None:
            return False

        row.status = status
        if status == "未解决":
            row.resolution = None
        else:
            row.resolution = (resolution or "").strip()
            row.resolved_at = datetime.now()
        self._session.commit()
        return True

    async def update_status(
        self, eval_id: str, status: str, resolution: str | None = None
    ) -> bool:
        """改处置状态。返回 ``False`` 表示台账里没有这个 ``eval_id``。"""
        return await _run(self._update_status, eval_id, status, resolution)


def _to_faith_case_row(row: FaithCase) -> FaithCaseRow:
    return FaithCaseRow(
        id=int(row.id),
        eval_id=row.eval_id,
        bucket=row.bucket,
        query=row.query,
        strategy=row.strategy,
        answer=row.answer,
        reason=row.reason,
        citations=row.citations,
        judge_model=row.judge_model,
        status=row.status,
        seen_count=int(row.seen_count),
        first_seen_at=row.first_seen_at,
        last_seen_at=row.last_seen_at,
        resolution=row.resolution,
        resolved_at=row.resolved_at,
    )


# ---------------------------------------------------------------------------
# 工具审计（ch08）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRow:
    """``tool_audit_logs`` 的只读投影（不把 ORM 对象漏出仓库层）。"""

    id: int
    conversation_id: int | None
    tool_call_id: str | None
    tool_name: str
    tool_source: str
    mcp_server: str | None
    arguments: dict[str, object] | None
    result_summary: str | None
    status: str
    error_message: str | None
    retry_count: int
    duration_ms: int | None
    created_at: datetime


class ToolAuditRepository:
    """``tool_audit_logs`` 的写入与最近记录查询。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _create(
        self,
        *,
        tool_name: str,
        tool_source: str,
        status: str,
        mcp_server: str | None = None,
        tool_call_id: str | None = None,
        conversation_id: int | None = None,
        arguments: dict[str, object] | None = None,
        result_summary: str | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
        duration_ms: int | None = None,
    ) -> None:
        self._session.add(
            ToolAuditLog(
                conversation_id=conversation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_source=tool_source,
                mcp_server=mcp_server,
                arguments=arguments,
                result_summary=result_summary,
                status=status,
                error_message=error_message,
                retry_count=retry_count,
                duration_ms=duration_ms,
            )
        )
        self._session.commit()

    async def create(self, **values: object) -> None:
        """落一条审计。参数名与 ``tool_audit_logs`` 的列语义一一对应。"""
        await anyio.to_thread.run_sync(partial(self._create, **values))

    def _list_recent(self, limit: int) -> list[AuditRow]:
        rows = self._session.scalars(
            select(ToolAuditLog).order_by(ToolAuditLog.id.desc()).limit(limit)
        ).all()
        return [
            AuditRow(
                id=int(row.id),
                conversation_id=row.conversation_id,
                tool_call_id=row.tool_call_id,
                tool_name=row.tool_name,
                tool_source=row.tool_source,
                mcp_server=row.mcp_server,
                arguments=row.arguments,
                result_summary=row.result_summary,
                status=row.status,
                error_message=row.error_message,
                retry_count=int(row.retry_count),
                duration_ms=row.duration_ms,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_recent(self, limit: int = 20) -> list[AuditRow]:
        """最近的若干条（新的在前）。"""
        return await _run(self._list_recent, limit)


# ---------------------------------------------------------------------------
# ch09：待审队列 / 评估记录
# ---------------------------------------------------------------------------


class ReviewQueueRepository:
    """``review_queue`` 的读写 —— 一行一个去重后的知识缺口。

    查重命中时只累加 ``occurrence_count``，绝不新增第二行；原始提问留在
    低置信度池里，用 ``matched_review_id`` 指回来。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _add(self, normalized_question: str, ai_suggested_answer: str | None) -> int:
        if not normalized_question.strip():
            raise ValueError("标准化问题不能为空")
        row = ReviewQueue(
            normalized_question=normalized_question,
            ai_suggested_answer=ai_suggested_answer,
        )
        self._session.add(row)
        self._session.commit()
        return int(row.id)

    async def add(self, normalized_question: str, ai_suggested_answer: str | None) -> int:
        return await _run(self._add, normalized_question, ai_suggested_answer)

    def _get(self, review_id: int) -> ReviewRow | None:
        row = self._session.get(ReviewQueue, review_id)
        return None if row is None else _review_row(row)

    async def get(self, review_id: int) -> ReviewRow | None:
        return await _run(self._get, review_id)

    def _list_rows(
        self, status: str | None, limit: int, offset: int
    ) -> list[ReviewRow]:
        statement = select(ReviewQueue).order_by(ReviewQueue.id.desc())
        if status is not None:
            statement = statement.where(ReviewQueue.review_status == status)
        rows = self._session.scalars(statement.limit(limit).offset(offset)).all()
        return [_review_row(row) for row in rows]

    async def list_rows(
        self, status: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[ReviewRow]:
        return await _run(self._list_rows, status, limit, offset)

    def _pending_questions(self, limit: int) -> list[tuple[int, str]]:
        rows = self._session.scalars(
            select(ReviewQueue)
            .where(ReviewQueue.review_status == "待审")
            .order_by(ReviewQueue.id.desc())
            .limit(limit)
        ).all()
        return [(int(row.id), row.normalized_question) for row in rows]

    async def pending_questions(self, limit: int = 50) -> list[tuple[int, str]]:
        """待审的 ``(id, 标准化问题)``，供查重时让模型判断是否同义。"""
        return await _run(self._pending_questions, limit)

    def _increment(self, review_id: int) -> bool:
        row = self._session.get(ReviewQueue, review_id)
        if row is None:
            return False
        row.occurrence_count = int(row.occurrence_count) + 1
        self._session.commit()
        return True

    async def increment(self, review_id: int) -> bool:
        return await _run(self._increment, review_id)

    def _set_status(
        self, review_id: int, status: str, approved_answer: str | None
    ) -> bool:
        if status not in REVIEW_STATUSES:
            raise ValueError(f"非法审核状态 {status!r}，只允许 {REVIEW_STATUSES}")
        row = self._session.get(ReviewQueue, review_id)
        if row is None:
            return False
        row.review_status = status
        if approved_answer is not None:
            row.approved_answer = approved_answer
        self._session.commit()
        return True

    async def set_status(
        self, review_id: int, status: str, approved_answer: str | None = None
    ) -> bool:
        return await _run(self._set_status, review_id, status, approved_answer)


class EvalRunRepository:
    """``eval_runs`` 的追加与读取（一行一轮，只追加不改写）。"""

    def __init__(self, session: Session) -> None:
        self._session = session

    def _add(self, triggered_by: str, dataset_size: int, metrics: dict[str, Any]) -> int:
        row = EvalRun(
            triggered_by=triggered_by,
            dataset_size=dataset_size,
            metrics=metrics,
        )
        self._session.add(row)
        self._session.commit()
        return int(row.id)

    async def add(
        self, triggered_by: str, dataset_size: int, metrics: dict[str, Any]
    ) -> int:
        return await _run(self._add, triggered_by, dataset_size, metrics)

    def _list_recent(self, limit: int) -> list[EvalRunRow]:
        rows = self._session.scalars(
            select(EvalRun).order_by(EvalRun.id.desc()).limit(limit)
        ).all()
        return [
            EvalRunRow(
                id=int(row.id),
                triggered_by=row.triggered_by,
                dataset_size=int(row.dataset_size),
                metrics=row.metrics,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def list_recent(self, limit: int = 20) -> list[EvalRunRow]:
        return await _run(self._list_recent, limit)


# ---------------------------------------------------------------------------
# ch10：主题分类结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PendingQuestionRow:
    """一道还没归类的问题。只带批跑真正要用的三列。"""

    id: int
    raw_question: str
    created_at: datetime


@dataclass(frozen=True)
class TopicCount:
    label: str
    count: int


class TopicClassificationRepository:
    """``topic_classifications`` 的读写 —— 旁路批跑的落库端。

    **只被旁路调用**：实时对话主链路不碰这张表（那是 ch10 的硬边界，
    加一次同步归类就等于给主链路加几十毫秒）。

    聚合类方法（``label_counts`` / ``multi_label_count``）把行读进 Python
    再数：``labels`` 是 JSON 数组，按标签拆行要依赖方言的 JSON 函数
    （MySQL ``JSON_TABLE`` / SQLite ``json_each`` 写法不同）。这张表的量级是
    「上万行、偶尔跑一次统计」，多读几 MB 换掉一处方言分支是划算的。
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _list_pending(self, limit: int) -> list[PendingQuestionRow]:
        stmt = (
            select(
                LowConfidenceQuestion.id,
                LowConfidenceQuestion.raw_question,
                LowConfidenceQuestion.created_at,
            )
            .outerjoin(
                TopicClassification,
                TopicClassification.question_id == LowConfidenceQuestion.id,
            )
            .where(TopicClassification.id.is_(None))
            .order_by(LowConfidenceQuestion.id)
            .limit(limit)
        )
        return [
            PendingQuestionRow(
                id=int(row.id), raw_question=row.raw_question, created_at=row.created_at
            )
            for row in self._session.execute(stmt).all()
        ]

    async def list_pending(self, limit: int = 50) -> list[PendingQuestionRow]:
        return await _run(self._list_pending, limit)

    def _count_pending(self) -> int:
        stmt = (
            select(func.count())
            .select_from(LowConfidenceQuestion)
            .outerjoin(
                TopicClassification,
                TopicClassification.question_id == LowConfidenceQuestion.id,
            )
            .where(TopicClassification.id.is_(None))
        )
        return int(self._session.scalar(stmt) or 0)

    async def count_pending(self) -> int:
        return await _run(self._count_pending)

    def _upsert_many(
        self, rows: Sequence[tuple[int, list[str]]], classified_at: datetime
    ) -> int:
        # 同批里同一道题出现两次时后写覆盖先写，否则第二次会撞唯一键。
        deduped = dict(rows)
        if not deduped:
            return 0
        existing = set(
            self._session.scalars(
                select(TopicClassification.question_id).where(
                    TopicClassification.question_id.in_(list(deduped))
                )
            ).all()
        )
        for question_id, labels in deduped.items():
            if question_id in existing:
                self._session.execute(
                    update(TopicClassification)
                    .where(TopicClassification.question_id == question_id)
                    .values(labels=list(labels), classified_at=classified_at)
                )
            else:
                self._session.add(
                    TopicClassification(
                        question_id=question_id,
                        labels=list(labels),
                        classified_at=classified_at,
                    )
                )
        # 单事务：一批要么全进，要么全不进，中断后重跑不会留下半批。
        self._session.commit()
        return len(deduped)

    async def upsert_many(
        self, rows: Sequence[tuple[int, list[str]]], *, classified_at: datetime
    ) -> int:
        return await _run(self._upsert_many, rows, classified_at)

    def _label_counts(self) -> list[TopicCount]:
        counter: Counter[str] = Counter()
        for labels in self._session.scalars(select(TopicClassification.labels)).all():
            counter.update(labels or [])
        ordered = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return [TopicCount(label=label, count=count) for label, count in ordered]

    async def label_counts(self) -> list[TopicCount]:
        return await _run(self._label_counts)

    def _total_classified(self) -> int:
        return int(
            self._session.scalar(select(func.count()).select_from(TopicClassification)) or 0
        )

    async def total_classified(self) -> int:
        return await _run(self._total_classified)

    def _multi_label_count(self) -> int:
        rows = self._session.scalars(select(TopicClassification.labels)).all()
        return sum(1 for labels in rows if labels and len(labels) > 1)

    async def multi_label_count(self) -> int:
        return await _run(self._multi_label_count)

    def _last_classified_at(self) -> datetime | None:
        return self._session.scalar(select(func.max(TopicClassification.classified_at)))

    async def last_classified_at(self) -> datetime | None:
        return await _run(self._last_classified_at)


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
    #: ch04：低置信度问题池
    low_confidence: LowConfidenceQuestionRepository
    #: ch04：编造个案台账
    faith_cases: FaithCaseRepository
    #: ch07：会话梗概（只追加）
    summaries: ConversationSummaryRepository
    #: ch08：工具调用审计（只追加，不挂外键）
    audit: ToolAuditRepository
    #: ch09：待审队列
    reviews: ReviewQueueRepository
    #: ch09：评估记录
    eval_runs: EvalRunRepository
    #: ch10：主题分类结果（旁路写入，主链路只读分布）
    topics: TopicClassificationRepository

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
            low_confidence=LowConfidenceQuestionRepository(session),
            faith_cases=FaithCaseRepository(session),
            summaries=ConversationSummaryRepository(session),
            audit=ToolAuditRepository(session),
            reviews=ReviewQueueRepository(session),
            eval_runs=EvalRunRepository(session),
            topics=TopicClassificationRepository(session),
        )
