"""后台异步摘要任务 —— 三层结构的第三层。

## 为什么是「后台任务」而不是「图里的一个节点」

摘要是**成本**：把一段历史压成一段事实卡片。用户并不需要等它 —— 这一轮的回复
用的是「层 1 原文 + 层 2 截短 + 已有梗概」，压不压得完都能答。所以它必须脱离
这一轮的请求生命周期：

- 放在图里当节点 → 图是同步推进的，节点返回之前这一轮就出不去，要么阻塞、
  要么用 ``interrupt`` 伪装后台（语义错位，且摘要失败会把流程拖进 error 分支）。
- 放在 HTTP 请求的 ``await`` 链里 → 上游一次慢调用就把用户的回复拖慢几秒。

``asyncio.create_task`` 是这里唯一诚实的做法：调度立刻返回，任务自己开 Session、
自己写库、自己 catch。

## 三条纪律

1. **只 append，不改旧段**。旧段是「这个事实被有损压过一次」的证据；每轮把
   「旧梗概 + 新消息」再压一次，同一个事实会被反复有损压缩，最后退化成
   「用户有售后需求」，订单号和手机号全丢。旧段只作为**背景**给模型看，
   并在提示词里明确「本次只总结【待压缩对话】」。
2. **乐观并发**。读历史 → 调模型 → 落库，中间隔着一次网络往返。这期间用户可能
   又聊了几轮、锚点已被本轮降级推走。落库前复核 ``layer1_from_msg_id``，对不上
   就**放弃、不重试**：重试会压到一段已经变了的区间。
3. **失败不冒泡**。任务跑在独立 Task 里，没人 ``await`` 它 —— 异常冒出去只会
   被 asyncio 打一行「Task exception was never retrieved」然后丢掉。全部
   catch 成 ``failed`` 状态 + 日志。

## 多进程怎么办

``_inflight`` 只在单进程内有效。真正的收口是数据库唯一键
``UNIQUE(conversation_id, segment_no)``：两个进程同时压同一段时，后到的那个
撞唯一键，翻译成 :class:`~app.db.repositories.SummarySegmentConflict`，
**放弃并留痕、不改号重试**（改号会让同一区间产生两段内容不同的梗概）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.ctxlog import AppLogger, log_summary_event
from app.agent.prompts import SUMMARY_PROMPT
from app.context.layers import LayerPlan, row_to_message
from app.context.tokens import count_tokens
from app.db.repositories import (
    MessageRow,
    Repositories,
    SummarySegmentConflict,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ContextSummarizer",
    "SummaryJob",
    "SummaryOutcome",
    "build_summary_job",
    "format_conversation",
]


def _row_speaker(row: MessageRow) -> str:
    """一条流水在摘要提示词里的说话人标签。"""
    if row.role == "user":
        return "用户"
    if row.role == "assistant":
        return "客服" if not row.tool_calls else "客服(调用工具)"
    return "工具结果"


def format_conversation(rows: Sequence[MessageRow]) -> str:
    """把待压缩的行渲染成提示词里的一段对话。

    **读原文**，不读 ``plan.layer2``（那是截短过的）。用截短版去压，正好落在
    60 字边界上的订单号会被截掉 —— 而那恰恰是唯一值得记住的东西。

    工具结果整段丢掉：摘要只要事实与诉求，工具返回的 JSON 一片噪音；工具**做过
    什么**由紧随其后的 assistant 文本体现。
    """
    lines: list[str] = []
    for row in rows:
        if row.role == "tool":
            continue
        content = " ".join(str(row.content or "").split())
        if not content:
            continue
        lines.append(f"{_row_speaker(row)}：{content}")
    return "\n".join(lines)


@dataclass(frozen=True)
class SummaryJob:
    """一次摘要任务的完整输入。**自带全部依赖**，不读全局状态。"""

    session_id: str
    conversation_id: int
    segment_no: int
    from_msg_id: int
    upto_msg_id: int
    rows: list[MessageRow]
    source_tokens: int
    background: str = ""
    """前几段梗概拼成的背景块（只读，不许搬进新段）。"""
    expected_layer1_from: int | None = None
    """落库前要求数据库里是这个层 1 起点；``None`` = 不检查。

    **必须是「调度方已经写进库的那个值」**，不是 ``LayerPlan`` 里的建议值。
    ``plan.layer1_from_msg_id`` 只表示「这次规划算出的起点」，本轮是否落库由
    调用方决定；拿建议值来比，会在「本轮还没降级过」时直接判成 anchor moved。
    """


@dataclass(frozen=True)
class SummaryOutcome:
    """一次摘要任务的结果。``status`` 只有三种，覆盖「成功/主动放弃/出错」。"""

    status: str
    """``done`` / ``skipped`` / ``failed``。"""
    segment_no: int | None = None
    content: str = ""
    elapsed_ms: int = 0
    reason: str = ""
    stored: bool = False
    """是否真的落了库。``skipped`` / ``failed`` 时为 ``False``。"""


def build_summary_job(
    *,
    plan: LayerPlan,
    session_id: str,
    conversation_id: int,
    segment_no: int,
    background: str = "",
    expected_layer1_from: int | None = None,
) -> SummaryJob:
    """从一次三层规划的结果造出摘要任务。

    用 ``plan.summary_rows``（**未截短**）而不是 ``plan.layer2``：
    ``plan.summary_due=False`` 时 ``summary_rows`` 为空，那时本就不该调度。
    """
    rows = list(plan.summary_rows)
    source_tokens = count_tokens([row_to_message(row) for row in rows])
    from_msg_id = rows[0].id if rows else (plan.summary_upto_msg_id or 0) + 1
    upto_msg_id = rows[-1].id if rows else plan.layer1_from_msg_id or 0
    return SummaryJob(
        session_id=session_id,
        conversation_id=conversation_id,
        segment_no=segment_no,
        from_msg_id=from_msg_id,
        upto_msg_id=upto_msg_id,
        rows=rows,
        source_tokens=source_tokens,
        background=background,
        expected_layer1_from=expected_layer1_from,
    )


@dataclass
class _State:
    """可变状态收在一处，避免 ``ContextSummarizer`` 上散着一堆属性。"""

    inflight: set[str] = field(default_factory=set)
    tasks: set[asyncio.Task] = field(default_factory=set)


class ContextSummarizer:
    """把层 2 的一批消息压成一段梗概，追加进 ``conversation_summaries``。

    :param model: 任意带 ``async ainvoke(messages)`` 的聊天模型。
    :param repositories_factory: **零参**工厂，每次任务调一次。任务在独立
        Task 里跑，不能借用请求那个 Session（它可能已经随请求结束被关了）。
    :param settings: 读 ``summary_enabled`` / ``summary_max_tokens`` /
        ``summary_timeout_seconds`` / ``summary_model``。
    :param logger: 上下文日志。默认按 ``settings.app_log_path`` 造一个。
    :param clock: 计时函数（默认 ``time.perf_counter``），测试可注入。
    """

    def __init__(
        self,
        *,
        model: Any,
        repositories_factory: Callable[[], Repositories],
        settings: Any,
        logger: AppLogger | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._model = model
        self._repositories_factory = repositories_factory
        self._settings = settings
        self._logger = logger if logger is not None else AppLogger(
            getattr(settings, "app_log_path", "log/app.log")
        )
        self._clock = clock or time.perf_counter
        self._state = _State()

    # -- 对外只读视图 ------------------------------------------------------

    @property
    def inflight(self) -> frozenset[str]:
        """正在被压的会话 id 集合（测试与排障用）。"""
        return frozenset(self._state.inflight)

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "summary_enabled", True))

    # -- 调度 --------------------------------------------------------------

    def schedule(
        self,
        *,
        session_id: str,
        conversation_id: int,
        plan: LayerPlan,
        background: str = "",
        segment_no: int | None = None,
        expected_layer1_from: int | None = None,
        layer2_budget: int | None = None,
        background_factory: Callable[[Repositories], Any] | None = None,
    ) -> bool:
        """起一个后台摘要任务，**立刻返回**。

        返回 ``False`` 的四种情形（都不是错误，是「不该压」）：
        摘要被配置关掉；``plan.summary_due`` 为假；该会话已有任务在飞；
        任务列表为空。

        ``segment_no`` 不传时由任务自己查（``next_segment_no``）—— 在调度时查
        会多一次数据库往返，而且调度方那个 Session 可能就是即将被关掉的那个。
        ``background_factory`` 同理：背景梗概由任务在自己的 Session 里读。

        ``expected_layer1_from`` 传**库里当前的值**（``anchors.layer1_from_msg_id``），
        不是 ``plan`` 的建议值 —— 后者还没落库，拿来比必然不等。
        """
        if not self.enabled:
            return False
        if not plan.summary_due:
            return False
        if not plan.summary_rows:
            return False
        if session_id in self._state.inflight:
            log_summary_event(
                logger=self._logger,
                event="skip",
                session_id=session_id,
                reason="inflight",
                layer2_tokens=plan.layer2_tokens,
            )
            return False

        self._state.inflight.add(session_id)
        job = build_summary_job(
            plan=plan,
            session_id=session_id,
            conversation_id=conversation_id,
            segment_no=segment_no if segment_no is not None else 0,
            background=background,
            expected_layer1_from=expected_layer1_from,
        )
        log_summary_event(
            logger=self._logger,
            event="trigger",
            session_id=session_id,
            layer2_tokens=plan.layer2_tokens,
            budget=layer2_budget if layer2_budget is not None else "?",
            msgs=len(job.rows),
        )
        task = asyncio.create_task(
            self._run_guarded(job, segment_no=segment_no, background_factory=background_factory)
        )
        self._state.tasks.add(task)
        task.add_done_callback(self._on_done(session_id))
        return True

    def _on_done(self, session_id: str) -> Callable[[asyncio.Task], None]:
        def _callback(task: asyncio.Task) -> None:
            self._state.tasks.discard(task)
            self._state.inflight.discard(session_id)
            # 任务本身已经把异常 catch 成 failed 了；这里再兜一层是为了
            # 兜住「构造 Outcome 之前就炸」的情形（比如日志模块自己抛）。
            if not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.warning("ch07 摘要任务异常（已吞）：%s", exc, exc_info=exc)

        return _callback

    async def _run_guarded(
        self,
        job: SummaryJob,
        *,
        segment_no: int | None,
        background_factory: Callable[[Repositories], Any] | None,
    ) -> SummaryOutcome:
        """包一层「不冒泡」的外壳，并补齐调度时没查的两样东西。"""
        try:
            repositories = self._repositories_factory()
            resolved_no = (
                segment_no
                if segment_no is not None
                else await repositories.summaries.next_segment_no(job.conversation_id)
            )
            background = job.background
            if not background and background_factory is not None:
                background = str(background_factory(repositories) or "")
            filled = SummaryJob(
                session_id=job.session_id,
                conversation_id=job.conversation_id,
                segment_no=int(resolved_no),
                from_msg_id=job.from_msg_id,
                upto_msg_id=job.upto_msg_id,
                rows=job.rows,
                source_tokens=job.source_tokens,
                background=background,
                expected_layer1_from=job.expected_layer1_from,
            )
            return await self.run_once(filled)
        except asyncio.CancelledError:  # pragma: no cover - 只在 shutdown 时
            log_summary_event(
                logger=self._logger, event="failed", session_id=job.session_id, error="cancelled"
            )
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务，任何异常都只留痕
            log_summary_event(
                logger=self._logger,
                event="failed",
                session_id=job.session_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            return SummaryOutcome(status="failed", reason=f"{type(exc).__name__}: {exc}")

    # -- 执行 --------------------------------------------------------------

    async def run_once(self, job: SummaryJob) -> SummaryOutcome:
        """跑一次摘要。**任何失败都返回 Outcome，不抛**（CancelledError 除外）。"""
        started = self._clock()
        try:
            return await self._run_once(job, started)
        except asyncio.CancelledError:  # pragma: no cover
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务只留痕
            reason = f"{type(exc).__name__}: {exc}"
            log_summary_event(
                logger=self._logger, event="failed", session_id=job.session_id, error=reason
            )
            return SummaryOutcome(
                status="failed",
                segment_no=job.segment_no,
                elapsed_ms=self._elapsed_ms(started),
                reason=reason,
            )

    async def _run_once(self, job: SummaryJob, started: float) -> SummaryOutcome:
        repositories = self._repositories_factory()

        # 1) 锚点复核 —— **先查、后调模型**：区间变了就省一次上游调用。
        anchors = await repositories.conversations.get_anchors(job.conversation_id)
        if anchors is None:
            return self._skip(job, started, "conversation missing")
        if (
            job.expected_layer1_from is not None
            and anchors.layer1_from_msg_id != job.expected_layer1_from
        ):
            return self._skip(
                job,
                started,
                f"anchor moved: expected layer1_from={job.expected_layer1_from} "
                f"now {anchors.layer1_from_msg_id}",
            )
        if not job.rows:
            return self._skip(job, started, "empty batch")

        # 2) 旧段背景。调度方没给就自己读 —— 「旧段只作背景」是这个任务的定义
        #    性质，不能靠调用方记得传。读失败退化成空背景，不让整个任务失败：
        #    没有背景只是少一句提示，梗概本身照样能压。
        background = job.background
        if not background:
            try:
                texts = await repositories.summaries.latest_texts(job.conversation_id)
                background = "\n".join(f"- {text}" for text in texts)
            except Exception as exc:  # noqa: BLE001 - 背景读不到不该让任务失败
                logger.warning("ch07 读取旧梗概背景失败（按无背景继续）：%s", exc)
                background = ""

        # 3) 调模型（有超时）。
        log_summary_event(
            logger=self._logger,
            event="start",
            session_id=job.session_id,
            seg=job.segment_no,
            from_msg=job.from_msg_id,
            upto_msg=job.upto_msg_id,
            msgs=len(job.rows),
        )
        prompt = SUMMARY_PROMPT.format(
            background=background or "（这是第一段，没有更早的记录）",
            conversation=format_conversation(job.rows),
        )
        timeout = float(getattr(self._settings, "summary_timeout_seconds", 20.0) or 20.0)
        answer = await asyncio.wait_for(
            self._model.ainvoke([SystemMessage(content=prompt)]), timeout=timeout
        )
        raw = answer.content if hasattr(answer, "content") else str(answer)
        if isinstance(raw, list):  # 多模态内容块
            raw = " ".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in raw
            )
        content = self._clamp(str(raw))
        if not content:
            # 空梗概落库等于「这段压过了但什么都没记」，比不压更糟。
            return self._skip(job, started, "empty summary")

        # 3) 落库：先 append 再推锚点。
        #    顺序反了会怎样：锚点先动、append 撞唯一键 → 摘要区间被覆盖但库里没有
        #    对应梗概，那一段历史就永久消失在上下文里（既不进层 2、也没梗概）。
        try:
            await repositories.summaries.append(
                conversation_id=job.conversation_id,
                segment_no=job.segment_no,
                from_msg_id=job.from_msg_id,
                upto_msg_id=job.upto_msg_id,
                content=content,
                msg_count=len(job.rows),
                source_tokens=job.source_tokens,
                model=self._model_name(),
                elapsed_ms=self._elapsed_ms(started),
            )
        except SummarySegmentConflict as exc:
            return self._skip(job, started, f"segment exists: {exc}")

        moved = await repositories.conversations.set_anchors(
            conversation_id=job.conversation_id,
            summary_upto_msg_id=job.upto_msg_id,
            layer1_from_msg_id=max(
                (anchors.layer1_from_msg_id or 0), job.upto_msg_id + 1
            ),
            only_if_layer1_from=job.expected_layer1_from,
        )
        if not moved:
            # 极窄的竞态：append 与 set_anchors 之间锚点又动了。梗概已经落库，
            # 但它覆盖的区间现在与锚点不符 —— 只能留痕，不能回滚（删梗概等于篡改证据）。
            log_summary_event(
                logger=self._logger,
                event="skip",
                session_id=job.session_id,
                reason="anchor moved after append",
                seg=job.segment_no,
                upto_msg_id=job.upto_msg_id,
            )

        elapsed = self._elapsed_ms(started)
        log_summary_event(
            logger=self._logger,
            event="done",
            session_id=job.session_id,
            seg=job.segment_no,
            chars=len(content),
            elapsed_ms=elapsed,
            upto_msg_id=job.upto_msg_id,
            msg_count=len(job.rows),
            stored=moved,
        )
        return SummaryOutcome(
            status="done",
            segment_no=job.segment_no,
            content=content,
            elapsed_ms=elapsed,
            stored=moved,
        )

    # -- 小工具 ------------------------------------------------------------

    def _skip(self, job: SummaryJob, started: float, reason: str) -> SummaryOutcome:
        elapsed = self._elapsed_ms(started)
        log_summary_event(
            logger=self._logger,
            event="skip",
            session_id=job.session_id,
            reason=reason,
            seg=job.segment_no,
            elapsed_ms=elapsed,
        )
        return SummaryOutcome(
            status="skipped", segment_no=job.segment_no, elapsed_ms=elapsed, reason=reason
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, int((self._clock() - started) * 1000))

    def _clamp(self, text: str) -> str:
        """压平空白并截断到 ``summary_max_tokens`` 对应的字数上限。

        字数上限直接取 ``summary_max_tokens`` 而不是它的倍数：本项目的中文校准
        口径是「1 个汉字 ≈ 1 token」（``CJK_CHARS_PER_TOKEN``），所以 token 预算
        与字数上限是同一个数。乘个系数等于自己偷偷把预算放大一倍，而梗概的总注入
        预算（``SUMMARY_RESERVE_TOKENS``）是按原始数字算的。
        """
        flat = " ".join(str(text).split())
        limit = max(int(getattr(self._settings, "summary_max_tokens", 300) or 300), 60)
        if len(flat) <= limit:
            return flat
        return flat[:limit]

    def _model_name(self) -> str:
        """记进 ``conversation_summaries.model`` 的模型名。"""
        configured = str(getattr(self._settings, "summary_model", "") or "")
        if configured:
            return configured
        return str(getattr(self._settings, "openai_model", "") or "")

    async def aclose(self) -> None:
        """等所有在飞任务收尾。lifespan 退出与测试用。

        ``return_exceptions=True``：单个任务失败不该阻止其余任务收敛；它们的
        异常本来就已经被 catch 成 ``failed``。
        """
        pending = list(self._state.tasks)
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        self._state.tasks.clear()
        self._state.inflight.clear()
