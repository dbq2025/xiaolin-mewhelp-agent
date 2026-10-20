"""后台摘要：不阻塞、可去重、只 append、失败不冒泡。

这组用例钉的是验收第 4 条：「日志里能看到早期轮次被后台摘要任务压缩的痕迹，
且**摘要生成没有阻塞该轮用户回复**」。所以除了功能正确，还有三条时序/并发性质：

- ``schedule()`` 必须在模型被调用之前就返回（``test_schedule_returns_immediately``）；
- 同一会话的第二次 ``schedule()`` 必须被去重（否则一轮超预算会起 N 个任务）；
- 任务失败只落日志、不冒泡（它跑在独立 Task 里，没人 ``await``，冒泡等于弃尸）。

另有三条是「只 append」纪律的守卫：旧段一个字不动、旧段只作背景、
落库前锚点变了就放弃。

沙箱不能建临时目录（见 ``conftest.py``），所以用工作区内的 ``log_dir`` 夹具。
"""

import asyncio
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.agent.summarizer import ContextSummarizer, build_summary_job
from app.context.layers import plan_layers
from app.context.tokens import count_tokens

#: 放临时日志的目录，位于工作区内。
LOG_FIXTURE_DIR = Path(".uv-cache") / "summarizer-fixtures"


@pytest.fixture
def log_dir():
    LOG_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    made = LOG_FIXTURE_DIR / uuid.uuid4().hex
    made.mkdir()
    yield made
    shutil.rmtree(made, ignore_errors=True)


def _read(path):
    return path.read_text(encoding="utf-8")


class _FakeModel:
    """可计次、可延时、可指定输出的假模型。"""

    def __init__(self, text="用户问过耳机，报了订单号 A123456789。", delay=0.0):
        self.text = text
        self.delay = delay
        self.calls = 0
        self.seen: list[str] = []

    async def ainvoke(self, messages, **kwargs):
        self.calls += 1
        self.seen.append("\n".join(str(getattr(m, "content", "")) for m in messages))
        if self.delay:
            await asyncio.sleep(self.delay)
        return AIMessage(self.text)


class _BoomModel:
    async def ainvoke(self, messages, **kwargs):
        raise RuntimeError("上游 500")


def _settings(**over):
    from app.config import Settings

    data = dict(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-test",
        openai_model="fake-model",
        turn_log_path="",
        app_log_path="",
    )
    data.update(over)
    return Settings(**data)


def _rows(n=4, *, prefix="消息"):
    from app.db.repositories import MessageRow

    now = datetime.now(timezone.utc)
    return [
        MessageRow(
            id=i + 1,
            role="user" if i % 2 == 0 else "assistant",
            content=f"{prefix}{i}",
            tool_calls=None,
            tool_call_id=None,
            created_at=now,
        )
        for i in range(n)
    ]


def _plan(rows, *, layer2_budget=1, layer1_from=None):
    """默认造一个「层 2 超预算」的计划：层 2 预算给 1，必然超。"""
    return plan_layers(
        rows,
        summary_upto=0,
        layer1_from=layer1_from if layer1_from is not None else len(rows) + 1,
        layer1_budget=99999,
        layer2_budget=layer2_budget,
    )


async def _repos_and_conversation(sqlite_session, session_id="s1"):
    from app.db.repositories import Repositories

    repos = Repositories.from_session(sqlite_session)
    cid = await repos.conversations.upsert_by_session(session_id, "anonymous")
    return repos, cid


async def test_schedule_returns_immediately_without_awaiting_the_model(sqlite_session):
    """验收 4：摘要不许阻塞该轮用户回复。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    model = _FakeModel(delay=0.5)
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    started = time.perf_counter()
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is True
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"schedule 等了 {elapsed:.2f}s，说明它 await 了模型"
    await summarizer.aclose()
    assert model.calls == 1


async def test_schedule_is_false_when_summary_not_due(sqlite_session):
    repos, cid = await _repos_and_conversation(sqlite_session)
    model = _FakeModel()
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    # 层 2 预算给足 → 不超预算 → 不该触发
    assert (
        summarizer.schedule(
            session_id="s1", conversation_id=cid, plan=_plan(_rows(), layer2_budget=99999)
        )
        is False
    )
    await summarizer.aclose()
    assert model.calls == 0


async def test_concurrent_schedule_for_same_session_is_deduplicated(sqlite_session, log_dir):
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    model = _FakeModel(delay=0.3)
    summarizer = ContextSummarizer(
        model=model,
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is True
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is False
    await summarizer.aclose()
    assert model.calls == 1, "同一会话起了两个任务"
    assert "reason=inflight" in _read(log_dir / "app.log")


async def test_inflight_is_cleared_after_the_task_finishes(sqlite_session):
    """跑完要能再次触发 —— 否则一个会话一辈子只压一次。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    model = _FakeModel()
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is True
    await summarizer.aclose()
    assert summarizer.inflight == frozenset()
    assert (
        summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows(6))) is True
    )
    await summarizer.aclose()


async def test_run_once_appends_one_segment_and_moves_the_anchor(sqlite_session):
    repos, cid = await _repos_and_conversation(sqlite_session)
    rows = _rows(4)
    summarizer = ContextSummarizer(
        model=_FakeModel(), repositories_factory=lambda: repos, settings=_settings()
    )
    job = build_summary_job(
        plan=_plan(rows),
        session_id="s1",
        conversation_id=cid,
        segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "done"
    assert out.segment_no == 1
    segments = await repos.summaries.list_for_conversation(cid)
    assert len(segments) == 1
    assert segments[0].content.startswith("用户问过耳机")
    assert segments[0].from_msg_id == 1
    assert segments[0].upto_msg_id == 4
    assert segments[0].msg_count == 4
    anchors = await repos.conversations.get_anchors(cid)
    assert anchors.summary_upto_msg_id == 4


async def test_run_once_records_source_tokens_and_elapsed(sqlite_session):
    """``source_tokens`` / ``elapsed_ms`` 是「没阻塞」在日志之外的第二份证据。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    rows = _rows(4)
    summarizer = ContextSummarizer(
        model=_FakeModel(), repositories_factory=lambda: repos, settings=_settings()
    )
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "done"
    segments = await repos.summaries.list_for_conversation(cid)
    assert segments[0].source_tokens == count_tokens(rows_to_messages(rows))
    assert segments[0].elapsed_ms is not None
    assert segments[0].elapsed_ms >= 0


def rows_to_messages(rows):
    from app.context.layers import row_to_message

    return [row_to_message(row) for row in rows]


async def test_run_once_never_rewrites_old_segments(sqlite_session):
    """只 append：压完不回头重写，避免同一事实被反复有损压缩。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=2,
        content="旧段原文",
        msg_count=2,
        source_tokens=5,
    )
    rows = _rows(4)
    summarizer = ContextSummarizer(
        model=_FakeModel("新段"), repositories_factory=lambda: repos, settings=_settings()
    )
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=2,
        expected_layer1_from=None,
    )
    await summarizer.run_once(job)
    segments = await repos.summaries.list_for_conversation(cid)
    assert segments[0].content == "旧段原文", "旧段被改写了"
    assert segments[1].content == "新段"


async def test_old_segments_are_passed_as_background_not_merged(sqlite_session):
    """Review Focus 3：旧段只作背景，不许把它的事实搬进新段。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=2,
        content="旧段：订单号 OLD-999",
        msg_count=2,
        source_tokens=5,
    )
    model = _FakeModel("新段：订单号 NEW-111")
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=2,
        expected_layer1_from=None,
    )
    await summarizer.run_once(job)
    prompt = model.seen[0]
    assert "OLD-999" in prompt, "旧段没有作为背景进去，模型看不到已经记过什么"
    assert "待压缩对话" in prompt
    assert "已记录背景" in prompt
    segments = await repos.summaries.list_for_conversation(cid)
    assert "OLD-999" not in segments[1].content, "旧段事实被搬进了新段"


async def test_anchor_moved_during_summary_is_skipped(sqlite_session, log_dir):
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=7)
    rows = _rows(4)
    summarizer = ContextSummarizer(
        model=_FakeModel(),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=5,
    )
    out = await summarizer.run_once(job)
    assert out.status == "skipped"
    assert "anchor" in out.reason
    assert await repos.summaries.list_for_conversation(cid) == []
    assert "summary skip" in _read(log_dir / "app.log")


async def test_anchor_check_happens_before_calling_the_model(sqlite_session):
    """锚点对不上就省一次上游调用 —— 别为一段已经变了的区间花钱。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    await repos.conversations.set_anchors(conversation_id=cid, layer1_from_msg_id=7)
    model = _FakeModel()
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=5,
    )
    await summarizer.run_once(job)
    assert model.calls == 0


async def test_segment_conflict_is_skipped_not_raised(sqlite_session, log_dir):
    """多进程并发压同一段：唯一键挡下，冲突方放弃、不改号重试。"""
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    await repos.summaries.append(
        conversation_id=cid,
        segment_no=1,
        from_msg_id=1,
        upto_msg_id=4,
        content="别人写的",
        msg_count=4,
        source_tokens=9,
    )
    summarizer = ContextSummarizer(
        model=_FakeModel("我写的"),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "skipped"
    segments = await repos.summaries.list_for_conversation(cid)
    assert len(segments) == 1 and segments[0].content == "别人写的"


async def test_model_failure_is_logged_not_raised(sqlite_session, log_dir):
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    summarizer = ContextSummarizer(
        model=_BoomModel(),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "failed"
    assert "上游 500" in out.reason
    text = _read(log_dir / "app.log")
    assert "summary failed" in text
    assert await repos.summaries.list_for_conversation(cid) == []


async def test_scheduled_failure_does_not_kill_the_caller(sqlite_session, log_dir):
    """任务在独立 Task 里跑，异常冒出去就没人接了（asyncio 只会打一行噪音）。"""
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    summarizer = ContextSummarizer(
        model=_BoomModel(),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is True
    await summarizer.aclose()
    assert summarizer.inflight == frozenset()
    assert "summary failed" in _read(log_dir / "app.log")


async def test_disabled_summarizer_never_calls_the_model(sqlite_session):
    repos, cid = await _repos_and_conversation(sqlite_session)
    model = _FakeModel()
    summarizer = ContextSummarizer(
        model=model,
        repositories_factory=lambda: repos,
        settings=_settings(summary_enabled=False),
    )
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows())) is False
    await summarizer.aclose()
    assert model.calls == 0


async def test_summary_text_is_within_the_configured_length(sqlite_session):
    """长度要求：几十到一两百字。模型话多时由我们截断，不能把整段贴回来。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    summarizer = ContextSummarizer(
        model=_FakeModel("很" * 5000),
        repositories_factory=lambda: repos,
        settings=_settings(summary_max_tokens=300),
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "done"
    assert len(out.content) <= 300


async def test_empty_model_answer_is_skipped_not_stored(sqlite_session, log_dir):
    """空梗概落库等于「这一段已经被压过了，但什么都没记下」，比不压更糟。"""
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    summarizer = ContextSummarizer(
        model=_FakeModel("   "),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    rows = _rows(4)
    job = build_summary_job(
        plan=_plan(rows), session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    out = await summarizer.run_once(job)
    assert out.status == "skipped"
    assert await repos.summaries.list_for_conversation(cid) == []


async def test_summary_events_carry_the_expected_fields(sqlite_session, log_dir):
    """五态留痕：trigger / start / done 都要能看到覆盖边界与耗时。"""
    from app.agent.ctxlog import AppLogger

    repos, cid = await _repos_and_conversation(sqlite_session)
    summarizer = ContextSummarizer(
        model=_FakeModel("一段事实"),
        repositories_factory=lambda: repos,
        settings=_settings(),
        logger=AppLogger(log_dir / "app.log"),
    )
    assert summarizer.schedule(session_id="s1", conversation_id=cid, plan=_plan(_rows(4))) is True
    await summarizer.aclose()
    text = _read(log_dir / "app.log")
    assert "summary trigger session=s1" in text
    assert "layer2_tokens=" in text and "> budget=" in text
    assert "summary start session=s1" in text
    assert "summary done session=s1" in text
    assert "upto_msg_id=" in text
    assert "elapsed_ms=" in text


async def test_summary_job_carries_the_original_rows_not_the_trimmed_ones(sqlite_session):
    """摘要必须读**原文**：拿截短过的 60 字去压，订单号可能正好被截掉。"""
    repos, cid = await _repos_and_conversation(sqlite_session)
    from app.db.repositories import MessageRow

    now = datetime.now(timezone.utc)
    rows = [
        MessageRow(id=1, role="user", content="我的订单号是 " + "A" * 100, tool_calls=None,
                   tool_call_id=None, created_at=now),
        MessageRow(id=2, role="assistant", content="B" * 500, tool_calls=None,
                   tool_call_id=None, created_at=now),
        MessageRow(id=3, role="user", content="后来呢", tool_calls=None, tool_call_id=None,
                   created_at=now),
        MessageRow(id=4, role="assistant", content="已经催了", tool_calls=None,
                   tool_call_id=None, created_at=now),
    ]
    model = _FakeModel("订单 A 在催")
    summarizer = ContextSummarizer(
        model=model, repositories_factory=lambda: repos, settings=_settings()
    )
    plan = _plan(rows, layer2_budget=1)
    job = build_summary_job(
        plan=plan, session_id="s1", conversation_id=cid, segment_no=1,
        expected_layer1_from=None,
    )
    await summarizer.run_once(job)
    prompt = model.seen[0]
    assert "B" * 500 in prompt, "摘要读到了截短后的内容，原文丢了"
    assert "A" * 100 in prompt


def test_summary_prompt_forbids_fabrication():
    from app.agent.prompts import SUMMARY_PROMPT

    assert "一个字都不许编" in SUMMARY_PROMPT
    assert "寒暄" in SUMMARY_PROMPT
    assert "待压缩对话" in SUMMARY_PROMPT
    assert "已记录背景" in SUMMARY_PROMPT
    assert "几十到一两百字" in SUMMARY_PROMPT
    assert "{background}" in SUMMARY_PROMPT
    assert "{conversation}" in SUMMARY_PROMPT
