"""上下文可观测：grep model_ctx / history_ctx 必须一行命中。

验收第 4 条要求「日志里能看到早期轮次被后台摘要任务压缩的痕迹」「grep
model_ctx / history_ctx 能直接看到每轮发给模型的摘要和滑窗内容」。所以这组
用例钉的不是「有没有写日志」，而是**写出来的东西能不能被 grep 直接读懂**：
一行里要有人能肉眼确认的锚点、条数、token 估算，后面跟着完整的摘要与滑窗正文。

日志是**旁路**：写失败只记 warning，绝不能让用户拿不到回答（对应
``test_write_failure_is_swallowed``）。
"""

import shutil
import uuid
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.ctxlog import (
    MAX_BODY_CHARS,
    AppLogger,
    log_history_ctx,
    log_model_ctx,
    log_summary_event,
)


#: 放临时日志的目录，位于工作区内。
#:
#: 为什么不用 ``tmp_path``：沙箱会把测试进程**新建的目录**在退出后收回权限
#: （``conftest.py`` 顶部有完整说明）。这里写进工作区里已存在的目录，用完即删。
LOG_FIXTURE_DIR = Path(".uv-cache") / "ctxlog-fixtures"


@pytest.fixture
def log_dir():
    """一个自建的可写目录，每次用例一个独立子目录（并行跑也不撞）。"""
    LOG_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    made = LOG_FIXTURE_DIR / uuid.uuid4().hex
    made.mkdir()
    yield made
    shutil.rmtree(made, ignore_errors=True)


def _read(path):
    return path.read_text(encoding="utf-8")


class _Budget:
    """`BudgetReport` 的鸭子替身：日志只读这几个字段，不必构造真报告。"""

    layer1_budget = 3954
    layer2_budget = 1695
    history_budget = 5650


def test_model_ctx_writes_one_greppable_record(log_dir):
    path = log_dir / "app.log"
    line = log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[SystemMessage("人设"), HumanMessage("我买了耳机"), AIMessage("好的")],
        layer1_from_msg_id=5,
        summary_upto_msg_id=3,
        budget=_Budget(),
        layer1_tokens=1200,
        layer2_tokens=300,
        summary_texts=["用户问过耳机"],
    )
    text = _read(path)
    assert "model_ctx session=s1 turn=t1" in text
    assert "用户问过耳机" in text
    assert "我买了耳机" in text
    assert "windows=" in text
    assert "tokens=" in text
    assert line.splitlines()[0].startswith("model_ctx")


def test_model_ctx_records_both_anchors_and_layer_budgets(log_dir):
    path = log_dir / "app.log"
    log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[HumanMessage("hi")],
        layer1_from_msg_id=5,
        summary_upto_msg_id=3,
        budget=_Budget(),
        layer1_tokens=10,
        layer2_tokens=2,
    )
    text = _read(path)
    for piece in (
        "summary_upto=3",
        "layer1_from=5",
        "layer1_budget=3954",
        "layer2_budget=1695",
    ):
        assert piece in text


def test_model_ctx_reports_window_count_and_token_estimate(log_dir):
    path = log_dir / "app.log"
    log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[HumanMessage("a"), AIMessage("b"), HumanMessage("c")],
        layer1_from_msg_id=None,
        summary_upto_msg_id=None,
        budget=_Budget(),
        layer1_tokens=7,
        layer2_tokens=0,
    )
    text = _read(path)
    assert "windows=3" in text
    assert "tokens=7" in text


def test_model_ctx_prints_every_message_on_its_own_line(log_dir):
    """逐条打印是为了能一眼看出「哪条进了上下文、哪条被截短」。"""
    path = log_dir / "app.log"
    log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[HumanMessage("第一句"), AIMessage("第二句")],
        layer1_from_msg_id=1,
        summary_upto_msg_id=None,
        budget=_Budget(),
        layer1_tokens=5,
        layer2_tokens=0,
    )
    text = _read(path)
    assert "[user] 第一句" in text
    assert "[ai] 第二句" in text
    assert text.count("\n") >= 3


def test_model_ctx_prints_each_summary_segment_with_its_number(log_dir):
    path = log_dir / "app.log"
    log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[],
        layer1_from_msg_id=None,
        summary_upto_msg_id=10,
        budget=_Budget(),
        layer1_tokens=0,
        layer2_tokens=0,
        summary_texts=["第一段事实", "第二段事实"],
    )
    text = _read(path)
    assert "[摘要 1] 第一段事实" in text
    assert "[摘要 2] 第二段事实" in text


def test_long_body_is_truncated(log_dir):
    """不截断的话一轮就能把日志写成几 MB，grep 反而更难用。"""
    path = log_dir / "app.log"
    log_model_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[HumanMessage("长" * (MAX_BODY_CHARS + 100))],
        layer1_from_msg_id=1,
        summary_upto_msg_id=None,
        budget=_Budget(),
        layer1_tokens=5,
        layer2_tokens=0,
    )
    text = _read(path)
    assert "长" * (MAX_BODY_CHARS + 100) not in text
    assert MAX_BODY_CHARS == 500


def test_history_ctx_is_written_even_without_summary(log_dir):
    """闲聊兜底轮也要打 —— 那些轮次不经过 Agent 节点，最容易没有日志。"""
    path = log_dir / "app.log"
    log_history_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        messages=[HumanMessage("上一句"), AIMessage("上一答")],
    )
    text = _read(path)
    assert "history_ctx session=s1 turn=t1" in text
    assert "上一句" in text


def test_history_ctx_marks_the_summary_line(log_dir):
    path = log_dir / "app.log"
    log_history_ctx(
        logger=AppLogger(path),
        session_id="s1",
        turn="t1",
        summary_texts=["用户报过订单号 A1"],
        messages=[HumanMessage("后来呢")],
    )
    text = _read(path)
    assert "[摘要 1] 用户报过订单号 A1" in text
    assert "summaries=1" in text


def test_history_ctx_handles_empty_history(log_dir):
    path = log_dir / "app.log"
    log_history_ctx(logger=AppLogger(path), session_id="s1", turn="t1", messages=[])
    text = _read(path)
    assert "history_ctx session=s1 turn=t1" in text
    assert "windows=0" in text


def test_summary_trigger_is_tagged_with_the_comparison(log_dir):
    path = log_dir / "app.log"
    log_summary_event(
        logger=AppLogger(path),
        event="trigger",
        session_id="s1",
        layer2_tokens=1832,
        budget=1695,
        msgs=14,
    )
    text = _read(path)
    assert "summary trigger session=s1" in text
    assert "layer2_tokens=1832 > budget=1695" in text


def test_summary_done_carries_segment_and_elapsed(log_dir):
    path = log_dir / "app.log"
    log_summary_event(
        logger=AppLogger(path),
        event="done",
        session_id="s1",
        seg=3,
        chars=118,
        elapsed_ms=742,
        upto_msg_id=54,
    )
    text = _read(path)
    assert "summary done session=s1" in text
    assert "seg=3" in text
    assert "elapsed_ms=742" in text
    assert "upto_msg_id=54" in text


@pytest.mark.parametrize("event", ["trigger", "start", "done", "skip", "failed"])
def test_all_five_summary_events_are_greppable(log_dir, event):
    """五态都要能单独 grep 出来 —— 只看 trigger 不看 skip 会以为摘要没触发过。"""
    path = log_dir / "app.log"
    log_summary_event(logger=AppLogger(path), event=event, session_id="s1")
    assert f"summary {event} session=s1" in _read(path)


def test_unknown_summary_event_is_rejected(log_dir):
    """手滑写成 `complet` 的日志会静静地少掉一类事件，必须当场炸。"""
    with pytest.raises(ValueError):
        log_summary_event(logger=AppLogger(log_dir / "a.log"), event="complet", session_id="s")


def test_missing_directory_is_created(log_dir):
    path = log_dir / "deep" / "nested" / "app.log"
    log_history_ctx(logger=AppLogger(path), session_id="s", turn="t", messages=[])
    assert path.is_file()


def test_write_failure_is_swallowed(log_dir):
    """日志是旁路：路径不可写时不能让用户拿不到回答。"""
    blocker = log_dir / "app.log"
    blocker.mkdir()
    AppLogger(blocker).write("x")  # 不抛即通过


def test_none_path_only_logs(log_dir):
    """`APP_LOG_PATH` 为空时不建文件、不报错 —— 测试环境默认就是这样。"""
    AppLogger(None).write("hello")
    assert list(log_dir.iterdir()) == []


def test_relative_path_is_resolved_against_project_root():
    from app.agent.ctxlog import resolve_log_path
    from app.config import PROJECT_ROOT

    assert resolve_log_path("log/app.log") == PROJECT_ROOT / "log" / "app.log"
    assert resolve_log_path(None) is None
