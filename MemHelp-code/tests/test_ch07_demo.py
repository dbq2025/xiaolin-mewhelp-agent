"""ch07 演示脚本的离线验证（Task 13）。

只钉脚本里**不需要网络与数据库**的部分：预算表数字、20 轮压测的规划结果。
真正的端到端演示由 ``scripts/demo_ch07.py`` 自己跑，不走 pytest —— 它要起服务、
要写真模型，那不属于单元测试的职责。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    """按文件路径导入 ``scripts/`` 下的模块。

    ``scripts`` 不是包（没有 ``__init__.py``），而 ``sys.path`` 里也没有它，
    所以直接 ``import scripts.demo_ch07`` 会失败。用 spec 从路径加载，
    顺带让脚本里 ``from app...`` 的绝对导入照常工作（ROOT 已在 sys.path 里）。
    """
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ch07_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_demo_budget_matches_acceptance_numbers():
    """验收第 2 条点名的三个数字必须逐项对上。

    这组数字是「窗口 − 固定开销」算出来的，不是拍脑袋定的：
    18000 − 2000(输出) − 5600(ReAct 峰值) − 1800(10% 余量) − 1200(系统提示)
    − 1250(5 条证据) − 500(梗概) = 5650；层 2 拿 30% = 1695；
    层 1 拿剩下的再让 1 个 token 给边界标记 = 3954。
    """
    demo = _load("demo_ch07")
    report = demo.demo_budget()
    assert report.history_budget == 5650
    assert report.layer1_budget == 3954
    assert report.layer2_budget == 1695
    assert report.overhead == 12350
    assert report.ok is True


def test_demo_budget_overrides_recompute_everything():
    """改一个输入，整张表都要跟着变 —— 预算若不是倒推的，这条会红。"""
    demo = _load("demo_ch07")
    wide = demo.demo_budget(window=64000)
    assert wide.history_budget > 5650
    assert wide.layer1_budget == wide.history_budget - wide.layer2_budget - 1

    narrow = demo.demo_budget(window=12000)
    assert narrow.ok is False
    assert narrow.reason.startswith("上下文预算不足")


def test_twenty_turns_produces_no_compression_under_default_window():
    """验收第 3 条：默认窗口下聊 20 轮，不该触发任何降级与摘要。

    「装得下就不压」和「超了就压」是同一个算法的两面，只验后者会让实现
    悄悄变成「每轮都压」也能过。
    """
    twenty = _load("ch07_twenty_turns")
    summary = twenty.run_twenty_turns_fake(turns=20)
    assert summary["degraded"] is False
    assert summary["summary_due"] is False
    assert summary["max_layer1_tokens"] <= summary["layer1_budget"]
    assert summary["max_layer2_tokens"] <= summary["layer2_budget"]
    assert summary["layer1_budget"] == summary["budget"].layer1_budget


def test_twenty_turns_under_demo_window_does_compress():
    """同一段对话放到演示窗口下，必须压 —— 否则说明 20 轮压根没增长。

    这条是上一条的反面对照：如果对话太短，`test_twenty_turns_produces_no_compression`
    就算实现坏了（永远不压）也会绿。
    """
    twenty = _load("ch07_twenty_turns")
    summary = twenty.run_twenty_turns_fake(turns=20, demo_window=True)
    assert summary["degraded"] is True or summary["summary_due"] is True


def test_summary_trigger_line_matches_the_acceptance_shape():
    """验收第 2 条要在日志里看到 `layer2_tokens=… > budget=…` 这个形状。"""
    from app.agent.ctxlog import log_summary_event
    from app.agent.ctxlog import AppLogger

    line = log_summary_event(
        logger=AppLogger(""),
        event="trigger",
        session_id="s-demo",
        layer2_tokens=1832,
        budget=1695,
        msgs=14,
    )
    assert "summary trigger" in line
    assert "layer2_tokens=1832 > budget=1695" in line
