"""token 预算：从模型窗口**倒推**，不写死成常量。

## 为什么不能写死

ch01–ch06 的历史预算是 ``CONTEXT_TOKEN_BUDGET=2000``，一个与模型无关的常量。
它有两个毛病：调大窗口不生效，调小窗口则会溢出。真正的约束是「模型窗口装不下
一次完整请求」，所以预算必须从 ``MODEL_CONTEXT_WINDOW`` 一路减下来。

## 扣减顺序（每一项都有对应的物理来源）

```
W                模型窗口
− 输出预留        模型要往外吐的 token（MAX_OUTPUT_TOKENS）
− 单轮 ReAct 峰值  一轮里用户输入 + 每一步工具结果（MAX_USER_INPUT_TOKENS
                   + MAX_AGENT_STEPS × TOOL_RESULT_MAX_TOKENS）
− 安全余量        窗口的 10%，用来吸收计数器误差与模板差异
− 系统提示        人设 + 红线 + 工具定义（每轮固定）
− 检索证据        RERANK_TOP_K × 每条估算
− 注入梗概        会话梗概合成的那一条
= 窗口实际匀得出来的历史空间
```

历史预算取「想留住的轮数 × 每轮稳态占用」与「窗口匀得出来的」**两个数里小的那个**：
窗口很大时不该无限塞历史（上下文越长越贵、也越容易跑偏），窗口很小时只能将就。

层 1 拿七成、层 2 拿三成。那 **1 个 token** 是给层 2 边界标记留的：
``layer1_budget + layer2_budget + 1 == history_budget``，这样两个数加起来永远不会
「刚好等于」总预算而在浮点/取整上溢出。

## 演示配置的账（验收第 2 条）

```
18000 − 2000 − 5600 − 1800 − 1200 − 1250 − 500 = 5650
5650 × 0.3 = 1695（层 2）
5650 − 1695 − 1 = 3954（层 1）
```

``tests/test_ch07_budget.py`` 把这几个数字钉死。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: ``BudgetReport.reason`` 的前缀。验收要 grep 这句话，所以它是契约的一部分。
INSUFFICIENT_BUDGET_PREFIX = "上下文预算不足"


@dataclass(frozen=True)
class BudgetReport:
    """一次预算计算的完整快照 —— 每一项都可被日志与断言读到。

    刻意做成 ``frozen``：同一轮里所有下游（分层、拼装、日志）必须看到**同一份**
    预算。可变对象被谁悄悄改一下，就会出现「日志说 3954、实际按 3000 裁」这种
    最难查的错。
    """

    model_context_window: int
    """模型窗口（``MODEL_CONTEXT_WINDOW``）。"""

    output_reserve: int
    """输出预留（``MAX_OUTPUT_TOKENS``）。"""

    react_peak: int
    """单轮 ReAct 瞬时峰值 = 用户输入 + 步数 × 单次工具结果。"""

    safety_margin: int
    """安全余量 = ceil(窗口 × 比例)。"""

    system_reserve: int
    """系统提示固定估算。"""

    evidence_reserve: int
    """检索证据固定估算 = 条数 × 每条。"""

    summary_reserve: int
    """注入梗概固定估算。"""

    overhead: int
    """以上固定开销之和。"""

    window_available: int
    """窗口匀得出来的历史空间（可能 ≤ 0）。"""

    desired_history: int
    """「想留住的轮数 × 每轮稳态占用」。"""

    history_budget: int
    """历史总预算 = min(窗口匀得出来的, 想留住的)。"""

    layer1_budget: int
    """层 1（原文滑窗）预算。"""

    layer2_budget: int
    """层 2（规则截短）预算。"""

    per_turn_floor: int
    """「至少能装下一轮」的门槛（``HISTORY_TOKENS_PER_TURN``）。"""

    ok: bool
    """预算是否健康。``False`` 时启动会打 ERROR，但不阻断。"""

    reason: str
    """``ok=False`` 时以「上下文预算不足」开头；``ok=True`` 时是一行摘要。"""


def compute_budget(settings: Any) -> BudgetReport:
    """按配置算一次预算。纯函数：不读环境、不碰 IO，同样输入永远同样输出。

    用 ``getattr`` 取值而不是类型标注 ``Settings``：``budget`` 是纯算法模块，
    不该反向依赖配置类。鸭子类型让它可以在测试里用一个 ``SimpleNamespace``
    喂进来，也能让 ``scripts/demo_ch07.py`` 用不同参数反复算。

    :param settings: 任何带下列属性的对象：``model_context_window`` /
        ``max_output_tokens`` / ``max_user_input_tokens`` /
        ``tool_result_max_tokens`` / ``agent_max_steps`` /
        ``safety_margin_ratio`` / ``system_prompt_reserve_tokens`` /
        ``rag_rerank_top_n`` / ``evidence_tokens_per_item`` /
        ``summary_reserve_tokens`` / ``history_desired_turns`` /
        ``history_tokens_per_turn`` / ``layer1_ratio`` / ``layer2_ratio``。
    """
    window = int(getattr(settings, "model_context_window"))
    output_reserve = int(getattr(settings, "max_output_tokens"))
    user_input = int(getattr(settings, "max_user_input_tokens"))
    tool_result = int(getattr(settings, "tool_result_max_tokens"))
    steps = int(getattr(settings, "agent_max_steps"))
    margin_ratio = float(getattr(settings, "safety_margin_ratio"))
    system_reserve = int(getattr(settings, "system_prompt_reserve_tokens"))
    evidence_items = int(getattr(settings, "rag_rerank_top_n"))
    evidence_per_item = int(getattr(settings, "evidence_tokens_per_item"))
    summary_reserve = int(getattr(settings, "summary_reserve_tokens"))
    desired_turns = int(getattr(settings, "history_desired_turns"))
    tokens_per_turn = int(getattr(settings, "history_tokens_per_turn"))
    layer1_ratio = float(getattr(settings, "layer1_ratio"))
    layer2_ratio = float(getattr(settings, "layer2_ratio"))

    react_peak = user_input + steps * tool_result
    safety_margin = math.ceil(window * margin_ratio)
    evidence_reserve = evidence_items * evidence_per_item

    overhead = (
        output_reserve
        + react_peak
        + safety_margin
        + system_reserve
        + evidence_reserve
        + summary_reserve
    )
    window_available = window - overhead
    desired_history = desired_turns * tokens_per_turn
    history_budget = min(window_available, desired_history)

    # 窗口被固定开销吃光：历史空间是负数，任何裁剪都无从谈起。
    if window_available <= 0:
        return BudgetReport(
            model_context_window=window,
            output_reserve=output_reserve,
            react_peak=react_peak,
            safety_margin=safety_margin,
            system_reserve=system_reserve,
            evidence_reserve=evidence_reserve,
            summary_reserve=summary_reserve,
            overhead=overhead,
            window_available=window_available,
            desired_history=desired_history,
            history_budget=0,
            layer1_budget=0,
            layer2_budget=0,
            per_turn_floor=tokens_per_turn,
            ok=False,
            reason=(
                f"{INSUFFICIENT_BUDGET_PREFIX}：窗口 {window} 装不下固定开销 {overhead}"
                f"（输出 {output_reserve} + ReAct 峰值 {react_peak} + 安全余量 {safety_margin}"
                f" + 系统提示 {system_reserve} + 证据 {evidence_reserve}"
                f" + 梗概 {summary_reserve}），匀得 {window_available}"
            ),
        )

    # 匀得出空间，但连一轮都装不下 —— 滑窗会归零，等于每轮都只能看当前这句。
    if history_budget < tokens_per_turn:
        return BudgetReport(
            model_context_window=window,
            output_reserve=output_reserve,
            react_peak=react_peak,
            safety_margin=safety_margin,
            system_reserve=system_reserve,
            evidence_reserve=evidence_reserve,
            summary_reserve=summary_reserve,
            overhead=overhead,
            window_available=window_available,
            desired_history=desired_history,
            history_budget=max(history_budget, 0),
            layer1_budget=0,
            layer2_budget=0,
            per_turn_floor=tokens_per_turn,
            ok=False,
            reason=(
                f"{INSUFFICIENT_BUDGET_PREFIX}：历史预算 {history_budget}"
                f" 连一轮 {tokens_per_turn} 都装不下（窗口 {window}，固定开销 {overhead}）"
            ),
        )

    layer2_budget = round(history_budget * layer2_ratio)
    layer1_budget = history_budget - layer2_budget - 1

    return BudgetReport(
        model_context_window=window,
        output_reserve=output_reserve,
        react_peak=react_peak,
        safety_margin=safety_margin,
        system_reserve=system_reserve,
        evidence_reserve=evidence_reserve,
        summary_reserve=summary_reserve,
        overhead=overhead,
        window_available=window_available,
        desired_history=desired_history,
        history_budget=history_budget,
        layer1_budget=layer1_budget,
        layer2_budget=layer2_budget,
        per_turn_floor=tokens_per_turn,
        ok=True,
        reason=format_budget_report_parts(window, overhead, history_budget, layer1_budget, layer2_budget),
    )


def format_budget_report_parts(
    window: int, overhead: int, history: int, layer1: int, layer2: int
) -> str:
    """一行摘要的拼装（拆出来是为了让 ``compute_budget`` 里那个 ``reason`` 短一点）。"""
    return f"window={window} overhead={overhead} history={history} layer1={layer1} layer2={layer2}"


def format_budget_report(report: BudgetReport) -> str:
    """把报告压成一行，供 lifespan 打 INFO、供脚本打印。"""
    return format_budget_report_parts(
        report.model_context_window,
        report.overhead,
        report.history_budget,
        report.layer1_budget,
        report.layer2_budget,
    )


__all__ = [
    "INSUFFICIENT_BUDGET_PREFIX",
    "BudgetReport",
    "compute_budget",
    "format_budget_report",
    "format_budget_report_parts",
]
