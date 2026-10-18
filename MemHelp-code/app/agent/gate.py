"""置信度闸 —— 知识类检索之后、Agent 之前。

## 本章只做「最简阈值版」

判据只有两条确定性检查：
1. 有没有证据（0 命中直接不过）；
2. 有重排分时，最高分是否达到 `rag_min_rerank_score`。

**重排降级时只看 0 命中**：降级后的 `chunks[0].score` 是 RRF 融合分，量级 0.03，
拿去比 0.3 会把每一轮知识类问题都拒掉（ch04 实测过同一个坑）。

正式的置信度检查（证据覆盖度、模型自评、分布标定）留给可观测章节。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GateDecision", "evaluate_gate"]


@dataclass(frozen=True)
class GateDecision:
    """闸的结论与理由。

    :param reason: 进日志/低置信度池的理由，必须能区分「0 命中」与「分数低」。
    """

    passed: bool
    reason: str
    score: float
    evidence_count: int


def evaluate_gate(
    *,
    refused: bool,
    evidence_count: int,
    top_score: float,
    rerank_degraded: bool,
    min_score: float,
) -> GateDecision:
    """确定性判断证据是否够格进 Agent。"""
    if refused:
        return GateDecision(
            passed=False,
            reason="检索层判定证据不足（0 命中或分数低于阈值）",
            score=top_score,
            evidence_count=evidence_count,
        )

    if evidence_count <= 0:
        return GateDecision(
            passed=False,
            reason="检索 0 命中，没有可交给 Agent 的证据",
            score=top_score,
            evidence_count=0,
        )

    if not rerank_degraded and top_score < min_score:
        return GateDecision(
            passed=False,
            reason=(
                f"最高检索分 {top_score:.3f} 低于阈值 {min_score}，证据相关性不足"
            ),
            score=top_score,
            evidence_count=evidence_count,
        )

    return GateDecision(
        passed=True,
        reason="证据通过置信度闸",
        score=top_score,
        evidence_count=evidence_count,
    )
