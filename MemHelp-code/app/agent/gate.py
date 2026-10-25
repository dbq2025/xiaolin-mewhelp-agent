"""置信度闸 —— 知识类检索之后、Agent 之前。

两个版本并存：

* ``evaluate_gate``：ch05 的最简阈值版（0 命中 / Top1 低于阈值），作为
  ch09 模型 artifact 缺失或损坏时的兜底；
* ``evaluate_evidence_gate``：ch09 正式版，用 ``evidence_confidence`` 模型
  对 Top1、Top1−Top2 分差、有效证据数打分，阈值来自 ch04 评估集留一法校准。

**重排降级时只看 0 命中**：降级后的 ``chunks[0].score`` 是 RRF 融合分，量级 0.03，
拿去比 0.3 会把每一轮知识类问题都拒掉（ch04 实测过同一个坑）。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.evidence_confidence import ConfidenceModel, EvidenceFeatures

__all__ = ["GateDecision", "evaluate_evidence_gate", "evaluate_gate"]


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


def evaluate_evidence_gate(
    *,
    evidence_count: int,
    features: EvidenceFeatures,
    confidence: float,
    rerank_degraded: bool,
    model: ConfidenceModel,
) -> GateDecision:
    """ch09 正式闸：模型概率与阈值比较，降级时放行。

    顺序是有意的：

    1. **0 证据**无论模型怎么说都不过 —— 没有证据就没有「相关性」可谈；
    2. **重排降级**直接放行 —— 此时分数是 RRF 融合分（量级 0.03），
       进模型只会系统性地判低；ch04/ch05 已经踩过这个坑；
    3. 其余情况用 ``confidence >= model.threshold``。
    """
    score = features.top1
    if evidence_count <= 0:
        return GateDecision(
            passed=False,
            reason="检索 0 命中，没有可交给 Agent 的证据",
            score=score,
            evidence_count=0,
        )

    if rerank_degraded:
        return GateDecision(
            passed=True,
            reason="重排降级，按融合分放行（不套置信度模型）",
            score=score,
            evidence_count=evidence_count,
        )

    if confidence < model.threshold:
        return GateDecision(
            passed=False,
            reason=(
                f"证据置信度 {confidence:.2f} 低于阈值 {model.threshold:.2f}"
                f"（Top1={features.top1:.3f}，分差={features.margin:.3f}，"
                f"有效证据={features.effective_count}）"
            ),
            score=score,
            evidence_count=evidence_count,
        )

    return GateDecision(
        passed=True,
        reason=f"证据置信度 {confidence:.2f} 达到阈值 {model.threshold:.2f}",
        score=score,
        evidence_count=evidence_count,
    )
