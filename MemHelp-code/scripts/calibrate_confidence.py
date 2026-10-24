"""用 ch04 评估集校准证据置信度（ch09 Task 7）。

跑法：``uv run python scripts/calibrate_confidence.py``

输出：
- ``evals/ch09_confidence_model.json``：仅当留一法 balanced accuracy ≥ 0.70 才发布；
- ``dev-notes/ch09-confidence-calibration.md``：无论是否发布都写，含混淆矩阵、
  阈值、逐桶统计与逐条预测。

低于 0.70 时返回码 2，闸退回 ch05 的阈值版 —— 不拿没把握的模型上线。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.evidence_confidence import (  # noqa: E402
    EvidenceFeatures,
    calibrate_loo,
    features_from_scores,
    model_to_dict,
)
from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402
from app.knowledge.embedder import get_embedder  # noqa: E402
from app.knowledge.hybrid import HybridRetriever  # noqa: E402
from app.knowledge.pipeline import RagPipeline  # noqa: E402
from app.knowledge.query import LlmRewriter, QueryUnderstander  # noqa: E402
from app.knowledge.rerank import get_reranker  # noqa: E402
from app.knowledge.sufficiency import SufficiencyJudge  # noqa: E402
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402
from app.llm import get_chat_model  # noqa: E402

CASES_PATH = PROJECT_ROOT / "evals" / "ch04_cases.jsonl"
MODEL_PATH = PROJECT_ROOT / "evals" / "ch09_confidence_model.json"
REPORT_PATH = PROJECT_ROOT / "dev-notes" / "ch09-confidence-calibration.md"
MIN_BALANCED_ACCURACY = 0.70

failed_ids: list[str] = []


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读 JSONL，跳过 ``#`` 注释行（评估集头部有两行说明）。"""
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cases.append(json.loads(stripped))
    return cases


def label_for(behavior: str) -> int:
    if behavior == "answer":
        return 1
    if behavior == "refuse":
        return 0
    raise ValueError(f"未知 behavior={behavior!r}，只支持 answer/refuse")


async def collect_samples(
    cases: Sequence[dict[str, Any]], pipeline: Any
) -> list[tuple[EvidenceFeatures, int]]:
    """逐条走 ``retrieve_for_gate`` 采集分数；单条失败不拖垮整轮。"""
    failed_ids.clear()
    samples: list[tuple[EvidenceFeatures, int]] = []
    for case in cases:
        try:
            outcome = await pipeline.retrieve_for_gate(
                case["query"], category=case.get("category")
            )
            scores = [float(item.score) for item in outcome.evidence]
        except Exception:  # noqa: BLE001 - 一条检索失败不该让校准整体中断
            failed_ids.append(str(case["id"]))
            scores = []
        samples.append((features_from_scores(scores), label_for(case["behavior"])))
    return samples


def _bucket_rows(
    cases: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> str:
    buckets: dict[str, dict[str, int]] = {}
    for case, probability, label in zip(cases, probabilities, labels):
        bucket = str(case.get("bucket") or "-")
        stats = buckets.setdefault(bucket, {"n": 0, "correct": 0})
        stats["n"] += 1
        predicted = 1 if probability >= threshold else 0
        stats["correct"] += int(predicted == label)
    lines = ["| 桶 | 条数 | 判对 | 准确率 |", "| --- | --- | --- | --- |"]
    for bucket in sorted(buckets):
        stats = buckets[bucket]
        accuracy = stats["correct"] / stats["n"] if stats["n"] else 0.0
        lines.append(
            f"| {bucket} | {stats['n']} | {stats['correct']} | {accuracy:.2%} |"
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    cases: Sequence[dict[str, Any]],
    samples: Sequence[tuple[EvidenceFeatures, int]],
    model: dict[str, Any],
    calibration_report: dict[str, Any],
    published: bool,
    failed_ids: Sequence[str],
) -> None:
    threshold = float(model["threshold"])
    probabilities = calibration_report.get("probabilities") or [0.0] * len(cases)
    labels = calibration_report.get("labels") or [label for _features, label in samples]
    confusion = calibration_report["confusion"]
    rows = [
        "# ch09 证据置信度校准报告",
        "",
        f"- 评估集：`{CASES_PATH}`",
        f"- 样本数：{calibration_report.get('n', len(cases))}（正类 {calibration_report.get('positives', sum(labels))}）",
        f"- 阈值：{threshold:.2f}",
        f"- balanced accuracy（留一法）：{calibration_report['balanced_accuracy']:.4f}",
        f"- 是否发布 artifact：{'是' if published else '否（低于 0.70，闸退回 ch05 阈值版）'}",
        "",
        "## 混淆矩阵",
        "",
        "| TP | FP | TN | FN |",
        "| --- | --- | --- | --- |",
        f"| {confusion['tp']} | {confusion['fp']} | {confusion['tn']} | {confusion['fn']} |",
        "",
        "## 分桶",
        "",
        _bucket_rows(cases, probabilities, labels, threshold),
        "",
        "## 逐条预测",
        "",
        "| 用例 | 桶 | 期望 | 置信度 | 判定 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case, probability, label in zip(cases, probabilities, labels):
        predicted = 1 if probability >= threshold else 0
        mark = "✓" if predicted == label else "✗"
        rows.append(
            f"| {case['id']} | {case.get('bucket') or '-'} | {case['behavior']} | "
            f"{probability:.3f} | {mark} |"
        )
    if failed_ids:
        rows += ["", "## 检索失败", "", "、".join(failed_ids)]
    rows += [
        "",
        "## 特征",
        "",
        "- `top1`：精排 Top1 相关性分。",
        "- `margin`：Top1 与 Top2 的分差；Top2 缺失按 0。",
        "- `effective_count`：分数 ≥ max(0.1, top1×0.5) 的证据条数。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _build_pipeline(settings: Any, embedder: Any, store: Any, reranker: Any, factory):
    with factory() as session:
        repositories = Repositories.from_session(session)
        understander = QueryUnderstander(
            LlmRewriter(get_chat_model(settings), timeout=settings.rag_rewrite_timeout),
            enabled=settings.rag_query_rewrite,
        )
        retriever = HybridRetriever(
            embedder=embedder,
            vector_store=store,
            reranker=reranker,
            chunk_repo=repositories.knowledge,
            hybrid_top_k=settings.rag_hybrid_top_k,
            candidate_k=settings.rag_candidate_k,
            rerank_top_n=settings.rag_rerank_top_n,
        )
        return RagPipeline(
            understander=understander,
            retriever=retriever,
            sufficiency=SufficiencyJudge(
                get_chat_model(settings), timeout=settings.rag_rewrite_timeout
            ),
            min_score=settings.rag_min_rerank_score,
            top_n=settings.rag_rerank_top_n,
            strategy="hybrid_rerank",
        )


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    settings = get_settings()
    settings.require_embedding_key()
    engine = create_engine_from(settings)
    factory = session_factory(engine)
    embedder = get_embedder(settings)
    store = MilvusVectorStore(
        settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
    )
    if await store.count() == 0:
        print("Milvus 集合为空 —— 先跑 scripts/build_knowledge.py 建库。", file=sys.stderr)
        return 1
    pipeline = _build_pipeline(settings, embedder, store, get_reranker(settings), factory)

    samples = await collect_samples(cases, pipeline)
    model, report = calibrate_loo(samples)
    published = report["balanced_accuracy"] >= MIN_BALANCED_ACCURACY
    if published:
        MODEL_PATH.write_text(
            json.dumps(model_to_dict(model), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    write_report(
        Path(args.report),
        cases=cases,
        samples=samples,
        model=model_to_dict(model),
        calibration_report=report,
        published=published,
        failed_ids=failed_ids,
    )
    print(f"balanced accuracy = {report['balanced_accuracy']:.4f}（阈值 {model.threshold:.2f}）")
    print(f"报告：{args.report}")
    if published:
        print(f"模型：{args.model}")
        return 0
    print("未达到 0.70，未发布模型；闸保持 ch05 阈值版。", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用 ch04 评估集校准证据置信度")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
