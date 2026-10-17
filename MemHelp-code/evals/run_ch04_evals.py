"""ch04 四策略评估：检索段（Recall@K / MRR）+ 生成段（Faithfulness / 拒答准确率）。

## 为什么四条策略必须在同一份索引上跑

换索引、换切分、换嵌入模型之后再比「谁更好」，比的就不是策略了。这个脚本
**只**切策略，其余全部共用：同一份 ``knowledge_chunks``、同一个 Milvus 集合、
同一个 Query 理解、同一个重排模型。

## 为什么拒答题不参与检索指标

拒答题没有 gold —— 它的正确行为是「检索不到」。把它算进 Recall 分母，
会让「检索越差、拒答越准」的策略看起来更好。所以检索指标只算
``behavior == "answer"`` 的题，拒答准确率单独统计。

## 为什么没跑的策略报 ``None`` 而不是 0

报告要被拿去决策。用 0 冒充「没测」会让读者以为某个策略在那一桶上真的很差。
`"—"` 与 `0.0` 是两个完全不同的信息。

## 生成段的提示词是「评估时提示词」

它与线上 System Prompt 共享**引用规则**（``app/prompts.py::CITATION_RULES``），
但不是同一条链路（线上由模型自主决定是否调用 ``query_faq``）。报告里会写清楚这点 ——
把评估口径说成「线上真实表现」是不诚实的。

用法::

    uv run --directory MemHelp-code python evals/run_ch04_evals.py --validate-only
    uv run --directory MemHelp-code python evals/run_ch04_evals.py --strategies vector,bm25,hybrid,hybrid_rerank
    uv run --directory MemHelp-code python evals/run_ch04_evals.py --only B_model --strategies bm25 -v
    uv run --directory MemHelp-code python evals/run_ch04_evals.py --generation hybrid,hybrid_rerank \\
        --json evals/reports/ch04.json --markdown dev-notes/ch04-eval-report.md

退出码：0 = ``hybrid_rerank`` 整体 Recall@5 ≥ ``vector``，且 ``B_model`` 桶
Recall@5 ≥ ``bm25``；否则 1。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402
from app.knowledge.citations import (  # noqa: E402
    build_evidence,
    evidence_payload,
    parse_citations,
    prompt_order,
)
from app.knowledge.embedder import get_embedder  # noqa: E402
from app.knowledge.hybrid import HybridRetriever, Strategy  # noqa: E402
from app.knowledge.pipeline import RagPipeline  # noqa: E402
from app.knowledge.query import LlmRewriter, QueryUnderstander  # noqa: E402
from app.knowledge.rerank import get_reranker  # noqa: E402
from app.knowledge.sufficiency import SufficiencyJudge  # noqa: E402
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402
from app.llm import get_chat_model  # noqa: E402
from app.prompts import CITATION_RULES, REFUSAL_TEXT  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "ch04_cases.jsonl"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"

BUCKETS: tuple[str, ...] = ("A_policy", "B_model", "C_colloquial", "E_multi")
#: 每桶在报告里都要出现 —— 即使这一轮没跑，也要有个 ``—`` 占位。
BUCKET_KEYS: tuple[str, ...] = (*BUCKETS, "overall")
RECALL_KS: tuple[int, ...] = (1, 3, 5, 10)

ALL_STRATEGIES: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank")


# ---------------------------------------------------------------------------
# 载入与金标
# ---------------------------------------------------------------------------


def load_cases(path: Path = CASES_FILE) -> list[dict]:
    cases: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_no} 不是合法 JSON：{exc}") from exc
    return cases


def resolve_gold(
    cases: Sequence[dict],
    rows: Sequence[Any],
    *,
    warnings: dict | None = None,
) -> dict[str, set[int]]:
    """把金标锚串解析成 chunk_id 集合（各锚串命中块的并集）。

    :param rows: 带 ``id`` / ``questions`` / ``answer`` 的行（``ChunkRow`` 或 dict）。
    :param warnings: 可选输出参数，记录每个锚串命中的块数（>3 说明锚串太泛）。
    :raises ValueError: 某个锚串一块都没命中 —— 那是标注写错或语料没建，不能静默跳过。
    """
    texts: list[tuple[int, str]] = []
    for row in rows:
        row_id = row["id"] if isinstance(row, dict) else row.id
        questions = row["questions"] if isinstance(row, dict) else row.questions
        answer = row["answer"] if isinstance(row, dict) else row.answer
        texts.append((int(row_id), f"{questions}\n{answer}"))

    gold: dict[str, set[int]] = {}
    for case in cases:
        anchors = case.get("gold_anchors") or []
        if not anchors:
            gold[case["id"]] = set()
            continue
        hits: set[int] = set()
        for anchor in anchors:
            matched = [row_id for row_id, text in texts if anchor in text]
            if not matched:
                raise ValueError(
                    f"评估集 {case['id']} 的锚串 {anchor!r} 在 knowledge_chunks 里一块都没命中"
                    "（标注写错，或索引没重建）"
                )
            if warnings is not None:
                warnings.setdefault(case["id"], {})[anchor] = len(matched)
            hits.update(matched)
        gold[case["id"]] = hits
    return gold


# ---------------------------------------------------------------------------
# 指标（纯函数，先被测试钉住）
# ---------------------------------------------------------------------------


def recall_at_k(hits: Sequence[int], gold: Iterable[int], k: int) -> float:
    """前 K 条里有没有任何一条金标块。单条用例的取值是 0 或 1。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if gold_set & set(list(hits)[:k]) else 0.0


def mrr(hits: Sequence[int], gold: Iterable[int]) -> float:
    """第一个相关结果的名次倒数；没有相关结果记 0。"""
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, chunk_id in enumerate(hits, start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


def _metrics(rows: Sequence[dict]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            **{f"recall@{k}": None for k in RECALL_KS},
            "mrr@10": None,
        }
    return {
        "n": len(rows),
        **{
            f"recall@{k}": sum(recall_at_k(row["hits"], row["gold"], k) for row in rows)
            / len(rows)
            for k in RECALL_KS
        },
        "mrr@10": sum(mrr(row["hits"], row["gold"]) for row in rows) / len(rows),
    }


def aggregate(rows: Sequence[dict], strategies: Sequence[str]) -> dict[str, dict]:
    """按策略 × 桶聚合检索指标。拒答题不参与（见模块 docstring）。"""
    report: dict[str, dict] = {}
    for strategy in strategies:
        subset = [
            row
            for row in rows
            if row["strategy"] == strategy and row["behavior"] == "answer"
        ]
        report[strategy] = {
            bucket: _metrics(
                subset if bucket == "overall" else [r for r in subset if r["bucket"] == bucket]
            )
            for bucket in BUCKET_KEYS
        }
    return report


def faithfulness(verdicts: Sequence[dict]) -> float | None:
    """claim 级忠实度：被证据支持的句子占比。没有任何断言时返回 ``None``。"""
    if not verdicts:
        return None
    return sum(1 for verdict in verdicts if verdict.get("supported")) / len(verdicts)


def refusal_accuracy(rows: Sequence[dict]) -> float | None:
    """拒答题里「确实拒答了」的比例。没有拒答题时返回 ``None``。"""
    refuse_rows = [row for row in rows if row.get("behavior") == "refuse"]
    if not refuse_rows:
        return None
    return sum(1 for row in refuse_rows if row.get("refused")) / len(refuse_rows)


def citation_validity(parses: Sequence[dict]) -> float | None:
    """能映射到证据的角标占比。一个角标都没出现时返回 ``None``。"""
    valid = sum(len(item.get("valid") or ()) for item in parses)
    dropped = sum(len(item.get("dropped") or ()) for item in parses)
    if valid + dropped == 0:
        return None
    return valid / (valid + dropped)


# ---------------------------------------------------------------------------
# 真机执行
# ---------------------------------------------------------------------------


class _CachingEmbedder:
    """按文本缓存嵌入 —— 同一道题在 3 条策略里会各嵌入一次，没必要真的算 3 遍。"""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._cache]
        if missing:
            vectors = await self._inner.embed(missing)
            for text, vector in zip(missing, vectors):
                self._cache[text] = list(vector)
        return [self._cache[text] for text in texts]

    async def warmup(self) -> None:
        warmup = getattr(self._inner, "warmup", None)
        if warmup is not None:
            await warmup()

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


class _StrategyRunner:
    """评估专用的「换策略跑一次」适配器。

    为什么不在 ``HybridRetriever`` 上开一个 ``forced_strategy`` 字段：线上只走
    ``hybrid_rerank``，为评估加一个会被线上误用的开关不划算。适配器留在这里，
    线上代码看不到它。
    """

    def __init__(self, understander, retriever) -> None:
        self._understander = understander
        self._retriever = retriever

    async def run(self, strategy: str, query: str, category: str | None):
        understood = await self._understander.understand(query)
        return await self._retriever.retrieve(strategy, understood, category=category)


async def _retrieval_rows(
    *,
    cases: Sequence[dict],
    gold: dict[str, set[int]],
    runner_factory,
    strategies: Sequence[str],
    verbose: bool,
) -> list[dict]:
    rows: list[dict] = []
    for strategy in strategies:
        for case in cases:
            async with runner_factory() as runner:
                result = await runner.run(strategy, case["query"], case.get("category"))
            hits = [chunk.chunk_id for chunk in result.chunks]
            rows.append(
                {
                    "case_id": case["id"],
                    "bucket": case["bucket"],
                    "behavior": case["behavior"],
                    "strategy": strategy,
                    "hits": hits,
                    "gold": sorted(gold.get(case["id"], set())),
                    "rerank_degraded": result.rerank_degraded,
                }
            )
            if verbose:
                print(
                    f"[{strategy}] {case['id']} {case['query']}\n"
                    f"    hits={hits} gold={sorted(gold.get(case['id'], set()))}"
                )
    return rows


#: 评估时的生成提示词。共享线上引用规则，但不是同一条链路（见模块 docstring）。
_ANSWER_SYSTEM = """你是电商平台的售后客服。只根据给定的编号证据回答用户问题。

规则：
""" + "\n".join(f"- {rule}" for rule in CITATION_RULES) + """

证据不够回答就只回复：无法回答。不要编造、不要使用证据之外的知识。"""

_CLAIM_SYSTEM = """你是忠实度裁判。把答案拆成若干条断言，逐条判断能否被给出的编号证据直接支持。

规则：
- 一句话里出现 [n] 且该条证据确实支撑这句话 → supported = true，evidence_n 填那个 n。
- 没有角标的句子一律 supported = false（证据无法追溯）。
- 只输出 JSON：{"claims": [{"text": "...", "supported": true, "evidence_n": 1}]}"""


def _evidence_block(evidence: Sequence) -> str:
    """首尾编排后渲染证据 —— 与线上塞进 prompt 的顺序一致。"""
    return "\n\n".join(
        f"[{item.n}] {item.section_path}\n{item.quote}" for item in prompt_order(evidence)
    )


async def _generate(
    model,
    question: str,
    evidence: Sequence,
) -> tuple[str, dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_ANSWER_SYSTEM),
        HumanMessage(content=f"用户问题：{question}\n\n证据：\n{_evidence_block(evidence)}"),
    ]
    response = await model.ainvoke(messages)
    answer = getattr(response, "content", "") or ""
    return str(answer), {"evidence": evidence_payload(evidence)}


async def _judge_claims(model, answer: str, evidence: Sequence) -> list[dict]:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_CLAIM_SYSTEM),
        HumanMessage(content=f"答案：\n{answer}\n\n证据：\n{_evidence_block(evidence)}"),
    ]
    response = await model.ainvoke(messages)
    raw = getattr(response, "content", "") or ""
    text = str(raw).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        # 裁判输出坏了：整段算「无法判定」，按 unsupported 处理并在报告里可见
        return [{"text": answer[:200], "supported": False, "evidence_n": None, "parse_error": True}]
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(claims, list):
        return [{"text": answer[:200], "supported": False, "evidence_n": None, "parse_error": True}]
    return [claim for claim in claims if isinstance(claim, dict)]


async def _generation_rows(
    *,
    cases: Sequence[dict],
    strategies: Sequence[str],
    pipeline_factory,
    model,
    faith_repo,
    judge_model_name: str,
    limit: int | None,
) -> list[dict]:
    """生成段。``limit`` 是**每桶**取多少条（不是全局前 N）——

    全局截断只会覆盖到前两个桶（用例就是按桶排的），分桶报告里另外两个桶永远是「—」。
    按桶取样才能让四个桶都有数字，成本也可控。
    """
    selected: list[dict] = []
    if limit is None:
        selected = list(cases)
    else:
        per_bucket: dict[str, list[dict]] = defaultdict(list)
        for case in cases:
            per_bucket[case["bucket"]].append(case)
        for bucket in BUCKETS:
            group = per_bucket[bucket]
            answer_cases = [case for case in group if case["behavior"] == "answer"]
            refuse_cases = [case for case in group if case["behavior"] == "refuse"]
            # 每桶取 ``limit-1`` 条可答题 + **1 条拒答题**：
            # 拒答题排在每桶末尾，纯按顺序截会一条都取不到，那样「拒答准确率」整列是 —
            take_answer = max(limit - 1, 0)
            selected.extend(answer_cases[:take_answer])
            selected.extend(refuse_cases[:1])

    rows: list[dict] = []
    for strategy in strategies:
        for case in selected:
            async with pipeline_factory(strategy) as pipeline:
                outcome = await pipeline.answer_context(case["query"])
            if outcome.refused:
                answer = REFUSAL_TEXT
                rows.append(
                    {
                        "case_id": case["id"],
                        "bucket": case["bucket"],
                        "behavior": case["behavior"],
                        "strategy": strategy,
                        "refused": True,
                        "refusal_source": outcome.refusal_source,
                        "refusal_reason": outcome.refusal_reason,
                        "answer": answer,
                        "claims": [],
                        "faithfulness": None,
                        "citations": {"valid": (), "dropped": ()},
                        "rerank_degraded": outcome.rerank_degraded,
                    }
                )
                continue

            answer, _ = await _generate(model, case["query"], outcome.evidence)
            claims = await _judge_claims(model, answer, outcome.evidence)
            parse = parse_citations(answer, outcome.evidence)
            score = faithfulness(claims)
            rows.append(
                {
                    "case_id": case["id"],
                    "bucket": case["bucket"],
                    "behavior": case["behavior"],
                    "strategy": strategy,
                    "refused": False,
                    "refusal_source": None,
                    "refusal_reason": None,
                    "answer": answer,
                    "claims": claims,
                    "faithfulness": score,
                    "citations": {"valid": parse.valid, "dropped": parse.dropped},
                    "rerank_degraded": outcome.rerank_degraded,
                }
            )

            # 编造个案入台账：只有「应该答」的题才算编造；拒答题答了才是另一种问题。
            unsupported = [claim for claim in claims if not claim.get("supported")]
            if unsupported and faith_repo is not None:
                await faith_repo.upsert_from_judgement(
                    eval_id=case["id"],
                    bucket=case["bucket"],
                    query=case["query"],
                    strategy=strategy,
                    answer=answer,
                    reason="；".join(
                        str(claim.get("text", ""))[:80] for claim in unsupported[:3]
                    ),
                    citations=evidence_payload(outcome.evidence),
                    judge_model=judge_model_name,
                )
    return rows


def _generation_report(rows: Sequence[dict], strategies: Sequence[str]) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for strategy in strategies:
        subset = [row for row in rows if row["strategy"] == strategy]
        report[strategy] = {}
        for bucket in BUCKET_KEYS:
            group = subset if bucket == "overall" else [r for r in subset if r["bucket"] == bucket]
            scored = [row["faithfulness"] for row in group if row["faithfulness"] is not None]
            report[strategy][bucket] = {
                "n": len(group),
                "faithfulness": (sum(scored) / len(scored)) if scored else None,
                "refusal_accuracy": refusal_accuracy(group),
                "citation_validity": citation_validity(
                    [row["citations"] for row in group]
                ),
            }
    return report


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_markdown(report: dict) -> str:
    lines = ["# ch04 四策略评估报告", ""]
    meta = report["meta"]
    lines += [
        f"- 生成时间：{meta['generated_at']}",
        f"- 评估集：{meta['cases']} 条（每桶 12 条：10 可答 + 2 拒答）",
        f"- 检索策略：{', '.join(meta['strategies'])}",
        f"- 生成段策略：{', '.join(meta['generation_strategies']) or '（未运行）'}",
        f"- 重排模型：{meta['rerank_model']}",
        f"- 阈值：RAG_MIN_RERANK_SCORE={meta['min_rerank_score']}",
        "",
        "> 生成段的提示词是**评估时提示词**：它与线上 System Prompt 共享引用规则，"
        "但不是同一条链路（线上由模型自主决定是否调用 `query_faq`）。",
        "",
    ]

    for title, key in (("Recall@5", "recall@5"), ("MRR@10", "mrr@10")):
        lines += [f"## 检索 · {title}（四桶 + 整体）", ""]
        header = "| 策略 | " + " | ".join(BUCKET_KEYS) + " |"
        lines += [header, "|" + "---|" * (len(BUCKET_KEYS) + 1)]
        for strategy, buckets in report["retrieval"].items():
            cells = [_fmt(buckets[bucket][key]) for bucket in BUCKET_KEYS]
            lines.append(f"| {strategy} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## 生成 · Faithfulness / 拒答准确率 / 引用有效率", ""]
    header = "| 策略 | 指标 | " + " | ".join(BUCKET_KEYS) + " |"
    lines += [header, "|" + "---|" * (len(BUCKET_KEYS) + 2)]
    for strategy, buckets in report["generation"].items():
        for label, key in (
            ("Faithfulness", "faithfulness"),
            ("拒答准确率", "refusal_accuracy"),
            ("引用有效率", "citation_validity"),
        ):
            cells = [_fmt(buckets[bucket][key]) for bucket in BUCKET_KEYS]
            lines.append(f"| {strategy} | {label} | " + " | ".join(cells) + " |")
    if not report["generation"]:
        lines.append("| — | 未运行 |" + " | ".join("—" for _ in BUCKET_KEYS) + " |")
    lines.append("")

    if report.get("warnings"):
        lines += ["## 锚串警告（命中 > 3 块，建议加长）", ""]
        for case_id, anchors in report["warnings"].items():
            for anchor, count in anchors.items():
                if count > 3:
                    lines.append(f"- {case_id} · `{anchor}` 命中 {count} 块")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [case for case in cases if case["bucket"] == args.only]
        if not cases:
            print(f"没有 {args.only} 桶的用例", file=sys.stderr)
            return 1

    settings = get_settings()
    settings.require_embedding_key()
    engine = create_engine_from(settings)
    factory = session_factory(engine)
    strategies: list[str] = [item for item in args.strategies.split(",") if item]

    try:
        with factory() as session:
            repositories = Repositories.from_session(session)
            chunk_rows = await repositories.knowledge.list_all()
            warnings: dict = {}
            gold = resolve_gold(cases, chunk_rows, warnings=warnings)
        if args.validate_only:
            print(f"锚串全部解析成功：{len(cases)} 条")
            for case in cases:
                print(f"  {case['id']}: gold={sorted(gold.get(case['id'], set()))}")
            return 0

        embedder = _CachingEmbedder(get_embedder(settings))
        store = MilvusVectorStore(
            settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
        )
        count = await store.count()
        if count == 0:
            print("Milvus 集合为空 —— 先跑 scripts/build_knowledge.py 建库。", file=sys.stderr)
            return 1
        reranker = get_reranker(settings)

        def build_runner():
            from contextlib import asynccontextmanager

            @asynccontextmanager
            async def factory_ctx():
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
                    yield _StrategyRunner(understander, retriever)

            return factory_ctx()

        retrieval_rows = await _retrieval_rows(
            cases=cases,
            gold=gold,
            runner_factory=build_runner,
            strategies=strategies,
            verbose=args.verbose,
        )
        report: dict = {
            "meta": {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "cases": len(cases),
                "strategies": strategies,
                "generation_strategies": [s for s in args.generation.split(",") if s]
                if args.generation
                else [],
                "rerank_model": settings.rerank_model,
                "min_rerank_score": settings.rag_min_rerank_score,
                "milvus_rows": count,
            },
            "retrieval": aggregate(retrieval_rows, strategies),
            "generation": {},
            "warnings": warnings,
        }

        if args.generation:
            generation_strategies = [s for s in args.generation.split(",") if s]
            model = get_chat_model(settings)

            def build_pipeline(strategy: str):
                from contextlib import asynccontextmanager

                @asynccontextmanager
                async def pipeline_ctx():
                    with factory() as session:
                        repositories = Repositories.from_session(session)
                        understander = QueryUnderstander(
                            LlmRewriter(model, timeout=settings.rag_rewrite_timeout),
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
                        pipeline = RagPipeline(
                            understander=understander,
                            retriever=retriever,
                            sufficiency=SufficiencyJudge(model, timeout=settings.rag_rewrite_timeout),
                            min_score=settings.rag_min_rerank_score,
                            top_n=settings.rag_rerank_top_n,
                            # 评估要换策略；线上默认 hybrid_rerank。
                            strategy=strategy,
                        )
                        yield pipeline

                return pipeline_ctx()

            with factory() as session:
                faith_repo = Repositories.from_session(session).faith_cases
                generation_rows = await _generation_rows(
                    cases=cases,
                    strategies=generation_strategies,
                    pipeline_factory=build_pipeline,
                    model=model,
                    faith_repo=faith_repo,
                    judge_model_name=settings.openai_model,
                    limit=args.limit,
                )
            report["generation"] = _generation_report(generation_rows, generation_strategies)

        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print(f"报告已写入 {args.json}")
        if args.markdown:
            Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
            Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
            print(f"报告已写入 {args.markdown}")

        if args.validate_only:
            return 0

        print(render_markdown(report))
        return _exit_code(report, strategies)
    finally:
        engine.dispose()


def _exit_code(report: dict, strategies: Sequence[str]) -> int:
    """验收判定：hybrid_rerank 整体 Recall@5 ≥ vector，且 B_model ≥ bm25。"""
    retrieval = report["retrieval"]

    def recall(strategy: str, bucket: str) -> float | None:
        return retrieval.get(strategy, {}).get(bucket, {}).get("recall@5")

    if "hybrid_rerank" not in strategies or "vector" not in strategies:
        print("未同时运行 vector 与 hybrid_rerank，跳过验收判定", file=sys.stderr)
        return 0

    overall_hybrid = recall("hybrid_rerank", "overall")
    overall_vector = recall("vector", "overall")
    if overall_hybrid is None or overall_vector is None:
        print("Recall@5 缺失，无法判定", file=sys.stderr)
        return 1

    if overall_hybrid < overall_vector:
        print(
            f"验收未过：hybrid_rerank 整体 Recall@5 {overall_hybrid:.3f} < vector {overall_vector:.3f}",
            file=sys.stderr,
        )
        return 1

    if "bm25" in strategies:
        model_hybrid = recall("hybrid_rerank", "B_model")
        model_bm25 = recall("bm25", "B_model")
        if model_hybrid is not None and model_bm25 is not None and model_hybrid < model_bm25:
            print(
                f"验收未过：B_model 桶 hybrid_rerank {model_hybrid:.3f} < bm25 {model_bm25:.3f}",
                file=sys.stderr,
            )
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ch04 四策略 RAG 评估")
    parser.add_argument("--cases", default=str(CASES_FILE), help="评估集路径")
    parser.add_argument(
        "--strategies", default=",".join(ALL_STRATEGIES), help="检索策略，逗号分隔"
    )
    parser.add_argument(
        "--generation", default="", help="生成段策略，逗号分隔（留空则不跑生成段）"
    )
    parser.add_argument("--only", default=None, choices=[*BUCKETS], help="只跑某个桶")
    parser.add_argument("--limit", type=int, default=None, help="生成段每题最多跑多少条")
    parser.add_argument("--validate-only", action="store_true", help="只校验金标解析")
    parser.add_argument("--json", default=None, help="JSON 报告路径")
    parser.add_argument("--markdown", default=None, help="Markdown 报告路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="逐条打印命中")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except Exception as exc:  # noqa: BLE001 - 脚本要把原因打给人看
        print(f"评估失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
