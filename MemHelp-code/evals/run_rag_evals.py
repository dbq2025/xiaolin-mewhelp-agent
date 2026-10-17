"""语义检索评估集：关键词基线 vs 向量检索，同一语料、同一判定。

## 为什么要做「同一语料」的对照

只证明「向量能召回」不算数 —— 那可能只是因为语料里恰好有那些字。
必须同时给出**同一批知识块**上关键词检索的命中情况，才能说明升级真的解决了漏召回
（ch02 的验收 3 正是「关键词查不到『邮费』」）。

## 三种基线（不能只报一个数）

- **kw-raw**：``knowledge_chunks`` 上对**用户整句**做 ``LIKE``。
- **kw-noun**：对 ch02 时代模型会截出的名词（``keyword`` 字段）做 ``LIKE``。
  ch02 的 ``query_faq`` 收到的其实是这个，所以它更接近历史行为。
- **vector**：走 ``KnowledgeRetriever``（BGE-M3 + Milvus）。

为什么要分两列：ch02 的 README 记录过「模型会把『邮费』归一化成『运费』再传参」，
所以 kw-noun 有可能本来就命中。**只报 kw-raw 会夸大向量的优势**，只报 kw-noun
又会掩盖「关键词检索依赖模型先猜对词」这件事。两列并排，读者自己判断。

## 用法

```powershell
uv run --directory MemHelp-code python evals/run_rag_evals.py
uv run --directory MemHelp-code python evals/run_rag_evals.py --baseline keyword
uv run --directory MemHelp-code python evals/run_rag_evals.py --json dev-notes/rag-evals.json
```

退出码：0 = 验收用例（``paraphrase-fee``）的向量检索命中；1 = 没命中或脚本出错。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

# 让脚本能 import app.*（它从 evals/ 下运行，不在 app 包内）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.db.session import create_engine_from, session_factory  # noqa: E402
from app.knowledge.embedder import get_embedder  # noqa: E402
from app.knowledge.retrieval import KnowledgeRetriever  # noqa: E402
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "rag_cases.jsonl"

#: 验收标准 1 的落点：这条用例的**向量**检索必须命中。
REQUIRED_VECTOR_CASE = "paraphrase-fee"

#: 关键词基线与向量基线都用同一个 Top-K，避免「多给几条就算赢」。
TOP_K = 5


@dataclass
class CaseResult:
    case_id: str
    query: str
    paraphrase: bool
    keyword_raw_hit: bool | None = None
    keyword_noun_hit: bool | None = None
    vector_hit: bool | None = None
    keyword_raw_hits: list[str] = field(default_factory=list)
    keyword_noun_hits: list[str] = field(default_factory=list)
    vector_hits: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 载入与判定
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:  # pragma: no cover - 数据错误直接报出来
            raise ValueError(f"{path.name}:{lineno} 不是合法 JSON：{exc}") from exc
    return cases


def _contains(answer: str, term: str) -> bool:
    """纯数字的判定词要求「两边不是数字」，其余按子串。

    为什么不直接 ``in``：``"99" in "满 199 元包邮"`` 为真，于是「99 元包邮」这条
    判定会被「199 元包邮」的答案满足 —— 那是一次**假通过**，比失败更糟。
    """
    if term.isdigit():
        return re.search(rf"(?<!\d){re.escape(term)}(?!\d)", answer) is not None
    return term in answer


def hit(case: dict, hits: Iterable[Any]) -> bool:
    """某**一条**命中的答案同时满足 ``expect_any`` 与 ``must_contain`` 才算命中。

    不能跨条目拼条件：一条命中「99」、另一条命中「包邮」，不等于系统答对了。
    """
    expect_any = list(case.get("expect_any") or [])
    must_contain = list(case.get("must_contain") or [])
    for item in hits:
        answer = getattr(item, "answer", "")
        if expect_any and not any(_contains(answer, term) for term in expect_any):
            continue
        if all(_contains(answer, term) for term in must_contain):
            return True
    return False


# ---------------------------------------------------------------------------
# 两种基线
# ---------------------------------------------------------------------------


def _keyword_hits(session, term: str, limit: int = TOP_K) -> list[SimpleNamespace]:
    pattern = f"%{term}%"
    rows = session.execute(
        text(
            "SELECT category, questions, answer FROM knowledge_chunks "
            "WHERE questions LIKE :p OR answer LIKE :p OR category LIKE :p "
            "ORDER BY id LIMIT :n"
        ),
        {"p": pattern, "n": limit},
    ).all()
    return [
        SimpleNamespace(category=category, question=questions, answer=answer)
        for category, questions, answer in rows
    ]


def run_baseline_keyword(session, cases: Sequence[dict]) -> list[CaseResult]:
    """关键词基线：整句与「截出的名词」各查一次，**分别记录**（不取并集）。"""
    results: list[CaseResult] = []
    for case in cases:
        raw_hits = _keyword_hits(session, case["query"])
        noun_hits = _keyword_hits(session, case["keyword"])
        results.append(
            CaseResult(
                case_id=case["id"],
                query=case["query"],
                paraphrase=bool(case.get("paraphrase")),
                keyword_raw_hit=hit(case, raw_hits),
                keyword_noun_hit=hit(case, noun_hits),
                keyword_raw_hits=[item.answer[:40] for item in raw_hits],
                keyword_noun_hits=[item.answer[:40] for item in noun_hits],
            )
        )
    return results


async def run_baseline_vector(
    retriever: KnowledgeRetriever, cases: Sequence[dict]
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        hits = await retriever.search(case["query"], top_k=TOP_K)
        results.append(
            CaseResult(
                case_id=case["id"],
                query=case["query"],
                paraphrase=bool(case.get("paraphrase")),
                vector_hit=hit(case, hits),
                vector_hits=[item.answer[:40] for item in hits],
            )
        )
    return results


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def _mark(value: bool | None) -> str:
    if value is None:
        return "  -  "
    return "  ✓  " if value else "  ✗  "


def render(results: Sequence[CaseResult]) -> str:
    lines = [f"{'case':32s} {'换说法':6s} {'kw-raw':8s} {'kw-noun':8s} {'vector':8s} query"]
    for result in results:
        lines.append(
            f"{result.case_id:32s} "
            f"{'是' if result.paraphrase else '否':6s} "
            f"{_mark(result.keyword_raw_hit)}"
            f"{_mark(result.keyword_noun_hit)}"
            f"{_mark(result.vector_hit)}"
            f"{result.query}"
        )

    def _rate(flag: str, subset: Sequence[CaseResult]) -> str:
        rows = [getattr(result, flag) for result in subset]
        scored = [value for value in rows if value is not None]
        if not scored:
            return "未运行"
        return f"{sum(scored)}/{len(scored)}"

    paraphrases = [result for result in results if result.paraphrase]
    for label, subset in (("全部用例", results), ("换说法用例", paraphrases)):
        lines.append(
            f"{label}：kw-raw {_rate('keyword_raw_hit', subset)} | "
            f"kw-noun {_rate('keyword_noun_hit', subset)} | "
            f"vector {_rate('vector_hit', subset)}"
        )
    return "\n".join(lines)


def _summarise_json(results: Sequence[CaseResult], milvus_count: int) -> dict:
    return {
        "milvus_rows": milvus_count,
        "cases": [
            {
                "id": result.case_id,
                "query": result.query,
                "paraphrase": result.paraphrase,
                "keyword_raw_hit": result.keyword_raw_hit,
                "keyword_noun_hit": result.keyword_noun_hit,
                "vector_hit": result.vector_hit,
                "keyword_raw_hits": result.keyword_raw_hits,
                "keyword_noun_hits": result.keyword_noun_hits,
                "vector_hits": result.vector_hits,
            }
            for result in results
        ],
    }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    settings = get_settings()
    engine = create_engine_from(settings)
    factory = session_factory(engine)

    results_by_id: dict[str, CaseResult] = {}
    milvus_count = 0

    if args.baseline in ("keyword", "both"):
        with factory() as session:
            keyword_results = run_baseline_keyword(session, cases)
        for result in keyword_results:
            results_by_id[result.case_id] = result

    if args.baseline in ("vector", "both"):
        settings.require_embedding_key()
        embedder = get_embedder(settings)
        store = MilvusVectorStore(
            settings.milvus_uri, settings.milvus_collection, settings.embedding_dim
        )
        milvus_count = await store.count()
        if milvus_count == 0:
            print(
                "Milvus 集合为空 —— 先跑 scripts/build_knowledge.py 建库，再评估。",
                file=sys.stderr,
            )
            return 1
        with factory() as session:
            repositories = Repositories.from_session(session)
            retriever = KnowledgeRetriever(embedder, store, repositories.knowledge)
            vector_results = await run_baseline_vector(retriever, cases)
        for result in vector_results:
            existing = results_by_id.get(result.case_id)
            if existing is not None:
                existing.vector_hit = result.vector_hit
                existing.vector_hits = result.vector_hits
            else:
                results_by_id[result.case_id] = result
        await embedder.aclose()

    ordered = [results_by_id[case["id"]] for case in cases if case["id"] in results_by_id]
    print(render(ordered))
    print(f"\nMilvus 行数：{milvus_count}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(_summarise_json(ordered, milvus_count), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"结果已写入 {args.json}")

    engine.dispose()

    required = next((r for r in ordered if r.case_id == REQUIRED_VECTOR_CASE), None)
    if required is None:
        print(f"评估集缺少验收用例 {REQUIRED_VECTOR_CASE}", file=sys.stderr)
        return 1
    if required.vector_hit is None:
        # 只跑关键词基线属于诊断模式：没有向量结果就无从判定验收，但不该算失败。
        print(
            f"未运行向量基线，跳过验收判定（{REQUIRED_VECTOR_CASE}）",
            file=sys.stderr,
        )
        return 0
    if not required.vector_hit:
        print(f"验收用例 {REQUIRED_VECTOR_CASE} 未被向量检索命中", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="语义检索评估：关键词基线 vs 向量检索")
    parser.add_argument("--baseline", choices=["keyword", "vector", "both"], default="both")
    parser.add_argument("--cases", default=str(CASES_FILE), help="评估集路径")
    parser.add_argument("--json", default=None, help="把结果写到 JSON 文件")
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
