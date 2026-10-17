"""ch04 评估脚本的判定逻辑测试（离线）。

脚本本体要连真库 / 真 Milvus / 真模型；但**判定规则**是纯函数，必须先钉住 ——
判定写错的后果比脚本报错严重得多：它会给出一个「混合检索更好」的假结论。

三条尤其重要：
1. 拒答题不参与检索指标（没有 gold），必须报 ``None`` 而不是 0；
2. 没跑的策略也必须是 ``None``，不能填 0 冒充；
3. MRR 取**第一个**相关结果的名次。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "evals" / "run_ch04_evals.py"
    spec = importlib.util.spec_from_file_location("run_ch04_evals_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evals_module():
    return _load_module()


# ---------------------------------------------------------------------------
# 金标解析
# ---------------------------------------------------------------------------


def test_resolve_gold_unions_multiple_anchors(evals_module):
    rows = [
        {"id": 1, "questions": "运费怎么算", "answer": "满 99 元包邮"},
        {"id": 2, "questions": "赔付标准", "answer": "单笔最高 500 元"},
        {"id": 3, "questions": "发票", "answer": "3 个工作日"},
    ]
    cases = [{"id": "X1", "gold_anchors": ["99 元包邮", "最高 500 元"]}]

    gold = evals_module.resolve_gold(cases, rows)
    assert gold["X1"] == {1, 2}


def test_resolve_gold_raises_on_zero_hit(evals_module):
    cases = [{"id": "X1", "gold_anchors": ["根本不存在的串"]}]
    with pytest.raises(ValueError, match="锚串"):
        evals_module.resolve_gold(cases, [{"id": 1, "questions": "q", "answer": "a"}])


def test_resolve_gold_reports_ambiguous_anchors(evals_module):
    rows = [
        {"id": 1, "questions": "q1", "answer": "型号 XH-3000A"},
        {"id": 2, "questions": "q2", "answer": "也是 XH-3000A"},
        {"id": 3, "questions": "q3", "answer": "还是 XH-3000A"},
        {"id": 4, "questions": "q4", "answer": "依然 XH-3000A"},
    ]
    cases = [{"id": "X1", "gold_anchors": ["XH-3000A"]}]

    report: dict = {}
    gold = evals_module.resolve_gold(cases, rows, warnings=report)
    assert gold["X1"] == {1, 2, 3, 4}
    assert report["X1"]["XH-3000A"] == 4


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def test_recall_at_k_is_one_when_gold_is_in_top_k(evals_module):
    assert evals_module.recall_at_k([5, 2, 9], {9}, 3) == 1.0
    assert evals_module.recall_at_k([5, 2, 9], {9}, 2) == 0.0


def test_mrr_uses_the_first_relevant_rank(evals_module):
    assert evals_module.mrr([5, 2, 9], {9}) == pytest.approx(1 / 3)
    assert evals_module.mrr([5, 2], {9}) == 0.0
    assert evals_module.mrr([5, 2], set()) == 0.0


def test_aggregate_skips_refuse_cases(evals_module):
    rows = [
        {"case_id": "A11", "bucket": "A_policy", "behavior": "refuse",
         "strategy": "vector", "hits": [], "gold": set()},
    ]
    report = evals_module.aggregate(rows, ["vector"])
    assert report["vector"]["A_policy"]["recall@5"] is None
    assert report["vector"]["A_policy"]["n"] == 0


def test_aggregate_reports_none_for_unrun_strategies(evals_module):
    rows = [
        {"case_id": "A01", "bucket": "A_policy", "behavior": "answer",
         "strategy": "vector", "hits": [1], "gold": {1}},
    ]
    report = evals_module.aggregate(rows, ["vector", "hybrid_rerank"])
    assert report["hybrid_rerank"]["overall"]["recall@5"] is None
    assert report["vector"]["overall"]["recall@5"] == 1.0


def test_aggregate_groups_by_bucket(evals_module):
    rows = [
        {"case_id": "A01", "bucket": "A_policy", "behavior": "answer",
         "strategy": "vector", "hits": [1], "gold": {1}},
        {"case_id": "B01", "bucket": "B_model", "behavior": "answer",
         "strategy": "vector", "hits": [2], "gold": {9}},
    ]
    report = evals_module.aggregate(rows, ["vector"])
    assert report["vector"]["A_policy"]["recall@5"] == 1.0
    assert report["vector"]["B_model"]["recall@5"] == 0.0
    assert report["vector"]["overall"]["n"] == 2


# ---------------------------------------------------------------------------
# 生成段指标
# ---------------------------------------------------------------------------


def test_faithfulness_counts_unsupported_claims(evals_module):
    verdicts = [
        {"text": "满 99 包邮", "supported": True},
        {"text": "破损赔 500", "supported": False},
    ]
    assert evals_module.faithfulness(verdicts) == 0.5


def test_faithfulness_without_claims_is_none(evals_module):
    assert evals_module.faithfulness([]) is None


def test_refusal_accuracy_only_counts_refuse_cases(evals_module):
    rows = [
        {"behavior": "refuse", "refused": True},
        {"behavior": "refuse", "refused": False},
        {"behavior": "answer", "refused": False},
    ]
    assert evals_module.refusal_accuracy(rows) == 0.5


def test_refusal_accuracy_without_refuse_cases_is_none(evals_module):
    assert evals_module.refusal_accuracy([{"behavior": "answer", "refused": False}]) is None


def test_citation_validity_is_none_when_no_citations(evals_module):
    assert evals_module.citation_validity([]) is None


def test_citation_validity_counts_valid_and_dropped(evals_module):
    parses = [
        {"valid": (1,), "dropped": (9,)},   # 1/2
        {"valid": (1, 2), "dropped": ()},    # 2/2
    ]
    assert evals_module.citation_validity(parses) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 退出码（验收判定）
# ---------------------------------------------------------------------------


def _report(strategy_rows: dict[str, dict]) -> dict:
    return {"retrieval": strategy_rows}


def test_exit_code_fails_when_bm25_model_recall_is_below_the_bar(evals_module):
    """`--only B_model --strategies bm25` 这条命令必须能红。

    初版只在「同时跑了 vector 与 hybrid_rerank」时才判定，于是验收 2 的这条命令
    永远返回 0 —— 退出码不承载信息，等于没验收。
    """
    report = _report(
        {"bm25": {"B_model": {"recall@5": 0.5}, "overall": {"recall@5": 0.5}}}
    )
    assert evals_module._exit_code(report, ["bm25"]) == 1


def test_exit_code_passes_when_bm25_model_recall_meets_the_bar(evals_module):
    report = _report(
        {"bm25": {"B_model": {"recall@5": 0.9}, "overall": {"recall@5": 0.9}}}
    )
    assert evals_module._exit_code(report, ["bm25"]) == 0


def test_exit_code_fails_when_hybrid_rerank_loses_to_vector(evals_module):
    report = _report(
        {
            "vector": {"overall": {"recall@5": 1.0}, "B_model": {"recall@5": 1.0}},
            "bm25": {"overall": {"recall@5": 1.0}, "B_model": {"recall@5": 1.0}},
            "hybrid_rerank": {"overall": {"recall@5": 0.9}, "B_model": {"recall@5": 1.0}},
        }
    )
    assert evals_module._exit_code(report, ["vector", "bm25", "hybrid_rerank"]) == 1
