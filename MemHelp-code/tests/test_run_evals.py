"""评估脚本的评分逻辑测试（离线，不联网）。

`evals/run_evals.py` 的评分代码要等有真实密钥才第一次真跑 —— 万一它有 bug，
就会白跑一轮评估。所以这里把评分逻辑本身当被测对象钉住。

重点验证 **「零编造」是硬性失败**：即使总体准确率很高，只要在「明确没提订单号」
的样例上抽出了订单号，就必须判不达标。这一条如果失效，整个评估就失去了意义。
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_run_evals():
    """按路径加载 evals/run_evals.py（它不在 app 包里）。"""
    path = ROOT / "evals" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def run_evals():
    return _load_run_evals()


@pytest.fixture(scope="module")
def cases(run_evals):
    return run_evals.load_cases(ROOT / "evals" / "aftersales_cases.jsonl")


def _perfect(cases) -> dict:
    """造一份「完全按标注」的模型产出。"""
    return {
        c["id"]: {
            "order_no": c["expected"]["order_no"],
            "request_type": c["expected"]["request_type"],
            "expected_solution": c["expected"]["expected_solution"],
            "detail": " ".join(c.get("detail_keywords", [])[:1]) or "摘要",
        }
        for c in cases
    }


def test_cases_file_is_wellformed(cases):
    assert len(cases) == 12
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert set(c["expected"]) >= {"order_no", "request_type", "expected_solution"}
        assert c["text"].strip()
        assert c["detail_keywords"], f"{c['id']} 缺 detail_keywords"


def test_cases_include_null_order_no_samples(cases):
    """必须有「明确没提订单号」的样例，否则测不出编造。"""
    null_cases = [c["id"] for c in cases if c["expected"]["order_no"] is None]
    assert len(null_cases) >= 3, f"order_no 期望为 null 的样例太少：{null_cases}"


def test_perfect_output_scores_full_marks(run_evals, cases):
    sc = run_evals.score(cases, _perfect(cases), {})
    assert sc.field_accuracy == 1.0
    assert not sc.fabricated
    ok, _ = sc.verdict(0.9)
    assert ok


def test_fabricated_order_no_is_a_hard_failure_even_above_threshold(run_evals, cases):
    """核心断言：准确率仍然很高（88.9% 只差一点）也绝不能判达标。"""
    null_cases = [c["id"] for c in cases if c["expected"]["order_no"] is None]
    results = _perfect(cases)
    for cid in null_cases:
        results[cid]["order_no"] = "99999999"

    sc = run_evals.score(cases, results, {})
    assert len(sc.fabricated) == len(null_cases)
    # 故意把阈值放到很低，证明「编造」这条是独立于阈值的硬门禁
    ok, reason = sc.verdict(threshold=0.1)
    assert not ok, "存在编造时，无论阈值多低都不能判达标"
    assert "编造" in reason


def test_missing_value_when_expected_is_present_fails(run_evals, cases):
    """期望有值却抽成 None，也要算失败（反向的宽松是不允许的）。"""
    results = _perfect(cases)
    assert run_evals.field_matches(None, "20240115001") is False


def test_null_matches_only_null(run_evals):
    assert run_evals.field_matches(None, None) is True
    assert run_evals.field_matches("", None) is False
    assert run_evals.field_matches("x", None) is False


def test_string_comparison_ignores_surrounding_whitespace(run_evals):
    assert run_evals.field_matches("  20240115001 ", "20240115001") is True


def test_extraction_error_is_recorded_as_failure(run_evals, cases):
    results = _perfect(cases)
    failed_id = cases[0]["id"]
    results.pop(failed_id)
    sc = run_evals.score(cases, results, {failed_id: "模型不按 schema 输出"})
    assert not sc.outcomes[0].passed
    assert any("提取失败" in line for line in sc.failures)


def test_detail_keyword_miss_is_recorded(run_evals, cases):
    results = _perfect(cases)
    cid = cases[0]["id"]
    results[cid]["detail"] = "完全无关的摘要内容"
    sc = run_evals.score(cases, results, {})
    assert not sc.outcomes[0].detail_ok
    assert any("detail" in line for line in sc.failures)


def test_report_contains_key_conclusions(run_evals, cases):
    sc = run_evals.score(cases, _perfect(cases), {})
    report = run_evals.format_report(sc, 0.9, "https://example.invalid/v1", "fake-model")
    assert "总体字段准确率：100.0%" in report
    assert "订单号编造次数：0" in report
    assert "结论：达标" in report


def test_report_flags_fabrication_with_pointer_to_prompt(run_evals, cases):
    null_cases = [c["id"] for c in cases if c["expected"]["order_no"] is None]
    results = _perfect(cases)
    for cid in null_cases:
        results[cid]["order_no"] = "12345"
    sc = run_evals.score(cases, results, {})
    report = run_evals.format_report(sc, 0.9, "https://example.invalid/v1", "fake-model")
    assert "编造订单号" in report
    assert "EXTRACT_PROMPT" in report, "未达标时要给出「先改 Prompt」的可操作指引"
    assert "不要改评估标准" in report


def test_self_test_runs_clean(run_evals):
    """`--self-test` 本身必须通过（它就是那种「等有密钥才跑」的代码）。"""
    assert run_evals.self_test() == 0


@pytest.mark.parametrize(
    "message",
    [
        "模型输出无法解析为结构化结果：Error code: 400 - The supported API model names are deepseek-flash",
        "模型输出无法解析为结构化结果：Error code: 401 - Incorrect API key provided",
        "模型输出无法解析为结构化结果：APIConnectionError: Connection error.",
        "模型输出无法解析为结构化结果：APITimeoutError: Request timed out",
    ],
    ids=["bad-model-name", "auth", "connection", "timeout"],
)
def test_environment_problems_are_detected(run_evals, message):
    """上游/配置故障必须能被识别，否则会误导使用者去重写 Prompt。"""
    assert run_evals.looks_like_environment_problem({"c01": message}) is True


def test_genuine_parse_failure_is_not_flagged_as_environment_problem(run_evals):
    """模型确实按不了 schema 输出时，不该被当成环境问题甩锅。"""
    assert (
        run_evals.looks_like_environment_problem({"c01": "模型未按 schema 返回：None"})
        is False
    )
    assert run_evals.looks_like_environment_problem({}) is False
