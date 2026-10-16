"""评估集脚本自身的测试（**不联网、不打模型**）。

评估集跑真实上游，那条路径天然慢且不稳定，不适合放进单元测试。
但**脚本自己**的逻辑必须测 —— 否则会出现最难堪的情况：
评估脚本的判定写错了，于是它报「全部通过」而实际有问题。

这里钉住三件事：
1. 用例文件能被解析，且每条都有必需字段；
2. 判定逻辑对「该调 / 不该调 / 调用次数不足 / 执行出错」四种情况的结论正确；
3. 验收标准 3 的用例**不**断言用户可见结果（那是实测会随模型改写而变的东西，
   把它写成断言等于把「模型当时怎么想的」当契约）。
"""

import json
import sys
from pathlib import Path

import pytest

# evals/ 不是包，按脚本自己的做法把仓库代码目录加进 sys.path
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evals.run_tool_evals import CASES_FILE, CaseResult, load_cases  # noqa: E402


def _case(**overrides) -> CaseResult:
    base = {
        "case_id": "x",
        "message": "m",
        "expected": ["query_faq"],
        "actual": ["query_faq"],
        "min_tool_calls": 1,
        "answer": "a",
    }
    base.update(overrides)
    return CaseResult(**base)


# ---------------------------------------------------------------------------
# 用例文件本身
# ---------------------------------------------------------------------------


def test_cases_file_parses_and_is_not_empty():
    cases = load_cases(CASES_FILE)
    assert cases, "评估集不能是空的"


def test_every_case_has_the_required_fields():
    for case in load_cases(CASES_FILE):
        assert case.get("id"), f"用例缺 id：{case}"
        assert case.get("message"), f"用例 {case.get('id')} 缺 message"
        assert "expect_tools" in case, f"用例 {case['id']} 缺 expect_tools"
        assert "expect_min_tool_calls" in case, f"用例 {case['id']} 缺 expect_min_tool_calls"


def test_case_ids_are_unique():
    """id 重复会让 ``--only`` 选到不止一条，也会让报告里两行同名难分辨。"""
    ids = [case["id"] for case in load_cases(CASES_FILE)]
    assert len(ids) == len(set(ids)), f"有重复 id：{ids}"


def test_the_three_acceptance_criteria_have_cases():
    """验收标准的用例必须真的在评估集里 —— 否则改了评估集也没人发现验收没了。"""
    ids = {case["id"] for case in load_cases(CASES_FILE)}
    for required in ("acc1", "acc2", "acc3"):
        assert required in ids, f"评估集缺少验收标准用例 {required}"


def test_negative_cases_expect_no_tools():
    """负例（寒暄/闲聊）必须 ``expect_tools`` 为空且不要求调用。

    这类用例的价值在于「不许多调」：模型对一句「你好」去查订单，
    用户会觉得莫名其妙，而且白花一次工具调用。
    """
    negatives = [c for c in load_cases(CASES_FILE) if c["id"].startswith("neg")]
    assert negatives, "评估集应有负例"
    for case in negatives:
        assert case["expect_tools"] == [], f"{case['id']} 的负例不该允许任何工具"
        assert case["expect_min_tool_calls"] == 0, f"{case['id']} 的负例不该要求调用"


def test_acceptance_criterion_3_case_does_not_assert_a_user_visible_outcome():
    """验收 3 的用例只断言**工具选择**，不断言用户看到「查不到」。

    实测：模型会把「邮费」归一化成「运费」从而命中语料并答对，
    所以「用户会看到查不出来」这个前提**不成立**（用户已确认留待后续章节修）。
    把那个预期写成断言，等于把「模型当时怎么改写」当成契约 ——
    模型换个说法评估就会无故变红，而真实的基线（数据层 count=0）
    由 ``tests/test_db_mysql.py`` 硬钉住。
    """
    case = next(c for c in load_cases(CASES_FILE) if c["id"] == "acc3")
    assert case["expect_tools"] == ["query_faq"]
    assert case["expect_min_tool_calls"] == 1
    # 用例里没有任何字段要求「查不到」这个结果
    assert "expect_max_faq_hits" not in case
    assert "query_faq" in case["note"]


# ---------------------------------------------------------------------------
# 判定逻辑
# ---------------------------------------------------------------------------


def test_pass_when_actual_is_within_the_allowed_set():
    assert _case(actual=["query_faq"]).passed


def test_pass_when_only_some_allowed_tools_were_called():
    """允许集合是「上限」，不要求全调 —— 调一个对的就算过。"""
    assert _case(expected=["query_logistics", "query_order"], actual=["query_logistics"]).passed


def test_fail_when_an_unexpected_tool_was_called():
    result = _case(expected=["query_faq"], actual=["query_order"])
    assert not result.passed
    assert "预期外" in result.why()


def test_fail_when_a_negative_case_called_a_tool():
    result = _case(expected=[], actual=["query_faq"], min_tool_calls=0)
    assert not result.passed


def test_fail_when_too_few_tool_calls():
    result = _case(expected=["query_faq"], actual=[], min_tool_calls=1)
    assert not result.passed
    assert "至少调" in result.why()


def test_pass_when_min_calls_is_zero_and_nothing_was_called():
    """模糊例：不强制调用，先问用户是允许的。"""
    result = _case(expected=["query_faq"], actual=[], min_tool_calls=0)
    assert result.passed


def test_fail_when_the_case_itself_errored():
    result = _case(error="RuntimeError: boom")
    assert not result.passed
    assert "执行出错" in result.why()


def test_zero_case_file_is_reported_not_silently_passed():
    """空用例文件的退出码行为由脚本负责，这里只钉住「加载器不吞掉空文件」。"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write("\n# 只有注释\n")
        path = Path(fh.name)
    try:
        assert load_cases(path) == []
    finally:
        path.unlink(missing_ok=True)


def test_malformed_json_line_raises_a_clear_error():
    """坏行必须报「第几行」，否则十几条用例里找一个错的会很痛苦。"""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as fh:
        fh.write('{"id": "ok", "message": "m", "expect_tools": [], "expect_min_tool_calls": 0}\n')
        fh.write("{not json}\n")
        path = Path(fh.name)
    try:
        with pytest.raises(SystemExit) as excinfo:
            load_cases(path)
        assert "第 2 行" in str(excinfo.value)
    finally:
        path.unlink(missing_ok=True)


def test_report_json_shape_is_stable():
    """``--json`` 的输出被留痕与对比使用，字段名不该随手改。"""
    result = _case()
    payload = json.loads(
        json.dumps(
            {
                "id": result.case_id,
                "expected": result.expected,
                "actual": result.actual,
                "passed": result.passed,
                "why": result.why(),
            },
            ensure_ascii=False,
        )
    )
    assert set(payload) == {"id", "expected", "actual", "passed", "why"}
