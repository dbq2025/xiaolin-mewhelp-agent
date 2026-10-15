"""售后信息提取评估脚本。

这是「TDD 替代方案」的第 2 层：纯 Prompt 与数据结构没法离线单测，所以拿
标注样例跑一遍真实模型，看字段级准确率。

**需要真实 .env**（`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`）。

用法::

    uv run python evals/run_evals.py
    uv run python evals/run_evals.py --threshold 0.95
    uv run python evals/run_evals.py --only c06 c07 c08   # 只跑信息缺失那几条
    uv run python evals/run_evals.py --self-test          # 不联网，验证评分逻辑

退出码：0 达标 / 1 未达标 / 2 配置缺失或输入文件坏。

评分逻辑（``score`` / ``format_report``）是纯函数，由 ``--self-test`` 和
``tests/test_run_evals.py`` 离线覆盖 —— 否则这段代码要等到有真实密钥才第一次
执行，万一有 bug 就白跑一轮。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 让脚本能直接 `python evals/run_evals.py` 跑（把仓库根加进 sys.path）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = ROOT / "evals" / "aftersales_cases.jsonl"

#: 参与字段级准确率统计的字段（detail 是自由文本，单独按关键词判）。
CHECKED_FIELDS = ("order_no", "request_type", "expected_solution")


# ---------------------------------------------------------------------------
# 输入
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"第 {lineno} 行不是合法 JSON：{exc}", file=sys.stderr)
                raise SystemExit(2) from exc
    return cases


# ---------------------------------------------------------------------------
# 评分（纯函数）
# ---------------------------------------------------------------------------


def field_matches(actual: Any, expected: Any) -> bool:
    """严格比对单个字段。

    ``None`` 期望必须**恰好**是 ``None`` —— 这是「零编造订单号」的核心断言，
    抽到任何值都算失败。
    """
    if expected is None:
        return actual is None
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    return actual == expected


def detail_recall(detail: str | None, keywords: list[str]) -> bool:
    """detail 是自由文本，用关键词命中率粗判（至少命中一个）。"""
    if not keywords:
        return True
    if not detail:
        return False
    return any(kw in detail for kw in keywords)


@dataclass
class Outcome:
    """一条样例的评估结果。"""

    case_id: str
    got: dict | None
    """模型抽出的字段；``None`` 表示提取本身失败。"""
    error: str | None = None
    field_ok: dict[str, bool] = field(default_factory=dict)
    detail_ok: bool = False
    fabricated_order_no: Any = None
    """期望 ``None`` 却抽出了值 —— 编造订单号，硬性失败。"""

    @property
    def passed(self) -> bool:
        return self.error is None and all(self.field_ok.values()) and self.detail_ok


@dataclass
class Score:
    """整份评估集的汇总。"""

    outcomes: list[Outcome]
    hits: dict[str, int]
    totals: dict[str, int]
    detail_hits: int
    detail_total: int

    @property
    def field_accuracy(self) -> float:
        hit = sum(self.hits.values())
        total = sum(self.totals.values())
        return hit / total if total else 0.0

    @property
    def fabricated(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.fabricated_order_no is not None]

    @property
    def failures(self) -> list[str]:
        lines: list[str] = []
        for o in self.outcomes:
            if o.error:
                lines.append(f"{o.case_id}: 提取失败 —— {o.error}")
                continue
            for name, ok in o.field_ok.items():
                if not ok:
                    actual = None if o.got is None else o.got.get(name)
                    lines.append(f"{o.case_id}.{name}: 实际 {actual!r}（期望不符）")
            if not o.detail_ok:
                lines.append(
                    f"{o.case_id}.detail: 关键词未命中 —— "
                    f"{None if o.got is None else o.got.get('detail')!r}"
                )
        return lines

    def verdict(self, threshold: float) -> tuple[bool, str]:
        if self.fabricated:
            return False, f"存在 {len(self.fabricated)} 次订单号编造（硬性失败）"
        if self.field_accuracy < threshold:
            return False, f"准确率 {self.field_accuracy:.1%} < 阈值 {threshold:.0%}"
        return True, f"准确率 {self.field_accuracy:.1%} >= {threshold:.0%} 且零编造"


def score(cases: list[dict], results: dict[str, dict | None], errors: dict[str, str]) -> Score:
    """把「模型产出」与「标注期望」比对成 Score。

    ``results``：``case_id -> 抽出的字段 dict``（缺键或值为 ``None`` 都表示没抽出来）。
    ``errors``：``case_id -> 错误说明``（提取抛异常时）。
    """
    outcomes: list[Outcome] = []
    hits = {f: 0 for f in CHECKED_FIELDS}
    totals = {f: 0 for f in CHECKED_FIELDS}
    detail_hits = detail_total = 0

    for case in cases:
        cid = case["id"]
        expected = case["expected"]
        got = results.get(cid)
        error = errors.get(cid)

        outcome = Outcome(case_id=cid, got=got, error=error)

        for name in CHECKED_FIELDS:
            totals[name] += 1
            exp = expected.get(name)
            act = None if got is None else got.get(name)
            ok = got is not None and field_matches(act, exp)
            outcome.field_ok[name] = ok
            if ok:
                hits[name] += 1
            if name == "order_no" and exp is None and act is not None:
                outcome.fabricated_order_no = act

        detail_total += 1
        outcome.detail_ok = got is not None and detail_recall(
            got.get("detail"), case.get("detail_keywords", [])
        )
        if outcome.detail_ok:
            detail_hits += 1

        outcomes.append(outcome)

    return Score(
        outcomes=outcomes,
        hits=hits,
        totals=totals,
        detail_hits=detail_hits,
        detail_total=detail_total,
    )


def format_report(score_result: Score, threshold: float, upstream: str, model_name: str) -> str:
    """把 Score 渲染成给人看的报告。"""
    sc = score_result
    lines: list[str] = []
    lines.append(f"上游：{upstream}")
    lines.append(f"模型：{model_name}")
    lines.append(f"样例：{len(sc.outcomes)} 条    阈值：{threshold:.0%}")
    lines.append("")

    header = f"{'id':<5} {'order_no':<20} {'type':<11} {'solution':<14} {'detail':<7} 结果"
    lines.append(header)
    lines.append("-" * len(header))
    for o in sc.outcomes:
        def cell(name: str) -> str:
            if o.error:
                return "ERR"
            mark = "OK" if o.field_ok.get(name) else "X"
            val = "" if o.got is None else o.got.get(name)
            return mark if o.field_ok.get(name) else f"{mark}({val})"

        lines.append(
            f"{o.case_id:<5} {cell('order_no'):<20} {cell('request_type'):<11} "
            f"{cell('expected_solution'):<14} {'OK' if o.detail_ok else 'X':<7} "
            f"{'PASS' if o.passed else 'FAIL'}"
        )

    lines.append("")
    lines.append("== 字段级准确率 ==")
    for name in CHECKED_FIELDS:
        hit, total = sc.hits[name], sc.totals[name]
        lines.append(f"  {name:<18} {hit}/{total}  {hit / total:.1%}")
    lines.append(
        f"  {'detail(关键词)':<18} {sc.detail_hits}/{sc.detail_total}  "
        f"{sc.detail_hits / sc.detail_total:.1%}"
    )
    lines.append("")
    lines.append(
        f"总体字段准确率：{sc.field_accuracy:.1%}"
        f"（{sum(sc.hits.values())}/{sum(sc.totals.values())}）"
    )
    lines.append(f"订单号编造次数：{len(sc.fabricated)}")

    if sc.fabricated:
        lines.append("")
        lines.append("!! 编造订单号（硬性失败，必须改 Prompt）:")
        for o in sc.fabricated:
            lines.append(f"   {o.case_id} -> {o.fabricated_order_no!r}")

    failures = sc.failures
    if failures:
        lines.append("")
        lines.append(f"== 失败明细（{len(failures)} 条）==")
        for item in failures:
            lines.append(f"  - {item}")

    ok, reason = sc.verdict(threshold)
    lines.append("")
    lines.append(f"结论：{'达标' if ok else '未达标'}（{reason}）")
    if not ok:
        lines.append("提示：先改 Prompt（app/prompts.py 的 EXTRACT_PROMPT）或字段描述")
        lines.append("      （app/schemas.py 的 Field(description=...)），不要改评估标准。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 联网执行
# ---------------------------------------------------------------------------


#: 出现这些片段说明「提取失败」的根因在上游/配置，而不是 Prompt 质量。
#: 用于在评估结论里区分「Prompt 该改」和「环境该修」—— 否则使用者会被告知去
#: 重写 Prompt，而真正的问题是 .env 里的模型名或密钥不对。
_ENV_PROBLEM_MARKERS = (
    "model names",
    "invalid_request_error",
    "401",
    "403",
    "400",
    "connection error",
    "connection refused",
    "timed out",
    "timeout",
    "unauthorized",
    "invalid api key",
)


def looks_like_environment_problem(errors: dict[str, str]) -> bool:
    """提取失败的消息是否指向环境/配置问题（而不是 Prompt 质量）。"""
    if not errors:
        return False
    joined = " ".join(errors.values()).lower()
    return any(marker in joined for marker in _ENV_PROBLEM_MARKERS)


async def evaluate(cases: list[dict], threshold: float) -> int:
    # 延迟导入：--self-test 不应该因为缺少 .env 或依赖而失败
    from app.config import get_settings
    from app.extract import ExtractionError, extract_request
    from app.llm import get_chat_model

    settings = get_settings()
    model = get_chat_model(settings)

    results: dict[str, dict | None] = {}
    errors: dict[str, str] = {}

    for case in cases:
        cid = case["id"]
        try:
            extracted = await extract_request(case["text"], model)
            results[cid] = extracted.model_dump()
        except ExtractionError as exc:
            results[cid] = None
            errors[cid] = str(exc)
        except Exception as exc:  # noqa: BLE001 - 上游故障（认证/连不上/超时/无效请求）
            print(
                f"\n!! 上游故障，评估中止：{type(exc).__name__}: {exc}\n"
                "   这通常意味着 .env 里的 OPENAI_BASE_URL / OPENAI_API_KEY / "
                "OPENAI_MODEL 不对，或网络不通。修好再跑。",
                file=sys.stderr,
            )
            return 2

    sc = score(cases, results, errors)
    print(format_report(sc, threshold, settings.openai_base_url, settings.openai_model))

    # 二次保险：万一上游故障被包成了 ExtractionError（历史上真实发生过 ——
    # 模型名写错时 400 曾被当成本地解析失败），这里不能再建议「去改 Prompt」。
    if looks_like_environment_problem(errors) and sc.field_accuracy == 0.0:
        print(
            "\n!! 全部样例都失败，且错误消息指向环境/配置问题。\n"
            "   先修 .env（BASE_URL / API_KEY / MODEL），不要改 Prompt。",
            file=sys.stderr,
        )
        return 2

    ok, _ = sc.verdict(threshold)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 离线自测（不联网）
# ---------------------------------------------------------------------------


def self_test() -> int:
    """用一份假的模型产出验证评分逻辑本身正确。

    这段逻辑要等有真实密钥才第一次真跑，所以先用假数据把它验了：
    满分样例应判达标，而「该为 null 却抽出订单号」必须判硬性失败。
    """
    cases = load_cases(DEFAULT_CASES)
    print(f"载入 {len(cases)} 条样例，用假产出验证评分逻辑……\n")

    # 场景 A：全部完美（order_no 按期望，包括 4 条应为 null 的）
    perfect = {
        c["id"]: {
            "order_no": c["expected"]["order_no"],
            "request_type": c["expected"]["request_type"],
            "expected_solution": c["expected"]["expected_solution"],
            "detail": " ".join(c.get("detail_keywords", [])[:1]) or "摘要",
        }
        for c in cases
    }
    sc_a = score(cases, perfect, {})
    ok_a, reason_a = sc_a.verdict(0.9)
    print(f"A 满分产出：准确率 {sc_a.field_accuracy:.1%}，编造 {len(sc_a.fabricated)} 次 -> {'达标' if ok_a else '未达标'}")
    assert sc_a.field_accuracy == 1.0, "满分产出应得 100%"
    assert not sc_a.fabricated
    assert ok_a, reason_a

    # 场景 B：把那 4 条应为 null 的 order_no 编造出来，必须判硬性失败
    null_cases = [c["id"] for c in cases if c["expected"]["order_no"] is None]
    assert null_cases, "样例里必须有 order_no 期望为 null 的用例，否则测不出编造"
    fabricated = {cid: dict(v) for cid, v in perfect.items()}
    for cid in null_cases:
        fabricated[cid]["order_no"] = "99999999"
    sc_b = score(cases, fabricated, {})
    ok_b, reason_b = sc_b.verdict(0.9)
    print(f"B 编造 {len(null_cases)} 个订单号：准确率 {sc_b.field_accuracy:.1%}，编造 {len(sc_b.fabricated)} 次 -> {'达标' if ok_b else '未达标'}")
    assert len(sc_b.fabricated) == len(null_cases), "编造必须被逐条抓出"
    assert not ok_b, "存在编造时绝不能判达标"
    assert "编造" in reason_b

    # 场景 C：一条提取失败
    partial = {k: v for k, v in perfect.items() if k != cases[0]["id"]}
    sc_c = score(cases, partial, {cases[0]["id"]: "boom"})
    print(f"C 一条提取失败：PASS {sum(o.passed for o in sc_c.outcomes)}/{len(sc_c.outcomes)}，失败明细 {len(sc_c.failures)} 条")
    assert not sc_c.outcomes[0].passed
    assert any("提取失败" in line for line in sc_c.failures)

    # 场景 D：字段值不对（枚举串对不上）
    wrong = {cid: dict(v) for cid, v in perfect.items()}
    wrong[cases[0]["id"]]["request_type"] = "complaint"
    sc_d = score(cases, wrong, {})
    ok_d, _ = sc_d.verdict(0.9)
    print(f"D 一个字段值错：准确率 {sc_d.field_accuracy:.1%} -> {'达标' if ok_d else '未达标'}")
    assert sc_d.field_accuracy < 1.0
    assert not sc_d.outcomes[0].field_ok["request_type"]

    print("\n报告渲染检查：")
    report = format_report(sc_a, 0.9, "https://example.invalid/v1", "fake-model")
    assert "总体字段准确率：100.0%" in report
    assert "订单号编造次数：0" in report
    print("  渲染正常，含准确率与编造计数两行关键结论")

    print("\n自测通过：评分逻辑在四种场景（满分 / 编造 / 提取失败 / 字段值错）下行为正确。")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="售后信息提取评估")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="不连网，用假产出验证评分逻辑（不需要 .env）",
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if not args.cases.exists():
        print(f"找不到样例文件：{args.cases}", file=sys.stderr)
        return 2

    # 延迟导入并先探配置，缺配置时给出可操作的提示而不是堆栈
    try:
        from app.config import get_settings

        get_settings()
    except Exception as exc:  # noqa: BLE001
        print("配置缺失，请先在仓库根目录创建 .env（参考 .env.example）。", file=sys.stderr)
        print(f"细节：{exc}", file=sys.stderr)
        return 2

    cases = load_cases(args.cases)
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c["id"] in wanted]
    if not cases:
        print("没有匹配的样例。", file=sys.stderr)
        return 2

    return asyncio.run(evaluate(cases, args.threshold))


if __name__ == "__main__":
    raise SystemExit(main())
