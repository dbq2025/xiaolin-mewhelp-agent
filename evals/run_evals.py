"""售后信息提取评估脚本。

这是「TDD 替代方案」的第 2 层：纯 Prompt 与数据结构没法离线单测，所以拿
标注样例跑一遍真实模型，看字段级准确率。

**需要真实 .env**（`OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`）。

用法::

    uv run python evals/run_evals.py
    uv run python evals/run_evals.py --threshold 0.95
    uv run python evals/run_evals.py --only c06 c07 c08   # 只跑信息缺失那几条

退出码：0 达标 / 1 未达标 / 2 配置缺失或输入文件坏。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 让脚本能直接 `python evals/run_evals.py` 跑（把仓库根加进 sys.path）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.extract import ExtractionError, extract_request  # noqa: E402
from app.llm import get_chat_model  # noqa: E402

DEFAULT_CASES = ROOT / "evals" / "aftersales_cases.jsonl"


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


def _match_expected(actual, expected) -> bool:
    """严格比对单个字段。

    ``None`` 期望必须**恰好**是 ``None`` —— 这是「零编造订单号」的核心断言，
    抽到任何值都算失败。
    """
    if expected is None:
        return actual is None
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip() == expected.strip()
    return actual == expected


def _detail_recall(detail: str | None, keywords: list[str]) -> bool:
    """detail 是自由文本，用关键词命中率粗判（至少命中一个）。"""
    if not keywords:
        return True
    if not detail:
        return False
    return any(kw in detail for kw in keywords)


async def run(cases: list[dict], threshold: float) -> int:
    settings = get_settings()
    print(f"上游：{settings.openai_base_url}")
    print(f"模型：{settings.openai_model}")
    print(f"样例：{len(cases)} 条    阈值：{threshold:.0%}\n")
    model = get_chat_model(settings)

    checked_fields = ("order_no", "request_type", "expected_solution")
    hits = {field: 0 for field in checked_fields}
    total = {field: 0 for field in checked_fields}
    fabricated_order_nos: list[str] = []
    failures: list[str] = []
    detail_ok = 0
    detail_total = 0

    header = f"{'id':<5} {'order_no':<20} {'type':<11} {'solution':<14} {'detail':<7} 结果"
    print(header)
    print("-" * len(header))

    for case in cases:
        cid = case["id"]
        expected = case["expected"]
        try:
            result = await extract_request(case["text"], model)
            got = result.model_dump()
        except ExtractionError as exc:
            got = None
            failures.append(f"{cid}: 提取失败 —— {exc}")

        row_ok = True
        cells: dict[str, str] = {}

        for field in checked_fields:
            total[field] += 1
            exp = expected.get(field)
            act = None if got is None else got.get(field)
            ok = got is not None and _match_expected(act, exp)
            if ok:
                hits[field] += 1
            else:
                row_ok = False
                failures.append(f"{cid}.{field}: 期望 {exp!r}，实际 {act!r}")
            cells[field] = "OK" if ok else f"X({act})"

            if field == "order_no" and exp is None and act is not None:
                fabricated_order_nos.append(f"{cid} -> {act!r}")

        detail_total += 1
        d_ok = got is not None and _detail_recall(
            got.get("detail"), case.get("detail_keywords", [])
        )
        if d_ok:
            detail_ok += 1
        else:
            failures.append(f"{cid}.detail: 关键词未命中 —— {None if got is None else got.get('detail')!r}")

        print(
            f"{cid:<5} {cells['order_no']:<20} {cells['request_type']:<11} "
            f"{cells['expected_solution']:<14} {'OK' if d_ok else 'X':<7} "
            f"{'PASS' if row_ok else 'FAIL'}"
        )

    print()
    print("== 字段级准确率 ==")
    grand_hit = grand_total = 0
    for field in checked_fields:
        grand_hit += hits[field]
        grand_total += total[field]
        print(f"  {field:<18} {hits[field]}/{total[field]}  {hits[field] / total[field]:.1%}")
    print(f"  {'detail(关键词)':<18} {detail_ok}/{detail_total}  {detail_ok / detail_total:.1%}")

    accuracy = grand_hit / grand_total
    print(f"\n总体字段准确率：{accuracy:.1%}（{grand_hit}/{grand_total}）")
    print(f"订单号编造次数：{len(fabricated_order_nos)}")

    if fabricated_order_nos:
        print("\n!! 编造订单号（这是硬性失败，必须改 Prompt）:")
        for item in fabricated_order_nos:
            print(f"   {item}")

    if failures:
        print(f"\n== 失败明细（{len(failures)} 条）==")
        for item in failures:
            print(f"  - {item}")

    passed = accuracy >= threshold and not fabricated_order_nos
    print()
    if passed:
        print(f"结论：达标（准确率 {accuracy:.1%} >= {threshold:.0%}，零编造）")
        return 0
    print(f"结论：未达标（准确率 {accuracy:.1%} < {threshold:.0%} 或存在编造）")
    print("提示：先改 Prompt（app/prompts.py 的 EXTRACT_PROMPT）或字段描述")
    print("      （app/schemas.py 的 Field(description=...)），不要改评估标准。")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="售后信息提取评估")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--only", nargs="*", help="只跑指定 id")
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"找不到样例文件：{args.cases}", file=sys.stderr)
        return 2

    try:
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

    return asyncio.run(run(cases, args.threshold))


if __name__ == "__main__":
    raise SystemExit(main())
