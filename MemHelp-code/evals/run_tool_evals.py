"""工具选择评估集 —— 拿标注样例跑一遍**真实上游**。

## 为什么这类产出不走 TDD，而是跑评估集

本文件要验的是「模型自己决定调什么工具」——那是**模型的判断行为**，不是代码逻辑。
断言替代不了它：代码可以断言「工具被正确执行」，但无法断言「模型会选对工具」。
所以按用户定的流程变体，这类产出用**标注样例 + 真实上游**验证。

评估集里的用例分三类：
- **该调**（正例）：规则类问 FAQ、订单/物流/商品类问对应工具、明确要人工就建工单；
- **不该调**（负例）：寒暄、闲聊、自我介绍 —— 调了就是多余的工具调用；
- **允许集合**（模糊例）：没给订单号时，问订单号或查规则都算合理，只要不编造。

## 判定规则（**未达标先改 Prompt / 工具描述，不许改评估标准**）

沿用 ch01 的既定规则。这条很重要：把标准改软能让评估「通过」，
但那只是把问题藏起来，而评估集存在的意义正是暴露它。

## 用法

    uv run --directory MemHelp-code python evals/run_tool_evals.py
    uv run --directory MemHelp-code python evals/run_tool_evals.py --only acc1,acc2
    uv run --directory MemHelp-code python evals/run_tool_evals.py --json dev-notes/tool-evals.json

退出码：0 = 全部符合预期；1 = 有未达标。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# 让脚本能 import app.*（它从 evals/ 下运行，不在 app 包内）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.orm import Session  # noqa: E402

from app.chat_tools import astream_chat_with_tools  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402

CASES_FILE = Path(__file__).resolve().parent / "tool_cases.jsonl"


@dataclass
class CaseResult:
    """一条用例的实测结果。"""

    case_id: str
    message: str
    expected: list[str]
    actual: list[str]
    min_tool_calls: int
    answer: str
    error: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        # 1) 实际调用的工具必须落在预期集合里（没预期就是不许调）
        if not set(self.actual) <= set(self.expected):
            return False
        # 2) 调用次数下限（正例要求至少调一次；负例要求 0 次）
        if len(self.actual) < self.min_tool_calls:
            return False
        return True

    def why(self) -> str:
        if self.error:
            return f"执行出错：{self.error}"
        unexpected = set(self.actual) - set(self.expected)
        if unexpected:
            return f"调了预期外的工具：{sorted(unexpected)}"
        if len(self.actual) < self.min_tool_calls:
            return f"应至少调 {self.min_tool_calls} 个工具，实际 {len(self.actual)} 个"
        return ""


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return cases


async def run_case(case: dict, settings: Settings, session: Session, run_id: str) -> CaseResult:
    """跑一条用例，收集「实际调了哪些工具」与最终回答。"""
    expected = list(case.get("expect_tools") or [])
    min_calls = int(case.get("expect_min_tool_calls") or 0)
    result = CaseResult(
        case_id=case["id"],
        message=case["message"],
        expected=expected,
        actual=[],
        min_tool_calls=min_calls,
        answer="",
    )

    # 每条用例独立会话：复用同一个 id 会把上一轮的对话当历史带上，
    # 于是「这次调了什么」会被上一轮影响（实测踩过）。
    session_id = f"eval-{case['id']}-{run_id}"

    try:
        async for event in astream_chat_with_tools(
            session_id=session_id,
            message=case["message"],
            model=settings_model(settings),
            repositories=Repositories.from_session(session),
            budget_tokens=settings.context_token_budget,
            model_name=settings.openai_model,
            timeout=settings.tool_timeout_seconds,
            max_retries=settings.tool_max_retries,
        ):
            if event.kind == "token":
                result.answer += event.payload.get("delta", "")
            elif event.kind == "tools":
                result.actual = [t["name"] for t in event.payload.get("tools") or []]
    except Exception as exc:  # noqa: BLE001 - 评估集要把失败记成结果，不是崩掉整轮
        result.error = f"{type(exc).__name__}: {exc}"

    return result


def settings_model(settings: Settings):
    """按配置建模型。单独抽出来是为了让 ``run_case`` 好读。"""
    from app.llm import get_chat_model

    return get_chat_model(settings)


def print_report(results: list[CaseResult]) -> int:
    failures = [r for r in results if not r.passed]

    print()
    print("=" * 100)
    print(f"{'id':<10} {'期望工具':<34} {'实际工具':<30} 判定")
    print("-" * 100)
    for r in results:
        expected = ",".join(r.expected) if r.expected else "(不调)"
        actual = ",".join(r.actual) if r.actual else "(未调)"
        mark = "PASS" if r.passed else "FAIL"
        print(f"{r.case_id:<10} {expected:<34} {actual:<30} {mark}")
        if not r.passed:
            print(f"{'':<10} └─ {r.why()}")

    print()
    print("-" * 100)
    total = len(results)
    passed = total - len(failures)
    print(f"命中 {passed}/{total}（{passed / total:.0%}）")

    # 未达标的用例把回答也打出来 —— 只看工具名不足以判断「为什么选错」
    if failures:
        print()
        print("未达标用例的回答（用于判断是工具选择问题还是话术问题）：")
        for r in failures:
            answer = " ".join(r.answer.split())[:160]
            print(f"  [{r.case_id}] {r.message}")
            print(f"     期望 {r.expected}，实际 {r.actual}")
            print(f"     回答：{answer or '(空)'}")

    print()
    print("提示：未达标时按既定规则**先改 Prompt / 工具描述，不要改评估标准**。")
    print("      把标准改软只会把问题藏起来，而暴露它正是评估集存在的意义。")
    return 1 if failures else 0


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.cases))
    if args.only:
        wanted = {piece.strip() for piece in args.only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
        if not cases:
            print(f"[ERR] --only 没有匹配到任何用例：{args.only}", file=sys.stderr)
            return 2

    settings = get_settings()
    print(f"[i] 模型 {settings.openai_model} @ {settings.openai_base_url}")
    print(f"[i] 数据库 {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    print(f"[i] 用例 {len(cases)} 条，来自 {CASES_FILE.name}")

    from app.db.session import create_engine_from, session_factory

    engine = create_engine_from(settings)
    factory = session_factory(engine)

    run_id = str(int(asyncio.get_event_loop().time() * 1000))
    results: list[CaseResult] = []
    try:
        for case in cases:
            with factory() as session:
                result = await run_case(case, settings, session, run_id)
            results.append(result)
            mark = "PASS" if result.passed else "FAIL"
            actual = ",".join(result.actual) if result.actual else "(未调)"
            print(f"  [{mark}] {result.case_id:<10} 实际工具：{actual}")
    finally:
        engine.dispose()

    exit_code = print_report(results)

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [
                    {
                        "id": r.case_id,
                        "message": r.message,
                        "expected": r.expected,
                        "actual": r.actual,
                        "passed": r.passed,
                        "why": r.why(),
                        "answer": r.answer,
                        "error": r.error,
                    }
                    for r in results
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[i] 明细已写入 {out}")

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="工具选择评估集")
    parser.add_argument("--cases", default=str(CASES_FILE), help="用例文件（jsonl）")
    parser.add_argument("--only", default="", help="只跑指定 id，逗号分隔")
    parser.add_argument("--json", default="", help="把明细写到这个 JSON 文件")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
