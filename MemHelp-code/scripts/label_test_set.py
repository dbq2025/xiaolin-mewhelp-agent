"""T8：测试集人工 gold 标注的工具链。

标注是人的判断，脚本只做三件事：出空白工作表、把填好的标签校验后回填、算与预标的分歧。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from app.topics.taxonomy import validate_labels

REVIEW_HEADER = ("id", "text", "my_labels", "my_rationale", "user_verdict", "user_note")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_worksheet(test_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """一行一个问题，`labels`/`rationale` 留空等人填；预标放进参考列。"""
    return [
        {
            "id": row["id"],
            "text": row["text"],
            "labels_prelabeled": list(row.get("labels") or []),
            "labels": [],
            "rationale": "",
        }
        for row in test_rows
    ]


def apply_gold(
    test_rows: Sequence[dict[str, Any]],
    worksheet: Sequence[dict[str, Any]],
    *,
    require_rationale: bool = False,
    labeled_by: str = "agent",
) -> dict[str, Any]:
    """把工作表回填进测试集，并返回 gold、合并后的测试行、与预标的分歧。"""
    by_id: dict[str, dict[str, Any]] = {}
    for row in worksheet:
        row_id = str(row.get("id", ""))
        if row_id in by_id:
            raise ValueError(f"工作表里 id 重复: {row_id}")
        by_id[row_id] = row

    test_ids = [str(row["id"]) for row in test_rows]
    missing = [row_id for row_id in test_ids if row_id not in by_id]
    if missing:
        raise ValueError("工作表缺少这些行: " + ", ".join(missing))
    extra = [row_id for row_id in by_id if row_id not in set(test_ids)]
    if extra:
        raise ValueError("工作表里有测试集之外的行: " + ", ".join(sorted(extra)))

    merged_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in test_rows:
        row_id = str(row["id"])
        filled = by_id[row_id]
        raw_labels = list(filled.get("labels") or [])
        if not raw_labels:
            raise ValueError(f"{row_id}: labels 不能为空（没诉求也至少标「其他」）")
        labels = validate_labels(raw_labels)
        rationale = str(filled.get("rationale") or "").strip()
        if require_rationale and not rationale:
            raise ValueError(f"{row_id}: 缺 rationale")

        prelabeled = list(row.get("labels") or [])
        merged = dict(row)
        merged["labels"] = labels
        merged["labels_prelabeled"] = prelabeled
        merged.pop("needs_human_label", None)
        merged_rows.append(merged)
        gold_rows.append(
            {
                "id": row_id,
                "text": row["text"],
                "labels": labels,
                "rationale": rationale,
                "labeled_by": labeled_by,
            }
        )
        if set(prelabeled) != set(labels):
            # 没有预标的行（真实问题）算「无参考」，不算分歧；否则分歧率会被真实语料稀释
            if prelabeled:
                disagreements.append(
                    {
                        "id": row_id,
                        "text": row["text"],
                        "prelabeled": prelabeled,
                        "gold": labels,
                    }
                )

    return {"rows": merged_rows, "gold": gold_rows, "disagreements": disagreements}


def render_review_csv(
    gold: Sequence[dict[str, Any]], worksheet: Sequence[dict[str, Any]]
) -> str:
    """抽审表：我的标签和理由填好，用户的判定列留空。"""
    reasons = {str(row.get("id", "")): str(row.get("rationale") or "") for row in worksheet}
    texts = {str(row.get("id", "")): str(row.get("text") or "") for row in worksheet}
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(REVIEW_HEADER)
    for row in gold:
        row_id = str(row["id"])
        writer.writerow(
            [
                row_id,
                row.get("text") or texts.get(row_id, ""),
                "|".join(row["labels"]),
                reasons.get(row_id, ""),
                "",
                "",
            ]
        )
    return buffer.getvalue()


def render_disagreement_report(
    disagreements: Sequence[dict[str, Any]], total: int
) -> str:
    rate = (len(disagreements) / total * 100) if total else 0.0
    lines = [
        "# 测试集 gold 与预标分歧",
        "",
        f"- 总数：{total}",
        f"- 分歧：{len(disagreements)}（{rate:.1f}%）",
        "- 经验区间：10–25% 正常；>40% 说明类目边界或预标 prompt 有问题，先修边界再训练。",
        "",
        "| id | 问题 | 预标 | gold |",
        "| --- | --- | --- | --- |",
    ]
    for row in disagreements:
        lines.append(
            "| {id} | {text} | {pre} | {gold} |".format(
                id=row["id"],
                text=str(row["text"]).replace("|", "\\|"),
                pre="、".join(row["prelabeled"]) or "（空）",
                gold="、".join(row["gold"]),
            )
        )
    return "\n".join(lines) + "\n"


def _cmd_dump(args: argparse.Namespace) -> int:
    test_rows = read_jsonl(Path(args.test))
    worksheet = dump_worksheet(test_rows)
    write_jsonl(Path(args.out), worksheet)
    print(f"[dump] {len(worksheet)} 行 → {args.out}")
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    test_rows = read_jsonl(Path(args.test))
    worksheet = read_jsonl(Path(args.worksheet))
    result = apply_gold(
        test_rows,
        worksheet,
        require_rationale=args.require_rationale,
        labeled_by=args.labeled_by,
    )

    write_jsonl(Path(args.gold_out), result["gold"])
    write_jsonl(Path(args.test), result["rows"])
    Path(args.review_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.review_out).write_text(
        render_review_csv(result["gold"], worksheet), encoding="utf-8", newline="\n"
    )
    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_out).write_text(
            render_disagreement_report(result["disagreements"], len(result["gold"])),
            encoding="utf-8",
            newline="\n",
        )

    total = len(result["gold"])
    conflicts = len(result["disagreements"])
    rate = (conflicts / total * 100) if total else 0.0
    print(f"[apply] gold {total} 行，分歧 {conflicts} 行（{rate:.1f}%）")
    if rate > 40:
        print("[apply] 分歧率 >40%，先改类目边界或预标 prompt，别急着训练", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测试集 gold 标注：出工作表 / 回填 gold")
    sub = parser.add_subparsers(dest="command", required=True)

    dump = sub.add_parser("dump", help="从 test.jsonl 出空白工作表")
    dump.add_argument("--test", default="data/topics/splits/test.jsonl")
    dump.add_argument("--out", default="data/topics/review/test_worksheet.jsonl")
    dump.set_defaults(func=_cmd_dump)

    apply = sub.add_parser("apply", help="把填好的工作表回填成 gold")
    apply.add_argument("--test", default="data/topics/splits/test.jsonl")
    apply.add_argument("--worksheet", default="data/topics/review/test_worksheet.jsonl")
    apply.add_argument("--gold-out", default="data/topics/splits/test_gold.jsonl")
    apply.add_argument("--review-out", default="data/topics/review/test_gold_review.csv")
    apply.add_argument("--report-out", default="data/topics/review/label_disagreements.md")
    apply.add_argument("--labeled-by", default="agent")
    apply.add_argument("--require-rationale", action="store_true")
    apply.set_defaults(func=_cmd_apply)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
