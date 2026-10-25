"""ch10 数据流水线 CLI：捞语料 → 清洗 → 预标 → 分层 → 增强 → 导出术语表。

每一步都是一次幂等的「读上一级文件、写下一级文件」，中间产物全部落在
``data/topics/`` 下，可以随时删掉重跑而不碰数据库。这样做的代价是磁盘上
多几份 jsonl，好处是**任何一步出错都能只重跑那一步**，而不是从飞轮重新捞
一遍、再烧一次大模型的 token。

真语料只有 12 条去重问题，所以 ``--generate`` 造 1500 条模拟问题补足；
真实问题**永远只进 test**（``iterative_stratify(real_rows=...)``），拿它们
训练等于让模型背答案，测试集分数就没有意义了。

用法::

    python scripts/build_topic_data.py --from-db --export-taxonomy --clean --split --augment
    python scripts/build_topic_data.py --generate --allow-llm --clean --split
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.topics.augment import augment_rows
from app.topics.clean import clean_text, is_noise, load_typo_map
from app.topics.split import iterative_stratify
from app.topics.synth import (
    ParsedQuestion,
    find_conflicts,
    generation_messages,
    labeling_messages,
    parse_generated,
    parse_labeled,
)
from app.topics.taxonomy import CATEGORIES, TaxonomyError, taxonomy_hash, taxonomy_payload

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "topics"

#: 多诉求句子的搭配：按「真实用户会一起问」的搭配造。随机配会把
#: 「发票 + 会员积分」这种不存在的组合喂进训练集，学到的是噪声。
MULTI_PAIRS: tuple[tuple[str, str], ...] = (
    ("退换货", "物流"),
    ("退换货", "运费"),
    ("退换货", "尺码"),
    ("物流", "运费"),
    ("发票", "订单修改"),
    ("优惠活动", "价保"),
    ("支付", "订单修改"),
    ("质量问题", "保修维修"),
    ("库存补货", "商品信息"),
    ("账号", "会员积分"),
    ("评价", "质量问题"),
    ("会员积分", "优惠活动"),
)

#: 执行顺序是固定的，跟命令行里写的顺序无关 —— 「先 split 再 clean」这种
#: 顺序在数据上没有任何意义，与其报错不如按正确顺序跑。
_STEP_ORDER: tuple[str, ...] = (
    "from_db",
    "generate",
    "clean",
    "prelabel",
    "split",
    "augment",
    "export_taxonomy",
)

_LLM_STEPS = {"generate", "prelabel"}
_MULTI_RATIO = 0.25
#: 单次生成的条数上限。一次要 88 条会被 max_tokens 截断，截断的 JSON 行
#: 全变废行 —— 每次 25 条是能稳定完整返回的量级。
_GENERATE_CHUNK = 25
#: 生成阶段的断点文件：每块写完就追加、flush，重跑时从它续。
_GENERATE_PARTIAL = "synthetic.partial.jsonl"


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content)


def _invoke(llm: Any, messages: list[dict[str, str]]) -> str:
    return _message_text(llm.invoke(messages))


# -- 各步骤 -----------------------------------------------------------------


def _raw_path(source_dir: Path, dirs: dict[str, Path], kind: str) -> Path:
    """优先吃 ``raw/`` 下的中间产物；没有就吃 ``source_dir`` 里的原始文件。

    测试用 12 行 fixture 直接喂进来，真跑时则吃 ``--from-db`` / ``--generate``
    写好的 ``raw/*.jsonl``，两边的清洗逻辑完全同一条。
    """

    inside = dirs["raw"] / f"{kind}.jsonl"
    if inside.is_file():
        return inside
    return source_dir / f"raw_{kind}.jsonl"


def _step_from_db(dirs: dict[str, Path], **_kwargs: Any) -> dict[str, Any]:
    from sqlalchemy import select

    from app.config import Settings
    from app.db.models import LowConfidenceQuestion
    from app.db.session import create_engine_from, session_factory

    settings = Settings()
    engine = create_engine_from(settings)
    try:
        session = session_factory(engine)()
        try:
            rows = session.execute(
                select(LowConfidenceQuestion.id, LowConfidenceQuestion.raw_question)
            ).all()
        finally:
            session.close()
    finally:
        engine.dispose()

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row_id, question in rows:
        text = (question or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append({"id": f"r{row_id}", "text": text, "labels": [], "source": "real"})
    written = _write_jsonl(dirs["raw"] / "real.jsonl", out)
    return {"fetched": len(rows), "kept": written, "dup_dropped": len(rows) - written}


def _partner(category_name: str) -> str | None:
    for left, right in MULTI_PAIRS:
        if category_name == left:
            return right
        if category_name == right:
            return left
    return None


def _generate_chunk(
    llm: Any, category: Category, take: int, second: str | None
) -> tuple[list[Any], str | None]:
    """一个「调模型 + 解析」的分块，**不碰任何共享状态**，所以能丢进线程池。"""
    messages = generation_messages(category, take, second_label=second)
    error: str | None = None
    for attempt in range(2):  # 一次坏输出重试一次，别让整批停在这一类
        try:
            raw = _invoke(llm, messages)
        except Exception as exc:  # noqa: BLE001 - 网络抖动不该让整批停掉
            print(
                f"[generate] {category.name} 调用失败（{attempt + 1}/2）：{exc}",
                file=sys.stderr,
                flush=True,
            )
            error = "error"
            continue
        try:
            parsed = parse_generated(raw)
        except TaxonomyError as exc:
            # 自造标签绝不能进语料：这里拦住、计数，重试一次还不行就跳过这段。
            print(f"[generate] {category.name} 标签越界：{exc}", file=sys.stderr, flush=True)
            error = "invalid"
            continue
        if parsed:
            return parsed, None
        print(f"[generate] {category.name} 第 {attempt + 1} 次为空", file=sys.stderr, flush=True)
        error = "empty"
    return [], error or "empty"


def _step_generate(
    dirs: dict[str, Path],
    *,
    llm: Any,
    count: int,
    seed: int,
    workers: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """按类目造模拟问题，**逐块落盘**，块与块之间可以并发。

    1500 条 = 68 次大模型调用，串行跑要一个多小时；中途断掉要是从头再来，
    既烧 token 又烧时间。所以：分块丢进线程池跑，**只有主线程**做去重、编号
    和追加写 ``synthetic.partial.jsonl``；重跑时先读断点、已够数的组直接跳过，
    全部跑完才汇总成 ``synthetic.jsonl`` 并删掉断点文件。
    """

    if llm is None:
        raise RuntimeError("--generate 需要大模型（--allow-llm）")
    per = max(1, count // len(CATEGORIES))
    partial = dirs["raw"] / _GENERATE_PARTIAL
    out: list[dict[str, Any]] = _read_jsonl(partial)
    seen = {str(row.get("text", "")) for row in out}
    done: Counter[tuple[str, str]] = Counter(
        (str(row.get("category", "")), str(row.get("gen_group", ""))) for row in out
    )
    resumed = len(out)
    state = {"fresh": 0, "skipped": 0, "invalid": 0}

    tasks: list[tuple[Category, str, str | None, int, tuple[str, str]]] = []
    for category in CATEGORIES:
        multi_n = int(round(per * _MULTI_RATIO))
        single_n = per - multi_n
        partner = _partner(category.name)
        if partner is None:
            single_n, multi_n = per, 0  # 没搭配就全造单诉求，不硬凑
        for group, n, second in (("single", single_n, None), ("multi", multi_n, partner)):
            key = (category.name, second or "")
            remaining = n - done.get(key, 0)
            while remaining > 0:
                take = min(_GENERATE_CHUNK, remaining)
                tasks.append((category, group, second, take, key))
                remaining -= take

    def merge(
        task: tuple[Category, str, str | None, int, tuple[str, str]],
        parsed: list[Any],
        error: str | None,
    ) -> None:
        category, group, _second, take, key = task
        if error == "invalid":
            state["invalid"] += 1
        state["skipped"] += take - len(parsed)
        batch: list[dict[str, Any]] = []
        for item in parsed:
            if item.text in seen:
                continue
            seen.add(item.text)
            batch.append(
                {
                    "id": f"p{len(out) + len(batch) + 1:05d}",
                    "text": item.text,
                    "labels": item.labels,
                    "source": "synthetic",
                    "category": category.name,
                    "gen_group": group,
                }
            )
        if batch:
            _append_jsonl(partial, batch)
            out.extend(batch)
            state["fresh"] += len(batch)
        done[key] += take
        print(
            f"[generate] {category.name}/{group} {len(parsed)}/{take}"
            f" 累计 {len(out)} 条",
            file=sys.stderr,
            flush=True,
        )

    if workers <= 1:
        for task in tasks:
            category, _group, second, take, _key = task
            parsed, error = _generate_chunk(llm, category, take, second)
            merge(task, parsed, error)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_generate_chunk, llm, task[0], task[3], task[2]): task
                for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    parsed, error = future.result()
                except Exception as exc:  # noqa: BLE001 - 单块炸了不该带走整批断点
                    print(f"[generate] 分块异常：{exc}", file=sys.stderr, flush=True)
                    parsed, error = [], "error"
                merge(task, parsed, error)

    random.Random(seed).shuffle(out)
    for index, row in enumerate(out, 1):
        row["id"] = f"s{index:04d}"
        row.pop("gen_group", None)
    written = _write_jsonl(dirs["raw"] / "synthetic.jsonl", out)
    partial.unlink(missing_ok=True)
    return {
        "requested": per * len(CATEGORIES),
        "kept": written,
        "fresh": state["fresh"],
        "resumed": resumed,
        "skipped": state["skipped"],
        "invalid_label_batches": state["invalid"],
    }


def _step_clean(
    source_dir: Path, dirs: dict[str, Path], *, seed: int, **kwargs: Any
) -> dict[str, Any]:
    typo_path = source_dir / "typos.json"
    typo_map = load_typo_map(typo_path) if typo_path.is_file() else None

    summary: dict[str, Any] = {"kept": 0, "noise_dropped": 0, "dup_dropped": 0}
    seen: set[str] = set()
    # real 先跑：同一句话既在真实池又在合成池时，留真实的那条。
    for kind in ("real", "synthetic"):
        kept_rows: list[dict[str, Any]] = []
        noise = dup = 0
        for row in _read_jsonl(_raw_path(source_dir, dirs, kind)):
            raw = str(row.get("text", ""))
            result = clean_text(raw, typo_map=typo_map)
            if is_noise(result):
                noise += 1
                continue
            if result.text in seen:
                dup += 1
                continue
            seen.add(result.text)
            kept_rows.append(
                {
                    "id": row.get("id", ""),
                    "text": result.text,
                    "text_raw": raw,
                    "labels": list(row.get("labels", [])),
                    "source": kind,
                    "changes": [
                        {"rule": c.rule, "before": c.before, "after": c.after}
                        for c in result.changes
                    ],
                }
            )
        _write_jsonl(dirs["clean"] / f"{kind}.jsonl", kept_rows)
        summary["kept"] += len(kept_rows)
        summary["noise_dropped"] += noise
        summary["dup_dropped"] += dup
    return summary


def _step_prelabel(
    dirs: dict[str, Path], *, llm: Any, batch_size: int, seed: int, **kwargs: Any
) -> dict[str, Any]:
    if llm is None:
        raise RuntimeError("--prelabel 需要大模型（--allow-llm）")

    conflicts: list[dict[str, Any]] = []
    labeled_counts: dict[str, int] = {}
    for kind in ("real", "synthetic"):
        rows = _read_jsonl(dirs["clean"] / f"{kind}.jsonl")
        if not rows:
            _write_jsonl(dirs["labeled"] / f"{kind}.jsonl", [])
            labeled_counts[kind] = 0
            continue
        labeled: list[dict[str, Any]] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            raw = _invoke(llm, labeling_messages([row["text"] for row in batch]))
            parsed_items = parse_labeled(raw)
            parsed = {item.text: item for item in parsed_items}
            generated_items = [
                ParsedQuestion(text=row["text"], labels=list(row["labels"]))
                for row in batch
                if row.get("labels")
            ]
            for conflict in find_conflicts(generated_items, parsed_items):
                conflicts.append(
                    {
                        "text": conflict.text,
                        "generated": "|".join(conflict.generated),
                        "prelabeled": "|".join(conflict.prelabeled),
                    }
                )
            for row in batch:
                item = parsed.get(row["text"])
                generated = list(row.get("labels", []))
                if item is None:
                    labeled.append({**row, "labels": generated, "rationale": ""})
                    continue
                # 真实问题的标签就取预标结果；合成问题保留生成标签作为权威，
                # 预标只用来发现分歧（两边都可能是错的，得人工看）。
                labels = item.labels if not generated else generated
                labeled.append({**row, "labels": labels, "rationale": item.rationale})
        _write_jsonl(dirs["labeled"] / f"{kind}.jsonl", labeled)
        labeled_counts[kind] = len(labeled)

    review_dir = dirs["review"]
    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "label_conflicts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("text,generated,prelabeled\n")
        for item in conflicts:
            handle.write(f'"{item["text"]}","{item["generated"]}","{item["prelabeled"]}"\n')
    return {"labeled": labeled_counts, "conflicts": len(conflicts)}


def _step_split(dirs: dict[str, Path], *, seed: int, **kwargs: Any) -> dict[str, Any]:
    gold = {
        str(row.get("id")): list(row["labels"])
        for row in _read_jsonl(dirs["labeled"] / "real.jsonl")
        if row.get("labels")
    }
    real_rows: list[dict[str, Any]] = []
    for row in _read_jsonl(dirs["clean"] / "real.jsonl"):
        labels = gold.get(str(row.get("id")), list(row.get("labels") or []))
        item = {**row, "labels": labels}
        if not labels:
            # 真实问题在人工标完之前保持「待标」状态，**不要**从 test 里悄悄删掉 ——
            # 评测时跳过并报数，比让它们消失安全得多。
            item["needs_human_label"] = True
        real_rows.append(item)

    synthetic = _read_jsonl(dirs["clean"] / "synthetic.jsonl")
    if not synthetic and not real_rows:
        raise RuntimeError("没有可切分的语料：先跑 --from-db / --generate 再 --clean")

    result = iterative_stratify(synthetic, seed=seed, real_rows=real_rows)
    for name, rows in (("train", result.train), ("val", result.val), ("test", result.test)):
        _write_jsonl(dirs["splits"] / f"{name}.jsonl", rows)
    (dirs["splits"] / "strata_report.json").write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "sizes": result.report["sizes"],
        "zero_in_split": len(result.report["zero_in_split"]),
        "real_in_test": result.report["real_in_test"],
    }


def _step_augment(
    dirs: dict[str, Path], *, seed: int, copies: int, **kwargs: Any
) -> dict[str, Any]:
    train = _read_jsonl(dirs["splits"] / "train.jsonl")
    if not train:
        # 真实语料只有 12 条且全在 test 时，train 就是空的：这不算错误，
        # 是「还没造模拟问题」的正常状态，空文件写出来让下游知道跑过了。
        _write_jsonl(dirs["splits"] / "train_aug.jsonl", [])
        return {"train": 0, "train_aug": 0, "added": 0, "skipped": "train 为空"}
    augmented = augment_rows(train, copies=copies, seed=seed)
    written = _write_jsonl(dirs["splits"] / "train_aug.jsonl", augmented)
    return {"train": len(train), "train_aug": written, "added": written - len(train)}


def export_taxonomy(out_dir: Path) -> str:
    """导出 sidecar 启动时要比对的那份快照 + 给人看的 markdown 表。"""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = taxonomy_payload()
    payload["hash"] = taxonomy_hash()
    (out_dir / "taxonomy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# ch10 主题类目表",
        "",
        f"共 {len(CATEGORIES)} 类，顺序即标签向量下标。sidecar 启动时用哈希比对，",
        "改了这张表就必须重训，否则标签会错位。",
        "",
        "| 编号 | 类目 | 边界说明 | 触发词 |",
        "| --- | --- | --- | --- |",
    ]
    for category in CATEGORIES:
        triggers = "、".join(category.triggers)
        lines.append(
            f"| {category.index} | {category.name} | {category.boundary} | {triggers} |"
        )
    (out_dir / "taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_dir)


def build_dataset(
    source_dir: Path | str,
    *,
    steps: Sequence[str],
    llm: Any = None,
    seed: int = 42,
    out_dir: Path | str | None = None,
    count: int = 1500,
    copies: int = 1,
    batch_size: int = 20,
    workers: int = 1,
) -> dict[str, Any]:
    """按 ``steps`` 跑流水线，返回每步的摘要 + 产物目录。

    没有副作用地「只读」是不成立的（流水线的产出就是副作用），所以真正的
    约定是：**只写 ``out_dir`` 之下**，测试传一个临时目录进来就不会碰到真语料。
    """

    source = Path(source_dir)
    root = Path(out_dir) if out_dir is not None else DEFAULT_OUT
    dirs = {
        "root": root,
        "raw": root / "raw",
        "clean": root / "clean",
        "labeled": root / "labeled",
        "review": root / "review",
        "splits": root / "splits",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    wanted = set(steps)
    unknown = wanted - set(_STEP_ORDER)
    if unknown:
        raise ValueError(f"未知步骤：{sorted(unknown)}")
    if wanted & _LLM_STEPS and llm is None:
        raise RuntimeError(f"这些步骤需要大模型（--allow-llm）：{sorted(wanted & _LLM_STEPS)}")

    summary: dict[str, Any] = {"dirs": {k: str(v) for k, v in dirs.items()}}
    for step in _STEP_ORDER:
        if step not in wanted:
            continue
        if step == "from_db":
            result = _step_from_db(dirs)
        elif step == "generate":
            result = _step_generate(dirs, llm=llm, count=count, seed=seed, workers=workers)
        elif step == "clean":
            result = _step_clean(source, dirs, seed=seed)
        elif step == "prelabel":
            result = _step_prelabel(dirs, llm=llm, batch_size=batch_size, seed=seed)
        elif step == "split":
            result = _step_split(dirs, seed=seed)
        elif step == "augment":
            result = _step_augment(dirs, seed=seed, copies=copies)
        else:  # export_taxonomy
            result = {"out": export_taxonomy(dirs["root"])}
        summary[step] = result
        print(json.dumps({step: result}, ensure_ascii=False))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ch10 主题分类语料流水线")
    parser.add_argument("--from-db", action="store_true", help="从低置信度问题池捞真实问题")
    parser.add_argument("--generate", action="store_true", help="调大模型造模拟问题")
    parser.add_argument("--clean", action="store_true", help="清洗（脱敏 / 错别字 / 空白）")
    parser.add_argument("--prelabel", action="store_true", help="大模型预标 + 冲突表")
    parser.add_argument("--split", action="store_true", help="分层抽样 80/10/10")
    parser.add_argument("--augment", action="store_true", help="增强训练集")
    parser.add_argument("--export-taxonomy", action="store_true", help="导出类目表快照")
    parser.add_argument("--all", action="store_true", help="按顺序跑全部步骤")
    parser.add_argument("--allow-llm", action="store_true", help="允许真的调大模型")
    parser.add_argument("--count", type=int, default=1500, help="模拟问题总数")
    parser.add_argument("--copies", type=int, default=1, help="每条训练样本的增强份数")
    parser.add_argument("--batch-size", type=int, default=20, help="预标每批条数")
    parser.add_argument("--workers", type=int, default=8, help="生成/预标并发分块数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT, help="原始语料目录")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    steps: list[str] = []
    if args.all:
        steps = list(_STEP_ORDER)
    else:
        for name in _STEP_ORDER:
            if getattr(args, name):
                steps.append(name)
    if not steps:
        parser.print_help()
        return 2
    if set(steps) & _LLM_STEPS and not args.allow_llm:
        print("调大模型的步骤要显式加 --allow-llm", file=sys.stderr)
        return 2
    if not args.allow_llm:
        steps = [s for s in steps if s not in _LLM_STEPS]

    llm = None
    if set(steps) & _LLM_STEPS:
        from app.llm import get_chat_model

        llm = get_chat_model()

    build_dataset(
        args.source,
        steps=steps,
        llm=llm,
        seed=args.seed,
        out_dir=args.out,
        count=args.count,
        copies=args.copies,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
