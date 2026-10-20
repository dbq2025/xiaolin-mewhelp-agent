"""上下文可观测日志 —— ``log/app.log``。

## 为什么单独一份日志，不塞进 ``logs/ch05-turns.jsonl``

那份是**每轮一行的结构化指标**（延迟、工具、token 用量），字段固定、给验收脚本
和看板读。而这里要的是**把实际发给模型的上下文原样打出来**：摘要全文、滑窗逐条、
两个锚点、两层预算与实占。两者的读者不同、体积量级也不同（一条 model_ctx 顶得上
几十条 turn 记录），混在一起会让那份 JSONL 变得没法扫。

## 为什么自己 ``open(..., "a")`` 而不用 ``logging.FileHandler``

``app/`` 里**没有任何** ``logging.basicConfig``，root logger 落不落盘取决于宿主
怎么配。挂 FileHandler 到自己的 logger 上又要处理重复 handler、多进程写交错、
以及 uvicorn 重载时的双写。写文件这件事只有五行的量，自己持有句柄反而更可控。

## 三个入口的分工

- :func:`log_model_ctx`：主力 Agent 每轮**调模型前**打。这是验收第 4 条的
  主要证据面 —— ``grep model_ctx log/app.log`` 要能直接看到摘要和滑窗。
- :func:`log_history_ctx`：指代消解/意图识别共用的历史，**每轮必打**。闲聊与
  兜底轮不经过 Agent 节点，只有这一条能证明「那一轮也有历史上下文」。
- :func:`log_summary_event`：后台摘要任务的五态（trigger / start / done / skip /
  failed）。少了 ``skip`` 这一态，看日志的人会以为摘要从没触发过。

日志是**旁路**：写失败只记 warning（见 :meth:`AppLogger.write`）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.messages import BaseMessage

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_BODY_CHARS",
    "SUMMARY_EVENTS",
    "AppLogger",
    "log_history_ctx",
    "log_model_ctx",
    "log_summary_event",
    "resolve_log_path",
]

#: 单条正文在日志里的最大字符数。
#:
#: 500 而不是不截：一次检索结果、一段客服长答复动辄几千字，一轮就能把日志写成
#: 几 MB，而 ``grep`` 一个几百 MB 的文件比翻十行日志还慢。500 字足够判断
#: 「这条说的是什么」，也足够确认「层 2 的截短确实生效了」（截短后是 60 字）。
MAX_BODY_CHARS = 500

#: 摘要任务的五种事件。写错拼写会让某一类事件静默消失，所以入口处校验。
SUMMARY_EVENTS: tuple[str, ...] = ("trigger", "start", "done", "skip", "failed")

#: 消息类型 → 日志里的短标签。用 ``user/ai`` 而不是 ``human/assistant``：
#: 这两个词在项目里到处都是（``role`` 枚举、前端渲染），日志里跟着用能直接对号。
_ROLE_LABELS: dict[str, str] = {
    "HumanMessage": "user",
    "AIMessage": "ai",
    "ToolMessage": "tool",
    "SystemMessage": "system",
}


def resolve_log_path(path: str | Path | None) -> Path | None:
    """把配置里的日志路径解析成绝对路径。空值 → ``None``（不写文件）。"""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    target = Path(text)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    return target


@dataclass(frozen=True)
class AppLogger:
    """往 ``log/app.log`` 追加一条记录（一行头 + N 行正文）。

    ``path`` 为空时只打 ``logger.info``：测试环境默认不写文件，否则每跑一次
    测试就往仓库里留一份日志。
    """

    path: str | Path | None = "log/app.log"

    def write(self, record: str) -> None:
        logger.info("ch07_ctx %s", record)
        target = resolve_log_path(self.path)
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(record if record.endswith("\n") else record + "\n")
        except OSError:  # noqa: BLE001 - 日志写失败不该吞掉回答
            logger.warning("ch07 上下文日志写入失败（已忽略）：%s", target, exc_info=True)


def _one_line(text: str) -> str:
    """折叠成一行并截断。

    折叠是必须的：正文里的换行会把「一条消息一行」的格式打散，于是
    ``grep model_ctx`` 后面跟的行数就不再等于窗口条数。
    """
    flat = " ".join(str(text).split())
    if len(flat) <= MAX_BODY_CHARS:
        return flat
    return flat[:MAX_BODY_CHARS] + "…"


def _role_label(message: BaseMessage) -> str:
    name = type(message).__name__
    return _ROLE_LABELS.get(name, name.removesuffix("Message").lower())


def _body(message: BaseMessage) -> str:
    """取一条消息在日志里的正文，带工具调用摘要。"""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        # 多模态 / 内容块：只留文本块，其余用类型名占位（不会有裸图片进日志）
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
            else:
                parts.append(f"<{type(block).__name__}>")
        text = " ".join(parts)
    else:
        text = "" if content is None else str(content)

    calls = getattr(message, "tool_calls", None) or []
    if calls:
        names = ", ".join(str(call.get("name", "?")) for call in calls if isinstance(call, dict))
        text = f"{text} (tool_calls: {names})" if text else f"(tool_calls: {names})"
    return _one_line(text)


def _summary_lines(summary_texts: Sequence[str]) -> list[str]:
    return [f"[摘要 {index}] {_one_line(text)}" for index, text in enumerate(summary_texts, 1)]


def _message_lines(messages: Sequence[BaseMessage]) -> list[str]:
    return [f"[{_role_label(message)}] {_body(message)}" for message in messages]


def log_model_ctx(
    *,
    logger: AppLogger,
    session_id: str,
    turn: str | int,
    messages: Sequence[BaseMessage],
    layer1_from_msg_id: int | None,
    summary_upto_msg_id: int | None,
    budget: Any,
    layer1_tokens: int,
    layer2_tokens: int,
    summary_texts: Sequence[str] = (),
) -> str:
    """打印主力 Agent 这一轮**实际发出去**的上下文。

    头行把「判断压缩是否该发生」需要的数全放上去：两个锚点（边界在哪）、
    两层预算（天花板）、两层实占（离天花板多远）。看一行就知道这一轮为什么
    压了或者为什么没压。

    :param messages: 拼装完成的完整列表（含第 0 条 system）。
    :param budget: ``BudgetReport`` 或任意带 ``layer1_budget`` / ``layer2_budget``
        属性的对象。用鸭子类型是为了让日志模块不必依赖预算模块。
    """
    header = (
        f"model_ctx session={session_id} turn={turn}"
        f" summary_upto={_none_str(summary_upto_msg_id)}"
        f" layer1_from={_none_str(layer1_from_msg_id)}"
        f" layer1_budget={budget.layer1_budget}"
        f" layer2_budget={budget.layer2_budget}"
        f" layer1_tokens={layer1_tokens}"
        f" layer2_tokens={layer2_tokens}"
        f" windows={len(messages)}"
        f" tokens={layer1_tokens + layer2_tokens}"
    )
    lines = [header, *_summary_lines(summary_texts), *_message_lines(messages)]
    record = "\n".join(lines)
    logger.write(record)
    return record


def log_history_ctx(
    *,
    logger: AppLogger,
    session_id: str,
    turn: str | int,
    summary_texts: Sequence[str] = (),
    messages: Sequence[BaseMessage],
) -> str:
    """打印指代消解 / 意图识别看到的历史。**每轮必打**，包括闲聊兜底轮。

    与 :func:`log_model_ctx` 的区别：那条是「主力 Agent 收到了什么」，这条是
    「前置节点收到了什么」。两者可以不同 —— 前置节点只看历史，不拿检索证据，
    也没有 system。分开打才能判断「答偏了」是检索的锅还是消解的锅。
    """
    header = (
        f"history_ctx session={session_id} turn={turn}"
        f" summaries={len(summary_texts)}"
        f" windows={len(messages)}"
    )
    lines = [header, *_summary_lines(summary_texts), *_message_lines(messages)]
    record = "\n".join(lines)
    logger.write(record)
    return record


def log_summary_event(
    *,
    logger: AppLogger,
    event: str,
    session_id: str,
    **fields: Any,
) -> str:
    """打印后台摘要任务的一个事件。

    ``fields`` 的顺序就是日志里的顺序，调用方按「先结论后证据」排：触发时
    ``layer2_tokens`` 在前（那是判据），完成时 ``seg`` / ``chars`` / ``elapsed_ms``
    在前（那是结果）。

    ``trigger`` 事件会把 ``layer2_tokens`` 与 ``budget`` 渲染成
    ``layer2_tokens=1832 > budget=1695`` 的不等式 —— 验收第 2 条要求日志里
    能直接看到「攒到超预算」这个判据，写成两个独立 k=v 就得自己去比。
    """
    if event not in SUMMARY_EVENTS:
        raise ValueError(f"未知摘要事件 {event!r}，只允许 {SUMMARY_EVENTS}")

    parts = [f"summary {event} session={session_id}"]
    consumed: set[str] = set()
    if event == "trigger" and "layer2_tokens" in fields and "budget" in fields:
        parts.append(f"layer2_tokens={fields['layer2_tokens']} > budget={fields['budget']}")
        consumed.update({"layer2_tokens", "budget"})
    for key, value in fields.items():
        if key in consumed or value is None:
            continue
        parts.append(f"{key}={_field_str(value)}")

    record = " ".join(parts)
    logger.write(record)
    return record


def _field_str(value: Any) -> str:
    """字段值转一行文本：多词的值加引号，否则 ``session=a b`` 会被读成三个字段。"""
    text = " ".join(str(value).split())
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or '"' in text:
        return '"' + text.replace('"', "'") + '"'
    return text


def _none_str(value: int | None) -> str:
    return "-" if value is None else str(value)
