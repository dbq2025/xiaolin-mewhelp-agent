"""每轮结构化日志（JSONL + logger）。

## 为什么写文件

验收 1 要能从日志里看见「强制检索节点被走到」。只打 `logger.info` 的话，
日志是否落盘取决于宿主怎么配 logging；写一份独立的 JSONL 让验收脚本和排查
都有稳定的读取面。

## 日志是旁路

JSONL 写失败（磁盘满、权限）只记 warning，不能让用户拿不到回答。
MySQL 落库失败才向上抛 —— 那是业务事实，不能被吞。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = ["TurnLogger", "build_turn_log"]


@dataclass(frozen=True)
class TurnLogger:
    """把一个 dict 追加进 JSONL；路径为空时只打 logger。"""

    path: str | Path | None = None

    def write(self, record: Mapping[str, Any]) -> None:
        logger.info("ch05_turn %s", json.dumps(record, ensure_ascii=False))
        if not self.path:
            return

        target = Path(self.path)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:  # noqa: BLE001 - 日志写失败不该吞掉回答
            logger.warning("ch05 turn log 写入失败（已忽略）：%s", target, exc_info=True)


def build_turn_log(state: Mapping[str, Any], latency_ms: int) -> dict[str, Any]:
    """从最终 state 组装一行日志。字段名是验收脚本的契约。"""
    evidence = list(state.get("evidence") or [])
    tool_records = list(state.get("tool_records") or [])
    token_usage = dict(state.get("token_usage") or {})
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": state.get("session_id"),
        "turn_id": state.get("turn_id"),
        "intent": state.get("intent"),
        "intent_source": state.get("intent_source"),
        "intent_confidence": state.get("intent_confidence"),
        "route": state.get("route"),
        "trace": list(state.get("trace") or []),
        "retrieval_attempted": bool(state.get("retrieval_attempted")),
        "retrieval_score": state.get("retrieval_score", 0.0),
        "retrieval_degraded": bool(state.get("retrieval_degraded")),
        "retrieval_refused": bool(state.get("retrieval_refused")),
        "evidence_count": len(evidence),
        "gate_passed": state.get("gate_passed"),
        "gate_reason": state.get("gate_reason"),
        "agent_steps": int(state.get("agent_steps") or 0),
        "llm_calls": int(state.get("llm_calls") or 0),
        "tools": [record.get("name") for record in tool_records],
        "token_usage": token_usage,
        "suggested_actions": list(state.get("suggested_actions") or []),
        "ticket_type": state.get("ticket_type"),
        "stop_reason": state.get("stop_reason"),
        "answer_chars": len(str(state.get("answer") or "")),
        "latency_ms": latency_ms,
        "turn_started_at": state.get("turn_started_at"),
        "logged_at": time.time(),
    }
