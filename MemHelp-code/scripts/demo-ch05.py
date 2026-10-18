"""ch05 五条验收的可复现演示（打本地服务）。

用法（先起服务）：

```powershell
uv run --directory MemHelp-code uvicorn app.main:app --port 8010
uv run --directory MemHelp-code python scripts/demo-ch05.py --base-url http://127.0.0.1:8010
```

退出码 0 = 五条后端可判定部分全过；1 = 有失败；2 = 服务/配置不可用。

> 「转人工」是纯前端模拟，脚本无法点击 DOM；它的浏览器验收单独在
> README 的演示步骤里说明，脚本会校验后端**没有**自动动作、两个动作都作为
> `actions` 返回。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx
import pymysql

from app.agent.prompts import CHITCHAT_REPLY
from app.config import PROJECT_ROOT, get_settings


def _log_path() -> Path:
    path = Path(get_settings().turn_log_path or "")
    return path if path.is_absolute() else PROJECT_ROOT / path


def _turn_records(session_id: str) -> list[dict]:
    path = _log_path()
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("session_id") == session_id:
            records.append(record)
    return records


def _last_turn(session_id: str) -> dict:
    records = _turn_records(session_id)
    return records[-1] if records else {}


def _ticket_row(ticket_no: str) -> dict | None:
    settings = get_settings()
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT description, ticket_type, status FROM tickets WHERE ticket_no = %s",
                (ticket_no,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {"description": row[0], "ticket_type": row[1], "status": row[2]}
    finally:
        conn.close()


def _chat(client: httpx.Client, base: str, session_id: str, message: str) -> dict:
    response = client.post(
        f"{base}/api/chat/completions",
        json={"session_id": session_id, "message": message, "stream": False},
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


def _names(payload: dict, key: str = "tools") -> list[str]:
    return [str(item.get("name")) for item in (payload.get(key) or [])]


def main() -> int:
    parser = argparse.ArgumentParser(description="ch05 五条验收演示")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--out", default="dev-notes/ch05-acceptance-output.txt")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    checks: list[tuple[str, bool, str]] = []
    details: list[str] = []

    try:
        with httpx.Client() as client:
            health = client.get(f"{args.base_url}/healthz", timeout=10.0)
            health.raise_for_status()

            # 1. 政策类：日志里必须看见强制检索节点
            s1 = f"ch05-acc-policy-{stamp}"
            b1 = _chat(client, args.base_url, s1, "退货政策是什么")
            r1 = _last_turn(s1)
            ok1 = (
                b1.get("route") == "knowledge"
                and "knowledge_retrieve" in (r1.get("trace") or [])
                and r1.get("retrieval_attempted") is True
            )
            checks.append(("1 政策问题走强制检索", ok1, f"route={b1.get('route')} trace={r1.get('trace')}"))
            details.append(
                f"[1] session={s1} intent={b1.get('intent')} route={b1.get('route')} "
                f"agent_steps={b1.get('agent_steps')} citations={len(b1.get('citations') or [])}\n"
                f"    log.trace={r1.get('trace')} retrieval_score={r1.get('retrieval_score')}"
            )

            # 2. 物流：Agent 自己调 query_logistics
            s2 = f"ch05-acc-logistics-{stamp}"
            b2 = _chat(client, args.base_url, s2, "订单 1001 的物流到哪了")
            ok2 = "query_logistics" in _names(b2)
            checks.append(("2 订单物流由 Agent 调工具", ok2, f"tools={_names(b2)}"))
            details.append(
                f"[2] session={s2} route={b2.get('route')} agent_steps={b2.get('agent_steps')} "
                f"tools={_names(b2)}\n    content={str(b2.get('content'))[:160]}"
            )

            # 3. 投诉：两个独立动作；建单端点才写 tickets；历史回读带 actions
            s3 = f"ch05-acc-complaint-{stamp}"
            b3 = _chat(client, args.base_url, s3, "我要投诉")
            actions = [item.get("action") for item in (b3.get("actions") or [])]
            ok3a = actions == ["transfer_human", "create_ticket"] and b3.get("agent_steps") == 0
            history = client.post(
                f"{args.base_url}/api/chat/sessions/{s3}/messages", timeout=10.0
            ).json()["messages"]
            ai = [message for message in history if message.get("role") == "ai"][-1]
            ok3b = [item.get("action") for item in (ai.get("actions") or [])] == actions

            ticket = client.post(
                f"{args.base_url}/api/actions/create-ticket",
                json={"session_id": s3, "description": "我要投诉", "ticket_type": "投诉"},
                timeout=30.0,
            )
            ticket.raise_for_status()
            ticket_body = ticket.json()
            row = _ticket_row(ticket_body["ticket_no"])
            ok3c = row is not None and row["ticket_type"] == "投诉"
            checks.append(("3 投诉两个动作 + 建单才写 tickets", ok3a and ok3b and ok3c, ticket_body.get("ticket_no", "")))
            details.append(
                f"[3] session={s3} actions={actions} ticket={ticket_body.get('ticket_no')} "
                f"db={row}\n    history.actions={ai.get('actions')}"
            )

            # 4. 闲聊：固定话术 + 日志 intent_source=rule（零模型调用）
            s4 = f"ch05-acc-chitchat-{stamp}"
            b4 = _chat(client, args.base_url, s4, "你好")
            r4 = _last_turn(s4)
            ok4 = b4.get("content") == CHITCHAT_REPLY and r4.get("intent_source") == "rule"
            checks.append(("4 闲聊固定话术", ok4, f"intent_source={r4.get('intent_source')}"))
            details.append(f"[4] session={s4} content={str(b4.get('content'))[:100]} log.intent_source={r4.get('intent_source')}")

            # 5. 复杂问题：先查订单再查物流，agent_steps >= 2
            s5 = f"ch05-acc-complex-{stamp}"
            b5 = _chat(
                client,
                args.base_url,
                s5,
                "订单 1001 是什么状态？如果发货了，物流到哪了",
            )
            steps = int(b5.get("agent_steps") or 0)
            ok5 = steps >= 2 and {"query_order", "query_logistics"} <= set(_names(b5))
            checks.append(("5 复杂问题 ReAct 多步", ok5, f"steps={steps} tools={_names(b5)}"))
            details.append(
                f"[5] session={s5} agent_steps={steps} tools={_names(b5)}\n"
                f"    content={str(b5.get('content'))[:200]}"
            )
    except Exception as exc:  # noqa: BLE001 - 演示脚本要把失败原因打出来
        print(f"服务不可用或请求失败：{exc}", file=sys.stderr)
        return 2

    print("ch05 验收（后端可判定部分）")
    print("=" * 60)
    for name, ok, note in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {note}")
    print("=" * 60)
    print("\n".join(details))

    output = PROJECT_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "ch05 验收输出\n"
        + "=" * 60
        + "\n"
        + "\n".join(f"{'PASS' if ok else 'FAIL'}  {name}  {note}" for name, ok, note in checks)
        + "\n\n"
        + "\n".join(details)
        + "\n\n注：转人工是纯前端模拟，需在浏览器点按钮验收；本脚本校验后端没有自动动作。\n",
        encoding="utf-8",
    )
    print(f"\n已写入 {output}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
