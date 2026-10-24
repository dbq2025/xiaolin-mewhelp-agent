"""ch08 功能演示：注册即用、MCP 接入、工单确认流、超时与审计。

用法（在 ``MemHelp-code/`` 下）：
    uv run python scripts/demo_ch08.py                 # 不依赖外部进程的三段
    uv run python scripts/run_mcp_servers.py           # 另开终端起两个 MCP Server
    uv run python scripts/demo_ch08.py --part mcp       # 问物流轨迹（验收 2）
    uv run python scripts/demo_ch08.py --part hot-add   # 热加工具（验收 3，自己起停物流 Server）

每段都把关键输出打到 stdout，方便原样贴进验收记录。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.agent.checkpoint import memory_checkpointer  # noqa: E402
from app.agent.react import run_react_agent  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.models import Ticket  # noqa: E402
from app.db.repositories import Repositories  # noqa: E402
from app.tools.contract import ToolSpec  # noqa: E402
from app.tools.executor import execute_tool  # noqa: E402
from app.tools.registry import build_registry  # noqa: E402
from tests.conftest import EmptyRetriever  # noqa: E402
from tests.fakes_agent import ScriptedChatModel, text_turn, tool_turn  # noqa: E402


def _settings(**overrides) -> Settings:
    values = dict(
        openai_base_url="https://gw.example.com/v1",
        openai_api_key="sk-demo",
        openai_model="demo-model",
        context_token_budget=2000,
        turn_log_path="",
    )
    values.update(overrides)
    return Settings(**values)


def _repositories() -> tuple[Repositories, object]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    return Repositories.from_session(session), session


def _hr(title: str) -> None:
    print(f"\n=== {title} ===")


async def _print_audit(repositories: Repositories, limit: int = 8) -> None:
    rows = await repositories.audit.list_recent(limit=limit)
    for row in rows:
        print(
            f"  audit id={row.id} tool={row.tool_name} source={row.tool_source}"
            f" status={row.status} retry={row.retry_count}"
            f" duration_ms={row.duration_ms} error={row.error_message or '-'}"
        )


# ---------------------------------------------------------------------------
# 验收 1：只做注册动作，Agent 就能用上新工具
# ---------------------------------------------------------------------------


async def demo_registry() -> None:
    _hr("验收 1：新工具只注册、不改核心代码")
    repositories, _ = _repositories()

    @tool
    def hello_tool(name: str) -> str:
        """向指定的人问好（ch08 演示工具）。"""
        return json.dumps({"hello": name, "source": "ch08"}, ensure_ascii=False)

    spec = ToolSpec(
        name="hello_tool",
        description="向指定的人问好（ch08 演示工具）。",
        args_schema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "要问候的人"}},
            "required": ["name"],
        },
        source="builtin",
        mcp_server=None,
        read_only=True,
        tool=hello_tool,
    )
    registry = build_registry(repositories)
    registry.register(spec)  # 唯一的一步：登记

    result = await execute_tool(
        "hello_tool",
        {"name": "ch08"},
        registry=registry,
        repositories=repositories,
        session_id="demo-registry",
        tool_call_id="call_hello",
        timeout=2.0,
    )
    print(f"execute_tool -> status={result.status} data={result.data}")

    model = ScriptedChatModel(
        script=[
            tool_turn("hello_tool", {"name": "Agent"}, call_id="call_agent_hello"),
            text_turn("已经替你打过招呼了。"),
        ]
    )
    outcome = await run_react_agent(
        model=model,
        messages=[HumanMessage(content="跟 ch08 打个招呼")],
        repositories=repositories,
        retriever=None,
        include_faq=False,
        registry=registry,
    )
    names = [str(trace.get("tool") or trace.get("name")) for trace in outcome.tool_traces]
    print(f"Agent 回答：{outcome.answer}")
    print(f"Agent 调到的工具：{names}")
    await _print_audit(repositories)


# ---------------------------------------------------------------------------
# 验收 2：两个 MCP Server 独立进程，Agent 通过 MCP 查数据
# ---------------------------------------------------------------------------


async def demo_mcp() -> None:
    _hr("验收 2：通过 MCP 查物流 / 在保 / 退货进度")
    from app.tools.mcp_client import (
        McpToolProvider,
        connections_from_settings,
        readonly_servers_from_settings,
    )

    settings = _settings(mcp_enabled=True)
    if not (_wait_port(8101, timeout=1.0) and _wait_port(8102, timeout=1.0)):
        print("（两个 MCP Server 没起：请先运行 `uv run python scripts/run_mcp_servers.py`）")
        return
    provider = McpToolProvider(
        connections_from_settings(settings),
        read_only_servers=readonly_servers_from_settings(settings),
        timeout=settings.mcp_timeout_seconds,
    )
    await provider.discover()
    specs = provider.cached_specs()
    print(f"发现工具：{[spec.name for spec in specs]}")
    if not specs:
        print("（没有发现工具：请先运行 `uv run python scripts/run_mcp_servers.py`）")
        return

    repositories, _ = _repositories()
    registry = build_registry(repositories, mcp_provider=provider, enable_mcp=True)
    await registry.refresh_mcp()

    for name, args in (
        ("track_shipment", {"order_id": "1001"}),
        ("check_warranty", {"order_id": "1002"}),
        ("check_return_progress", {"order_id": "1003"}),
    ):
        if registry.get(name) is None:
            print(f"{name}: 本 Server 未提供")
            continue
        result = await execute_tool(
            name,
            args,
            registry=registry,
            repositories=repositories,
            session_id="demo-mcp",
            timeout=5.0,
        )
        print(f"{name} -> status={result.status} data={result.data}")
    await _print_audit(repositories)


# ---------------------------------------------------------------------------
# 验收 3：MCP Server 侧加工具、只重启该 Server
# ---------------------------------------------------------------------------


def _wait_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(0.4)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _spawn_logistics(*, extra: bool) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env["MEMHELP_LOGISTICS_EXTRA"] = "1"
    else:
        env.pop("MEMHELP_LOGISTICS_EXTRA", None)
    return subprocess.Popen(
        [sys.executable, "-m", "mcp_servers.logistics_server"], cwd=ROOT, env=env
    )


def _stop(process: subprocess.Popen[bytes]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


async def demo_hot_add() -> None:
    _hr("验收 3：只重启物流 Server，客服系统侧代码与进程不动")
    from app.tools.mcp_client import McpToolProvider

    connection = {
        "logistics": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8101/mcp",
            "timeout": 5.0,
        }
    }
    provider = McpToolProvider(connection, read_only_servers={"logistics"}, timeout=5.0)

    first = _spawn_logistics(extra=False)
    try:
        if not _wait_port(8101):
            print("物流 Server 没起来（端口 8101）。")
            return
        await provider.discover()
        before = [spec.name for spec in provider.cached_specs()]
        print(f"重启前工具：{before}")
    finally:
        _stop(first)

    second = _spawn_logistics(extra=True)
    try:
        if not _wait_port(8101):
            print("带演示工具的物流 Server 没起来。")
            return
        # 同一个 provider、同一个进程：只重新发现，不重启客服系统。
        await provider.discover()
        after = [spec.name for spec in provider.cached_specs()]
        print(f"只重启物流 Server 后工具：{after}")
    finally:
        _stop(second)

    print(f"结论：新工具 {'estimate_delivery' in after}（客户端进程未重启、未改代码）")


# ---------------------------------------------------------------------------
# 验收 4 / 5：工单预览 -> 确认落库 / 取消拒绝
# ---------------------------------------------------------------------------


def _intent() -> object:
    return text_turn(
        json.dumps({"intent": "logistics", "confidence": 0.9}, ensure_ascii=False)
    )


def _start_preview(client, model, session: str, args: dict) -> dict:
    model.script.extend(
        [_intent(), tool_turn("create_ticket", args, call_id="call_ticket")]
    )
    with client.stream(
        "POST",
        "/api/chat/completions",
        json={"session_id": session, "message": "帮我建个工单"},
    ) as resp:
        text = "".join(resp.iter_text())
    frames = _frames(text)
    awaiting = next(data for name, data in frames if name == "awaiting_input")
    return awaiting


def _frames(text: str) -> list[tuple[str | None, dict]]:
    frames: list[tuple[str | None, dict]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        frames.append((name, json.loads(data) if data else {}))
    return frames


def _tokens(text: str) -> str:
    return "".join(
        str(data.get("delta") or "")
        for name, data in _frames(text)
        if name == "token"
    )


async def demo_ticket() -> None:
    _hr("验收 4 / 5：工单预览卡片的确认与取消")
    from fastapi.testclient import TestClient

    from app import main as app_main

    repositories, session = _repositories()
    settings = _settings(mcp_enabled=False)
    model = ScriptedChatModel()
    app = app_main.create_app(
        model=model,
        settings=settings,
        repositories=repositories,
        db_ping=lambda: True,
        retriever=EmptyRetriever(),
        rag_warmup=False,
        checkpointer=memory_checkpointer(),
    )
    client = TestClient(app)

    session_id = "demo-ticket-confirm"
    args = {
        "description": "耳机左声道没有声音",
        "ticket_type": "售后",
        "conversation_id": session_id,
    }
    awaiting = _start_preview(client, model, session_id, args)
    print(f"暂停并出示预览：kind={awaiting['kind']} preview={awaiting['preview']}")
    with client.stream(
        "POST",
        "/api/chat/ticket/confirm",
        json={"session_id": session_id, "confirmed": True},
    ) as resp:
        reply = _tokens("".join(resp.iter_text()))
    tickets = session.scalars(select(Ticket)).all()
    print(f"确认后回复：{reply}")
    print(f"tickets 落库：{[t.ticket_no for t in tickets]}")
    rows = await repositories.audit.list_recent(5)
    print(
        "create_ticket 审计："
        f"{[(row.status, row.retry_count) for row in rows if row.tool_name == 'create_ticket']}"
    )

    cancel_model = ScriptedChatModel()
    cancel_app = app_main.create_app(
        model=cancel_model,
        settings=settings,
        repositories=repositories,
        db_ping=lambda: True,
        retriever=EmptyRetriever(),
        rag_warmup=False,
        checkpointer=memory_checkpointer(),
    )
    cancel_client = TestClient(cancel_app)
    cancel_session = "demo-ticket-cancel"
    cancel_args = {
        "description": "包装破损想退货",
        "ticket_type": "投诉",
        "conversation_id": cancel_session,
    }
    _start_preview(cancel_client, cancel_model, cancel_session, cancel_args)
    with cancel_client.stream(
        "POST",
        "/api/chat/ticket/confirm",
        json={"session_id": cancel_session, "confirmed": False},
    ) as resp:
        cancel_reply = _tokens("".join(resp.iter_text()))
    tickets_after = session.scalars(select(Ticket)).all()
    rows = await repositories.audit.list_recent(5)
    denied = [
        row
        for row in rows
        if row.tool_name == "create_ticket" and row.status == "权限拒绝"
    ]
    print(f"取消后回复：{cancel_reply}")
    print(f"tickets 数量仍为：{len(tickets_after)}")
    print(f"取消审计：{[(row.status, row.error_message) for row in denied]}")


# ---------------------------------------------------------------------------
# 验收 6：只读超时会重试；写超时不自动重试
# ---------------------------------------------------------------------------


async def demo_timeout() -> None:
    _hr("验收 6：超时、重试与审计")
    repositories, _ = _repositories()

    @tool
    async def slow_read(query: str) -> str:
        """一个故意很慢的只读工具。"""
        await asyncio.sleep(1.0)
        return json.dumps({"query": query}, ensure_ascii=False)

    @tool
    async def slow_write(description: str) -> str:
        """一个故意很慢的写工具。"""
        await asyncio.sleep(1.0)
        return json.dumps({"description": description}, ensure_ascii=False)

    registry = build_registry(repositories)
    registry.register(
        ToolSpec(
            name="slow_read",
            description="故意很慢的只读工具（超时演示）。",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            source="builtin",
            mcp_server=None,
            read_only=True,
            tool=slow_read,
        )
    )
    registry.register(
        ToolSpec(
            name="slow_write",
            description="故意很慢的写工具（超时演示）。",
            args_schema={
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            },
            source="builtin",
            mcp_server=None,
            read_only=False,
            tool=slow_write,
        )
    )

    read_result = await execute_tool(
        "slow_read",
        {"query": "物流到哪了"},
        registry=registry,
        repositories=repositories,
        session_id="demo-timeout",
        tool_call_id="call_slow_read",
        timeout=0.1,
        max_retries=1,
    )
    print(
        f"只读超时：status={read_result.status} attempts={read_result.attempts}"
        f" error={read_result.error}"
    )

    write_result = await execute_tool(
        "slow_write",
        {"description": "写操作超时演示"},
        registry=registry,
        repositories=repositories,
        session_id="demo-timeout",
        tool_call_id="call_slow_write",
        timeout=0.1,
        max_retries=1,
        confirmed=True,
    )
    print(
        f"写超时：status={write_result.status} attempts={write_result.attempts}"
        f" error={write_result.error}"
    )
    await _print_audit(repositories)


async def main_async(part: str) -> None:
    if part in ("all", "registry"):
        await demo_registry()
    if part in ("all", "mcp"):
        await demo_mcp()
    if part == "hot-add":
        await demo_hot_add()
    if part in ("all", "ticket"):
        await demo_ticket()
    if part in ("all", "timeout"):
        await demo_timeout()


def main() -> int:
    parser = argparse.ArgumentParser(description="ch08 演示脚本")
    parser.add_argument(
        "--part",
        choices=("all", "registry", "mcp", "hot-add", "ticket", "timeout"),
        default="all",
        help="默认 all=registry+ticket+timeout；mcp 需先起两个 Server",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.part))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
