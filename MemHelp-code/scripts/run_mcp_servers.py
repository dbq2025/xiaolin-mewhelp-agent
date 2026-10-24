"""同时起两个业务 MCP Server（各自独立进程，Streamable HTTP）。

用法：
    uv run python scripts/run_mcp_servers.py
    # 验收 3：多带一个演示工具启动物流 Server
    uv run python scripts/run_mcp_servers.py --logistics-extra

Ctrl+C 会一起收掉两个子进程。客服系统（uvicorn）不在这个脚本里，需要另起。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SERVERS = (
    ("logistics", "mcp_servers.logistics_server", 8101),
    ("aftersales", "mcp_servers.aftersales_server", 8102),
)


def _spawn(module: str, *, extra: bool = False) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env["MEMHELP_LOGISTICS_EXTRA"] = "1"
    return subprocess.Popen([sys.executable, "-m", module], cwd=ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="启动 ch08 的两个业务 MCP Server")
    parser.add_argument(
        "--logistics-extra",
        action="store_true",
        help="给物流 Server 多注册一个演示工具 estimate_delivery（验收 3 用）",
    )
    args = parser.parse_args()

    processes: dict[str, subprocess.Popen[bytes]] = {}
    try:
        for name, module, port in SERVERS:
            processes[name] = _spawn(
                module, extra=args.logistics_extra and name == "logistics"
            )
            print(f"[mcp] {name} 已启动：http://127.0.0.1:{port}/mcp", flush=True)
        while True:
            for name, process in processes.items():
                code = process.poll()
                if code is not None:
                    print(f"[mcp] {name} 退出（code={code}），一起收掉其余进程。", flush=True)
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[mcp] 收到 Ctrl+C，正在退出…", flush=True)
        return 0
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
