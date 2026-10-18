"""ch06 前端验收：用 CDP 驱动真实聊天页，走一遍订单卡片/续跑/退款单。

前置：uvicorn 跑在 8000；Chrome 以 ``--remote-debugging-port=9222`` 启动。
这是验收脚本，不是产品代码；截图写进 dev-notes/。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
CDP = "http://127.0.0.1:9222"


def pick_page() -> str:
    with urllib.request.urlopen(f"{CDP}/json/list") as response:
        pages = json.load(response)
    pages = [page for page in pages if page.get("type") == "page"]
    if not pages:
        raise RuntimeError("没有可用的 CDP page target")
    return pages[0]["webSocketDebuggerUrl"]


class Session:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.seq = 0
        self.errors: list[str] = []

    async def send(self, method: str, params: dict | None = None) -> dict:
        self.seq += 1
        message_id = self.seq
        await self.ws.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        while True:
            raw = await self.ws.recv()
            message = json.loads(raw)
            if message.get("id") == message_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result") or {}
            if message.get("method") == "Runtime.exceptionThrown":
                detail = (message.get("params") or {}).get("exceptionDetails") or {}
                self.errors.append(str(detail.get("text") or detail))
            if message.get("method") == "Runtime.consoleAPICalled":
                params = message.get("params") or {}
                if params.get("type") == "error":
                    self.errors.append(
                        " ".join(
                            str(item.get("value") or item.get("description") or "")
                            for item in params.get("args") or []
                        )
                    )

    async def js(self, expression: str):
        result = await self.send(
            "Runtime.evaluate",
            {
                "expression": expression,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        return (result.get("result") or {}).get("value")

    async def wait_for(self, expression: str, *, timeout: float = 60.0) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if await self.js(expression):
                return True
            await asyncio.sleep(0.4)
        return False

    async def screenshot(self, path: Path) -> None:
        result = await self.send("Page.captureScreenshot", {"format": "png"})
        path.write_bytes(base64.b64decode(result["data"]))
        print(f"screenshot: {path}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat/")
    args = parser.parse_args()

    async with websockets.connect(pick_page(), max_size=20 * 1024 * 1024) as ws:
        session = Session(ws)
        await session.send("Page.enable")
        await session.send("Runtime.enable")
        await session.send("Network.enable")
        await session.send("Network.setCacheDisabled", {"cacheDisabled": True})
        await session.send(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1200, "height": 900, "deviceScaleFactor": 1, "mobile": False},
        )
        await session.send("Page.navigate", {"url": args.url})
        assert await session.wait_for("!!document.getElementById('input')", timeout=20)
        print("page: loaded")
        await asyncio.sleep(0.8)

        async def ask(text: str) -> None:
            await session.js(
                f"""
                (() => {{
                  const input = document.getElementById('input');
                  input.value = {json.dumps(text, ensure_ascii=False)};
                  input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                  document.getElementById('send').click();
                  return true;
                }})()
                """
            )

        async def wait_done() -> None:
            assert await session.wait_for(
                """
                (() => {
                  const rows = document.querySelectorAll('.row.bot');
                  const last = rows[rows.length - 1];
                  return !!last
                    && (last.querySelector('.bubble-content')?.innerText || '').trim().length > 0
                    && !last.querySelector('.caret');
                })()
                """,
                timeout=120,
            ), "上一轮没有结束"

        # 第一段：多轮指代。先聊某张订单的物流，再问「那它能退吗」。
        await session.js(
            """
            (() => {
              sessionStorage.clear();
              document.getElementById('btn-clear').click();
              return true;
            })()
            """
        )
        await ask("订单 20240115001 的物流到哪了")
        await wait_done()
        await ask("那它能退吗")
        await wait_done()
        coref_note = await session.js(
            "Array.from(document.querySelectorAll('.flow-note')).map(n => n.textContent).find(t => t.includes('已补全')) || ''"
        )
        print("coref note:", coref_note)
        assert "已补全" in coref_note, "多轮指代没有显示补全结果"
        await session.screenshot(ROOT / "dev-notes" / "ch06-coref.png")

        # 第二段：新会话，输入一个没有订单号的退款问题。
        await session.js(
            """
            (() => {
              sessionStorage.clear();
              document.getElementById('btn-clear').click();
              return true;
            })()
            """
        )
        await ask("这个能退吗")
        assert await session.wait_for(
            "!!document.querySelector('.order-picker .order-card')", timeout=60
        ), "订单卡片没有出现"
        print("order picker: visible")
        picker_note = await session.js(
            "document.querySelector('.flow-note')?.textContent || ''"
        )
        print("coref note:", picker_note)
        await session.screenshot(ROOT / "dev-notes" / "ch06-order-picker.png")

        # 点第一张卡，继续流程。
        await session.js("document.querySelector('.order-card').click(); true")
        assert await session.wait_for(
            "!!document.querySelector('.refund-form')", timeout=90
        ), "退款表单没有出现"
        print("refund form: visible")
        form_text = await session.js(
            "document.querySelector('.refund-form')?.innerText || ''"
        )
        print("refund form text:", form_text.replace('\n', ' | ')[:200])
        await session.screenshot(ROOT / "dev-notes" / "ch06-refund-form.png")

        # 选固定原因并提交。
        await session.js(
            """
            (() => {
              const form = document.querySelector('.refund-form');
              const select = form.querySelector('select');
              select.value = '商品质量问题';
              select.dispatchEvent(new Event('change', { bubbles: true }));
              form.querySelector('button[type="submit"]').click();
              return select.value;
            })()
            """
        )
        assert await session.wait_for(
            "!!document.querySelector('.refund-success')", timeout=40
        ), "退款单没有提交成功"
        success = await session.js(
            "document.querySelector('.refund-success')?.textContent || ''"
        )
        print("refund success:", success)

        print("js errors:", session.errors)
        assert not session.errors, "页面出现 JS 异常"


if __name__ == "__main__":
    asyncio.run(main())
