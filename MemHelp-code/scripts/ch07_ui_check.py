"""ch07 前端验收：用 CDP 驱动真实聊天页，走一遍侧栏的多会话、切换、回载、续聊。

**为什么不用 httpx 直打接口**（plan 里原话是 httpx）：接口通不通已经由
``tests/test_ch07_api.py`` 的 8 条用例钉住了。验收第 5 条要的是
「侧栏能开多个会话对照、切回旧会话时历史完整回载、能接着聊」，
这三件事只发生在 DOM 与两个 id 的联动里 —— http 客户端一个都看不见。
所以这里照 ch06 的做法用 CDP 驱动真页面，再用 httpx 回查库里的条数，
两边对上才算过。

**不按「侧栏有几项」断言**：侧栏走 ``limit=50``，接的是真库，跑几次就满 50 了。
所以全部判定都按「带唯一标记的那句话在不在」来做 —— 数量会饱和，标记不会。

前置：
  1. uvicorn 跑在 ``--api`` 指的那个地址（默认 http://127.0.0.1:8000）；
  2. Chrome 以 ``--remote-debugging-port=9222`` 启动。

这是验收脚本，不是产品代码；截图写进 dev-notes/。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import urllib.request
import uuid
from pathlib import Path

import httpx
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
    """薄薄一层 CDP 封装：只做 evaluate / wait_for / screenshot。"""

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


async def ask(session: Session, text: str) -> None:
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


async def wait_done(session: Session) -> None:
    """等最后一条客服气泡收尾（有字了、光标没了）。"""
    ok = await session.wait_for(
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
    )
    assert ok, "上一轮没有结束"


BOT_BUBBLES = "document.querySelectorAll('#messages .row.bot').length"
USER_BUBBLES = "document.querySelectorAll('#messages .row.user').length"
SIDEBAR_PREVIEWS = (
    "Array.from(document.querySelectorAll('#sidebar-list .sidebar-item'))"
    ".map(n => (n.querySelector('.sidebar-preview')?.textContent || '').trim())"
)


def _item_for(marker: str) -> str:
    """定位侧栏里 preview 含 marker 的那一项（用 find，不用下标）。"""
    return (
        "Array.from(document.querySelectorAll('#sidebar-list .sidebar-item'))"
        f".find(n => (n.querySelector('.sidebar-preview')?.textContent || '').includes({json.dumps(marker)}))"
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat/")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    # 每次跑两个唯一标记（A/B 各一），真库上反复跑也不会互撞。
    # 两个会话**必须用不同的标记**：共用一个标记时 ``find`` 命中的是第一项，
    # 「点回会话 A」实际点到了 B，回载正文自然对不上。
    marker_a = uuid.uuid4().hex[:6]
    marker_b = uuid.uuid4().hex[:6]
    qa = f"订单 20240115001 的物流到哪了 #{marker_a}"
    qb = f"耳机怎么退货 #{marker_b}"
    follow_up = f"那能改地址吗 #{marker_a}"
    print(f"本轮标记：A={marker_a} B={marker_b}")

    async with websockets.connect(pick_page(), max_size=20 * 1024 * 1024) as ws:
        session = Session(ws)
        await session.send("Page.enable")
        await session.send("Runtime.enable")
        await session.send("Network.enable")
        await session.send("Network.setCacheDisabled", {"cacheDisabled": True})
        await session.send(
            "Emulation.setDeviceMetricsOverride",
            {
                "width": 1200,
                "height": 900,
                "deviceScaleFactor": 1,
                "mobile": False,
            },
        )
        await session.js("(() => { sessionStorage.clear(); return true; })()")
        await session.send("Page.navigate", {"url": args.url})
        assert await session.wait_for(
            "!!document.getElementById('input')", timeout=20
        ), "聊天页没加载出来"
        assert await session.wait_for(
            "!!document.getElementById('sidebar-list')", timeout=20
        ), "侧栏没有渲染（index.html 里缺 #sidebar-list？）"
        await asyncio.sleep(1.0)
        print("page: loaded（侧栏存在）")

        # ---- 会话 A ----
        await session.js("document.getElementById('btn-new-chat').click(); true")
        await asyncio.sleep(0.3)
        await ask(session, qa)
        await wait_done(session)
        assert await session.wait_for(f"!!({_item_for(marker_a)})", timeout=40), (
            "第一个会话没有出现在侧栏"
        )
        print("会话 A 已在侧栏")

        # ---- 会话 B（新对话只开新的，A 必须还在）----
        await session.js("document.getElementById('btn-new-chat').click(); true")
        await asyncio.sleep(0.3)
        assert await session.js("document.querySelectorAll('#messages .row').length") == 0, (
            "「新对话」没有清空视图"
        )
        assert await session.js(f"!!({_item_for(marker_a)})"), "「新对话」把旧会话弄丢了"
        await ask(session, qb)
        await wait_done(session)
        assert await session.wait_for(
            f"!!({_item_for(marker_a)}) && !!({_item_for(marker_b)})",
            timeout=40,
        ), "两个会话没有同时出现在侧栏"
        previews = await session.js(SIDEBAR_PREVIEWS)
        print(f"侧栏前 3 项：{previews[:3]}")
        assert previews[0] == qb, f"最新会话不在最前：{previews[:2]}"
        print("两个会话都在侧栏，新在前")

        await session.screenshot(ROOT / "dev-notes" / "ch07-sidebar.png")

        # ---- 点回会话 A：历史要完整回载 ----
        await session.js(f"({_item_for(marker_a)}).click(); true")
        assert await session.wait_for(
            f"({BOT_BUBBLES}) >= 1 && ({USER_BUBBLES}) >= 1", timeout=40
        ), "切回旧会话后历史没有回载"
        users = await session.js(USER_BUBBLES)
        bots = await session.js(BOT_BUBBLES)
        loaded_text = await session.js(
            "Array.from(document.querySelectorAll('#messages .row.user .bubble-content'))"
            ".map(n => n.innerText.trim()).join(' | ')"
        )
        print(f"回载会话 A：user {users} 条 / bot {bots} 条 → {loaded_text[:60]}")
        assert users == 1, f"会话 A 只有一句用户消息，回载到 {users} 条"
        assert qa[:20] in loaded_text, "回载的正文不是会话 A 的那句"

        active = await session.js(
            "document.querySelector('#sidebar-list .sidebar-item.is-active')"
            "?.querySelector('.sidebar-preview')?.textContent || ''"
        )
        print(f"高亮项：{active}")
        assert marker_a in active and "20240115001" in active, "切过去的会话没有高亮"

        conversation_id = await session.js(
            "document.querySelector('#sidebar-list .sidebar-item.is-active')"
            "?.dataset.conversationId || ''"
        )
        assert conversation_id, "侧栏项上没有 data-conversation-id"

        # ---- 续聊：这一句必须落进会话 A，不能串到 B ----
        await ask(session, follow_up)
        await wait_done(session)
        await asyncio.sleep(1.5)  # 让侧栏刷新把 message_count 拉新

        async with httpx.AsyncClient(base_url=args.api, timeout=20) as client:
            listing = (await client.get("/api/conversations")).json()["conversations"]
            ours = [row for row in listing if str(row["id"]) == conversation_id]
            assert ours, f"回查不到会话 {conversation_id}"

            async def contents(cid: str) -> list[str]:
                rows = (
                    await client.get(f"/api/conversations/{cid}/messages")
                ).json()["messages"]
                return [str(row.get("content") or "") for row in rows]

            target_rows = await contents(conversation_id)
            # 不按「条数」断言：``message_count`` 是**消息行**，一轮里除了
            # human / ai 还有工具调用与工具结果行（实测一轮 6 行）。
            # 要验的是「这句话落进了哪个会话」，所以直接找那句话本身。
            print(f"会话 {conversation_id} 共 {len(target_rows)} 行")
            assert any(follow_up in text for text in target_rows), (
                f"续聊那句话没有落进被点开的会话 {conversation_id}"
            )
            assert sum(1 for text in target_rows if qa[:20] in text) == 1, (
                "会话 A 的首问丢了或重复了"
            )
            same_marker = [
                row
                for row in listing
                if row["id"] != int(conversation_id)
                and marker_b in (row["preview"] or "")
            ]
            assert same_marker, "同批的会话 B 没找到"
            for other in same_marker:
                other_rows = await contents(str(other["id"]))
                assert not any(follow_up in text for text in other_rows), (
                    f"续聊串进了会话 {other['id']}（session 串台，Review Focus 5）"
                )
                print(f"会话 {other['id']} 共 {len(other_rows)} 行，未串入续聊 ✓")
            # 角色序列只校验「用户说的在这一轮里、客服答过」，不假设行数
            roles = [
                row["role"]
                for row in (
                    await client.get(f"/api/conversations/{conversation_id}/messages")
                ).json()["messages"]
            ]
            print(f"会话 A 角色序列：{roles}")
            assert roles[0] == "human", f"会话首行不是用户消息：{roles[:3]}"
            assert roles.count("human") == 2, f"用户消息应恰好 2 条：{roles}"
            assert "ai" in roles, f"没有客服答复：{roles}"

        # ---- 侧栏加载失败要静默降级：拦掉接口，再刷新，聊天必须仍可用 ----
        await session.send("Network.enable")
        await session.send("Network.setBlockedURLs", {"urls": ["*/api/conversations*"]})
        await session.send("Page.reload")
        assert await session.wait_for(
            "!!document.getElementById('sidebar-list')", timeout=20
        ), "侧栏接口被拦之后页面没起来"
        await asyncio.sleep(1.2)
        blocked_items = await session.js(
            "document.querySelectorAll('#sidebar-list .sidebar-item').length"
        ) or 0
        print(f"侧栏接口被拦：侧栏 {blocked_items} 项（应为 0）")
        assert blocked_items == 0, "接口失败时侧栏没有静默清空"
        assert await session.js("!!document.getElementById('input')"), "侧栏失败拖垮了聊天"
        # 拦着接口也要能聊一句
        await session.js("document.getElementById('btn-new-chat').click(); true")
        await asyncio.sleep(0.3)
        await ask(session, "侧栏挂了还能聊吗")
        await wait_done(session)
        print("拦着侧栏接口仍然聊成了一句")
        await session.send("Network.setBlockedURLs", {"urls": []})

        print("js errors:", session.errors)
        assert not session.errors, "页面出现 JS 异常"
        await session.screenshot(ROOT / "dev-notes" / "ch07-sidebar-after.png")
        print(
            "\n✓ 侧栏多会话 / 切换回载 / 续聊不串台 / 新对话不丢旧会话 / 接口失败静默降级"
        )
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
