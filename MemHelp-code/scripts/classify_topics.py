"""ch10：手动归一批主题（CLI）。

为什么走主应用的 HTTP 接口而不是直接调 ``run_batch()``：批跑要写库，库连接与
配置都在主应用那侧；脚本自己在进程里再搭一套 engine/session 只会多一份口径。
**批跑实现只有一份**（``app/topics/batch.py``），这里只负责触发。

用法::

    # 先把 sidecar 和主应用起起来（见 README / dev-notes）
    python scripts/classify_topics.py --limit 27

退出码：0 = 归完了（可能 0 条）；1 = 请求失败；2 = sidecar 没起（提示启动命令）。
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="手动触发一次主题批跑（调用主应用接口）")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="本次最多归多少条；不传则用服务端 TOPIC_BATCH_LIMIT。",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"主应用地址，默认 {DEFAULT_BASE_URL}。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="整个请求的超时（秒）。一批 100 条通常几秒，留足余量给冷启动。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = f"{args.base_url.rstrip('/')}/api/topics/classify-batch"
    payload = {"limit": args.limit} if args.limit is not None else {}
    try:
        response = httpx.post(url, json=payload, timeout=args.timeout)
    except httpx.HTTPError as exc:
        print(f"请求失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"主应用起了吗？{args.base_url}", file=sys.stderr)
        return 1

    if response.status_code == 503:
        message = response.json().get("error", {}).get("message", response.text)
        print(f"sidecar 不可用：{message}", file=sys.stderr)
        return 2
    if response.status_code != 200:
        print(f"HTTP {response.status_code}: {response.text}", file=sys.stderr)
        return 1

    body = response.json()
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print(
        "本次归了 {processed} 条，兜底「其他」{fallback_other} 条，"
        "耗时 {elapsed_ms}ms；待归类 {pending_before} → {pending_after}".format(**body)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
