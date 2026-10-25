"""ch10：主应用侧的两个接口 —— 主题分布、手动批跑。

主应用**不 import torch**：这里只发 HTTP 给 sidecar。这正是方案 B 的全部意义 ——
对话主链路的进程不必背 2.7GB 的轮子，测试也不用等模型加载。

为什么批跑是同步返回而不是「202 + 轮询」：一批最多 100 条，sidecar 上是几秒的
事；把它做成异步任务要再加一张任务表和一页轮询 UI，收益不成比例。真到了几分钟
的量级，先调小 ``TOPIC_BATCH_LIMIT`` 分几批跑即可。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Sequence

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db.repositories import Repositories
from app.topics.batch import TopicServiceUnavailable, run_batch
from app.topics.infer import TopicPrediction
from app.topics.taxonomy import CATEGORY_NAMES

LOGGER = logging.getLogger("app.topics.routes")

#: 与 ``app/topics/service.py`` 的 ``MAX_BATCH`` 对齐：sidecar 单请求上限 64 条，
#: 这里按同样大小切块。刻意不 import 那个模块 —— 它有模块级 ``app`` 实例。
CHUNK_SIZE = 64


class BatchRequest(BaseModel):
    limit: int | None = Field(
        default=None,
        ge=1,
        le=500,
        description="本次最多归多少条；不传则用 TOPIC_BATCH_LIMIT。",
    )


def _error(status_code: int, type_: str, message: str, details: Any = None) -> JSONResponse:
    """与主应用一致的 ``{"error": {...}}`` 外壳（不能 import app.api，会循环）。"""
    body: dict[str, Any] = {"error": {"type": type_, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _settings_for(request: Request) -> Settings:
    """优先用注入的 settings（测试不读宿主机 .env）。"""
    injected = getattr(request.app.state, "settings", None)
    if injected is not None:
        return injected
    return get_settings()


async def _classify_via_http(
    texts: Sequence[str], settings: Settings
) -> Sequence[TopicPrediction]:
    """把文本切块发给 sidecar，拼回与输入同序的预测。

    **空输入不发请求**：sidecar 对空列表返回空结果，但白跑一次网络没有意义。
    """
    if not texts:
        return []
    url = f"{settings.topic_service_url.rstrip('/')}/classify"
    results: list[TopicPrediction] = []
    try:
        async with httpx.AsyncClient(timeout=settings.topic_service_timeout) as client:
            for start in range(0, len(texts), CHUNK_SIZE):
                chunk = list(texts[start : start + CHUNK_SIZE])
                response = await client.post(url, json={"texts": chunk})
                if response.status_code != 200:
                    raise TopicServiceUnavailable(
                        f"sidecar 返回 {response.status_code}：{response.text[:200]}"
                    )
                payload = response.json()
                results.extend(
                    TopicPrediction(
                        labels=list(item["labels"]),
                        scores=dict(item["scores"]),
                        fallback=bool(item.get("fallback", False)),
                    )
                    for item in payload["results"]
                )
    except httpx.HTTPError as exc:
        # ConnectError / ReadTimeout / 协议错误都归一类：sidecar 没起或没答完。
        raise TopicServiceUnavailable(f"连不上 sidecar（{type(exc).__name__}）：{exc}") from exc
    return results


def register_topic_routes(
    app: FastAPI, *, repositories_dependency: Callable[..., Awaitable[Repositories]]
) -> None:
    """挂上两个接口。

    ``repositories_dependency`` 由 ``app/api.py`` 传进来（它自己那份
    ``get_repositories``）。之所以不在这里 ``from app.api import get_repositories``：
    ``app.api`` 会在 ``create_app`` 里调本函数，模块级互引会成环。
    """

    @app.get("/api/topics/distribution")
    async def topic_distribution(
        request: Request,
        repositories: Repositories = Depends(repositories_dependency),
    ) -> JSONResponse:
        """主题分布看板的数据源。

        ``pending`` 与 ``min_pending`` 一起构成水位提示：后端只报数，**拦不拦由前端
        决定** —— 拦在后端会让调试时跑不了小批。
        """
        total = await repositories.topics.total_classified()
        counts = await repositories.topics.label_counts()
        multi = await repositories.topics.multi_label_count()
        pending = await repositories.topics.count_pending()
        last = await repositories.topics.last_classified_at()
        settings = _settings_for(request)
        return JSONResponse(
            status_code=200,
            content={
                "total": total,
                "by_label": [{"label": row.label, "count": row.count} for row in counts],
                # 17 类的完整清单（含计数为 0 的）：前端据此画满 17 行，而不是把
                # 类目表在 JS 里再抄一份 —— 抄一份就一定会和 taxonomy.py 漂移。
                "all_labels": list(CATEGORY_NAMES),
                # 分母为 0 时给 0.0 而不是 None：前端可以直接乘 100 画百分比。
                "multi_label_share": (multi / total) if total else 0.0,
                "pending": pending,
                "min_pending": settings.topic_batch_min_pending,
                "last_classified_at": _iso(last),
            },
        )

    @app.post("/api/topics/classify-batch")
    async def topic_classify_batch(
        payload: BatchRequest,
        request: Request,
        repositories: Repositories = Depends(repositories_dependency),
    ) -> JSONResponse:
        """手动归一批。sidecar 不可达时返 503 **并带上启动命令**。"""
        settings = _settings_for(request)
        limit = payload.limit or settings.topic_batch_limit

        async def classify(texts: Sequence[str]) -> Sequence[TopicPrediction]:
            return await _classify_via_http(texts, settings)

        try:
            outcome = await run_batch(
                repositories=repositories, classify=classify, limit=limit
            )
        except TopicServiceUnavailable as exc:
            LOGGER.warning("批跑失败（sidecar 不可用）：%s", exc)
            return _error(
                503,
                "topic_service_unavailable",
                f"{exc}；先启动 sidecar：uvicorn app.topics.service:app --port 8100",
            )
        return JSONResponse(
            status_code=200,
            content={
                "processed": outcome.processed,
                "fallback_other": outcome.fallback_other,
                "elapsed_ms": outcome.elapsed_ms,
                "pending_before": outcome.pending_before,
                "pending_after": outcome.pending_after,
            },
        )
