"""ch10：批跑编排 —— 把一批「还没归类」的问题归完并落库。

**全项目只有这一份批跑实现**：HTTP 按钮和 CLI 脚本都调它。批跑最容易出的两类
问题（水位算错、失败写了半批）都只在这里修一次。

sidecar 通过 ``classify`` 注入而不是在这里 import httpx：编排逻辑因此不绑定
传输方式，测试也不用起一个服务。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from app.db.repositories import Repositories
from app.topics.infer import TopicPrediction
from app.topics.taxonomy import validate_labels

#: 「一批文本 → 一批预测」。路由层传 HTTP 实现，测试传假函数。
Classifier = Callable[[Sequence[str]], Awaitable[Sequence[TopicPrediction]]]


class TopicServiceUnavailable(RuntimeError):
    """sidecar 连不上、超时、或返回条数不对。

    这几种情况**一律不写库**：写一半等于在统计里掺进一批「看起来归过类、
    其实没归」的题，主题分布会静默偏低，而且查不出来。
    """


@dataclass(frozen=True)
class BatchOutcome:
    """一次批跑的结果。字段同时被 HTTP 响应与 CLI 打印使用。"""

    processed: int
    fallback_other: int
    elapsed_ms: int
    pending_before: int
    pending_after: int


async def run_batch(
    *,
    repositories: Repositories,
    classify: Classifier,
    limit: int,
    now: datetime | None = None,
) -> BatchOutcome:
    """归类至多 ``limit`` 条 pending 问题。

    ``pending_before`` / ``pending_after`` 是**同一批的前后水位**：前端「还剩多少
    没归」直接读它，不必自己再算一次（算两次就会有两份口径）。
    """

    started = time.perf_counter()
    pending_before = await repositories.topics.count_pending()
    pending = await repositories.topics.list_pending(limit)
    if not pending:
        return BatchOutcome(
            processed=0,
            fallback_other=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            pending_before=pending_before,
            pending_after=pending_before,
        )

    texts = [row.raw_question for row in pending]
    predictions = await classify(texts)
    if len(predictions) != len(texts):
        raise TopicServiceUnavailable(
            f"sidecar 返回 {len(predictions)} 条结果，但请求了 {len(texts)} 条："
            "条数对不上就不落库（错位一格等于整批标签全串行）"
        )

    rows: list[tuple[int, list[str]]] = []
    for row, prediction in zip(pending, predictions):
        # 落库前再过一遍类目表：「其他」独占、未知标签这类问题必须在写库前炸，
        # 而不是等前端画图时才发现有个「尺碼」。
        rows.append((row.id, validate_labels(prediction.labels)))

    classified_at = now or datetime.now()
    await repositories.topics.upsert_many(rows, classified_at=classified_at)
    pending_after = await repositories.topics.count_pending()
    return BatchOutcome(
        processed=len(rows),
        fallback_other=sum(1 for p in predictions if p.fallback),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        pending_before=pending_before,
        pending_after=pending_after,
    )
