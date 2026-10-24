"""``eval_runs`` 的周期调度。

启动时立刻返回一个 ``asyncio.Task``，第一轮在一个间隔之后才跑（不在启动路径上
抢数据库/模型资源）；任何一轮失败只记日志，不终止循环，也不影响聊天。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import Settings, get_settings
from app.db.repositories import Repositories
from app.eval_runs import run_evaluation

logger = logging.getLogger(__name__)


def _settings_of(app: Any) -> Settings:
    return getattr(app.state, "settings", None) or get_settings()


def _open_repositories(app: Any):
    """返回 ``(repositories, session_or_none)``：测试注入的仓库不归这里关。"""
    injected = getattr(app.state, "repositories", None)
    if injected is not None:
        return injected, None
    database = getattr(app.state, "_database", None)
    if database is None:
        raise RuntimeError("数据库尚未初始化，跳过本轮评估")
    session = database[0]()
    return Repositories.from_session(session), session


async def _run_once(app: Any, settings: Settings) -> int:
    repositories, session = _open_repositories(app)
    try:
        return await run_evaluation(
            repositories=repositories, trigger="定时", settings=settings
        )
    finally:
        if session is not None:
            session.close()


async def _loop(app: Any, settings: Settings) -> None:
    interval = max(1.0, float(settings.eval_schedule_interval_hours) * 3600.0)
    logger.info("ch09 评估调度已启动：每 %.1f 小时一轮", interval / 3600.0)
    while True:
        await asyncio.sleep(interval)
        try:
            await _run_once(app, settings)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - 定时任务失败不能拖垮进程
            logger.exception("ch09 定时评估失败，等待下一轮")


def start_scheduler(app: Any) -> asyncio.Task | None:
    """按配置启动调度；关闭时返回 ``None``。"""
    settings = _settings_of(app)
    if not settings.eval_schedule_enabled:
        logger.info("ch09 评估调度未启用（EVAL_SCHEDULE_ENABLED=false）")
        return None
    task = asyncio.create_task(_loop(app, settings), name="ch09-eval-scheduler")
    app.state.eval_scheduler = task
    return task


async def stop_scheduler(task: asyncio.Task | None) -> None:
    """取消调度并等它退出；``None`` 或已结束的任务是安全的。"""
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:  # noqa: BLE001 - 退出路径不该再抛
        logger.debug("关闭评估调度时出错（已忽略）", exc_info=True)
