"""ch10：旁路推理 sidecar —— 无状态、不连库、只做「文本 → 标签」。

**为什么单独一个进程**（spec 方案 B）：

- 主应用零 ML 依赖：不装 torch 也能起，测试不用等 2.7GB 轮子；
- 模型加载要几秒、显存常驻，不该和对话主链路抢资源；
- **实时对话不调它**：攒够一批由后台按钮或脚本触发，归一次类写一次库。

它自己不碰数据库：写 ``topic_classifications`` 是主应用那份 ``run_batch()``
的事。sidecar 只回答「这批文本各是什么主题」，于是它可以被随意重启、扩容，
也不会和主应用抢连接池。

启动即自检（见 ``lifespan``）：类目表哈希对不上、阈值缺键、权重加载失败，
一律**拒绝启动**。半坏的 sidecar 比没起的 sidecar 更糟 —— 后者主应用会返 503，
前者会静默把标签错位地写进库。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.topics.infer import TopicClassifier
from app.topics.taxonomy import taxonomy_hash

LOGGER = logging.getLogger("app.topics.service")

#: 单次请求的文本条数上限。批跑端按这个值切块 —— 64 条 × 64 token 在 8GB 显存上很宽裕，
#: 再大就要开始担心 OOM 和请求超时了。
MAX_BATCH = 64

#: 项目根（sidecar 可能从任意 cwd 启动，默认路径必须锚在仓库上）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 默认模型目录（可用 ``TOPIC_MODEL_DIR`` 或 ``Settings.topic_model_dir`` 覆盖）。
#: 指向 ``latest`` 软拷贝：训练产物落在 ``models/topics/<run_id>/``，服务不裸指 run。
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "topics" / "latest"


def _configured_model_dir() -> Path:
    """没有显式参数/环境变量时，读 ``Settings.topic_model_dir``（相对项目根）。

    sidecar 只做推理，不需要 OpenAI 凭据；配置不完整时退回默认目录，别为了读一个
    路径把服务卡死在启动阶段。
    """

    override = os.environ.get("TOPIC_MODEL_DIR")
    if override:
        return Path(override)
    try:
        from app.config import get_settings

        configured = Path(get_settings().topic_model_dir)
    except Exception:  # noqa: BLE001 - 配置缺失/非法都不该拦住推理服务
        return DEFAULT_MODEL_DIR
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=MAX_BATCH)


def create_app(
    *, classifier: TopicClassifier | None = None, model_dir: Path | None = None
) -> FastAPI:
    """造一个 sidecar。``classifier`` 只给测试注入，线上走 ``model_dir``。"""

    model_path = Path(model_dir) if model_dir is not None else _configured_model_dir()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        active = classifier
        if active is None:
            LOGGER.info("加载主题分类模型：%s", model_path)
            # 加载是同步阻塞（几秒），丢进线程池，别卡住事件循环的启动钩子。
            active = await run_in_threadpool(TopicClassifier.load, model_path)
        else:
            # 哈希自检对注入的 classifier 同样生效：训练/评测用的是同一份判定器。
            if active.model_taxonomy_hash and active.model_taxonomy_hash != taxonomy_hash():
                raise RuntimeError("注入的 classifier 与当前类目表哈希不一致")
        application.state.classifier = active
        application.state.run_id = model_path.name if classifier is None else "inline"
        LOGGER.info(
            "sidecar 就绪：run_id=%s taxonomy_hash=%s",
            application.state.run_id,
            active.model_taxonomy_hash,
        )
        yield
        application.state.classifier = None

    app = FastAPI(title="ch10 topic classifier sidecar", lifespan=lifespan)

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        active: TopicClassifier | None = getattr(request.app.state, "classifier", None)
        return {
            "status": "ok",
            "run_id": getattr(request.app.state, "run_id", ""),
            "taxonomy_hash": active.model_taxonomy_hash if active else taxonomy_hash(),
            "model_loaded": active is not None,
        }

    @app.post("/classify")
    async def classify(payload: ClassifyRequest, request: Request) -> Any:
        active: TopicClassifier | None = getattr(request.app.state, "classifier", None)
        if active is None:
            return JSONResponse(
                status_code=503,
                content={"error": {"type": "model_not_loaded", "message": "模型未加载"}},
            )
        try:
            predictions = await run_in_threadpool(active.predict, payload.texts)
        except Exception:
            # 原始异常只进日志：它常带绝对路径/权重名，不该出现在 HTTP 响应里。
            LOGGER.exception("归类失败（%d 条）", len(payload.texts))
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "type": "internal_error",
                        "message": "归类失败，详情见 sidecar 日志",
                    }
                },
            )
        return {
            "results": [
                {"labels": p.labels, "scores": p.scores, "fallback": p.fallback}
                for p in predictions
            ]
        }

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("未处理异常：%s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_error", "message": "内部错误，详情见 sidecar 日志"}},
        )

    return app


#: uvicorn 的入口：``uvicorn app.topics.service:app --port 8100``。
#: 这里**不**加载模型（import 就该是便宜的），加载发生在 lifespan。
app = create_app()
