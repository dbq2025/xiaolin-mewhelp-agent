"""BGE-M3 嵌入器 —— 全章唯一的「文本 → 向量」入口。

## 为什么要懒加载 + 进程单例

BGE-M3 是 568M 参数的模型（2.27GB 权重），一次加载 20-40s、常驻约 1.5GB 内存。
每个请求 new 一个会把内存吃穿，每个进程里加载两份则是纯粹浪费。
所以：``get_embedder(settings)`` 是 ``lru_cache`` 单例，模型在**第一次真正要编码时**
才加载（``warmup`` 可以提前触发）。

## 为什么所有调用都进工作线程

``model.encode`` 是 CPU 密集的同步调用。直接在 ``async def`` 里调它，阻塞的是
事件循环本身 —— 单请求看不出问题，并发时所有请求排成一队等同一次前向。
与仓库层同一条纪律：``async def`` 不等于不阻塞。

## 为什么返回 list 而不是 numpy

Milvus 的 gRPC 客户端只认真实的 Python list；numpy 的 ``float32`` 传进去会在
**写入时**才报类型错误。在这一层就转成 ``list[float]``，把错误挡在最近的地方。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import anyio

from app.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["BgeM3Embedder", "Embedder", "get_embedder"]


class Embedder(Protocol):
    """上层对嵌入器的最小依赖。测试注入假实现即可完全离线。"""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass
class BgeM3Embedder:
    """BGE-M3 的薄封装。

    :param model_path: 模型目录（本地）。不存在时抛 ``FileNotFoundError``，
        并在消息里指向 ``scripts/download_model.py``。
    :param batch_size: 编码批大小。CPU 上 32 是稳妥值。
    :param model_factory: 仅供测试注入；默认延迟 import ``sentence_transformers``。
        延迟 import 很重要：否则 ``import app`` 就会把 torch 拖起来，
        连 ``--status`` 这种不碰模型的命令都要等好几秒。
    """

    model_path: Path
    batch_size: int = 32
    model_factory: Callable[[str], Any] | None = None
    _model: Any = field(default=None, init=False, repr=False)

    # -- 模型加载 -----------------------------------------------------

    def _default_factory(self, path: str) -> Any:
        from sentence_transformers import SentenceTransformer

        logger.info("加载嵌入模型：%s（首次约 20-40s）", path)
        model = SentenceTransformer(path, device="cpu")
        model.eval()
        return model

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        path = str(self.model_path)
        if self.model_factory is None and not self.model_path.exists():
            raise FileNotFoundError(
                f"嵌入模型目录不存在：{path}。"
                "先跑 `uv run --directory MemHelp-code python scripts/download_model.py` 下载模型。"
            )
        factory = self.model_factory or self._default_factory
        self._model = factory(path)
        return self._model

    async def warmup(self) -> None:
        """提前把模型加载进内存（lifespan 用）。

        **必须**送进工作线程：这一步就是 20-40s 的同步 IO/CPU，直接在事件循环里
        做会让整个应用在这段时间里不响应任何请求。
        """
        await anyio.to_thread.run_sync(self._load)

    # -- 编码 ---------------------------------------------------------

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        array = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in array]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本编码成 L2 归一化的向量，顺序与输入一一对应。

        空输入直接返回 ``[]`` 且**不加载模型** —— 建库空跑时不该付 30s 的加载成本。
        """
        if not texts:
            return []
        return await anyio.to_thread.run_sync(self._encode, list(texts))


_EMBEDDERS: dict[str, BgeM3Embedder] = {}


def get_embedder(settings: Settings) -> BgeM3Embedder:
    """进程级单例（按模型目录区分）。

    为什么不用 ``lru_cache(settings)``：pydantic 的 ``Settings`` 不可哈希，
    直接当缓存键会抛 ``TypeError: unhashable type`` —— 而且只有真跑到这条路径
    才会发现。用模型目录字符串作键，既哈希得了，也正好是「换模型要换实例」的语义。
    """
    key = str(settings.embedding_model_dir())
    cached = _EMBEDDERS.get(key)
    if cached is None:
        cached = BgeM3Embedder(Path(key))
        _EMBEDDERS[key] = cached
    return cached


def reset_embedders() -> None:
    """丢掉进程内已加载的模型（测试与「热换模型」用）。生产代码不要调。"""
    _EMBEDDERS.clear()
