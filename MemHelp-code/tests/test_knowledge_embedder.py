"""BGE-M3 嵌入器。

**默认不加载真模型**：2.3GB / 首载 20-40s，套件必须能离线秒跑。真模型用例由
``MEMHELP_TEST_EMBED=1`` 显式开启（见文件末尾）。

这一层要钉住三件事：
1. 模型**只加载一次**（每个进程一份，否则一次检索要等两次加载）；
2. 输入顺序与输出顺序一一对应（顺序错位在检索里表现为「答非所问」，且极难复现）；
3. 输出是 **Python list**，不是 numpy 数组 —— Milvus 不接受 numpy，而错误信息
   只在真实写入时才出现。
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.knowledge.embedder import BgeM3Embedder


class _FakeModel:
    """假的 sentence-transformers 模型：记录调用参数，返回固定维度向量。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self.max_seq_length = 8192

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.kwargs.append(kwargs)
        # 用文本长度区分向量，方便断言「顺序没被打乱」
        return np.array([[0.1 + len(t) / 100.0] * 1024 for t in texts], dtype="float32")


async def test_embed_returns_python_lists_in_input_order():
    model = _FakeModel()
    embedder = BgeM3Embedder(Path("unused"), model_factory=lambda _: model)

    vectors = await embedder.embed(["甲", "乙乙"])

    assert isinstance(vectors, list)
    assert all(isinstance(v, list) and len(v) == 1024 for v in vectors)
    assert all(isinstance(x, float) for x in vectors[0])
    assert vectors[0][0] != vectors[1][0]  # 顺序没有被打乱
    assert model.calls == [["甲", "乙乙"]]


async def test_embed_asks_for_normalised_vectors():
    """向量必须 L2 归一化：Milvus 用 COSINE，但换度量/换库时不想依赖实现细节。"""
    model = _FakeModel()
    embedder = BgeM3Embedder(Path("unused"), model_factory=lambda _: model)

    await embedder.embed(["甲"])

    kwargs = model.kwargs[0]
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == 32


async def test_model_is_loaded_only_once():
    loaded: list[str] = []

    def factory(path: str):
        loaded.append(path)
        return _FakeModel()

    embedder = BgeM3Embedder(Path("some/dir"), model_factory=factory)
    await embedder.embed(["a"])
    await embedder.embed(["b"])
    await embedder.warmup()

    assert loaded == [str(Path("some/dir"))]


async def test_empty_input_does_not_load_the_model():
    def factory(path: str):  # pragma: no cover - 不该被调用
        raise AssertionError("空输入不应加载模型")

    embedder = BgeM3Embedder(Path("unused"), model_factory=factory)
    assert await embedder.embed([]) == []


async def test_missing_model_dir_raises_actionable_error():
    embedder = BgeM3Embedder(Path("definitely/not/here"))
    with pytest.raises(FileNotFoundError, match="download_model"):
        await embedder.embed(["甲"])


def test_get_embedder_is_a_process_singleton():
    """整个进程只该有一份模型：uvicorn 每请求 new 一个会直接把内存吃穿。"""
    from app.config import Settings
    from app.knowledge.embedder import get_embedder, reset_embedders

    reset_embedders()
    settings = Settings(
        openai_base_url="https://x/v1", openai_api_key="k", openai_model="m",
        embedding_model_path="models/bge-m3",
    )
    first = get_embedder(settings)
    second = get_embedder(settings)
    assert first is second
    reset_embedders()


# ---------------------------------------------------------------------------
# 真模型（默认 skip）
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.getenv("MEMHELP_TEST_EMBED") != "1", reason="需要真实 BGE-M3")
async def test_real_model_is_1024_dim_and_semantic():
    """真模型验收：维度、归一化、以及「同义句比无关句更近」这条前提。

    配置**显式构造**而不是 ``get_settings()``：套件的 autouse fixture 会把
    ``env_file`` 置空（让结果与宿主机 .env 无关），真模型用例只需要模型目录。
    """
    from app.config import PROJECT_ROOT, Settings
    from app.knowledge.embedder import get_embedder, reset_embedders

    reset_embedders()
    settings = Settings(
        openai_base_url="https://x/v1",
        openai_api_key="k",
        openai_model="m",
        embedding_model_path=str(PROJECT_ROOT / "models" / "bge-m3"),
    )
    embedder = get_embedder(settings)
    vectors = await embedder.embed(["邮费是多少", "运费怎么收", "发票怎么开"])

    assert all(len(v) == 1024 for v in vectors)
    a, b, c = (np.array(v) for v in vectors)
    assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-3  # 已 L2 归一化
    assert float(a @ b) > float(a @ c) + 0.1  # 同义句明显比无关句更近
    reset_embedders()
