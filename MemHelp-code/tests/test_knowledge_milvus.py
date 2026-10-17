"""真 Milvus 上的集成验证（默认 skip）。

    $env:MEMHELP_TEST_MILVUS = "1"
    uv run --directory MemHelp-code python -m pytest tests/test_knowledge_milvus.py -v

## 为什么必须有这个文件

假客户端能钉住「我们发了什么请求」，但钉不住**服务端是否接受**：

- 主键类型/长度写错 → 建集合时才报 ``PrimaryKeyException``；
- 索引参数名改了（pymilvus 2.6 ↔ 3.x）→ 只有在真客户端上才炸；
- ``upsert`` 的幂等性、``Strong`` 读己写 —— 都是**服务端行为**，假实现怎么断言都不算数。

这三类正是「单测全绿、真机挂」的典型。
"""

import os

import pytest

from app.knowledge.vector_store import MilvusVectorStore

pytestmark = pytest.mark.skipif(
    os.getenv("MEMHELP_TEST_MILVUS") != "1", reason="需要真实 Milvus（MEMHELP_TEST_MILVUS=1）"
)

URI = os.getenv("MEMHELP_TEST_MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = "ch03_integration_test"


@pytest.fixture
async def store():
    s = MilvusVectorStore(URI, COLLECTION, 8)
    await s.reset()
    yield s
    await s.reset()


async def test_server_version_matches_the_pinned_client():
    """pymilvus 2.6.X 是给 Milvus 2.6.* 的；版本错配是本章最容易踩的返工点。"""
    from pymilvus import MilvusClient

    version = MilvusClient(uri=URI).get_server_version()
    assert str(version).startswith("2.6.")


async def test_upsert_is_idempotent_and_read_your_writes(store):
    vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    await store.upsert(["1"], [vector])
    await store.upsert(["1"], [vector])  # 同主键再写一次

    assert await store.count() == 1
    hits = await store.search(vector, top_k=1)
    assert hits[0].id == "1"
    assert hits[0].score > 0.99  # 同一向量，余弦≈1


async def test_reset_clears_the_collection(store):
    await store.upsert(["1", "2"], [[0.1] * 8, [0.2] * 8])
    assert await store.count() == 2

    await store.reset()

    assert await store.count() == 0
