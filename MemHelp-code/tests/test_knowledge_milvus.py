"""真 Milvus 上的集成验证（默认 skip）。

    $env:MEMHELP_TEST_MILVUS = "1"
    uv run --directory MemHelp-code python -m pytest tests/test_knowledge_milvus.py -v

## 为什么必须有这个文件

假客户端能钉住「我们发了什么请求」，但钉不住**服务端是否接受**：

- 主键类型/长度写错 → 建集合时才报 ``PrimaryKeyException``；
- 索引参数名改了（pymilvus 2.6 ↔ 3.x）→ 只有在真客户端上才炸；
- BM25 函数与 chinese analyzer 的**切分行为**只有服务端说了算；
- ``upsert`` 的幂等性、``Strong`` 读己写、per-request ``expr`` 过滤 —— 都是
  **服务端行为**，假实现怎么断言都不算数。

这几类正是「单测全绿、真机挂」的典型。
"""

import os

import pytest

from app.knowledge.vector_store import MilvusVectorStore, VectorRecord

pytestmark = pytest.mark.skipif(
    os.getenv("MEMHELP_TEST_MILVUS") != "1", reason="需要真实 Milvus（MEMHELP_TEST_MILVUS=1）"
)

URI = os.getenv("MEMHELP_TEST_MILVUS_URI", "http://127.0.0.1:19530")
COLLECTION = "ch04_integration_test"
DIM = 8


def _record(chunk_id: str, text: str, category: str, seed: float) -> VectorRecord:
    return VectorRecord(
        id=chunk_id,
        dense=[seed] * DIM,
        text=text,
        category=category,
        content_type="faq",
    )


@pytest.fixture
async def store():
    s = MilvusVectorStore(URI, COLLECTION, DIM)
    await s.reset()
    yield s
    await s.reset()


async def test_server_version_matches_the_pinned_client():
    """pymilvus 2.6.X 是给 Milvus 2.6.* 的；版本错配是本章最容易踩的返工点。"""
    from pymilvus import MilvusClient

    version = MilvusClient(uri=URI).get_server_version()
    assert str(version).startswith("2.6.")


async def test_upsert_is_idempotent_and_read_your_writes(store):
    record = _record("1", "问法：运费怎么算\n答案：满 99 元包邮", "售后政策", 0.1)

    await store.upsert([record])
    await store.upsert([record])  # 同主键再写一次

    assert await store.count() == 1
    hits = await store.search_dense(record.dense, top_k=1)
    assert hits[0].id == "1"
    assert hits[0].score > 0.99  # 同一向量，余弦≈1


async def test_bm25_hits_a_model_number(store):
    """验收 2 的最小版本：型号串必须能被 BM25 那一路命中。"""
    await store.upsert(
        [
            _record("1", "问法：XH-3000A 续航\n答案：XH-3000A 续航 18 小时", "商品参数", 0.1),
            _record("2", "问法：XH-3000B 续航\n答案：XH-3000B 续航 9 小时", "商品参数", 0.2),
            _record("3", "问法：运费怎么算\n答案：满 99 元包邮", "售后政策", 0.3),
        ]
    )

    hits = await store.search_bm25("XH-3000A", top_k=3)

    assert hits, "BM25 没命中型号 —— chinese analyzer 的切分与预期不符"
    assert hits[0].id == "1"


async def test_per_request_expr_filters_both_paths(store):
    """过滤必须能在 hybrid 里生效（顶层 filter 会被静默忽略，见探针结论）。"""
    await store.upsert(
        [
            _record("1", "问法：XH-3000A 续航\n答案：续航 18 小时", "商品参数", 0.1),
            _record("2", "问法：运费怎么算\n答案：满 99 元包邮", "售后政策", 0.2),
        ]
    )

    hits = await store.hybrid_search(
        [0.1] * DIM, "续航 包邮", top_k=10, expr='category == "商品参数"'
    )

    assert [hit.id for hit in hits] == ["1"]

    empty = await store.hybrid_search(
        [0.1] * DIM, "续航 包邮", top_k=10, expr='category == "不存在的品类"'
    )
    assert empty == []


async def test_reset_clears_the_collection(store):
    await store.upsert(
        [
            _record("1", "问法：a\n答案：b", "售后政策", 0.1),
            _record("2", "问法：c\n答案：d", "售后政策", 0.2),
        ]
    )
    assert await store.count() == 2

    await store.reset()

    assert await store.count() == 0
