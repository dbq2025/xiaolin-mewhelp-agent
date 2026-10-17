"""Milvus 向量库封装。

**默认不连真 Milvus**：假客户端记录调用，把「建集合的参数、写入的形状、
检索的一致性级别」这些容易写错又不容易发现的地方钉住。真服务端的用例在
``tests/test_knowledge_milvus.py``（``MEMHELP_TEST_MILVUS=1`` 才跑）。

几个已实测的 Milvus 2.6 事实（写在这里当下一个改代码的人的提示）：
- ``get_collection_stats().row_count`` **不可信**（写完仍可能报 0），校验一律用
  ``query(count(*))``；
- ``search(consistency_level="Strong")`` 才能「写完立刻搜得到」，默认的
  bounded staleness 会让刚灌完库的演示查不到东西；
- ``upsert`` 按主键覆盖，这就是「重跑不产生重复向量」的机制。
"""

from __future__ import annotations

import pytest

from app.knowledge.vector_store import MilvusVectorStore, VectorHit


class _Schema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict]] = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class _IndexParams:
    def __init__(self) -> None:
        self.indexes: list[tuple[str, str, str]] = []

    def add_index(self, field_name, index_type, metric_type):
        self.indexes.append((field_name, index_type, metric_type))


class _FakeClient:
    """只实现被用到的那几个方法，并记录每一次调用。"""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created: list[str] = []
        self.schemas: list[_Schema] = []
        self.index_params: list[_IndexParams] = []
        self.upserts: list[list[dict]] = []
        self.searches: list[tuple[int, str]] = []
        self.vector_batches: list[list[list[float]]] = []
        self.dropped: list[str] = []
        self.loaded: list[str] = []

    def has_collection(self, name):
        return name in self.collections

    def create_schema(self, auto_id=False, enable_dynamic_field=False):
        assert auto_id is False
        schema = _Schema()
        self.schemas.append(schema)
        return schema

    def prepare_index_params(self):
        params = _IndexParams()
        self.index_params.append(params)
        return params

    def create_collection(self, name, schema=None, index_params=None):
        self.collections.add(name)
        self.created.append(name)

    def describe_collection(self, name):
        # 只有 id + vector 两列；维度取自最近一次 create_schema
        fields = []
        if self.schemas:
            for field_name, _dtype, kwargs in self.schemas[-1].fields:
                params = {"dim": kwargs["dim"]} if "dim" in kwargs else {}
                fields.append({"name": field_name, "params": params})
        return {"collection_name": name, "fields": fields}

    def load_collection(self, name):
        self.loaded.append(name)

    def upsert(self, name, rows):
        self.upserts.append(rows)

    def search(self, name, data, limit, output_fields, consistency_level):
        self.searches.append((limit, consistency_level))
        self.vector_batches.append(data)
        return [[{"id": "7", "distance": 0.83}]]

    def query(self, name, filter, output_fields, consistency_level=None):
        assert output_fields == ["count(*)"]
        # 真机实测：不带 Strong 时刚写完立刻数会得到 0，所以这里也钉住。
        assert consistency_level == "Strong"
        return [{"count(*)": 3}]

    def drop_collection(self, name):
        self.collections.discard(name)
        self.dropped.append(name)


@pytest.fixture
def client():
    return _FakeClient()


@pytest.fixture
def store(client):
    return MilvusVectorStore("http://x", "knowledge", 1024, client_factory=lambda: client)


# ---------------------------------------------------------------------------
# 建集合
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_once_with_expected_schema(store, client):
    await store.ensure_collection()
    await store.ensure_collection()  # 幂等：第二次不能再建一遍

    assert client.created == ["knowledge"]
    fields = client.schemas[0].fields
    assert fields[0][0] == "id"
    assert fields[0][2] == {"is_primary": True, "max_length": 64}
    assert fields[1][0] == "vector"
    assert fields[1][2] == {"dim": 1024}
    assert client.index_params[0].indexes == [("vector", "AUTOINDEX", "COSINE")]
    assert client.loaded == ["knowledge"]


async def test_ensure_collection_is_noop_when_already_present(client):
    client.collections.add("knowledge")
    store = MilvusVectorStore("http://x", "knowledge", 1024, client_factory=lambda: client)

    await store.ensure_collection()

    assert client.created == []


async def test_existing_collection_with_wrong_dim_fails_with_an_actionable_error(client):
    """换过嵌入模型/维度后，第一次写入会收到 Milvus 原生维度错误 —— 太难查。

    这里提前校验并给出「该重建了」的可执行提示。
    """
    client.collections.add("knowledge")
    client.schemas.append(_Schema())
    client.schemas[-1].fields.append(("id", object, {"max_length": 64}))
    client.schemas[-1].fields.append(("vector", object, {"dim": 4}))
    store = MilvusVectorStore("http://x", "knowledge", 1024, client_factory=lambda: client)

    with pytest.raises(RuntimeError, match="--rebuild"):
        await store.ensure_collection()


# ---------------------------------------------------------------------------
# 写入 / 检索 / 计数
# ---------------------------------------------------------------------------


async def test_upsert_sends_string_ids_and_python_float_lists(client):
    store = MilvusVectorStore("http://x", "knowledge", 4, client_factory=lambda: client)
    await store.upsert(["12"], [[0.1, 0.2, 0.3, 0.4]])

    assert client.upserts == [[{"id": "12", "vector": [0.1, 0.2, 0.3, 0.4]}]]
    assert isinstance(client.upserts[0][0]["vector"][0], float)


async def test_search_uses_strong_consistency_and_maps_hits(store, client):
    hits = await store.search([0.1] * 1024, top_k=5)

    assert hits == [VectorHit(id="7", score=0.83)]
    assert client.searches == [(5, "Strong")]
    assert client.vector_batches[0] == [[0.1] * 1024]


async def test_search_on_empty_result_returns_empty_list(client):
    client.search = lambda *a, **kw: [[]]  # type: ignore[assignment]
    store = MilvusVectorStore("http://x", "knowledge", 4, client_factory=lambda: client)
    assert await store.search([0.1] * 4, top_k=5) == []


async def test_count_uses_query_not_collection_stats(store):
    assert await store.count() == 3


# ---------------------------------------------------------------------------
# 重建
# ---------------------------------------------------------------------------


async def test_reset_drops_then_recreates(client):
    client.collections.add("knowledge")
    store = MilvusVectorStore("http://x", "knowledge", 1024, client_factory=lambda: client)

    await store.reset()

    assert client.dropped == ["knowledge"]
    assert client.created == ["knowledge"]
