"""Milvus 向量库封装（ch04：检索索引副本 + 三路检索）。

**默认不连真 Milvus**：假客户端记录调用，把「建集合的参数、BM25 函数与索引、
写入的形状、过滤挂在哪个参数上」这些容易写错又不容易发现的地方钉住。
真服务端的用例在 ``tests/test_knowledge_milvus.py``（``MEMHELP_TEST_MILVUS=1`` 才跑）。

几条已实测的 Milvus 2.6 / pymilvus 2.6 事实（写在这里当下一个改代码的人的提示）：

- ``get_collection_stats().row_count`` **不可信**（写完仍可能报 0），校验一律用
  ``query(count(*))``；
- ``search(consistency_level="Strong")`` 才能「写完立刻搜得到」；
- **BM25 函数只能在 ``create_collection`` 之前加到 schema 上**，事后补不了；
- **``hybrid_search`` 的顶层 ``filter`` 被静默忽略**、顶层 ``expr`` 直接崩，
  过滤只能挂到每个 ``AnnSearchRequest.expr`` 上（下面有专门的用例钉住）。
"""

from __future__ import annotations

import pytest

from app.knowledge.vector_store import (
    CATEGORY_FIELD,
    CONTENT_TYPE_FIELD,
    DENSE_FIELD,
    ID_FIELD,
    SPARSE_FIELD,
    TEXT_FIELD,
    MilvusVectorStore,
    VectorHit,
    VectorRecord,
)

DIM = 4


class _Schema:
    def __init__(self) -> None:
        self.fields: list[tuple[str, object, dict]] = []
        #: ``add_function`` 收到的是真的 ``pymilvus.Function``（实现里直接 import），
        #: 所以这里不做包装，测试直接读它的属性。
        self.functions: list[object] = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))

    def add_function(self, function):
        self.functions.append(function)

    def field(self, name):
        for field_name, _dtype, kwargs in self.fields:
            if field_name == name:
                return kwargs
        raise AssertionError(f"schema 里没有字段 {name!r}")


class _IndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, field_name, index_type, metric_type, params=None):
        self.indexes.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params,
            }
        )


class _FakeClient:
    """只实现被用到的那几个方法，并记录每一次调用。"""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.created: list[str] = []
        self.schemas: list[_Schema] = []
        self.index_params: list[_IndexParams] = []
        self.upserts: list[list[dict]] = []
        self.searches: list[dict] = []
        self.hybrids: list[dict] = []
        self.dropped: list[str] = []
        self.loaded: list[str] = []
        #: 老集合的字段清单；默认空（表示「用最近一次 schema 描述」）。
        self.describe_fields: list[dict] | None = None
        #: 检索返回值，测试可替换。
        self.search_result: list[list[dict]] = [[{"id": "7", "distance": 0.83}]]
        self.hybrid_result: list[list[dict]] = [[{"id": "7", "distance": 0.83}]]

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
        if self.describe_fields is not None:
            return {"collection_name": name, "fields": self.describe_fields}
        fields = []
        for field_name, _dtype, kwargs in self.schemas[-1].fields:
            params = {"dim": kwargs["dim"]} if "dim" in kwargs else {}
            fields.append({"name": field_name, "params": params})
        return {"collection_name": name, "fields": fields}

    def load_collection(self, name):
        self.loaded.append(name)

    def upsert(self, name, rows):
        self.upserts.append(rows)

    def search(self, name, **kwargs):
        self.searches.append(kwargs)
        return self.search_result

    def hybrid_search(self, name, **kwargs):
        self.hybrids.append(kwargs)
        return self.hybrid_result

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
    return MilvusVectorStore("http://x", "knowledge", DIM, client_factory=lambda: client)


def record(**overrides) -> VectorRecord:
    base = {
        "id": "12",
        "dense": [0.1, 0.2, 0.3, 0.4],
        "text": "问法：运费怎么算\n答案：满 99 元包邮",
        "category": "售后政策",
        "content_type": "faq",
    }
    base.update(overrides)
    return VectorRecord(**base)


# ---------------------------------------------------------------------------
# 建集合
# ---------------------------------------------------------------------------


async def test_ensure_collection_creates_bm25_schema(store, client):
    await store.ensure_collection()

    schema = client.schemas[0]
    assert [f[0] for f in schema.fields] == [
        ID_FIELD,
        TEXT_FIELD,
        CATEGORY_FIELD,
        CONTENT_TYPE_FIELD,
        DENSE_FIELD,
        SPARSE_FIELD,
    ]
    assert schema.field(TEXT_FIELD)["enable_analyzer"] is True
    assert schema.field(TEXT_FIELD)["analyzer_params"] == {"type": "chinese"}
    assert schema.field(DENSE_FIELD) == {"dim": DIM}

    # BM25 函数：输入 text、输出 sparse —— 建集合前必须挂上
    assert len(schema.functions) == 1
    function = schema.functions[0]
    assert function.input_field_names == [TEXT_FIELD]
    assert function.output_field_names == [SPARSE_FIELD]
    assert function.type.name == "BM25"

    indexes = {item["field_name"]: item for item in client.index_params[0].indexes}
    assert indexes[DENSE_FIELD]["metric_type"] == "COSINE"
    assert indexes[SPARSE_FIELD]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert indexes[SPARSE_FIELD]["metric_type"] == "BM25"
    assert indexes[SPARSE_FIELD]["params"]["bm25_k1"] == 1.2
    assert client.loaded == ["knowledge"]


async def test_ensure_collection_is_idempotent(store, client):
    await store.ensure_collection()
    await store.ensure_collection()
    assert client.created == ["knowledge"]


async def test_legacy_collection_without_text_field_fails_with_rebuild_hint(client):
    """ch03 的老集合只有 id+vector：必须报「该重建了」，不能静默让 BM25 永远空。"""
    client.collections.add("knowledge")
    client.describe_fields = [
        {"name": ID_FIELD, "params": {"max_length": 64}},
        {"name": "vector", "params": {"dim": DIM}},
    ]
    store = MilvusVectorStore("http://x", "knowledge", DIM, client_factory=lambda: client)

    with pytest.raises(RuntimeError, match="--rebuild"):
        await store.ensure_collection()


async def test_wrong_dimension_fails_with_rebuild_hint(store, client):
    client.collections.add("knowledge")
    client.describe_fields = [
        {"name": TEXT_FIELD, "params": {}},
        {"name": CATEGORY_FIELD, "params": {}},
        {"name": SPARSE_FIELD, "params": {}},
        {"name": DENSE_FIELD, "params": {"dim": 1024}},
    ]

    with pytest.raises(RuntimeError, match="--rebuild"):
        await store.ensure_collection()


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------


async def test_upsert_writes_text_category_and_dense(store, client):
    await store.upsert([record()])

    row = client.upserts[0][0]
    assert row[ID_FIELD] == "12"
    assert row[TEXT_FIELD] == "问法：运费怎么算\n答案：满 99 元包邮"
    assert row[CATEGORY_FIELD] == "售后政策"
    assert row[CONTENT_TYPE_FIELD] == "faq"
    assert row[DENSE_FIELD] == [0.1, 0.2, 0.3, 0.4]
    # 稀疏向量由服务端 BM25 函数生成，客户端**不写**
    assert SPARSE_FIELD not in row
    assert isinstance(row[DENSE_FIELD][0], float)


async def test_upsert_uses_empty_string_for_missing_content_type(store, client):
    """Milvus 标量字段不接受 null；可空字段必须落成空串。"""
    await store.upsert([record(content_type=None)])
    assert client.upserts[0][0][CONTENT_TYPE_FIELD] == ""


# ---------------------------------------------------------------------------
# 三路检索
# ---------------------------------------------------------------------------


async def test_search_dense_uses_cosine_and_strong_consistency(store, client):
    hits = await store.search_dense([0.1] * DIM, top_k=5)

    assert hits == [VectorHit(id="7", score=0.83)]
    call = client.searches[0]
    assert call["anns_field"] == DENSE_FIELD
    assert call["search_params"] == {"metric_type": "COSINE"}
    assert call["consistency_level"] == "Strong"
    assert call["data"] == [[0.1] * DIM]
    assert "expr" not in call  # 不过滤时不传空表达式


async def test_search_dense_passes_expr_when_given(store, client):
    await store.search_dense([0.1] * DIM, top_k=5, expr='category == "售后政策"')
    assert client.searches[0]["expr"] == 'category == "售后政策"'


async def test_search_bm25_sends_raw_text_not_vector(store, client):
    await store.search_bm25("XH-3000A 续航", top_k=50)

    call = client.searches[0]
    assert call["data"] == ["XH-3000A 续航"]  # 文本，不是向量
    assert call["anns_field"] == SPARSE_FIELD
    assert call["search_params"] == {"metric_type": "BM25"}


async def test_search_bm25_blank_text_short_circuits(store, client):
    assert await store.search_bm25("   ", top_k=50) == []
    assert client.searches == []


async def test_hybrid_puts_expr_on_both_requests(store, client):
    await store.hybrid_search(
        [0.1] * DIM, "满 99 包邮", top_k=20, expr='category == "售后政策"'
    )

    call = client.hybrids[0]
    reqs = call["reqs"]
    assert [r.anns_field for r in reqs] == [DENSE_FIELD, SPARSE_FIELD]
    # 实测坑：顶层 filter 会被静默忽略、顶层 expr 直接崩
    assert [r.expr for r in reqs] == ['category == "售后政策"'] * 2
    assert "filter" not in call and "expr" not in call
    assert call["limit"] == 20
    assert call["ranker"]._k == 60  # pymilvus 把 k 存在私有属性里


async def test_hybrid_per_path_limit_and_payload(store, client):
    await store.hybrid_search([0.1] * DIM, "包邮", top_k=20, per_path_limit=50)

    reqs = client.hybrids[0]["reqs"]
    assert [r.limit for r in reqs] == [50, 50]
    assert reqs[0].data == [[0.1] * DIM]
    assert reqs[1].data == ["包邮"]
    assert reqs[0].param == {"metric_type": "COSINE"}
    assert reqs[1].param == {"metric_type": "BM25"}


async def test_hybrid_maps_hits(store, client):
    client.hybrid_result = [[{"id": "7", "distance": 0.0328}, {"id": "3", "distance": 0.0322}]]
    hits = await store.hybrid_search([0.1] * DIM, "包邮", top_k=20)
    assert [h.id for h in hits] == ["7", "3"]


async def test_empty_results_return_empty_list(store, client):
    client.search_result = [[]]
    client.hybrid_result = [[]]
    assert await store.search_dense([0.1] * DIM, top_k=5) == []
    assert await store.hybrid_search([0.1] * DIM, "包邮", top_k=5) == []


async def test_search_alias_keeps_ch03_callers_working(store, client):
    """ch03 的 KnowledgeRetriever 还在调 ``search``。"""
    hits = await store.search([0.1] * DIM, top_k=5)
    assert hits == [VectorHit(id="7", score=0.83)]
    assert client.searches[0]["anns_field"] == DENSE_FIELD


# ---------------------------------------------------------------------------
# 计数 / 重建
# ---------------------------------------------------------------------------


async def test_count_uses_query_not_collection_stats(store):
    assert await store.count() == 3


async def test_reset_drops_then_recreates(client):
    client.collections.add("knowledge")
    store = MilvusVectorStore("http://x", "knowledge", DIM, client_factory=lambda: client)

    await store.reset()

    assert client.dropped == ["knowledge"]
    assert client.created == ["knowledge"]
