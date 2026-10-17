"""四策略检索：vector / bm25 / hybrid / hybrid_rerank。

## 为什么四条策略共用一个接口

评估脚本与线上必须跑**同一份实现** —— 否则报告里的数字与用户实际得到的结果
是两套东西，那份报告就只能用来交差，不能用来做决定。

## 为什么融合必须交给 Milvus

自己实现 RRF 再和服务端的算法比大小，两边口径迟早分叉；``hybrid_search`` 已经
做了融合，我们只负责传两路输入与过滤表达式。

## 过滤挂在每个 request 上

实测（pymilvus 2.6.17）：``hybrid_search`` 的顶层 ``filter`` **被静默忽略**、
顶层 ``expr`` 直接抛错。所以过滤表达式必须挂到两个 ``AnnSearchRequest.expr`` 上。
"""

from __future__ import annotations

import pytest

from app.knowledge.hybrid import HybridRetriever, build_expr
from app.knowledge.query import UnderstoodQuery
from app.knowledge.rerank import NoopReranker, RerankError, RerankHit
from app.knowledge.vector_store import VectorHit


class _FakeEmbedder:
    def __init__(self, vectors=None) -> None:
        self.vectors = vectors if vectors is not None else [[0.1, 0.2]]
        self.texts: list[str] = []
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        return self.vectors


class _Row:
    def __init__(self, chunk_id, questions, answer, category="售后政策",
                 section_path="售后手册 / 运费", content_type="manual") -> None:
        self.id = chunk_id
        self.questions = questions
        self.answer = answer
        self.category = category
        self.section_path = section_path
        self.content_type = content_type


class _FakeChunkRepo:
    def __init__(self, rows) -> None:
        self.rows = {row.id: row for row in rows}
        self.asked: list[list[int]] = []

    async def get_many(self, ids):
        self.asked.append(list(ids))
        return [self.rows[i] for i in ids if i in self.rows]


class _FakeStore:
    def __init__(self, dense=None, bm25=None, hybrid=None) -> None:
        self.dense_result = dense or []
        self.bm25_result = bm25 or []
        self.hybrid_result = hybrid or []
        self.dense_calls: list[dict] = []
        self.bm25_calls: list[dict] = []
        self.hybrid_calls: list[dict] = []

    async def search_dense(self, vector, top_k, *, expr=None):
        self.dense_calls.append({"vector": vector, "top_k": top_k, "expr": expr})
        return self.dense_result

    async def search_bm25(self, text, top_k, *, expr=None):
        self.bm25_calls.append({"text": text, "top_k": top_k, "expr": expr})
        return self.bm25_result

    async def hybrid_search(self, dense_vector, bm25_text, top_k, *, expr=None,
                            per_path_limit=50):
        self.hybrid_calls.append(
            {"dense": dense_vector, "text": bm25_text, "top_k": top_k,
             "expr": expr, "per_path_limit": per_path_limit}
        )
        return self.hybrid_result


class _FakeReranker:
    def __init__(self, order=None, error: Exception | None = None) -> None:
        self.order = order or []
        self.error = error
        self.calls: list[dict] = []

    async def rerank(self, query, documents, top_n):
        self.calls.append({"query": query, "documents": list(documents), "top_n": top_n})
        if self.error is not None:
            raise self.error
        return self.order


def _understood(standard="运费怎么算", bm25_text="运费怎么算 邮费 快递费"):
    return UnderstoodQuery(
        original=standard, standard=standard, bm25_text=bm25_text, expanded=("邮费",)
    )


def _retriever(store, repo, reranker=None, **overrides):
    options = {
        "hybrid_top_k": 50,
        "candidate_k": 20,
        "rerank_top_n": 10,
    }
    options.update(overrides)
    return HybridRetriever(
        embedder=_FakeEmbedder(),
        vector_store=store,
        reranker=reranker or _FakeReranker(),
        chunk_repo=repo,
        **options,
    )


# ---------------------------------------------------------------------------
# 过滤表达式
# ---------------------------------------------------------------------------


def test_build_expr_quotes_and_escapes():
    assert build_expr("售后政策", None) == 'category == "售后政策"'
    assert build_expr('含"引号', None) == 'category == "含\\"引号"'


def test_build_expr_ignores_blank_values():
    assert build_expr("", None) is None
    assert build_expr("   ", "") is None
    assert build_expr(None, None) is None


def test_build_expr_combines_two_fields():
    assert build_expr("售后政策", "policy") == (
        'category == "售后政策" && content_type == "policy"'
    )


# ---------------------------------------------------------------------------
# vector
# ---------------------------------------------------------------------------


async def test_vector_uses_standard_query_and_hybrid_top_k():
    store = _FakeStore(dense=[VectorHit(id="1", score=0.9)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])
    retriever = _retriever(store, repo)

    result = await retriever.retrieve("vector", _understood())

    assert [chunk.chunk_id for chunk in result.chunks] == [1]
    assert result.chunks[0].source == "vector"
    assert store.dense_calls[0]["top_k"] == 50
    assert store.bm25_calls == []


async def test_vector_passes_expr_for_prefiltering():
    store = _FakeStore(dense=[VectorHit(id="1", score=0.9)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])
    await _retriever(store, repo).retrieve("vector", _understood(), category="售后政策")
    assert store.dense_calls[0]["expr"] == 'category == "售后政策"'


# ---------------------------------------------------------------------------
# bm25
# ---------------------------------------------------------------------------


async def test_bm25_uses_bm25_text_with_synonyms():
    store = _FakeStore(bm25=[VectorHit(id="1", score=8.2)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])

    result = await _retriever(store, repo).retrieve("bm25", _understood())

    assert store.bm25_calls[0]["text"] == "运费怎么算 邮费 快递费"
    assert result.chunks[0].source == "bm25"
    assert result.chunks[0].score == 8.2


async def test_bm25_skips_queries_without_searchable_characters():
    """纯标点/空白不该发一次远端请求，也不该报错。"""
    store = _FakeStore(bm25=[VectorHit(id="1", score=1.0)])
    repo = _FakeChunkRepo([_Row(1, "x", "y")])
    result = await _retriever(store, repo).retrieve(
        "bm25", _understood(standard="？？？", bm25_text="？？？")
    )

    assert result.chunks == ()
    assert store.bm25_calls == []


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------


async def test_hybrid_passes_both_paths_and_filter():
    store = _FakeStore(hybrid=[VectorHit(id="1", score=0.032)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])

    result = await _retriever(store, repo).retrieve(
        "hybrid", _understood(), category="售后政策"
    )

    call = store.hybrid_calls[0]
    assert call["text"] == "运费怎么算 邮费 快递费"
    assert call["expr"] == 'category == "售后政策"'
    assert call["top_k"] == 20          # 进重排的候选数
    assert call["per_path_limit"] == 50  # 每路召回数
    assert result.chunks[0].source == "hybrid"


async def test_hybrid_returns_empty_when_store_is_empty():
    store = _FakeStore(hybrid=[])
    repo = _FakeChunkRepo([])
    result = await _retriever(store, repo).retrieve(
        "hybrid", _understood(), category="不存在的品类"
    )
    assert result.chunks == ()
    assert result.rerank_degraded is False


# ---------------------------------------------------------------------------
# hybrid_rerank
# ---------------------------------------------------------------------------


async def test_hybrid_rerank_orders_by_rerank_score():
    store = _FakeStore(hybrid=[VectorHit(id="1", score=0.03), VectorHit(id="2", score=0.02)])
    repo = _FakeChunkRepo(
        [_Row(1, "运费怎么算", "满 99 包邮"), _Row(2, "发票怎么开", "3 个工作日")]
    )
    reranker = _FakeReranker(order=[RerankHit(index=1, score=0.9), RerankHit(index=0, score=0.1)])

    result = await _retriever(store, repo, reranker).retrieve("hybrid_rerank", _understood())

    assert [chunk.chunk_id for chunk in result.chunks] == [2, 1]
    assert [chunk.score for chunk in result.chunks] == [0.9, 0.1]
    assert result.chunks[0].source == "rerank"
    assert result.rerank_degraded is False
    assert reranker.calls[0]["query"] == "运费怎么算"
    assert reranker.calls[0]["documents"] == ["运费怎么算\n满 99 包邮", "发票怎么开\n3 个工作日"]
    assert reranker.calls[0]["top_n"] == 10


async def test_rerank_failure_degrades_to_rrf_order():
    store = _FakeStore(hybrid=[VectorHit(id="1", score=0.03), VectorHit(id="2", score=0.02)])
    repo = _FakeChunkRepo(
        [_Row(1, "运费怎么算", "满 99 包邮"), _Row(2, "发票怎么开", "3 个工作日")]
    )
    reranker = _FakeReranker(error=RerankError("超时"))

    result = await _retriever(store, repo, reranker, rerank_top_n=1).retrieve(
        "hybrid_rerank", _understood()
    )

    assert result.rerank_degraded is True
    assert [chunk.chunk_id for chunk in result.chunks] == [1]  # RRF 原序，截到 top_n


async def test_noop_reranker_is_marked_degraded():
    """没配 Key 时不许假装「重排过」——报告里要能看出来。"""
    store = _FakeStore(hybrid=[VectorHit(id="1", score=0.03)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])

    result = await _retriever(store, repo, NoopReranker()).retrieve(
        "hybrid_rerank", _understood()
    )

    assert result.rerank_degraded is True
    assert [chunk.chunk_id for chunk in result.chunks] == [1]


# ---------------------------------------------------------------------------
# 脏数据
# ---------------------------------------------------------------------------


async def test_orphan_ids_are_skipped():
    """Milvus 有、MySQL 没有的 id 是可预期脏数据（沿用 ch03 的处置）。"""
    store = _FakeStore(dense=[VectorHit(id="1", score=0.9), VectorHit(id="9", score=0.8)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])

    result = await _retriever(store, repo).retrieve("vector", _understood())

    assert [chunk.chunk_id for chunk in result.chunks] == [1]


async def test_non_numeric_vector_ids_are_skipped():
    store = _FakeStore(dense=[VectorHit(id="abc", score=0.9), VectorHit(id="1", score=0.8)])
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])

    result = await _retriever(store, repo).retrieve("vector", _understood())

    assert [chunk.chunk_id for chunk in result.chunks] == [1]


async def test_empty_embedding_returns_empty_result():
    store = _FakeStore(dense=[VectorHit(id="1", score=0.9)])
    repo = _FakeChunkRepo([_Row(1, "x", "y")])
    retriever = HybridRetriever(
        embedder=_FakeEmbedder(vectors=[]),
        vector_store=store,
        reranker=_FakeReranker(),
        chunk_repo=repo,
    )

    result = await retriever.retrieve("vector", _understood())

    assert result.chunks == ()
    assert store.dense_calls == []


@pytest.mark.parametrize("strategy", ["vector", "bm25", "hybrid", "hybrid_rerank"])
async def test_every_strategy_returns_strategy_result(strategy):
    store = _FakeStore(
        dense=[VectorHit(id="1", score=0.9)],
        bm25=[VectorHit(id="1", score=5.0)],
        hybrid=[VectorHit(id="1", score=0.03)],
    )
    repo = _FakeChunkRepo([_Row(1, "运费怎么算", "满 99 包邮")])
    reranker = _FakeReranker(order=[RerankHit(index=0, score=0.8)])

    result = await _retriever(store, repo, reranker).retrieve(strategy, _understood())

    assert result.strategy == strategy
    assert len(result.chunks) == 1
