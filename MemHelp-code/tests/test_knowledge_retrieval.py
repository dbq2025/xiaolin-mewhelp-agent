"""在线检索：问题 → 向量 → Milvus Top-K → MySQL 回填原文。

这一层最容易出的两个错都不会报错：

- **顺序丢了**：按 id 从 MySQL 取回来再原序返回，否则最高分那条可能排最后；
- **孤儿 id 崩了**：Milvus 里有 id 而 MySQL 行已删（`--rebuild` 的中间态、人工删行），
  必须跳过并告警，不能 `KeyError` —— 否则一次检索就把整个聊天连接打断。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.db.repositories import KnowledgeChunkRepository
from app.knowledge.retrieval import KnowledgeHit, KnowledgeRetriever
from app.knowledge.vector_store import VectorHit


class _FakeEmbedder:
    def __init__(self) -> None:
        self.warmup_calls = 0
        self.texts: list[list[str]] = []

    async def embed(self, texts):
        self.texts.append(list(texts))
        return [[0.5, 0.5, 0.5, 0.5] for _ in texts]

    async def warmup(self) -> None:
        self.warmup_calls += 1


class _FakeStore:
    def __init__(self, hits) -> None:
        self.hits = hits
        self.ensure_calls = 0
        self.asked_top_k: list[int] = []

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def search(self, vector, top_k):
        self.asked_top_k.append(top_k)
        return self.hits[:top_k]

    async def upsert(self, ids, vectors):  # pragma: no cover - 检索不该写
        raise AssertionError("检索路径不应写入向量库")

    async def count(self) -> int:  # pragma: no cover
        return len(self.hits)

    async def reset(self) -> None:  # pragma: no cover
        pass


async def _seed_chunks(session, items):
    """按给定的 (问法, 答案, 分类) 造块，返回 (仓库, [id...])。"""
    repo = KnowledgeChunkRepository(session)
    ids: list[int] = []
    for index, (questions, answer, category) in enumerate(items):
        chunk = SimpleNamespace(
            category=category,
            questions=questions,
            answer=answer,
            section_path=f"物流 / {category}",
            content_type="faq",
            is_key_clause=False,
            ordinal=index,
        )
        chunk_id, _ = await repo.upsert_chunk(chunk)
        ids.append(chunk_id)
    return repo, ids


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------


async def test_search_preserves_score_order_and_maps_fields(sqlite_session):
    repo, (a, b) = await _seed_chunks(
        sqlite_session,
        [("运费怎么算", "满 99 包邮", "物流"), ("退货政策", "七天无理由", "售后")],
    )
    store = _FakeStore([VectorHit(str(b), 0.9), VectorHit(str(a), 0.8)])
    retriever = KnowledgeRetriever(_FakeEmbedder(), store, repo)

    hits = await retriever.search("邮费是多少", top_k=2)

    assert isinstance(hits[0], KnowledgeHit)
    # 分数序 = 返回顺序（b 分高），不是 id 序
    assert [h.question for h in hits] == ["退货政策", "运费怎么算"]
    assert hits[1].category == "物流"
    assert hits[1].score == 0.8


async def test_search_embeds_the_query_verbatim(sqlite_session):
    """用户原话整句进嵌入（不截成名词）—— 这是「换说法能召回」的前提。"""
    repo, _ = await _seed_chunks(sqlite_session, [("运费怎么算", "满 99 包邮", "物流")])
    embedder = _FakeEmbedder()
    retriever = KnowledgeRetriever(embedder, _FakeStore([]), repo)

    await retriever.search("邮费是多少")

    assert embedder.texts == [["邮费是多少"]]


async def test_search_passes_top_k_through(sqlite_session):
    repo, _ = await _seed_chunks(sqlite_session, [("运费怎么算", "满 99 包邮", "物流")])
    store = _FakeStore([])
    await KnowledgeRetriever(_FakeEmbedder(), store, repo).search("q", top_k=3)
    assert store.asked_top_k == [3]


async def test_search_falls_back_to_the_configured_default_top_k(sqlite_session):
    """``RAG_TOP_K`` 必须真的生效：不传 top_k 时用构造时给的默认值。"""
    repo, _ = await _seed_chunks(sqlite_session, [("运费怎么算", "满 99 包邮", "物流")])
    store = _FakeStore([])

    await KnowledgeRetriever(_FakeEmbedder(), store, repo, default_top_k=9).search("q")

    assert store.asked_top_k == [9]


async def test_search_returns_empty_when_store_is_empty(sqlite_session):
    repo, _ = await _seed_chunks(sqlite_session, [])
    assert await KnowledgeRetriever(_FakeEmbedder(), _FakeStore([]), repo).search("q") == []


# ---------------------------------------------------------------------------
# 脏数据兜底（Review Focus：Milvus 有 id 而 MySQL 已删）
# ---------------------------------------------------------------------------


async def test_search_drops_ids_missing_from_mysql(sqlite_session):
    repo, (a,) = await _seed_chunks(sqlite_session, [("运费怎么算", "满 99 包邮", "物流")])
    store = _FakeStore([VectorHit("999", 0.95), VectorHit(str(a), 0.8)])

    hits = await KnowledgeRetriever(_FakeEmbedder(), store, repo).search("q")

    assert [h.question for h in hits] == ["运费怎么算"]


async def test_search_treats_non_numeric_vector_id_as_missing(sqlite_session):
    repo, _ = await _seed_chunks(sqlite_session, [("运费怎么算", "满 99 包邮", "物流")])
    store = _FakeStore([VectorHit("not-a-number", 0.99)])

    assert await KnowledgeRetriever(_FakeEmbedder(), store, repo).search("q") == []


# ---------------------------------------------------------------------------
# 预热
# ---------------------------------------------------------------------------


async def test_warmup_loads_model_and_ensures_collection(sqlite_session):
    repo, _ = await _seed_chunks(sqlite_session, [])
    embedder, store = _FakeEmbedder(), _FakeStore([])

    await KnowledgeRetriever(embedder, store, repo).warmup()

    assert embedder.warmup_calls == 1
    assert store.ensure_calls == 1
