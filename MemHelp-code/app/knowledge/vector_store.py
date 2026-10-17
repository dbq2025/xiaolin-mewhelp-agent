"""Milvus 封装 —— ch04 起是**检索索引副本**，不再是「只存 id + 向量」。

## 为什么集合里多了 text / category / content_type

ch03 的原则是「MySQL 是原文权威源，Milvus 只存 id + 向量」。ch04 要两条新能力，
这两条都要求数据物理上在 Milvus 里：

1. **BM25 全文检索**：Milvus 2.5+ 的原生 BM25 要求一个挂了 analyzer 的 ``text``
   字段 + 一个由 BM25 函数产出的稀疏向量字段，两者都必须在**建集合时**声明。
2. **按品类先过滤再检索**：后过滤会先把 Top-50 截断再丢一半，等于自残召回，
   所以过滤字段必须是 Milvus 的标量字段。

于是口径改成：**Milvus 是检索索引副本，MySQL 仍是原文权威**。
``text`` / ``category`` / ``content_type`` 是从 ``knowledge_chunks`` 派生出来的只读副本，
不参与人工编辑；改了原文必须重建索引（``build_knowledge.py --rebuild``）。

## BM25 函数建集合时必须加

``schema.add_function(Function(FunctionType.BM25, ...))`` 只能在 ``create_collection``
之前调用，事后补不了（官方文档明确写了）。所以 ``ensure_collection`` 的自检不只查维度，
还查 ``text`` / ``sparse`` 是否存在 —— 老集合必须重建，而不是静默地「BM25 永远查不到」。

## 过滤必须挂在每个 AnnSearchRequest 上（实测坑）

pymilvus 2.6.17 的 ``hybrid_search``：

- 顶层 ``filter=`` 参数**被静默忽略**（不报错、不过滤 —— 最危险的一类失败）；
- 顶层 ``expr=`` 会直接抛 ``TypeError: got multiple values for keyword argument 'expr'``。

唯一可靠的写法是把过滤表达式挂到**每个** ``AnnSearchRequest.expr`` 上。两路都过滤，
融合后的候选才都满足条件。探针实测：不存在的品类返回空、组合条件返回正确子集。

## 一致性级别

检索固定 ``consistency_level="Strong"``：刚灌完库就演示/评估时必须读得到。

## 已实测的坑（沿用 ch03）

``get_collection_stats().row_count`` 不可信（写完仍可能报 0），所以 :meth:`count`
走 ``query(count(*))``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import anyio

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORY_FIELD",
    "CONTENT_TYPE_FIELD",
    "DENSE_FIELD",
    "ID_FIELD",
    "SPARSE_FIELD",
    "TEXT_FIELD",
    "MilvusVectorStore",
    "VectorHit",
    "VectorRecord",
    "VectorStore",
]

#: 列名（建集合、写入、检索必须一致）。
ID_FIELD = "id"
TEXT_FIELD = "text"
CATEGORY_FIELD = "category"
CONTENT_TYPE_FIELD = "content_type"
DENSE_FIELD = "dense"
SPARSE_FIELD = "sparse"

#: 主键最大长度，与 MySQL 侧 ``vector_id VARCHAR(64)`` 对齐。
ID_MAX_LENGTH = 64

#: BM25 文本字段的长度上限。单块上限 600 字（``CHUNK_MAX_CHARS``），8192 有充足余量。
TEXT_MAX_LENGTH = 8192

#: 元数据字段长度，与 ``knowledge_chunks`` 的列长对齐。
CATEGORY_MAX_LENGTH = 255
CONTENT_TYPE_MAX_LENGTH = 32

#: RRF 的 k 参数（原论文与 Milvus 的默认值都是 60）。
RRF_K = 60


@dataclass(frozen=True)
class VectorRecord:
    """一条写入 Milvus 的记录：检索要用的全部东西都在这里。

    :param dense: BGE-M3 的稠密向量（由 ``render_embed_text`` 编码而来）。
    :param text: BM25 用原文（由 ``render_bm25_text`` 生成，**不带**「分类/问法/答案」标签）。
    :param content_type: 可空；Milvus 标量字段不接受 null，实现里落成空串。
    """

    id: str
    dense: Sequence[float]
    text: str
    category: str
    content_type: str | None = None


@dataclass(frozen=True)
class VectorHit:
    """一次检索命中：id 与分数，原文去 MySQL 拿。

    分数的含义随策略变化：``search_dense`` 是余弦相似度，``search_bm25`` 是 BM25 分，
    ``hybrid_search`` 是 RRF 融合分（量级在 0.0x，不可与余弦比大小）。
    """

    id: str
    score: float


class VectorStore(Protocol):
    """上层对向量库的最小依赖。测试注入假实现即可离线。"""

    async def ensure_collection(self) -> None: ...
    async def upsert(self, records: Sequence[VectorRecord]) -> None: ...
    async def delete(self, ids: Sequence[str]) -> None: ...
    async def search_dense(
        self, vector: Sequence[float], top_k: int, *, expr: str | None = None
    ) -> list[VectorHit]: ...
    async def search_bm25(
        self, text: str, top_k: int, *, expr: str | None = None
    ) -> list[VectorHit]: ...
    async def hybrid_search(
        self,
        dense_vector: Sequence[float],
        bm25_text: str,
        top_k: int,
        *,
        expr: str | None = None,
        per_path_limit: int = 50,
    ) -> list[VectorHit]: ...
    async def search(self, vector: Sequence[float], top_k: int) -> list[VectorHit]: ...
    async def count(self) -> int: ...
    async def reset(self) -> None: ...


class MilvusVectorStore:
    """``MilvusClient`` 的薄封装。

    :param client_factory: 仅供测试注入。生产用 ``MilvusClient(uri=...)``。
        客户端**延迟构造**：导入本模块不应该就需要连上 Milvus。
    """

    def __init__(
        self,
        uri: str,
        collection: str,
        dim: int,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._uri = uri
        self._collection = collection
        self._dim = dim
        self._client_factory = client_factory
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                from pymilvus import MilvusClient

                self._client = MilvusClient(uri=self._uri)
        return self._client

    # -- 建集合 -------------------------------------------------------

    def _create_schema_and_collection(self) -> None:
        from pymilvus import DataType, Function, FunctionType

        client = self._get_client()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH
        )
        schema.add_field(
            TEXT_FIELD,
            DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(CATEGORY_FIELD, DataType.VARCHAR, max_length=CATEGORY_MAX_LENGTH)
        schema.add_field(
            CONTENT_TYPE_FIELD, DataType.VARCHAR, max_length=CONTENT_TYPE_MAX_LENGTH
        )
        schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
        # BM25 函数必须在建集合前加上（官方文档：事后无法补）。
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=DENSE_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            },
        )
        client.create_collection(
            self._collection, schema=schema, index_params=index_params
        )
        client.load_collection(self._collection)

    def _ensure_collection(self) -> None:
        client = self._get_client()
        if not client.has_collection(self._collection):
            self._create_schema_and_collection()
            return

        # 集合已存在时**校验结构**：换过嵌入模型/维度、或还是 ch03 的「只有 id+向量」
        # 老集合时，第一次写入/检索会收到一条原生错误，对使用者毫无操作性。
        # 这里提前给出「该重建了」的可执行提示。
        described = client.describe_collection(self._collection)
        names = {field.get("name") for field in described.get("fields", [])}
        missing = {TEXT_FIELD, SPARSE_FIELD, CATEGORY_FIELD} - names
        if missing:
            raise RuntimeError(
                f"Milvus 集合 {self._collection!r} 缺少字段 {sorted(missing)}（ch04 的 BM25 "
                "全文检索与品类过滤需要它们，且只能在建集合时声明）。请重建索引："
                "运行 scripts/build_knowledge.py --docs knowledge --faq-table --rebuild"
            )
        for field in described.get("fields", []):
            if field.get("name") != DENSE_FIELD:
                continue
            params = field.get("params") or {}
            existing_dim = params.get("dim")
            if existing_dim is not None and int(existing_dim) != self._dim:
                raise RuntimeError(
                    f"Milvus 集合 {self._collection!r} 的向量维度是 {existing_dim}，"
                    f"而当前配置是 {self._dim}。改了嵌入模型/维度后需要重建索引："
                    "运行 scripts/build_knowledge.py --docs knowledge --faq-table --rebuild"
                )

    async def ensure_collection(self) -> None:
        """集合不存在就建（含索引、BM25 函数与 load）。幂等。"""
        await anyio.to_thread.run_sync(self._ensure_collection)

    # -- 写入 ---------------------------------------------------------

    def _upsert(self, records: list[VectorRecord]) -> None:
        rows = [
            {
                ID_FIELD: str(record.id),
                TEXT_FIELD: record.text,
                CATEGORY_FIELD: record.category,
                # Milvus 标量字段不接受 null，可空字段落成空串。
                CONTENT_TYPE_FIELD: record.content_type or "",
                DENSE_FIELD: [float(x) for x in record.dense],
            }
            for record in records
        ]
        self._get_client().upsert(self._collection, rows)

    async def upsert(self, records: Sequence[VectorRecord]) -> None:
        """按主键 upsert：同 id 覆盖，不产生第二行。"""
        await anyio.to_thread.run_sync(self._upsert, list(records))

    # -- 删除 ---------------------------------------------------------

    def _delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._get_client().delete(self._collection, ids=ids)

    async def delete(self, ids: Sequence[str]) -> None:
        """按主键删除（增量重建时清理已不存在的旧块）。"""
        await anyio.to_thread.run_sync(self._delete, [str(i) for i in ids])

    # -- 检索 ---------------------------------------------------------

    @staticmethod
    def _hits(rows: list[dict]) -> list[VectorHit]:
        return [VectorHit(id=str(h[ID_FIELD]), score=float(h["distance"])) for h in rows]

    def _search_dense(
        self, vector: list[float], top_k: int, expr: str | None
    ) -> list[VectorHit]:
        kwargs: dict[str, Any] = {}
        if expr:
            kwargs["expr"] = expr
        results = self._get_client().search(
            self._collection,
            data=[[float(x) for x in vector]],
            anns_field=DENSE_FIELD,
            limit=top_k,
            output_fields=[ID_FIELD],
            search_params={"metric_type": "COSINE"},
            consistency_level="Strong",
            **kwargs,
        )
        return self._hits(results[0]) if results and results[0] else []

    async def search_dense(
        self, vector: Sequence[float], top_k: int, *, expr: str | None = None
    ) -> list[VectorHit]:
        """稠密向量 Top-K。空集合返回空列表（不是错误）。"""
        return await anyio.to_thread.run_sync(self._search_dense, list(vector), top_k, expr)

    async def search(
        self, vector: Sequence[float], top_k: int
    ) -> list[VectorHit]:
        """ch03 的旧入口，等价于 :meth:`search_dense`（不带过滤）。

        保留它是因为 ``KnowledgeRetriever`` 与 ch03 的评估基线还在用；
        ch04 的新代码一律走显式的三路方法。
        """
        return await self.search_dense(vector, top_k)

    def _search_bm25(self, text: str, top_k: int, expr: str | None) -> list[VectorHit]:
        kwargs: dict[str, Any] = {}
        if expr:
            kwargs["expr"] = expr
        results = self._get_client().search(
            self._collection,
            # BM25 检索传的是**原始文本**，不是稀疏向量 —— 分析器在服务端跑。
            data=[text],
            anns_field=SPARSE_FIELD,
            limit=top_k,
            output_fields=[ID_FIELD],
            search_params={"metric_type": "BM25"},
            consistency_level="Strong",
            **kwargs,
        )
        return self._hits(results[0]) if results and results[0] else []

    async def search_bm25(
        self, text: str, top_k: int, *, expr: str | None = None
    ) -> list[VectorHit]:
        """BM25 全文检索 Top-K。``text`` 是查询串，空串返回空列表。"""
        if not text.strip():
            return []
        return await anyio.to_thread.run_sync(self._search_bm25, text, top_k, expr)

    def _hybrid(
        self,
        dense_vector: list[float],
        bm25_text: str,
        top_k: int,
        expr: str | None,
        per_path_limit: int,
    ) -> list[VectorHit]:
        from pymilvus import AnnSearchRequest, RRFRanker

        request_kwargs: dict[str, Any] = {}
        if expr:
            # 过滤只能挂在每个 request 上：顶层 filter 会被静默忽略。
            request_kwargs["expr"] = expr
        dense_req = AnnSearchRequest(
            data=[[float(x) for x in dense_vector]],
            anns_field=DENSE_FIELD,
            limit=per_path_limit,
            param={"metric_type": "COSINE"},
            **request_kwargs,
        )
        bm25_req = AnnSearchRequest(
            data=[bm25_text],
            anns_field=SPARSE_FIELD,
            limit=per_path_limit,
            param={"metric_type": "BM25"},
            **request_kwargs,
        )
        results = self._get_client().hybrid_search(
            self._collection,
            reqs=[dense_req, bm25_req],
            ranker=RRFRanker(k=RRF_K),
            limit=top_k,
            output_fields=[ID_FIELD],
            consistency_level="Strong",
        )
        return self._hits(results[0]) if results and results[0] else []

    async def hybrid_search(
        self,
        dense_vector: Sequence[float],
        bm25_text: str,
        top_k: int,
        *,
        expr: str | None = None,
        per_path_limit: int = 50,
    ) -> list[VectorHit]:
        """dense + BM25 双路召回，**由 Milvus 做 RRF 融合**。

        :param top_k: 融合后返回几条（= 进重排的候选数）。
        :param per_path_limit: 每一路各召回几条。
        """
        return await anyio.to_thread.run_sync(
            self._hybrid, list(dense_vector), bm25_text, top_k, expr, per_path_limit
        )

    # -- 计数 / 重建 ---------------------------------------------------

    def _count(self) -> int:
        # **必须带 Strong**：默认的 bounded staleness 下，刚 upsert 完立刻
        # `query(count(*))` 会返回 0 —— 真机实测踩过。
        rows = self._get_client().query(
            self._collection,
            filter="",
            output_fields=["count(*)"],
            consistency_level="Strong",
        )
        return int(rows[0]["count(*)"]) if rows else 0

    async def count(self) -> int:
        """集合里的实体数。用 ``query(count(*))`` 而不是 ``get_collection_stats``。"""
        return await anyio.to_thread.run_sync(self._count)

    def _reset(self) -> None:
        client = self._get_client()
        if client.has_collection(self._collection):
            client.drop_collection(self._collection)
        self._create_schema_and_collection()

    async def reset(self) -> None:
        """drop 后重建 —— ``--rebuild`` 用。

        与 ``KnowledgeChunkRepository.truncate`` 配对使用，顺序是**先清向量再清表**：
        反了的话，中断会留下指向已删原文的孤儿向量。
        """
        await anyio.to_thread.run_sync(self._reset)
