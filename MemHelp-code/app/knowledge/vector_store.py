"""Milvus 封装 —— 只存 ``id + vector`` 的向量库。

## 为什么集合里只有两列

需求的边界是「MySQL 是原文权威源，元数据只存不进向量」。所以 Milvus 里只留
``id``（= ``str(knowledge_chunks.id)``）与 ``vector``；检索拿到 id 后回 MySQL
取原文与元数据。好处是**没有第二份真相**：改原文只改一处，Milvus 里不会躺着
过期的分类/章节路径。

## 确定性主键 = 幂等

主键不用 Milvus 的 ``auto_id``，而是由调用方给出 ``str(chunk.id)``，写用 ``upsert``。
于是「Milvus 写成功、回填 MySQL 时进程被杀」这种情况，重跑会用同一主键覆盖 ——
不会出现第二条向量。这是 Context7 上 Milvus 官方文档给的 RAG 重摄入做法。

## 一致性级别

检索固定 ``consistency_level="Strong"``：刚灌完库就演示/评估时必须读得到。
默认的 bounded staleness 会让「建库成功但查不到」看起来像检索坏了。

## 已实测的坑

``get_collection_stats().row_count`` 不可信（写完仍可能报 0），所以 :meth:`count`
走 ``query(count(*))``。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import anyio

logger = logging.getLogger(__name__)

__all__ = ["MilvusVectorStore", "VectorHit", "VectorStore"]

#: 主键列名与向量列名（建集合与写入必须一致）。
ID_FIELD = "id"
VECTOR_FIELD = "vector"

#: 主键最大长度，与 MySQL 侧 ``vector_id VARCHAR(64)`` 对齐。
ID_MAX_LENGTH = 64


@dataclass(frozen=True)
class VectorHit:
    """一次向量检索命中：只有 id 与相似度，原文去 MySQL 拿。"""

    id: str
    score: float


class VectorStore(Protocol):
    """上层对向量库的最小依赖。测试注入假实现即可离线。"""

    async def ensure_collection(self) -> None: ...
    async def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None: ...
    async def delete(self, ids: Sequence[str]) -> None: ...
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
        from pymilvus import DataType

        client = self._get_client()
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            ID_FIELD, DataType.VARCHAR, is_primary=True, max_length=ID_MAX_LENGTH
        )
        schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=self._dim)

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
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

        # 集合已存在时**校验维度**：换过 EMBEDDING_MODEL / EMBEDDING_DIM 之后，
        # 第一次写入会收到一条 Milvus 原生的维度错误，对使用者毫无操作性。
        # 这里提前给出「该重建了」的可执行提示。
        described = client.describe_collection(self._collection)
        for field in described.get("fields", []):
            if field.get("name") != VECTOR_FIELD:
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
        """集合不存在就建（含索引与 load）。幂等。"""
        await anyio.to_thread.run_sync(self._ensure_collection)

    # -- 写入 ---------------------------------------------------------

    def _upsert(self, ids: list[str], vectors: list[list[float]]) -> None:
        rows = [
            {ID_FIELD: str(row_id), VECTOR_FIELD: [float(x) for x in vector]}
            for row_id, vector in zip(ids, vectors)
        ]
        self._get_client().upsert(self._collection, rows)

    async def upsert(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        """按主键 upsert：同 id 覆盖，不产生第二行。"""
        await anyio.to_thread.run_sync(self._upsert, [str(i) for i in ids], list(vectors))

    # -- 删除 ---------------------------------------------------------

    def _delete(self, ids: list[str]) -> None:
        if not ids:
            return
        self._get_client().delete(self._collection, ids=ids)

    async def delete(self, ids: Sequence[str]) -> None:
        """按主键删除向量（增量重建时清理已不存在的旧块）。"""
        await anyio.to_thread.run_sync(self._delete, [str(i) for i in ids])

    # -- 检索 ---------------------------------------------------------

    def _search(self, vector: list[float], top_k: int) -> list[VectorHit]:
        results = self._get_client().search(
            self._collection,
            data=[[float(x) for x in vector]],
            limit=top_k,
            output_fields=[ID_FIELD],
            consistency_level="Strong",
        )
        if not results:
            return []
        return [VectorHit(id=str(h[ID_FIELD]), score=float(h["distance"])) for h in results[0]]

    async def search(self, vector: Sequence[float], top_k: int) -> list[VectorHit]:
        """按余弦相似度取 Top-K。空集合返回空列表（不是错误）。"""
        return await anyio.to_thread.run_sync(self._search, list(vector), top_k)

    # -- 计数 / 重建 ---------------------------------------------------

    def _count(self) -> int:
        # **必须带 Strong**：默认的 bounded staleness 下，刚 upsert 完立刻
        # `query(count(*))` 会返回 0 —— 真机实测踩过。检索与计数必须用同一级一致性，
        # 否则「建库成功、校验说 0 条」会把人引向完全错误的方向。
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
