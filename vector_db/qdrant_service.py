"""
Qdrant — collection, upsert, search cho semantic product search (BGE-M3).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    UpdateStatus,
    VectorParams,
)

from vector_db.config import get_settings

LOGGER = logging.getLogger(__name__)

VECTOR_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def make_vector_id(source: str, product_id: str) -> str:
    """Khớp create_product_vectors.py — id ổn định theo source + product_id."""
    key = f"{source}_{product_id}"
    return str(uuid.uuid5(VECTOR_NAMESPACE, key))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and np_is_nan(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def np_is_nan(v: Any) -> bool:
    import math

    return isinstance(v, float) and math.isnan(v)


class QdrantService:
    _instance: "QdrantService | None" = None

    def __new__(cls) -> "QdrantService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.client: QdrantClient | None = None
        self.collection_name = get_settings().QDRANT_COLLECTION
        self._initialized = True

    def _ensure_client(self) -> QdrantClient:
        if self.client is None:
            url = get_settings().QDRANT_URL
            LOGGER.info("Connecting Qdrant: %s", url)
            client_kwargs: dict[str, Any] = {
                "url": url,
                "timeout": 120,
                "check_compatibility": False,
            }
            if get_settings().QDRANT_API_KEY:
                client_kwargs["api_key"] = get_settings().QDRANT_API_KEY
            # Chỉ dùng https=True khi URL là http:// (client tự nâng cấp TLS)
            if get_settings().QDRANT_HTTPS and url.startswith("http://"):
                client_kwargs["https"] = True
            self.client = QdrantClient(**client_kwargs)
            LOGGER.info("Collection: %s", self.collection_name)
        return self.client

    def create_collection(self, vector_size: int, recreate: bool = False) -> None:
        client = self._ensure_client()
        existing = [c.name for c in client.get_collections().collections]

        if self.collection_name in existing:
            if recreate:
                LOGGER.warning("Recreating collection '%s'...", self.collection_name)
                client.delete_collection(self.collection_name)
            else:
                LOGGER.info("Collection '%s' already exists — skip create.", self.collection_name)
                return

        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        for field, schema in [
            ("product_id", PayloadSchemaType.KEYWORD),
            ("source", PayloadSchemaType.KEYWORD),
            ("category", PayloadSchemaType.KEYWORD),
            ("brand", PayloadSchemaType.KEYWORD),
            ("price", PayloadSchemaType.FLOAT),
        ]:
            client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=schema,
            )

        LOGGER.info(
            "Created collection '%s' (size=%d, COSINE)",
            self.collection_name,
            vector_size,
        )

    def get_collection_info(self) -> dict[str, Any]:
        try:
            info = self._ensure_client().get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
            }
        except Exception as e:
            return {"error": str(e)}

    def upsert_products(
        self,
        payloads: list[dict[str, Any]],
        embeddings: list[list[float]],
        batch_size: int = 50,
    ) -> int:
        if len(payloads) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(payloads)} payloads vs {len(embeddings)} embeddings"
            )

        points: list[PointStruct] = []
        for payload, vector in zip(payloads, embeddings):
            pid = str(payload.get("product_id", ""))
            source = str(payload.get("source", ""))
            point_id = payload.get("vector_id") or make_vector_id(source, pid)

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "vector_id": point_id,
                        "product_id": pid,
                        "source": source,
                        "title": str(payload.get("title", "") or ""),
                        "category": str(payload.get("category", "") or ""),
                        "brand": str(payload.get("brand", "") or ""),
                        "price": _safe_float(payload.get("price")),
                        "rating": _safe_float(payload.get("rating"), default=0.0),
                        "reviews_count": int(payload.get("reviews_count") or 0),
                        "image_url": str(payload.get("image_url", "") or ""),
                        "tags": str(payload.get("tags", "") or ""),
                        "color": str(payload.get("color", "") or ""),
                        "size": str(payload.get("size", "") or ""),
                        "searchable_text": str(payload.get("searchable_text", "") or ""),
                        "description": str(payload.get("description", "") or ""),
                    },
                )
            )

        uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            result = self._ensure_client().upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            if result.status == UpdateStatus.COMPLETED:
                uploaded += len(batch)
            LOGGER.info("Upserted %d / %d", uploaded, len(points))

        return uploaded

    def search(
        self,
        query_vector: list[float],
        top_k: int | None = None,
        category_filter: str | None = None,
        brand_filter: str | None = None,
        price_range: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        top_k = top_k or get_settings().DEFAULT_TOP_K
        conditions = []

        if category_filter:
            conditions.append(
                FieldCondition(key="category", match=MatchValue(value=category_filter))
            )
        if brand_filter:
            conditions.append(
                FieldCondition(key="brand", match=MatchValue(value=brand_filter))
            )
        if price_range:
            min_p, max_p = price_range
            range_args: dict[str, float] = {}
            if min_p is not None:
                range_args["gte"] = min_p
            if max_p is not None:
                range_args["lte"] = max_p
            if range_args:
                conditions.append(FieldCondition(key="price", range=Range(**range_args)))

        query_filter = Filter(must=conditions) if conditions else None

        response = self._ensure_client().query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        output: list[dict[str, Any]] = []
        for r in response.points:
            p = r.payload or {}
            output.append(
                {
                    "vector_id": p.get("vector_id", ""),
                    "product_id": p.get("product_id", ""),
                    "source": p.get("source", ""),
                    "title": p.get("title", ""),
                    "category": p.get("category", ""),
                    "brand": p.get("brand", ""),
                    "price": p.get("price", 0),
                    "image_url": p.get("image_url", ""),
                    "searchable_text": p.get("searchable_text", ""),
                    "description": p.get("description", ""),
                    "score": float(r.score),
                }
            )
        return output


qdrant_service = QdrantService()
