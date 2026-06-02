"""
Encode products → BGE-M3 fine-tuned → upsert Qdrant.

Mặc định: embedding_project/data/merged_products_vi_cleaned.csv (searchable_text)

Usage (repo root):
  python vector_db/03_index_to_qdrant.py --recreate
  python vector_db/03_index_to_qdrant.py --products-csv embedding_project/data/merged_products_vi_cleaned.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vector_db.config import settings
from vector_db.data_loader import load_products
from vector_db.embedding_service import embedding_service
from vector_db.qdrant_service import make_vector_id, qdrant_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("index_to_qdrant")


def build_payload(product: dict) -> dict:
    source = str(product["source"])
    product_id = str(product["product_id"])
    return {
        "vector_id": make_vector_id(source, product_id),
        "product_id": product_id,
        "source": source,
        "title": product.get("title", ""),
        "description": product.get("description", ""),
        "category": product.get("category", ""),
        "brand": product.get("brand", ""),
        "price": product.get("price"),
        "rating": product.get("rating"),
        "reviews_count": product.get("reviews_count"),
        "image_url": product.get("image_url", ""),
        "tags": product.get("tags", ""),
        "color": product.get("color", ""),
        "size": product.get("size", ""),
        "searchable_text": str(product["searchable_text"]),
    }


def main(
    products_path: Path,
    batch_size: int,
    encode_batch_size: int,
    recreate_collection: bool,
    from_files: bool,
    vectors_npy: Path | None,
    payloads_json: Path | None,
) -> None:
    import numpy as np

    if from_files and vectors_npy and payloads_json:
        vectors = np.load(vectors_npy)
        payloads = json.loads(payloads_json.read_text(encoding="utf-8"))
        LOGGER.info("Loaded precomputed: %s (%d vectors)", vectors_npy, len(vectors))
    else:
        LOGGER.info("Loading: %s", products_path)
        products = load_products(products_path)
        payloads = [build_payload(p) for p in products]
        texts = [p["searchable_text"] for p in payloads]
        LOGGER.info("Products: %d | Encode searchable_text...", len(texts))
        vectors = embedding_service.encode(texts, batch_size=encode_batch_size)

    if len(payloads) != len(vectors):
        raise ValueError(f"Mismatch: {len(payloads)} payloads vs {len(vectors)} vectors")

    embeddings = vectors.tolist()
    vector_size = len(embeddings[0])
    qdrant_service.create_collection(vector_size=vector_size, recreate=recreate_collection)

    LOGGER.info("Upsert → collection '%s'", settings.QDRANT_COLLECTION)
    uploaded = qdrant_service.upsert_products(
        payloads=payloads,
        embeddings=embeddings,
        batch_size=batch_size,
    )
    LOGGER.info("Collection: %s", qdrant_service.get_collection_info())
    LOGGER.info("Uploaded: %d", uploaded)

    results = qdrant_service.search(embeddings[0], top_k=3)
    for r in results:
        LOGGER.info("  [%.4f] %s", r["score"], (r.get("title") or "")[:70])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index products to Qdrant (BGE-M3).")
    p.add_argument(
        "--products-csv",
        default=settings.PRODUCTS_CSV,
        help="CSV merged_products_vi_cleaned (searchable_text)",
    )
    p.add_argument(
        "--products-jsonl",
        default=None,
        help="Nếu set: dùng JSONL thay CSV",
    )
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--encode-batch-size", type=int, default=settings.EMBEDDING_BATCH_SIZE)
    p.add_argument("--recreate", action="store_true")
    p.add_argument("--from-files", action="store_true")
    p.add_argument(
        "--vectors",
        default=str(_REPO_ROOT / "embedding_project/outputs/embeddings/product_vectors_bge_m3.npy"),
    )
    p.add_argument(
        "--payloads",
        default=str(_REPO_ROOT / "embedding_project/outputs/embeddings/product_payloads_bge_m3.json"),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    products_path = Path(args.products_jsonl) if args.products_jsonl else Path(args.products_csv)
    main(
        products_path=products_path,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        recreate_collection=args.recreate,
        from_files=args.from_files,
        vectors_npy=Path(args.vectors) if args.from_files else None,
        payloads_json=Path(args.payloads) if args.from_files else None,
    )
