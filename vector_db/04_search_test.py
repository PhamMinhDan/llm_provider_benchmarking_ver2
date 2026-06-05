"""Thử search Qdrant — hỗ trợ BGE-M3 và E5 fine-tuned."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vector_db.config import apply_runtime_config, settings

logging.basicConfig(level=logging.INFO, format="%(message)s")

DEFAULT_E5_2EP_MODEL = _REPO_ROOT / "embedding_project/models/e5_base_finetuned_2ep_final"
DEFAULT_E5_2EP_COLLECTION = "products_vi_e5_2ep"


def main(
    query: str,
    top_k: int,
    collection: str | None = None,
    model_path: str | None = None,
    use_e5_prefix: bool | None = None,
    preset_e5_2ep: bool = False,
) -> None:
    from vector_db.embedding_service import EmbeddingService
    from vector_db.qdrant_service import QdrantService

    if preset_e5_2ep:
        collection = collection or DEFAULT_E5_2EP_COLLECTION
        model_path = model_path or str(DEFAULT_E5_2EP_MODEL)
        use_e5_prefix = True

    cfg = apply_runtime_config(
        collection=collection,
        model_path=model_path,
        use_e5_prefix=use_e5_prefix,
    )
    print(f"Collection: {cfg.QDRANT_COLLECTION} | E5 prefix: {cfg.EMBEDDING_USE_E5_PREFIX}")

    emb = EmbeddingService()
    qdr = QdrantService()
    vec = emb.encode_query(query)
    results = qdr.search(vec, top_k=top_k)
    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {(r.get('title') or '')[:80]}")
        print(f"   {r['product_id']} | {r.get('brand', '')} | {r.get('price', '')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="giày chạy bộ nam nhẹ")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--collection", default=None)
    p.add_argument("--model-path", default=None)
    p.add_argument("--e5-prefix", action="store_true")
    p.add_argument("--preset-e5-2ep", action="store_true")
    args = p.parse_args()

    main(
        args.query,
        args.top_k,
        collection=args.collection,
        model_path=args.model_path,
        use_e5_prefix=True if args.e5_prefix else None,
        preset_e5_2ep=args.preset_e5_2ep,
    )
