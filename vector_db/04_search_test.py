"""Thử search Qdrant với query tiếng Việt + BGE-M3 fine-tuned."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vector_db.embedding_service import embedding_service
from vector_db.qdrant_service import qdrant_service

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main(query: str, top_k: int) -> None:
    vec = embedding_service.encode_query(query)
    results = qdrant_service.search(vec, top_k=top_k)
    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {r['title'][:80]}")
        print(f"   {r['product_id']} | {r['brand']} | {r['price']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="giày chạy bộ nam nhẹ")
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    main(args.query, args.top_k)
