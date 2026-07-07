"""
Benchmark 1 — Exact Retrieval Evaluation (trên tập test_v2.jsonl)
================================================================
So sánh 3 retrieval models trên catalog 28k products:

  1. BM25                 (rank_bm25.OkapiBM25, keyword-based baseline)
  2. E5-base (pretrained) (intfloat/multilingual-e5-base, multilingual semantic)
  3. E5-FT-V2             (finetuned trên triplets, semantic, in-domain)

Queries: test_v2.jsonl (đã có ground truth product_id)

Mỗi query có ground truth product_id → tính:
  - Recall@K  (1 nếu GT nằm trong top-K, 0 nếu không)
  - MRR       (1 / rank của GT, 0 nếu không trong top-K)
  - NDCG@K    (binary relevance, dùng công thức chuẩn)

Run:
  python embedding_project/scripts/evaluate_llm_judge.py
  python embedding_project/scripts/evaluate_llm_judge.py --top-k 10
  python embedding_project/scripts/evaluate_llm_judge.py --skip-retrieval --cache cache.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

BASE_DIR = Path("embedding_project")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" / "benchmark1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths (local)
MODEL_PRETRAINED = "intfloat/multilingual-e5-base"
MODEL_FT_V2 = str(BASE_DIR / "models" / "e5_base_finetune_v2")

# Corpus (dùng catalog 28k cho cả BM25 + semantic)
CORPUS_CSV = DATA_DIR / "Dataset_DATN_28k.csv"

# Test queries (test_v2.jsonl đã có ground truth product_id)
TEST_V2_JSONL = DATA_DIR / "test_v2.jsonl"

# Evaluation params
RETRIEVE_TOP_K = 10

LOGGER = logging.getLogger("benchmark1_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


# ──────────────────────────────────────────────
# CORPUS LOADER
# ──────────────────────────────────────────────

def load_corpus(corpus_csv: Path) -> tuple[list[str], list[dict]]:
    """Load catalog CSV, trả về (product_ids, product_dicts)."""
    df = pd.read_csv(corpus_csv)
    df = df.dropna(subset=["product_id", "product_name"]).copy()
    df["product_id"] = df["product_id"].astype(str)
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    df = df.reset_index(drop=True)

    product_ids = df["product_id"].tolist()

    # Build corpus text (consistent với training pipeline)
    def build_text(row):
        parts = []
        for col in ["product_name", "description", "category_name", "brand"]:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return " | ".join(parts)[:512]

    corpus_texts = [build_text(row) for _, row in df.iterrows()]

    product_dicts = df.to_dict("records")
    LOGGER.info(f"Loaded corpus: {len(product_ids)} products")
    return product_ids, product_dicts


# ──────────────────────────────────────────────
# BM25 RETRIEVAL
# ──────────────────────────────────────────────

class BM25Retriever:
    def __init__(self, corpus_csv: Path):
        self.product_ids, self.product_dicts = load_corpus(corpus_csv)
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if len(t) > 1 or t.isdigit()]

    def _build_index(self):
        corpus_texts = []
        for p in self.product_dicts:
            parts = []
            for col in ["product_name", "description", "category_name", "brand"]:
                val = p.get(col)
                if pd.notna(val):
                    parts.append(str(val))
            corpus_texts.append(" ".join(parts))
        self.corpus_texts_raw = corpus_texts
        tokenized = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
        LOGGER.info(f"BM25 index: {len(self.product_ids)} products")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(top_indices, 1):
            p = self.product_dicts[idx]
            results.append(self._make_result(p, rank, float(scores[idx])))
        return results

    def _make_result(self, p: dict, rank: int, raw_score: float) -> dict:
        return {
            "rank": rank,
            "product_id": str(p.get("product_id", "")),
            "title": str(p.get("product_name", "")),
            "brand": str(p.get("brand", "")) if pd.notna(p.get("brand")) else "",
            "category": str(p.get("category_name", "")) if pd.notna(p.get("category_name")) else "",
            "description": str(p.get("description", ""))[:300] if pd.notna(p.get("description")) else "",
            "raw_score": raw_score,
        }


# ──────────────────────────────────────────────
# SEMANTIC RETRIEVAL (local in-memory)
# ──────────────────────────────────────────────

class SemanticRetriever:
    """
    Semantic retrieval dùng local in-memory cosine similarity.
    Encode corpus 1 lần → reuse cho tất cả queries.
    Dùng cho pretrained và từng fine-tuned model riêng biệt.
    """
    def __init__(self, model_path: str, corpus_csv: Path, batch_size: int = 64):
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info(f"Loading model: {model_path} on {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)

        # Load corpus
        self.product_ids, self.product_dicts = load_corpus(corpus_csv)

        # Encode corpus
        LOGGER.info(f"Encoding corpus ({len(self.product_ids)} products)...")
        corpus_texts = []
        for p in self.product_dicts:
            parts = []
            for col in ["product_name", "description", "category_name", "brand"]:
                val = p.get(col)
                if pd.notna(val):
                    parts.append(str(val))
            # E5 format: thêm prefix
            corpus_texts.append("passage: " + " | ".join(parts))
        self.corpus_embs = self.model.encode(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.corpus_embs_np = self.corpus_embs.cpu().numpy()
        LOGGER.info(f"Corpus encoded: {self.corpus_embs.shape}")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        start = time.time()
        # Encode query (E5 format)
        query_emb = self.model.encode(
            [f"query: {query}"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        ).cpu().numpy()[0]
        latency_ms = int((time.time() - start) * 1000)

        # Cosine similarity (vectors đã normalized)
        scores = np.dot(self.corpus_embs_np, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            p = self.product_dicts[idx]
            results.append(self._make_result(p, rank, float(scores[idx]), latency_ms))
        return results

    def _make_result(self, p: dict, rank: int, score: float, latency_ms: int) -> dict:
        return {
            "rank": rank,
            "product_id": str(p.get("product_id", "")),
            "title": str(p.get("product_name", "")),
            "brand": str(p.get("brand", "")) if pd.notna(p.get("brand")) else "",
            "category": str(p.get("category_name", "")) if pd.notna(p.get("category_name")) else "",
            "description": str(p.get("description", ""))[:300] if pd.notna(p.get("description")) else "",
            "vector_score": score,
            "latency_ms": latency_ms,
        }


# ──────────────────────────────────────────────
# TEST QUERY LOADER (test_v2.jsonl)
# ──────────────────────────────────────────────

def load_test_v2(path: Path) -> list[dict]:
    """Load test_v2.jsonl, mỗi item có product_id (GT) + anchors (queries)."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_test_queries(test_v2: list[dict], catalog_df: pd.DataFrame) -> list[dict]:
    """
    Từ test_v2.jsonl (triplets format), mỗi item có:
      - 'query': query string
      - 'product_id': ground truth product
    Trả về list[dict] với (query, product_id).
    Chỉ giữ các query có product_id tồn tại trong catalog.
    """
    catalog_ids = set(catalog_df["product_id"].astype(str).tolist())
    queries: list[dict] = []
    for item in test_v2:
        gt_pid = str(item["product_id"])
        if gt_pid not in catalog_ids:
            continue
        q = str(item.get("query", "")).strip()
        if q:
            queries.append({"query": q, "product_id": gt_pid})
    return queries


# ──────────────────────────────────────────────
# METRIC COMPUTATION (binary relevance với ground truth)
# ──────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], gt_id: str, k: int = 10) -> int:
    """1 nếu ground truth nằm trong top-K, 0 nếu không."""
    return int(gt_id in retrieved_ids[:k])


def mrr(retrieved_ids: list[str], gt_id: str) -> float:
    """Mean Reciprocal Rank: 1/rank của ground truth, 0 nếu không có."""
    for i, pid in enumerate(retrieved_ids, start=1):
        if pid == gt_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gt_id: str, k: int = 10) -> float:
    """NDCG@K với binary relevance (1 nếu GT ở rank i trong top-K, 0 nếu không)."""
    k = min(k, len(retrieved_ids))
    actual = [1.0 if pid == gt_id else 0.0 for pid in retrieved_ids[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(actual))
    ideal = [1.0] + [0.0] * (k - 1)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────
# REPORT GENERATION
# ──────────────────────────────────────────────

def generate_report(
    model_metrics: dict[str, dict],
    n_queries: int,
    top_k: int,
    output_path: Optional[Path] = None,
) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append(f"BENCHMARK 1 — EXACT RETRIEVAL (test_v2.jsonl, {n_queries} queries, top-{top_k})")
    lines.append("=" * 70)

    model_order = ["bm25", "e5_pretrained", "e5_ft_v2"]
    model_labels = {
        "bm25": "BM25 (baseline)",
        "e5_pretrained": "E5-base (pretrained)",
        "e5_ft_v2": "E5-FT-V2 (finetuned)",
    }

    lines.append("")
    lines.append("## Kết quả chính")
    lines.append("")
    header = f"{'Model':<24} | {'Recall@5':>9} | {'Recall@10':>10} | {'MRR':>8} | {'NDCG@10':>9} | {'Latency':>9}"
    lines.append(header)
    lines.append("-" * len(header))

    for key in model_order:
        if key not in model_metrics:
            continue
        m = model_metrics[key]
        label = model_labels.get(key, key)
        lines.append(
            f"{label:<24} | "
            f"{m['recall_at_5']*100:>8.1f}% | "
            f"{m['recall_at_10']*100:>9.1f}% | "
            f"{m['mrr']:>8.3f} | "
            f"{m['ndcg_at_10']:>9.3f} | "
            f"{m['mean_latency_ms']:>8.0f}ms"
        )

    lines.append("")
    lines.append("**Chú thích:**")
    lines.append(f"  - Recall@K: tỷ lệ query có ground truth nằm trong top-{K if (K:=5) else ''}{'' if False else ''}")
    lines.append("  - MRR: trung bình 1/rank của ground truth")
    lines.append("  - NDCG@10: normalized discounted cumulative gain (binary relevance)")
    lines.append("  - Latency: thời gian retrieval trung bình (ms)")

    # Winner highlight
    if "e5_ft_v2" in model_metrics and "e5_pretrained" in model_metrics:
        ft_recall = model_metrics["e5_ft_v2"]["recall_at_10"]
        pre_recall = model_metrics["e5_pretrained"]["recall_at_10"]
        delta = (ft_recall - pre_recall) * 100
        sign = "+" if delta >= 0 else ""
        lines.append("")
        lines.append(f"**Δ Recall@10 (FT-V2 vs Pretrained): {sign}{delta:.1f}%**")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    if output_path:
        output_path.write_text(report_text, encoding="utf-8")
        LOGGER.info(f"Report saved: {output_path}")
    return report_text


# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def run_pipeline(
    top_k: int = RETRIEVE_TOP_K,
    skip_retrieval: bool = False,
    retrieval_cache_path: Optional[Path] = None,
):
    session_id = str(uuid.uuid4())[:8]
    LOGGER.info(f"Session ID: {session_id}")

    # ── Step 1: Load data ──
    LOGGER.info("Loading data...")
    test_v2 = load_test_v2(TEST_V2_JSONL)
    LOGGER.info(f"  test_v2: {len(test_v2)} items")

    catalog_df = pd.read_csv(CORPUS_CSV)
    catalog_df["product_id"] = catalog_df["product_id"].astype(str)
    catalog_df = catalog_df.drop_duplicates(subset=["product_id"], keep="first")
    catalog_df = catalog_df.reset_index(drop=True)
    LOGGER.info(f"  catalog: {len(catalog_df)} products")

    # ── Step 2: Build queries (từ test_v2) ──
    queries = build_test_queries(test_v2, catalog_df)
    for i, q in enumerate(queries):
        q["query_id"] = i
    LOGGER.info(f"  queries: {len(queries)} (ground truth product_id đã có)")

    # ── Step 3: Retrieval (3 models) ──
    if skip_retrieval and retrieval_cache_path and retrieval_cache_path.exists():
        LOGGER.info(f"Loading cached retrieval from {retrieval_cache_path}")
        all_retrieval = json.loads(retrieval_cache_path.read_text(encoding="utf-8"))
    else:
        LOGGER.info(f"Running retrieval cho {len(queries)} queries × 3 models (top-{top_k})...")

        bm25 = BM25Retriever(CORPUS_CSV)

        semantic_retrievers: dict[str, SemanticRetriever] = {}

        def get_semantic(model_key: str, model_path: str) -> SemanticRetriever:
            if model_key not in semantic_retrievers:
                LOGGER.info(f"  Loading {model_key} ({model_path})...")
                semantic_retrievers[model_key] = SemanticRetriever(model_path, CORPUS_CSV)
            return semantic_retrievers[model_key]

        # retrieved[model_key] = list of (query_id, retrieved_pids, latency_ms)
        all_retrieval: dict[str, list[dict]] = {
            "bm25": [],
            "e5_pretrained": [],
            "e5_ft_v2": [],
        }

        model_tasks = [
            ("bm25", None),
            ("e5_pretrained", MODEL_PRETRAINED),
            ("e5_ft_v2", MODEL_FT_V2),
        ]

        for i, qitem in enumerate(queries):
            query = qitem["query"]
            qid = qitem["query_id"]
            if i % 100 == 0:
                LOGGER.info(f"  [{i}/{len(queries)}] '{query[:40]}...'")

            for model_key, model_path in model_tasks:
                if model_key == "bm25":
                    results = bm25.search(query, top_k=top_k)
                else:
                    try:
                        retriever = get_semantic(model_key, model_path)
                        results = retriever.search(query, top_k=top_k)
                    except Exception as e:
                        LOGGER.warning(f"  [{model_key}] error: {e}")
                        results = []

                retrieved_pids = [str(r["product_id"]) for r in results]
                latency_ms = results[0].get("latency_ms", 0.0) if results else 0.0
                all_retrieval[model_key].append({
                    "query_id": qid,
                    "query": query,
                    "ground_truth_pid": qitem["product_id"],
                    "retrieved_pids": retrieved_pids,
                    "latency_ms": latency_ms,
                })

        if retrieval_cache_path:
            retrieval_cache_path.write_text(
                json.dumps(all_retrieval, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            LOGGER.info(f"Cached retrieval → {retrieval_cache_path}")

    # ── Step 4: Compute metrics ──
    LOGGER.info("Computing metrics...")
    model_metrics: dict[str, dict] = {}

    for model_key, results in all_retrieval.items():
        recalls_5, recalls_10, mrrs, ndcgs, latencies = [], [], [], [], []
        for r in results:
            retrieved = r["retrieved_pids"]
            gt = r["ground_truth_pid"]
            recalls_5.append(recall_at_k(retrieved, gt, k=5))
            recalls_10.append(recall_at_k(retrieved, gt, k=10))
            mrrs.append(mrr(retrieved, gt))
            ndcgs.append(ndcg_at_k(retrieved, gt, k=10))
            latencies.append(r.get("latency_ms", 0.0))

        model_metrics[model_key] = {
            "recall_at_5": float(np.mean(recalls_5)) if recalls_5 else 0.0,
            "recall_at_10": float(np.mean(recalls_10)) if recalls_10 else 0.0,
            "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
            "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "n_queries": len(results),
        }

    # ── Step 5: Save outputs ──
    output_file = OUT_DIR / f"benchmark1_results_{session_id}.json"
    output_file.write_text(json.dumps({
        "session_id": session_id,
        "n_queries": len(queries),
        "top_k": top_k,
        "model_metrics": model_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(f"Results saved: {output_file}")

    report_path = OUT_DIR / f"benchmark1_report_{session_id}.md"
    report_text = generate_report(model_metrics, len(queries), top_k, output_path=report_path)
    print("\n" + report_text)

    LOGGER.info("Benchmark 1 complete!")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark 1 — Exact Retrieval Evaluation (test_v2.jsonl)")
    parser.add_argument("--top-k", type=int, default=RETRIEVE_TOP_K, help="Top-K retrieval (mặc định 10)")
    parser.add_argument("--skip-retrieval", action="store_true", help="Dùng cached retrieval")
    parser.add_argument("--cache", type=str, default="", help="Path đến cached retrieval results")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else OUT_DIR / "retrieval_cache.json"

    run_pipeline(
        top_k=args.top_k,
        skip_retrieval=args.skip_retrieval,
        retrieval_cache_path=cache_path if args.skip_retrieval else None,
    )


if __name__ == "__main__":
    main()
