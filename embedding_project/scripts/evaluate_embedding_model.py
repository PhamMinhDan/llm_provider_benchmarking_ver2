from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from model_presets import get_preset


LOGGER = logging.getLogger("evaluate_embedding_model")


def load_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_labels(path: Path) -> dict[str, set[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for r in rows:
        out[r["query"]] = set(r["relevant_product_ids"])
    return out


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def build_corpus_from_csv(csv_path: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(csv_path)
    required = {"product_id", "searchable_text"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required - set(df.columns)}")
    df = df.dropna(subset=["product_id", "searchable_text"]).copy()
    df["product_id"] = df["product_id"].astype(str)
    df["searchable_text"] = df["searchable_text"].astype(str)
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    return df["product_id"].tolist(), df["searchable_text"].tolist()


def top_k_search(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    scores = np.matmul(query_embeddings, corpus_embeddings.T)
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    row_idx = np.arange(scores.shape[0])[:, None]
    topk_scores = scores[row_idx, topk_idx]
    order = np.argsort(-topk_scores, axis=1)
    return topk_idx[row_idx, order]


def precision_recall_mrr_ndcg_at_k(
    retrieved_ids: list[list[str]],
    query_texts: list[str],
    labels: dict[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    precision_list = []
    recall_list = []
    mrr_list = []
    ndcg_list = []

    for q, hits in zip(query_texts, retrieved_ids):
        relevant = labels.get(q, set())
        if not relevant:
            continue

        top_hits = hits[:k]
        hit_flags = [1 if pid in relevant else 0 for pid in top_hits]
        hit_count = sum(hit_flags)
        precision = hit_count / k
        recall = hit_count / len(relevant)

        rr = 0.0
        for rank, flag in enumerate(hit_flags, start=1):
            if flag:
                rr = 1.0 / rank
                break

        dcg = sum((flag / np.log2(i + 2)) for i, flag in enumerate(hit_flags))
        ideal_flags = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
        idcg = sum((flag / np.log2(i + 2)) for i, flag in enumerate(ideal_flags))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precision_list.append(precision)
        recall_list.append(recall)
        mrr_list.append(rr)
        ndcg_list.append(ndcg)

    return {
        f"Precision@{k}": float(np.mean(precision_list)) if precision_list else 0.0,
        f"Recall@{k}": float(np.mean(recall_list)) if recall_list else 0.0,
        f"MRR@{k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
    }


def load_model(model_name_or_path: str, trust_remote_code: bool = False) -> SentenceTransformer:
    if trust_remote_code:
        return SentenceTransformer(model_name_or_path, trust_remote_code=True)
    return SentenceTransformer(model_name_or_path)


def model_cache_key(model_name_or_path: str) -> str:
    return hashlib.sha256(model_name_or_path.encode("utf-8")).hexdigest()[:16]


def evaluate_model(
    model_name_or_path: str,
    product_ids: list[str],
    corpus_texts: list[str],
    query_texts: list[str],
    labels: dict[str, set[str]],
    k: int = 10,
    trust_remote_code: bool = False,
    encode_batch_size: int = 128,
    cache_dir: Path | None = None,
) -> dict[str, float]:
    LOGGER.info("Loading model for evaluation: %s", model_name_or_path)
    model = load_model(model_name_or_path, trust_remote_code=trust_remote_code)

    key = model_cache_key(model_name_or_path)
    corpus_cache = cache_dir / f"{key}_corpus.npy" if cache_dir else None
    query_cache = cache_dir / f"{key}_queries.npy" if cache_dir else None

    if corpus_cache and corpus_cache.is_file():
        corpus_embeddings = np.load(corpus_cache)
    else:
        corpus_embeddings = model.encode(
            corpus_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        corpus_embeddings = normalize(np.asarray(corpus_embeddings, dtype=np.float32))
        if corpus_cache:
            corpus_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(corpus_cache, corpus_embeddings)

    if query_cache and query_cache.is_file():
        query_embeddings = np.load(query_cache)
    else:
        query_embeddings = model.encode(
            query_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        query_embeddings = normalize(np.asarray(query_embeddings, dtype=np.float32))
        if query_cache:
            np.save(query_cache, query_embeddings)

    topk_indices = top_k_search(query_embeddings, corpus_embeddings, k=k)
    retrieved_ids = [[product_ids[idx] for idx in row] for row in topk_indices]
    return precision_recall_mrr_ndcg_at_k(retrieved_ids, query_texts, labels, k=k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained vs fine-tuned embedding models.")
    parser.add_argument("--preset", choices=["minilm", "bge-m3", "e5-base"], default="minilm")
    parser.add_argument("--test-jsonl", type=Path, default=Path("embedding_project/data/test_cleaned.jsonl"))
    parser.add_argument(
        "--labels-json",
        type=Path,
        default=Path("embedding_project/data/query_product_labels_cleaned.json"),
    )
    parser.add_argument(
        "--products-csv",
        type=Path,
        default=Path("embedding_project/data/merged_products_vi_cleaned.csv"),
    )
    parser.add_argument("--pretrained-model", default=None)
    parser.add_argument("--finetuned-model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--encode-batch-size", type=int, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("embedding_project/outputs/evaluation/embedding_cache"),
        help="Cache corpus/query embeddings (giảm thời gian chạy lại).",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--only-pretrained", action="store_true")
    parser.add_argument("--only-finetuned", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    preset = get_preset(args.preset)

    pretrained = args.pretrained_model or preset.base_model
    finetuned = args.finetuned_model or preset.finetuned_rel_path
    output = args.output or Path("embedding_project/outputs/evaluation") / preset.metrics_filename
    if torch.cuda.is_available():
        encode_batch = args.encode_batch_size or (8 if args.preset == "bge-m3" else 128)
        LOGGER.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    else:
        encode_batch = args.encode_batch_size or (4 if args.preset == "bge-m3" else 64)
        LOGGER.warning("Không có GPU — BGE-M3 trên CPU có thể mất 1–3 giờ.")

    cache_dir = None if args.no_cache else args.cache_dir / args.preset

    test_records = load_jsonl(args.test_jsonl)
    labels = load_labels(args.labels_json)
    product_ids, corpus_texts = build_corpus_from_csv(args.products_csv)
    query_texts = sorted({r["query"] for r in test_records if r.get("query") in labels})

    if not query_texts:
        raise ValueError("No valid queries found for evaluation.")

    trust = preset.trust_remote_code
    result = defaultdict(dict)
    result["preset"] = preset.name
    result["k"] = args.k

    if args.preset == "e5-base":
        # E5 retrieval convention: query/passage prefixes improve quality.
        query_texts = [f"query: {q}" for q in query_texts]
        corpus_texts = [f"passage: {t}" for t in corpus_texts]

    run_pretrained = not args.only_finetuned
    run_finetuned = not args.only_pretrained

    if run_pretrained:
        result["pretrained"] = evaluate_model(
            pretrained,
            product_ids,
            corpus_texts,
            query_texts,
            labels,
            k=args.k,
            trust_remote_code=trust,
            encode_batch_size=encode_batch,
            cache_dir=cache_dir,
        )
    if run_finetuned:
        result["finetuned"] = evaluate_model(
            finetuned,
            product_ids,
            corpus_texts,
            query_texts,
            labels,
            k=args.k,
            trust_remote_code=trust,
            encode_batch_size=encode_batch,
            cache_dir=cache_dir,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Saved metrics to %s", output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
