"""Đánh giá E5 + Reranker với TREC-style evaluation data.

Tách biệt INPUT vs GROUND TRUTH:
- queries.jsonl: Input (qid, query) - model nhìn thấy
- qrels.jsonl: Ground truth (qid, product_id, relevance) - chấm điểm

1. Precision@10, Recall@10, F1@10 với reranking
2. Ngưỡng tối ưu (EER, FPR=FNR)
3. Tìm n và k tối ưu
4. Cải thiện Precision@10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("evaluate_reranker_trec")

# ===== FILE LOADING =====

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_trec_queries(queries_path: Path) -> tuple[dict[str, str], list[str]]:
    """Load queries TREC-style: {qid, query} -> returns (qid_to_query, query_list)"""
    rows = load_jsonl(queries_path)
    qid_to_query = {}
    query_list = []
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        query = str(row.get("query", "")).strip()
        if qid and query:
            qid_to_query[qid] = query
            query_list.append(query)
    return qid_to_query, query_list


def load_trec_qrels(qrels_path: Path) -> dict[str, set[str]]:
    """Load qrels TREC-style: {qid, product_id, relevance} -> returns {qid: {product_ids}}"""
    rows = load_jsonl(qrels_path)
    qid_to_pids = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        pid = str(row.get("product_id", "")).strip()
        rel = row.get("relevance", 1)
        if qid and pid and rel > 0:
            qid_to_pids.setdefault(qid, set()).add(pid)
    return qid_to_pids


def load_corpus(corpus_path: Path) -> pd.DataFrame:
    df = pd.read_csv(corpus_path)
    df = df.dropna(subset=["product_id", "searchable_text"])
    df = df.drop_duplicates(subset=["product_id"])
    df["product_id"] = df["product_id"].astype(str)
    return df


# ===== EMBEDDING =====

def normalize(vectors: np.ndarray) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


def topk_indices(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> np.ndarray:
    scores = query_emb @ corpus_emb.T
    k_eff = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = np.arange(scores.shape[0])[:, None]
    top_scores = scores[rows, idx]
    order = np.argsort(-top_scores, axis=1)
    return idx[rows, order]


def encode_corpus_and_queries(
    emb: SentenceTransformer,
    corpus_texts: list[str],
    query_texts: list[str],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    corpus_e5 = [f"passage: {t}" for t in corpus_texts]
    queries_e5 = [f"query: {q}" for q in query_texts]
    corpus_emb = normalize(
        emb.encode(corpus_e5, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    )
    query_emb = normalize(
        emb.encode(queries_e5, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    )
    return corpus_emb, query_emb


# ===== RERANKER =====

def load_reranker(model_dir: Path, device: str):
    sys.path.insert(0, str(model_dir.resolve()))
    from transformers import AutoConfig, AutoModelForSequenceClassification

    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    if hasattr(config, "use_flash_attn"):
        config.use_flash_attn = device != "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        config=config,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.name_or_path = str(model_dir)
    return model


def score_pairs(
    reranker,
    pairs: list[tuple[str, str]],
    batch_size: int,
    device: str,
) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float32)
    scores = reranker.compute_score(pairs, batch_size=batch_size)
    return np.asarray(scores, dtype=np.float32)


# ===== METRICS =====

def compute_metrics(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int,
) -> dict[str, float]:
    """Compute Precision@k, Recall@k, F1@k for lists of retrieved and relevant IDs."""
    p_list, r_list, f1_list, mrr_list, ndcg_list = [], [], [], [], []
    
    for hits, rel in zip(retrieved_ids, relevant_ids):
        if not rel:
            continue
        top_hits = hits[:k]
        hit_count = sum(1 for pid in top_hits if pid in rel)
        
        p = hit_count / k if k > 0 else 0.0
        r = hit_count / len(rel) if len(rel) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        p_list.append(p)
        r_list.append(r)
        f1_list.append(f1)
        
        # MRR
        rr = 0.0
        for i, pid in enumerate(top_hits, 1):
            if pid in rel:
                rr = 1.0 / i
                break
        mrr_list.append(rr)
        
        # NDCG
        flags = [1 if pid in rel else 0 for pid in top_hits]
        dcg = sum(f / np.log2(i + 2) for i, f in enumerate(flags))
        ideal = [1] * min(len(rel), k) + [0] * (k - min(len(rel), k))
        idcg = sum(f / np.log2(i + 2) for i, f in enumerate(ideal))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    
    n = len(p_list)
    return {
        f"Precision@{k}": float(np.mean(p_list)) if p_list else 0.0,
        f"Recall@{k}": float(np.mean(r_list)) if r_list else 0.0,
        f"F1@{k}": float(np.mean(f1_list)) if f1_list else 0.0,
        f"MRR@{k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "n_queries": n,
    }


# ===== THRESHOLD ANALYSIS =====

def threshold_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 201,
) -> dict:
    """Analyze threshold: FPR, FNR, EER, min error rate."""
    if len(scores) == 0:
        return {
            "curve": [],
            "EER": {},
            "min_error": {},
            "optimal": {},
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
        }
    
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if hi <= lo:
        hi = lo + 1e-6
    
    thresholds = np.linspace(lo, hi, n_steps)
    curve = []
    
    for t in thresholds:
        pred = scores >= t
        tp = int(np.sum(pred & (labels == 1)))
        fp = int(np.sum(pred & (labels == 0)))
        tn = int(np.sum(~pred & (labels == 0)))
        fn = int(np.sum(~pred & (labels == 1)))
        
        n_pos = tp + fn
        n_neg = fp + tn
        
        fpr = fp / (n_neg) if n_neg > 0 else 0.0
        fnr = fn / (n_pos) if n_pos > 0 else 0.0
        err = (fp + fn) / len(labels) if len(labels) > 0 else 0.0
        
        curve.append({
            "threshold": float(t),
            "FPR": float(fpr),
            "FNR": float(fnr),
            "error_rate": float(err),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        })
    
    # EER: FPR ≈ FNR
    diff = [abs(c["FPR"] - c["FNR"]) for c in curve]
    eer_idx = int(np.argmin(diff))
    
    # Min error rate
    err_idx = int(np.argmin([c["error_rate"] for c in curve]))
    
    # Find optimal threshold where FPR = FNR (EER point)
    eer_threshold = curve[eer_idx]["threshold"]
    
    return {
        "curve": curve,
        "EER": {
            "threshold": eer_threshold,
            "FPR": curve[eer_idx]["FPR"],
            "FNR": curve[eer_idx]["FNR"],
            "error_rate": curve[eer_idx]["error_rate"],
        },
        "min_error": {
            "threshold": curve[err_idx]["threshold"],
            "FPR": curve[err_idx]["FPR"],
            "FNR": curve[err_idx]["FNR"],
            "error_rate": curve[err_idx]["error_rate"],
        },
        "optimal": {
            "threshold": eer_threshold,
            "description": "Ngưỡng tối ưu tại điểm EER (FPR = FNR)",
            "FPR": curve[eer_idx]["FPR"],
            "FNR": curve[eer_idx]["FNR"],
            "error_rate": curve[eer_idx]["error_rate"],
        },
        "n_pairs": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
    }


# ===== MAIN EVALUATION =====

def build_threshold_dataset(
    query_texts: list[str],
    relevant_ids: list[set[str]],
    bi_topn: list[list[str]],
    id_to_text: dict[str, str],
    max_neg_per_query: int = 5,
    n_pool: int = 30,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Build dataset for threshold analysis: positives + hard negatives from E5 top-n."""
    pairs = []
    labels = []
    
    for qi, (q, rel) in enumerate(zip(query_texts, relevant_ids)):
        if not rel:
            continue
        
        # Positive: first relevant in top-n, or first relevant
        pos_pid = next((p for p in bi_topn[qi] if p in rel), None)
        if pos_pid is None:
            pos_pid = next(iter(rel))
        
        pos_text = id_to_text.get(pos_pid, "")
        if pos_text:
            pairs.append((q, pos_text))
            labels.append(1)
        
        # Hard negatives: from top-n but not relevant
        negs = [p for p in bi_topn[qi][:n_pool] if p not in rel][:max_neg_per_query]
        for pid in negs:
            neg_text = id_to_text.get(pid, "")
            if neg_text:
                pairs.append((q, neg_text))
                labels.append(0)
    
    return pairs, np.asarray(labels, dtype=np.int32)


def run_evaluation(
    embedding_model: Path,
    reranker_model: Path,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    output_path: Path,
    n_values: list[int],
    k_values: list[int],
    eval_k: int = 10,
    encode_batch: int = 128,
    rerank_batch: int = 32,
    max_neg_per_query: int = 5,
    target_recall: float = 0.95,
    device: str = "cuda",
) -> dict:
    """Main evaluation pipeline."""
    
    LOGGER.info("Loading data...")
    corpus_df = load_corpus(corpus_path)
    qid_to_query, query_list = load_trec_queries(queries_path)
    qid_to_rel = load_trec_qrels(qrels_path)
    
    # Map queries to their relevant IDs (aligned by index)
    relevant_ids = []
    query_order = []
    for q in query_list:
        # Find qid for this query
        qid = next((qid for qid, qtext in qid_to_query.items() if qtext == q), None)
        if qid and qid in qid_to_rel:
            relevant_ids.append(qid_to_rel[qid])
            query_order.append(q)
    
    LOGGER.info(f"Corpus: {len(corpus_df)} products")
    LOGGER.info(f"Queries: {len(query_order)} (matched with qrels)")
    
    product_ids = corpus_df["product_id"].tolist()
    corpus_texts = corpus_df["searchable_text"].astype(str).tolist()
    id_to_text = dict(zip(product_ids, corpus_texts))
    
    # Encode
    LOGGER.info("Loading embedding model...")
    emb = SentenceTransformer(str(embedding_model))
    emb.max_seq_length = 512
    
    LOGGER.info("Encoding corpus and queries...")
    corpus_emb, query_emb = encode_corpus_and_queries(emb, corpus_texts, query_order, encode_batch)
    
    # Bi-encoder retrieval at max_n
    max_n = max(n_values)
    LOGGER.info(f"Retrieving top-{max_n} with bi-encoder...")
    topn_idx = topk_indices(query_emb, corpus_emb, k=max_n)
    bi_topn = [[product_ids[i] for i in row] for row in topn_idx]
    
    # Compute metrics for different n values
    recall_by_n = {}
    for n in n_values:
        metrics = compute_metrics(
            [row[:n] for row in bi_topn],
            relevant_ids,
            k=n
        )
        recall_by_n[n] = metrics[f"Recall@{n}"]
        LOGGER.info(f"Bi-encoder Recall@{n}: {metrics[f'Recall@{n}']:.4f}")
    
    # Select n based on target recall
    selected_n = max_n
    for n in sorted(n_values):
        if recall_by_n[n] >= target_recall:
            selected_n = n
            break
    
    LOGGER.info(f"Selected n={selected_n} for target recall={target_recall}")
    
    # Reranking
    LOGGER.info(f"Loading reranker: {reranker_model}")
    reranker = load_reranker(reranker_model, device)
    
    # Score pairs for reranking
    search_rows = [row[:selected_n] for row in bi_topn]
    score_cache = [{} for _ in query_order]
    
    pairs_for_rerank = []
    pair_meta = []
    for qi, (q, pids) in enumerate(zip(query_order, search_rows)):
        for pid in pids:
            text = id_to_text.get(pid, "")
            if text:
                pairs_for_rerank.append((q, text))
                pair_meta.append((qi, pid))
    
    LOGGER.info(f"Scoring {len(pairs_for_rerank)} pairs for reranking...")
    scores = score_pairs(reranker, pairs_for_rerank, rerank_batch, device)
    for (qi, pid), sc in zip(pair_meta, scores):
        score_cache[qi][pid] = float(sc)
    
    # Rerank
    reranked = []
    for qi, pids in enumerate(search_rows):
        ranked = sorted(
            ((pid, score_cache[qi].get(pid, float("-inf"))) for pid in pids),
            key=lambda x: -x[1]
        )
        reranked.append([pid for pid, _ in ranked])
    
    # Bi-encoder only metrics at eval_k
    bi_metrics = compute_metrics(
        [row[:eval_k] for row in bi_topn],
        relevant_ids,
        k=eval_k
    )
    
    # Reranker metrics at eval_k
    reranker_metrics = compute_metrics(
        [row[:eval_k] for row in reranked],
        relevant_ids,
        k=eval_k
    )
    
    # Grid search over k
    grid_results = []
    best_f1 = -1.0
    best_k = eval_k
    
    for k in sorted(k_values):
        if k > selected_n:
            continue
        metrics = compute_metrics(
            [row[:k] for row in reranked],
            relevant_ids,
            k=k
        )
        entry = {"k": k, **metrics}
        grid_results.append(entry)
        if metrics[f"F1@{k}"] > best_f1:
            best_f1 = metrics[f"F1@{k}"]
            best_k = k
    
    # Threshold analysis
    LOGGER.info("Building threshold dataset...")
    thr_pairs, thr_labels = build_threshold_dataset(
        query_order, relevant_ids, bi_topn, id_to_text,
        max_neg_per_query=max_neg_per_query
    )
    
    LOGGER.info(f"Scoring {len(thr_pairs)} pairs for threshold analysis...")
    thr_scores = score_pairs(reranker, thr_pairs, rerank_batch, device)
    thr_result = threshold_analysis(thr_scores, thr_labels)
    
    LOGGER.info(f"EER threshold: {thr_result['EER']['threshold']:.4f}")
    LOGGER.info(f"EER FPR: {thr_result['EER']['FPR']:.4f}, FNR: {thr_result['EER']['FNR']:.4f}")
    
    # Build result
    result = {
        "embedding_model": str(embedding_model),
        "reranker_model": str(reranker_model),
        "corpus_size": len(corpus_df),
        "n_queries": len(query_order),
        "eval_k": eval_k,
        "selected_n": selected_n,
        "target_recall": target_recall,
        "recall_by_n": recall_by_n,
        
        # Bi-encoder vs Reranker
        "bi_encoder_only": bi_metrics,
        "reranker": reranker_metrics,
        "improvement": {
            "precision_delta": reranker_metrics[f"Precision@{eval_k}"] - bi_metrics[f"Precision@{eval_k}"],
            "recall_delta": reranker_metrics[f"Recall@{eval_k}"] - bi_metrics[f"Recall@{eval_k}"],
            "f1_delta": reranker_metrics[f"F1@{eval_k}"] - bi_metrics[f"F1@{eval_k}"],
        },
        
        # Grid search k
        "grid_search_k": grid_results,
        "optimal_k": {"k": best_k, "f1": best_f1},
        
        # Threshold
        "threshold_analysis": {
            "EER": thr_result["EER"],
            "min_error": thr_result["min_error"],
            "optimal": thr_result["optimal"],
            "n_pairs": thr_result["n_pairs"],
            "n_positive": thr_result["n_positive"],
            "n_negative": thr_result["n_negative"],
        },
        "threshold_curve": thr_result["curve"],
    }
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info(f"Saved: {output_path}")
    
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate E5 + Reranker với TREC-style data")
    
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--queries-jsonl", type=Path, required=True)
    parser.add_argument("--qrels-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 20, 30, 50, 75, 100])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.95)
    
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-neg-per-query", type=int, default=5)
    
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    
    result = run_evaluation(
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model,
        corpus_path=args.eval_csv,
        queries_path=args.queries_jsonl,
        qrels_path=args.qrels_jsonl,
        output_path=args.output,
        n_values=args.n_values,
        k_values=args.k_values,
        eval_k=args.eval_k,
        encode_batch=args.encode_batch_size,
        rerank_batch=args.rerank_batch_size,
        max_neg_per_query=args.max_neg_per_query,
        target_recall=args.target_recall,
        device=args.device,
    )
    
    print(json.dumps(result, ensure_ascii=False, indent=2))
