"""Đánh giá pipeline E5 bi-encoder + cross-encoder reranker trên tập test.

Corpus: ecommerce.csv (5000 SP). Query + nhãn: test jsonl (metadata.product_id).

1. Precision@k, Recall@k, F1@k, MRR@k, NDCG@k (bi-encoder vs reranker)
2. Chọn n theo Recall@n, grid k
3. FPR, FNR, EER, ngưỡng τ (hard negative từ top-n E5)

Ví dụ:
  python embedding_project/scripts/evaluate_reranker_pipeline.py \\
    --embedding-model embedding_project/models/e5_base_finetuned_5000 \\
    --reranker-model embedding_project/models/reranker \\
    --query-jsonl data/training/test_5000.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger("evaluate_reranker_pipeline")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_test_queries(
    jsonl_paths: list[Path],
    products_df: pd.DataFrame,
) -> tuple[list[str], dict[str, set[str]], list[dict], str]:
    corpus_ids = set(products_df["product_id"].astype(str))
    rows: list[dict] = []
    for path in jsonl_paths:
        if path.is_file():
            rows.extend(load_jsonl(path))

    labels: dict[str, set[str]] = {}
    query_order: list[str] = []
    for row in rows:
        q = str(row.get("query", "")).strip()
        pid = str(row.get("metadata", {}).get("product_id", "")).strip()
        if not q or not pid or pid not in corpus_ids:
            continue
        if q not in labels:
            query_order.append(q)
        labels.setdefault(q, set()).add(pid)

    source = "jsonl:" + ",".join(p.name for p in jsonl_paths)
    return query_order, labels, rows, source


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


def compute_bi_encoder_topn(
    args: argparse.Namespace,
    products_df: pd.DataFrame,
    query_texts: list[str],
    pool_n: int,
) -> list[list[str]]:
    product_ids = products_df["product_id"].astype(str).tolist()
    corpus_texts = products_df["searchable_text"].astype(str).tolist()
    pool_n = min(pool_n, len(product_ids))

    LOGGER.info("Loading embedding model: %s", args.embedding_model)
    emb = SentenceTransformer(str(args.embedding_model))
    emb.max_seq_length = 512
    corpus_emb, query_emb = encode_corpus_and_queries(
        emb, corpus_texts, query_texts, args.encode_batch_size
    )
    topn_idx = topk_indices(query_emb, corpus_emb, k=pool_n)
    return [[product_ids[i] for i in row] for row in topn_idx]


def retrieval_metrics(
    retrieved_ids: list[list[str]],
    query_texts: list[str],
    labels: dict[str, set[str]],
    k: int,
) -> dict[str, float]:
    p_list, r_list, mrr_list, ndcg_list = [], [], [], []
    for q, hits in zip(query_texts, retrieved_ids):
        rel = labels.get(q, set())
        if not rel:
            continue
        top_hits = hits[:k]
        flags = [1 if pid in rel else 0 for pid in top_hits]
        hit_count = sum(flags)
        p = hit_count / k
        r = hit_count / len(rel)
        p_list.append(p)
        r_list.append(r)
        rr = next((1.0 / i for i, f in enumerate(flags, 1) if f), 0.0)
        mrr_list.append(rr)
        dcg = sum(f / np.log2(i + 2) for i, f in enumerate(flags))
        ideal = [1] * min(len(rel), k) + [0] * (k - min(len(rel), k))
        idcg = sum(f / np.log2(i + 2) for i, f in enumerate(ideal))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    p = float(np.mean(p_list)) if p_list else 0.0
    r = float(np.mean(r_list)) if r_list else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        f"Precision@{k}": p,
        f"Recall@{k}": r,
        f"F1@{k}": f1,
        f"MRR@{k}": float(np.mean(mrr_list)) if mrr_list else 0.0,
        f"NDCG@{k}": float(np.mean(ndcg_list)) if ndcg_list else 0.0,
        "n_queries": len(p_list),
    }


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


def build_rerank_score_cache(
    queries: list[str],
    candidate_ids: list[list[str]],
    id_to_text: dict[str, str],
    reranker,
    batch_size: int,
    device: str,
) -> list[dict[str, float]]:
    pairs: list[tuple[str, str]] = []
    meta: list[tuple[int, str]] = []
    for qi, (q, pids) in enumerate(zip(queries, candidate_ids)):
        for pid in pids:
            text = id_to_text.get(pid, "")
            if text:
                pairs.append((q, text))
                meta.append((qi, pid))

    scores = score_pairs(reranker, pairs, batch_size, device)
    cache: list[dict[str, float]] = [{} for _ in queries]
    for (qi, pid), sc in zip(meta, scores):
        cache[qi][pid] = float(sc)
    return cache


def rerank_from_cache(
    candidate_ids: list[list[str]],
    score_cache: list[dict[str, float]],
) -> list[list[str]]:
    reranked: list[list[str]] = []
    for qi, pids in enumerate(candidate_ids):
        ranked = sorted(
            ((pid, score_cache[qi].get(pid, float("-inf"))) for pid in pids),
            key=lambda x: -x[1],
        )
        reranked.append([pid for pid, _ in ranked if pid in score_cache[qi]])
    return reranked


def build_threshold_dataset_from_retrieval(
    query_texts: list[str],
    labels: dict[str, set[str]],
    bi_encoder_topn: list[list[str]],
    id_to_text: dict[str, str],
    max_neg_per_query: int = 5,
    n_pool: int = 30,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs: list[tuple[str, str]] = []
    label_arr: list[int] = []

    for qi, q in enumerate(query_texts):
        rel = labels.get(q, set())
        if not rel:
            continue
        pos_pid = next((p for p in bi_encoder_topn[qi] if p in rel), None)
        if pos_pid is None:
            pos_pid = next(iter(rel))
        pos_text = id_to_text.get(pos_pid, "")
        if not pos_text:
            continue
        pairs.append((q, pos_text))
        label_arr.append(1)

        negs: list[str] = []
        for pid in bi_encoder_topn[qi][:n_pool]:
            if pid not in rel:
                negs.append(pid)
            if len(negs) >= max_neg_per_query:
                break
        for pid in negs:
            neg_text = id_to_text.get(pid, "")
            if neg_text:
                pairs.append((q, neg_text))
                label_arr.append(0)

    return pairs, np.asarray(label_arr, dtype=np.int32)


def threshold_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 201,
) -> dict:
    if len(scores) == 0:
        return {
            "curve": [],
            "EER": {},
            "min_error": {},
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
        fpr = fp / (fp + tn + 1e-12)
        fnr = fn / (fn + tp + 1e-12)
        err = (fp + fn) / len(labels)
        curve.append(
            {
                "threshold": float(t),
                "FPR": float(fpr),
                "FNR": float(fnr),
                "error_rate": float(err),
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
            }
        )

    diff = [abs(c["FPR"] - c["FNR"]) for c in curve]
    eer_idx = int(np.argmin(diff))
    err_idx = int(np.argmin([c["error_rate"] for c in curve]))

    return {
        "curve": curve,
        "EER": {
            "threshold": curve[eer_idx]["threshold"],
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
        "n_pairs": int(len(labels)),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate E5 + reranker on test jsonl.")
    parser.add_argument(
        "--embedding-model",
        type=Path,
        default=root / "embedding_project/models/e5_base_finetuned_5000",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=root / "embedding_project/models/reranker",
    )
    parser.add_argument(
        "--eval-csv",
        type=Path,
        default=root / "embedding_project/data/ecommerce.csv",
        help="Corpus sản phẩm (searchable_text)",
    )
    parser.add_argument(
        "--query-jsonl",
        type=Path,
        nargs="+",
        default=[root / "data/training/test_5000.jsonl"],
        help="Tập test (query + metadata.product_id)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "embedding_project/outputs/evaluation/reranker_pipeline_eval_test.json",
    )
    parser.add_argument("--n-values", type=int, nargs="+", default=[10, 20, 30, 50, 75, 100])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--eval-k", type=int, default=10, help="k báo cáo chính P/R/F1")
    parser.add_argument(
        "--best-metric",
        choices=["F1", "MRR", "NDCG"],
        default="NDCG",
        help="Metric chọn cấu hình tốt nhất tại eval-k",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Chọn n nhỏ nhất sao cho Recall@n >= ngưỡng này",
    )
    parser.add_argument(
        "--n-search-values",
        type=int,
        nargs="+",
        default=[10, 20, 30, 50, 75, 100],
        help="Các n dùng để dò Recall trước khi chốt n",
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None, help="Giới hạn query (smoke test)")
    parser.add_argument(
        "--grid-max-queries",
        type=int,
        default=None,
        help="Chỉ dùng N query đầu cho grid k (metric báo cáo vẫn full)",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Bỏ phân tích FPR/FNR/EER trong lần chạy eval chính",
    )
    parser.add_argument(
        "--threshold-only",
        action="store_true",
        help="Chỉ chạy phân tích ngưỡng reranker",
    )
    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=None,
        help="File JSON ngưỡng (mặc định: reranker_threshold_test.json)",
    )
    parser.add_argument(
        "--max-neg-per-query",
        type=int,
        default=3,
        help="Số hard negative mỗi query cho phân tích ngưỡng",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def run_threshold_analysis(
    args: argparse.Namespace,
    device: str,
    query_texts: list[str],
    labels: dict[str, set[str]],
    bi_encoder_topn: list[list[str]],
    id_to_text: dict[str, str],
) -> dict:
    LOGGER.info("Loading reranker: %s", args.reranker_model)
    reranker = load_reranker(args.reranker_model, device)

    LOGGER.info("Threshold analysis (%d queries, hard neg từ E5 top-n)...", len(query_texts))
    thr_pairs, thr_labels = build_threshold_dataset_from_retrieval(
        query_texts,
        labels,
        bi_encoder_topn,
        id_to_text,
        max_neg_per_query=args.max_neg_per_query,
    )
    thr_scores = score_pairs(reranker, thr_pairs, args.rerank_batch_size, device)
    thr_result = threshold_analysis(thr_scores, thr_labels)
    LOGGER.info(
        "EER τ=%.4f (FPR=%.4f, FNR=%.4f) | min-error τ=%.4f (err=%.4f)",
        thr_result["EER"].get("threshold", 0.0),
        thr_result["EER"].get("FPR", 0.0),
        thr_result["EER"].get("FNR", 0.0),
        thr_result["min_error"].get("threshold", 0.0),
        thr_result["min_error"].get("error_rate", 0.0),
    )
    return {
        "reranker_model": str(args.reranker_model),
        "eval_csv": str(args.eval_csv),
        "n_eval_queries": len(query_texts),
        "max_neg_per_query": args.max_neg_per_query,
        "n_pairs": thr_result["n_pairs"],
        "threshold_analysis": {
            "EER": thr_result["EER"],
            "min_error_rate": thr_result["min_error"],
            "n_pairs": thr_result["n_pairs"],
            "n_positive": thr_result["n_positive"],
            "n_negative": thr_result["n_negative"],
        },
        "threshold_curve": thr_result["curve"],
        "deployment_threshold": {
            "eer": thr_result["EER"],
            "min_error_rate": thr_result["min_error"],
        },
    }


def run_threshold_only(args: argparse.Namespace, device: str) -> None:
    products_df = pd.read_csv(args.eval_csv)
    products_df = products_df.dropna(subset=["product_id", "searchable_text"]).drop_duplicates("product_id")

    query_texts, labels, _jsonl_rows, eval_source = load_test_queries(args.query_jsonl, products_df)
    if args.max_queries:
        query_texts = query_texts[: args.max_queries]
        labels = {q: labels[q] for q in query_texts}

    product_ids = products_df["product_id"].astype(str).tolist()
    id_to_text = dict(zip(product_ids, products_df["searchable_text"].astype(str)))
    pool_n = max(args.n_search_values) if args.n_search_values else 30
    bi_encoder_topn = compute_bi_encoder_topn(args, products_df, query_texts, pool_n)

    result = run_threshold_analysis(args, device, query_texts, labels, bi_encoder_topn, id_to_text)
    result["eval_source"] = eval_source
    out = args.threshold_output or (args.output.parent / "reranker_threshold_test.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved threshold analysis: %s", out)
    print(json.dumps(result["deployment_threshold"], ensure_ascii=False, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    if args.threshold_only:
        run_threshold_only(args, device)
        return

    products_df = pd.read_csv(args.eval_csv)
    products_df = products_df.dropna(subset=["product_id", "searchable_text"]).drop_duplicates("product_id")

    query_texts, labels, _jsonl_rows, eval_source = load_test_queries(args.query_jsonl, products_df)
    if args.max_queries:
        query_texts = query_texts[: args.max_queries]
        labels = {q: labels[q] for q in query_texts}

    LOGGER.info("Eval queries: %d | source: %s", len(query_texts), eval_source)
    product_ids = products_df["product_id"].astype(str).tolist()
    corpus_texts = products_df["searchable_text"].astype(str).tolist()
    id_to_text = dict(zip(product_ids, corpus_texts))

    max_n = max(args.n_values)
    bi_encoder_topn = compute_bi_encoder_topn(args, products_df, query_texts, max_n)

    eval_k = args.eval_k
    bi_metrics = retrieval_metrics(bi_encoder_topn, query_texts, labels, k=eval_k)
    LOGGER.info("Bi-encoder only @%d: %s", eval_k, bi_metrics)

    n_search_values = sorted(set(n for n in args.n_search_values if n <= max_n))
    if not n_search_values:
        n_search_values = [max_n]

    recall_by_n: dict[int, float] = {}
    for n in n_search_values:
        recall_by_n[n] = retrieval_metrics(
            [row[:n] for row in bi_encoder_topn],
            query_texts,
            labels,
            k=n,
        )[f"Recall@{n}"]

    selected_n = n_search_values[-1]
    if args.target_recall is not None:
        for n in n_search_values:
            if recall_by_n[n] >= args.target_recall:
                selected_n = n
                break

    LOGGER.info("Recall by n: %s", recall_by_n)
    LOGGER.info("Selected n = %d", selected_n)

    LOGGER.info("Loading reranker: %s", args.reranker_model)
    reranker = load_reranker(args.reranker_model, device)

    search_rows = [row[:selected_n] for row in bi_encoder_topn]
    LOGGER.info(
        "Reranking tại selected_n=%d (%d queries × %d = %d cặp)...",
        selected_n,
        len(query_texts),
        selected_n,
        selected_n * len(query_texts),
    )
    score_cache = build_rerank_score_cache(
        query_texts,
        search_rows,
        id_to_text,
        reranker,
        args.rerank_batch_size,
        device,
    )

    grid_query_texts = query_texts
    grid_labels = labels
    grid_topn = bi_encoder_topn
    grid_cache = score_cache
    if args.grid_max_queries and args.grid_max_queries < len(query_texts):
        gq = args.grid_max_queries
        grid_query_texts = query_texts[:gq]
        grid_labels = {q: labels[q] for q in grid_query_texts}
        grid_topn = bi_encoder_topn[:gq]
        grid_cache = score_cache[:gq]
        LOGGER.info("Grid k trên %d/%d query (mẫu)", gq, len(query_texts))

    grid_results = []
    metric_key = {"F1": "F1", "MRR": "MRR", "NDCG": "NDCG"}[args.best_metric]
    best_score = -1.0
    best_nk: dict | None = None

    cand_ids = [row[:selected_n] for row in grid_topn]
    reranked_ids = rerank_from_cache(cand_ids, grid_cache)
    for k in sorted(args.k_values):
        if k > selected_n:
            continue
        m = retrieval_metrics(reranked_ids, grid_query_texts, grid_labels, k=k)
        entry = {"n": selected_n, "k": k, **m}
        grid_results.append(entry)
        score_key = f"{metric_key}@{k}"
        if m[score_key] > best_score:
            best_score = m[score_key]
            best_nk = entry
        if k == eval_k:
            LOGGER.info(
                "Reranker n=%d @%d: P=%.4f R=%.4f F1=%.4f MRR=%.4f NDCG=%.4f",
                selected_n,
                k,
                m[f"Precision@{k}"],
                m[f"Recall@{k}"],
                m[f"F1@{k}"],
                m[f"MRR@{k}"],
                m[f"NDCG@{k}"],
            )

    best_at_eval_k = max(
        (g for g in grid_results if g["k"] == eval_k),
        key=lambda g: g[f"{metric_key}@{eval_k}"],
        default=None,
    )
    if best_at_eval_k:
        reranked_best = rerank_from_cache(
            [row[:selected_n] for row in bi_encoder_topn],
            score_cache,
        )
        reranker_metrics = retrieval_metrics(reranked_best, query_texts, labels, k=eval_k)
        LOGGER.info(
            "Best n=%d @%d (full %d queries): P=%.4f R=%.4f F1=%.4f MRR=%.4f NDCG=%.4f",
            selected_n,
            eval_k,
            len(query_texts),
            reranker_metrics[f"Precision@{eval_k}"],
            reranker_metrics[f"Recall@{eval_k}"],
            reranker_metrics[f"F1@{eval_k}"],
            reranker_metrics[f"MRR@{eval_k}"],
            reranker_metrics[f"NDCG@{eval_k}"],
        )
    else:
        reranker_metrics = {}

    if args.skip_threshold:
        thr_block = {
            "threshold_analysis": {
                "EER": {},
                "min_error_rate": {},
                "n_pairs": 0,
                "n_positive": 0,
                "n_negative": 0,
            },
            "threshold_curve": [],
            "deployment_threshold": {},
        }
    else:
        thr = run_threshold_analysis(
            args, device, query_texts, labels, bi_encoder_topn, id_to_text
        )
        thr_block = {
            "threshold_analysis": thr["threshold_analysis"],
            "threshold_curve": thr["threshold_curve"],
            "deployment_threshold": thr["deployment_threshold"],
        }

    result = {
        "embedding_model": str(args.embedding_model),
        "reranker_model": str(args.reranker_model),
        "eval_csv": str(args.eval_csv),
        "eval_source": eval_source,
        "query_jsonl": [str(p) for p in args.query_jsonl],
        "corpus_size": len(product_ids),
        "n_eval_queries": len(query_texts),
        "grid_max_queries": args.grid_max_queries,
        "rerank_pairs_scored_once": selected_n * len(query_texts),
        "selected_n": selected_n,
        "best_metric": args.best_metric,
        f"bi_encoder_only@{eval_k}": bi_metrics,
        f"reranker_best_n@{eval_k}": {
            "n": selected_n,
            "metrics": reranker_metrics,
        },
        "grid_search_n_k": grid_results,
        "optimal_n_k": {
            f"by_max_{args.best_metric.lower()}_any_k": best_nk,
            f"by_max_{args.best_metric.lower()}_at_k{eval_k}": best_at_eval_k,
            "selected_n_by_target_recall": selected_n,
            "recall_by_n": recall_by_n,
        },
        **thr_block,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved: %s", args.output)
    print(
        json.dumps(
            {
                "bi_encoder": bi_metrics,
                "reranker_best": reranker_metrics,
                "optimal_n_k": result["optimal_n_k"],
                "threshold": result.get("threshold_analysis", {}),
                "deployment_threshold": result.get("deployment_threshold", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
