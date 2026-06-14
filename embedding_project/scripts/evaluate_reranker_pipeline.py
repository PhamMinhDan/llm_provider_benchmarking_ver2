"""Đánh giá pipeline E5 bi-encoder + cross-encoder reranker trên corpus 5000 SP.

1. Precision@k, Recall@k, F1@k, MRR@k, NDCG@k (bi-encoder vs bi-encoder + reranker)
2. Grid search n (retrieve) và k (final top-k)
3. FPR, FNR, EER và ngưỡng tối ưu trên điểm reranker

Ví dụ (đánh giá trên toàn bộ ecommerce.csv — 5000 SP):
  python embedding_project/scripts/evaluate_reranker_pipeline.py \\
    --embedding-model embedding_project/models/e5_base_finetuned_5000 \\
    --reranker-model embedding_project/models/reranker \\
    --eval-csv embedding_project/data/ecommerce.csv
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


def build_labels_from_jsonl(rows: list[dict]) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    for row in rows:
        query = row.get("query")
        pid = str(row.get("metadata", {}).get("product_id", "")).strip()
        if query and pid:
            labels.setdefault(query, set()).add(pid)
    return labels


def build_eval_from_ecommerce(
    products_df: pd.DataFrame,
    query_col: str = "title",
) -> tuple[list[str], dict[str, set[str]]]:
    """Query = cột title (hoặc query_col), ground truth = product_id cùng dòng."""
    labels: dict[str, set[str]] = {}
    query_order: list[str] = []

    for _, row in products_df.iterrows():
        query = str(row.get(query_col, "")).strip()
        pid = str(row.get("product_id", "")).strip()
        if not query or not pid:
            continue
        if query not in labels:
            query_order.append(query)
        labels.setdefault(query, set()).add(pid)

    return query_order, labels


def load_query_jsonl_files(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if path.is_file():
            rows.extend(load_jsonl(path))
    return rows


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
    """Chấm điểm reranker một lần; dùng lại cho mọi n <= len(candidates)."""
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


def rerank_retrieved(
    queries: list[str],
    candidate_ids: list[list[str]],
    id_to_text: dict[str, str],
    reranker,
    batch_size: int,
    device: str,
) -> list[list[str]]:
    cache = build_rerank_score_cache(
        queries, candidate_ids, id_to_text, reranker, batch_size, device
    )
    return rerank_from_cache(candidate_ids, cache)


def build_threshold_dataset_from_ecommerce(
    products_df: pd.DataFrame,
    query_col: str = "title",
    max_neg_per_query: int = 3,
    seed: int = 42,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []

    df = products_df.dropna(subset=["product_id", "searchable_text", query_col]).drop_duplicates("product_id")
    pids = df["product_id"].astype(str).tolist()
    id_to_text = dict(zip(pids, df["searchable_text"].astype(str)))

    for _, row in df.iterrows():
        query = str(row[query_col]).strip()
        pid = str(row["product_id"]).strip()
        positive = str(row["searchable_text"]).strip()
        if not query or not positive:
            continue
        pairs.append((query, positive))
        labels.append(1)

        others = [p for p in pids if p != pid]
        if not others:
            continue
        n_neg = min(max_neg_per_query, len(others))
        for neg_pid in rng.choice(others, size=n_neg, replace=False):
            pairs.append((query, id_to_text[neg_pid]))
            labels.append(0)

    return pairs, np.asarray(labels, dtype=np.int32)


def build_threshold_dataset_from_jsonl(
    eval_rows: list[dict],
    id_to_text: dict[str, str],
    max_neg_per_query: int = 3,
) -> tuple[list[tuple[str, str]], np.ndarray]:
    pairs: list[tuple[str, str]] = []
    labels: list[int] = []

    for row in eval_rows:
        q = row.get("query")
        pos = row.get("positive")
        if not q or not pos:
            continue
        pairs.append((q, pos))
        labels.append(1)

        negs: list[str] = []
        for key in ("hard_negative", "negative"):
            for t in row.get(key) or []:
                if t and t not in negs:
                    negs.append(t)
        for t in negs[:max_neg_per_query]:
            pairs.append((q, t))
            labels.append(0)

    return pairs, np.asarray(labels, dtype=np.int32)


def threshold_analysis(
    scores: np.ndarray,
    labels: np.ndarray,
    n_steps: int = 201,
) -> dict:
    thresholds = np.linspace(0.0, 1.0, n_steps)
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
    parser = argparse.ArgumentParser(description="Evaluate E5 + reranker pipeline.")
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
        help="CSV đánh giá + corpus (mặc định: toàn bộ 5000 SP)",
    )
    parser.add_argument(
        "--query-col",
        default="title",
        help="Cột dùng làm query khi đánh giá từ CSV (mặc định: title)",
    )
    parser.add_argument(
        "--query-jsonl",
        type=Path,
        nargs="*",
        default=None,
        help="Tùy chọn: dùng query từ jsonl thay vì cột CSV (vd. train+valid, không gồm test)",
    )
    parser.add_argument("--output", type=Path, default=root / "embedding_project/outputs/evaluation/reranker_pipeline_eval.json")
    parser.add_argument("--n-values", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--eval-k", type=int, default=10, help="k báo cáo chính P/R/F1")
    parser.add_argument(
        "--best-metric",
        choices=["F1", "MRR", "NDCG"],
        default="NDCG",
        help="Metric dùng để chọn cấu hình tốt nhất tại eval-k",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Nếu đặt, tự chọn n nhỏ nhất sao cho Recall@n >= ngưỡng này và chỉ rerank tới n cần thiết",
    )
    parser.add_argument(
        "--n-search-values",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100, 200, 500],
        help="Các n dùng để dò Recall trước khi chốt n",
    )
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--max-queries", type=int, default=None, help="Giới hạn query (smoke test)")
    parser.add_argument(
        "--grid-max-queries",
        type=int,
        default=None,
        help="Chỉ dùng N query đầu cho grid n×k (nhanh hơn; metric báo cáo vẫn full)",
    )
    parser.add_argument(
        "--skip-threshold",
        action="store_true",
        help="Bỏ phân tích FPR/FNR/EER (tiết kiệm thời gian)",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    products_df = pd.read_csv(args.eval_csv)
    products_df = products_df.dropna(subset=["product_id", "searchable_text"]).drop_duplicates("product_id")

    if args.query_jsonl:
        jsonl_rows = load_query_jsonl_files(args.query_jsonl)
        labels = build_labels_from_jsonl(jsonl_rows)
        eco_ids = set(products_df["product_id"].astype(str))
        labels = {q: rel & eco_ids for q, rel in labels.items() if rel & eco_ids}
        query_texts = sorted(labels)
        eval_source = "jsonl:" + ",".join(str(p.name) for p in args.query_jsonl)
    else:
        query_texts, labels = build_eval_from_ecommerce(products_df, query_col=args.query_col)
        eval_source = f"ecommerce.csv[{args.query_col}]"

    if args.max_queries:
        query_texts = query_texts[: args.max_queries]
        labels = {q: labels[q] for q in query_texts}
    product_ids = products_df["product_id"].astype(str).tolist()
    corpus_texts = products_df["searchable_text"].astype(str).tolist()
    id_to_text = dict(zip(product_ids, corpus_texts))

    LOGGER.info("Loading embedding model: %s", args.embedding_model)
    emb = SentenceTransformer(str(args.embedding_model))
    emb.max_seq_length = 512

    corpus_e5 = [f"passage: {t}" for t in corpus_texts]
    queries_e5 = [f"query: {q}" for q in query_texts]

    LOGGER.info("Encoding corpus (%d products)...", len(corpus_e5))
    corpus_emb = normalize(
        emb.encode(corpus_e5, batch_size=args.encode_batch_size, show_progress_bar=True, convert_to_numpy=True)
    )
    LOGGER.info("Encoding queries (%d)...", len(queries_e5))
    query_emb = normalize(
        emb.encode(queries_e5, batch_size=args.encode_batch_size, show_progress_bar=True, convert_to_numpy=True)
    )

    max_n = max(args.n_values)
    topn_idx = topk_indices(query_emb, corpus_emb, k=max_n)
    bi_encoder_topn = [[product_ids[i] for i in row] for row in topn_idx]

    eval_k = args.eval_k
    bi_metrics = retrieval_metrics(bi_encoder_topn, query_texts, labels, k=eval_k)
    LOGGER.info("Bi-encoder only @%d: %s", eval_k, bi_metrics)

    # 1. Tìm n tối ưu trước bằng Recall@n, chưa load reranker
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

    # 2. Chỉ sau khi có selected_n mới load reranker
    LOGGER.info("Loading reranker: %s", args.reranker_model)
    reranker = load_reranker(args.reranker_model, device)

    # 3. Chỉ rerank selected_n, không rerank max 100/500 nữa
    max_n_eff = selected_n
    search_rows = [row[:max_n_eff] for row in bi_encoder_topn]
    LOGGER.info(
        "Reranking tại selected_n=%d (%d queries × %d = %d cặp)...",
        max_n_eff,
        len(query_texts),
        max_n_eff,
        max_n_eff * len(query_texts),
    )
    score_cache = build_rerank_score_cache(
        query_texts,
        search_rows,
        id_to_text,
        reranker,
        args.rerank_batch_size,
        device,
    )
    LOGGER.info("Đã cache điểm reranker tại selected_n=%d.", selected_n)

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
        LOGGER.info("Grid n×k trên %d/%d query (mẫu)", gq, len(query_texts))

    grid_results = []
    metric_key = {"F1": "F1", "MRR": "MRR", "NDCG": "NDCG"}[args.best_metric]
    best_score = -1.0
    best_nk: dict | None = None

    for n in [selected_n]:
        n_eff = min(n, max_n_eff)
        cand_ids = [row[:n_eff] for row in grid_topn]
        reranked_ids = rerank_from_cache(cand_ids, grid_cache)
        for k in sorted(args.k_values):
            if k > n_eff:
                continue
            m = retrieval_metrics(reranked_ids, grid_query_texts, grid_labels, k=k)
            entry = {"n": n_eff, "k": k, **m}
            grid_results.append(entry)
            score_key = f"{metric_key}@{k}"
            if m[score_key] > best_score:
                best_score = m[score_key]
                best_nk = entry
            if k == eval_k:
                LOGGER.info(
                    "Reranker n=%d @%d: P=%.4f R=%.4f F1=%.4f MRR=%.4f NDCG=%.4f",
                    n_eff,
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
        n_best = best_at_eval_k["n"]
        reranked_best = rerank_from_cache(
            [row[:n_best] for row in bi_encoder_topn],
            score_cache,
        )
        reranker_metrics = retrieval_metrics(reranked_best, query_texts, labels, k=eval_k)
        LOGGER.info(
            "Best n=%d @%d (full %d queries): P=%.4f R=%.4f F1=%.4f MRR=%.4f NDCG=%.4f",
            n_best,
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
        thr_result = {
            "curve": [],
            "EER": {},
            "min_error": {},
            "n_pairs": 0,
            "n_positive": 0,
            "n_negative": 0,
        }
    else:
        LOGGER.info("Threshold analysis on query-passage pairs...")
        if args.query_jsonl:
            thr_rows = [r for r in jsonl_rows if r.get("query") in set(query_texts)]
            thr_pairs, thr_labels = build_threshold_dataset_from_jsonl(thr_rows, id_to_text)
        else:
            sub_df = products_df[products_df[args.query_col].astype(str).str.strip().isin(query_texts)]
            thr_pairs, thr_labels = build_threshold_dataset_from_ecommerce(
                sub_df, query_col=args.query_col
            )
        thr_scores = score_pairs(reranker, thr_pairs, args.rerank_batch_size, device)
        thr_result = threshold_analysis(thr_scores, thr_labels)

    result = {
        "embedding_model": str(args.embedding_model),
        "reranker_model": str(args.reranker_model),
        "eval_csv": str(args.eval_csv),
        "eval_source": eval_source,
        "query_col": args.query_col,
        "corpus_size": len(product_ids),
        "n_eval_queries": len(query_texts),
        "grid_max_queries": args.grid_max_queries,
        "rerank_pairs_scored_once": max_n_eff * len(query_texts),
        "selected_n": selected_n,
        "best_metric": args.best_metric,
        f"bi_encoder_only@{eval_k}": bi_metrics,
        f"reranker_best_n@{eval_k}": {
            "n": best_at_eval_k["n"] if best_at_eval_k else None,
            "metrics": reranker_metrics,
        },
        "grid_search_n_k": grid_results,
        "optimal_n_k": {
            f"by_max_{args.best_metric.lower()}_any_k": best_nk,
            f"by_max_{args.best_metric.lower()}_at_k{eval_k}": best_at_eval_k,
            "selected_n_by_target_recall": selected_n,
            "recall_by_n": recall_by_n,
        },
        "alternative_metrics_summary": {
            "primary_metric": f"Precision@{eval_k}",
            "alternative_metrics": [f"Recall@{eval_k}", f"F1@{eval_k}", f"MRR@{eval_k}", f"NDCG@{eval_k}"],
            "bi_encoder_only": {k: bi_metrics[k] for k in [f"Precision@{eval_k}", f"Recall@{eval_k}", f"F1@{eval_k}", f"MRR@{eval_k}", f"NDCG@{eval_k}"] if k in bi_metrics},
            "reranker_best_n": {k: reranker_metrics.get(k) for k in [f"Precision@{eval_k}", f"Recall@{eval_k}", f"F1@{eval_k}", f"MRR@{eval_k}", f"NDCG@{eval_k}"]},
        },
        "threshold_analysis": {
            "EER": thr_result["EER"],
            "min_error_rate": thr_result["min_error"],
            "n_pairs": thr_result["n_pairs"],
            "n_positive": thr_result["n_positive"],
            "n_negative": thr_result["n_negative"],
        },
        "threshold_curve": thr_result["curve"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved: %s", args.output)
    print(json.dumps(
        {
            "bi_encoder": bi_metrics,
            "reranker_best": reranker_metrics,
            "alternative_metrics_summary": result["alternative_metrics_summary"],
            "optimal_n_k": result["optimal_n_k"],
            "threshold": result["threshold_analysis"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
