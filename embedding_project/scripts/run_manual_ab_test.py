"""
Xuất kết quả top-K search của pretrained vs fine-tuned cho chấm tay A/B.

Usage:
  python embedding_project/scripts/run_manual_ab_test.py \
    --queries embedding_project/data/manual_eval_queries.csv \
    --top-k 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from model_presets import get_preset

LOGGER = logging.getLogger("run_manual_ab_test")


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def top_k_indices(query_emb: np.ndarray, corpus_emb: np.ndarray, k: int) -> np.ndarray:
    scores = query_emb @ corpus_emb.T
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    top_scores = scores[rows, idx]
    order = np.argsort(-top_scores, axis=1)
    return idx[rows, order]


def load_queries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "query" not in df.columns:
        raise ValueError("File query phải có cột 'query'.")
    df = df.copy()
    df["query"] = df["query"].astype(str).str.strip()
    df = df[df["query"] != ""].drop_duplicates(subset=["query"], keep="first")
    if "query_id" not in df.columns:
        df["query_id"] = [f"q{i+1:03d}" for i in range(len(df))]
    if "group" not in df.columns:
        df["group"] = ""
    return df


def load_corpus(csv_path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(csv_path)
    required = {"product_id", "searchable_text", "title", "category", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")
    df = df.dropna(subset=["product_id", "searchable_text"]).copy()
    df["product_id"] = df["product_id"].astype(str)
    df = df.drop_duplicates(subset=["product_id"], keep="first").reset_index(drop=True)
    return df, df["product_id"].tolist(), df["searchable_text"].astype(str).tolist()


def search_top_k(
    model_path: str,
    queries: list[str],
    corpus_texts: list[str],
    k: int,
    batch_size: int,
    trust_remote_code: bool = False,
) -> list[list[int]]:
    kwargs = {"trust_remote_code": True} if trust_remote_code else {}
    model = SentenceTransformer(model_path, **kwargs)
    corpus_emb = normalize(
        model.encode(corpus_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    )
    query_emb = normalize(
        model.encode(queries, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    )
    return top_k_indices(query_emb, corpus_emb, k=k).tolist()


def build_result_rows(
    model_label: str,
    query_df: pd.DataFrame,
    corpus_df: pd.DataFrame,
    topk_idx: list[list[int]],
) -> list[dict]:
    rows: list[dict] = []
    for qpos, (_, qrow) in enumerate(query_df.iterrows()):
        for rank, cidx in enumerate(topk_idx[qpos], start=1):
            product = corpus_df.iloc[cidx]
            rows.append(
                {
                    "query_id": qrow["query_id"],
                    "query": qrow["query"],
                    "group": qrow.get("group", ""),
                    "model": model_label,
                    "rank": rank,
                    "product_id": product["product_id"],
                    "title": product.get("title", ""),
                    "category": product.get("category", ""),
                    "source": product.get("source", ""),
                    "label": "",  # 0=irrelevant, 1=partial, 2=relevant
                    "notes": "",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export manual A/B search results (pretrained vs finetuned).")
    p.add_argument(
        "--queries",
        type=Path,
        default=Path("embedding_project/data/manual_eval_queries.csv"),
    )
    p.add_argument(
        "--products-csv",
        type=Path,
        default=Path("embedding_project/data/merged_products_vi_cleaned.csv"),
    )
    p.add_argument("--preset", choices=["minilm", "bge-m3"], default="minilm")
    p.add_argument("--pretrained-model", default=None)
    p.add_argument("--finetuned-model", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--blind-output",
        type=Path,
        default=None,
        help="Ẩn tên model (model_a/model_b) để chấm không bias.",
    )
    return p.parse_args()


def default_outputs(preset_name: str) -> tuple[Path, Path, Path]:
    suffix = "" if preset_name == "minilm" else f"_{preset_name.replace('-', '_')}"
    base = Path("embedding_project/outputs/evaluation")
    return (
        base / f"manual_ab_results{suffix}.csv",
        base / f"manual_ab_blind{suffix}.csv",
        base / f"manual_ab_model_mapping{suffix}.txt",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    preset = get_preset(args.preset)
    default_out, default_blind, default_mapping = default_outputs(preset.name)
    output_path = args.output or default_out
    blind_path = args.blind_output or default_blind
    mapping_path = default_mapping

    pretrained = args.pretrained_model or preset.base_model
    finetuned = args.finetuned_model or preset.finetuned_rel_path
    batch_size = args.batch_size or (4 if args.preset == "bge-m3" else 64)
    trust = preset.trust_remote_code

    LOGGER.info("Preset: %s | pretrained=%s | finetuned=%s", preset.name, pretrained, finetuned)

    query_df = load_queries(args.queries)
    corpus_df, _, corpus_texts = load_corpus(args.products_csv)
    queries = query_df["query"].tolist()

    LOGGER.info("Queries: %d | Corpus products: %d | top_k=%d", len(queries), len(corpus_texts), args.top_k)

    LOGGER.info("Searching with pretrained model...")
    pretrained_idx = search_top_k(
        pretrained, queries, corpus_texts, args.top_k, batch_size, trust_remote_code=trust
    )
    LOGGER.info("Searching with fine-tuned model...")
    finetuned_idx = search_top_k(
        finetuned, queries, corpus_texts, args.top_k, batch_size, trust_remote_code=trust
    )

    all_rows = []
    all_rows.extend(build_result_rows("pretrained", query_df, corpus_df, pretrained_idx))
    all_rows.extend(build_result_rows("finetuned", query_df, corpus_df, finetuned_idx))

    out_df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    blind_df = out_df.copy()
    blind_map = {"pretrained": "model_a", "finetuned": "model_b"}
    blind_df["model"] = blind_df["model"].map(blind_map)
    blind_df.to_csv(blind_path, index=False, encoding="utf-8-sig")

    mapping_path.write_text(
        f"preset: {preset.name}\n"
        f"pretrained: {pretrained}\n"
        f"finetuned: {finetuned}\n"
        "model_a = pretrained\n"
        "model_b = finetuned\n",
        encoding="utf-8",
    )

    LOGGER.info("Saved labeled export: %s", output_path)
    LOGGER.info("Saved blind export: %s", blind_path)
    LOGGER.info("Model mapping: %s", mapping_path)
    print(f"\nHoàn tất [{preset.name}].")
    print(f"- Full (có tên model): {output_path.resolve()}")
    print(f"- Chấm blind:         {blind_path.resolve()}")
    print("Label: 0=irrelevant, 1=partial, 2=relevant (top-5 mỗi query)")
    print("\nChấm nhanh top-1 (30 query):")
    print(
        "  python embedding_project/scripts/create_manual_ab_fast_sample.py "
        f"--input {output_path.as_posix()} "
        f"--output embedding_project/outputs/evaluation/manual_ab_fast_top1_{preset.name.replace('-', '_')}.csv"
    )
    print("\nSau khi chấm:")
    print("  python embedding_project/scripts/score_manual_ab.py --input <file_da_cham.csv>")


if __name__ == "__main__":
    main()
