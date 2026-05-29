"""
Tạo vector embedding từ merged_products_vi_cleaned.csv bằng model fine-tuned.

Usage:
  python create_product_vectors.py
  python create_product_vectors.py --preset bge-m3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent / "embedding_project" / "scripts"))
from model_presets import get_preset

LOGGER = logging.getLogger("create_product_vectors")

INPUT_CSV = Path("embedding_project/data/merged_products_vi_cleaned.csv")

VECTOR_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

PAYLOAD_FIELDS = [
    "product_id",
    "source",
    "title",
    "category",
    "brand",
    "price",
    "rating",
    "reviews_count",
    "image_url",
    "tags",
    "color",
    "size",
    "searchable_text",
]


def load_products(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"product_id", "source", "searchable_text", *PAYLOAD_FIELDS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {sorted(missing)}")

    df = df.dropna(subset=["product_id", "searchable_text"]).copy()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["searchable_text"] = df["searchable_text"].astype(str).str.strip()
    df = df[df["searchable_text"] != ""]
    df = df.drop_duplicates(subset=["source", "product_id"], keep="first").reset_index(drop=True)
    return df


def make_vector_id(source: str, product_id: str) -> str:
    key = f"{source}_{product_id}"
    return str(uuid.uuid5(VECTOR_NAMESPACE, key))


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def encode_searchable_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 64,
) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_vectors(np.asarray(vectors, dtype=np.float32))


def build_payloads(df: pd.DataFrame) -> list[dict]:
    payloads: list[dict] = []
    for _, row in df.iterrows():
        source = str(row["source"])
        product_id = str(row["product_id"])
        item = {
            "vector_id": make_vector_id(source, product_id),
            "product_id": product_id,
            "source": source,
            "title": str(row.get("title", "") or ""),
            "category": str(row.get("category", "") or ""),
            "brand": str(row.get("brand", "") or ""),
            "price": row.get("price") if pd.notna(row.get("price")) else None,
            "rating": row.get("rating") if pd.notna(row.get("rating")) else None,
            "reviews_count": (
                int(row["reviews_count"]) if pd.notna(row.get("reviews_count")) else None
            ),
            "image_url": str(row.get("image_url", "") or ""),
            "tags": str(row.get("tags", "") or ""),
            "color": str(row.get("color", "") or ""),
            "size": str(row.get("size", "") or ""),
            "searchable_text": str(row["searchable_text"]),
        }
        payloads.append(item)
    return payloads


def save_outputs(vectors: np.ndarray, payloads: list[dict], vectors_path: Path, payloads_path: Path) -> None:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    payloads_path.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tạo product vectors từ CSV.")
    p.add_argument("--preset", choices=["minilm", "bge-m3"], default="minilm")
    p.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("embedding_project/outputs/embeddings"))
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    preset = get_preset(args.preset)

    model_path = args.model or Path(preset.finetuned_rel_path)
    suffix = preset.name.replace("-", "_")
    vectors_path = args.output_dir / f"product_vectors_{suffix}.npy"
    payloads_path = args.output_dir / f"product_payloads_{suffix}.json"
    batch_size = args.batch_size or (16 if args.preset == "bge-m3" else 64)

    if not args.input_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy input: {args.input_csv.resolve()}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path.resolve()}")

    LOGGER.info("Preset: %s", preset.name)
    LOGGER.info("Loading products from %s", args.input_csv)
    df = load_products(args.input_csv)

    LOGGER.info("Loading model from %s", model_path)
    model = SentenceTransformer(str(model_path), trust_remote_code=preset.trust_remote_code)

    texts = df["searchable_text"].tolist()
    LOGGER.info("Encoding %d products (batch=%d)...", len(texts), batch_size)
    vectors = encode_searchable_texts(model, texts, batch_size=batch_size)

    payloads = build_payloads(df)
    save_outputs(vectors, payloads, vectors_path, payloads_path)

    print(f"Số sản phẩm: {len(df)}")
    print(f"Vector dimension: {vectors.shape[1]}")
    print(f"Vectors: {vectors_path.resolve()}")
    print(f"Payloads: {payloads_path.resolve()}")


if __name__ == "__main__":
    main()
