"""Load sản phẩm từ CSV hoặc JSONL (trường searchable_text)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def infer_source(product_id: str) -> str:
    pid = str(product_id).strip()
    if pid.upper().startswith("B") and len(pid) >= 10:
        return "amazon"
    if pid.isdigit():
        return "shein"
    return "unknown"


def tags_to_str(tags: Any) -> str:
    if tags is None:
        return ""
    if isinstance(tags, list):
        return " | ".join(str(t) for t in tags if t)
    return str(tags)


def _clean_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def load_products_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {path}")

    df = pd.read_csv(path)
    required = {"product_id", "source", "searchable_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột: {sorted(missing)}")

    df = df.dropna(subset=["product_id", "searchable_text"]).copy()
    df["product_id"] = df["product_id"].astype(str).str.strip()
    df["source"] = df["source"].fillna("").astype(str).str.strip()
    df["searchable_text"] = df["searchable_text"].astype(str).str.strip()
    df = df[df["searchable_text"] != ""]
    df = df.drop_duplicates(subset=["source", "product_id"], keep="first")

    products: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        products.append(
            {
                "product_id": str(row["product_id"]),
                "source": str(row["source"]) or infer_source(str(row["product_id"])),
                "title": _clean_str(row.get("title")),
                "description": _clean_str(row.get("description")),
                "category": _clean_str(row.get("category")),
                "brand": _clean_str(row.get("brand")),
                "price": row.get("price") if pd.notna(row.get("price")) else None,
                "rating": row.get("rating") if pd.notna(row.get("rating")) else None,
                "reviews_count": (
                    int(row["reviews_count"]) if pd.notna(row.get("reviews_count")) else None
                ),
                "image_url": _clean_str(row.get("image_url")),
                "tags": _clean_str(row.get("tags")),
                "color": _clean_str(row.get("color")),
                "size": _clean_str(row.get("size")),
                "searchable_text": str(row["searchable_text"]),
            }
        )
    return products


def load_products_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy JSONL: {path}")

    products: list[dict[str, Any]] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL lỗi dòng {line_no}: {e}") from e

            pid = str(row.get("product_id", "")).strip()
            text = str(row.get("searchable_text", "") or "").strip()
            if not pid or not text:
                continue

            source = str(row.get("source", "")).strip() or infer_source(pid)
            key = f"{source}_{pid}"
            if key in seen:
                continue
            seen.add(key)

            products.append(
                {
                    "product_id": pid,
                    "source": source,
                    "title": _clean_str(row.get("title")),
                    "description": _clean_str(row.get("description")),
                    "category": _clean_str(row.get("category")),
                    "brand": _clean_str(row.get("brand")),
                    "price": row.get("price"),
                    "rating": row.get("rating"),
                    "reviews_count": row.get("reviews_count"),
                    "image_url": _clean_str(row.get("image_url")),
                    "tags": tags_to_str(row.get("tags")),
                    "color": _clean_str(row.get("color")),
                    "size": _clean_str(row.get("size")),
                    "searchable_text": text,
                }
            )

    return products


def load_products(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_products_csv(path)
    if suffix in (".jsonl", ".json"):
        return load_products_jsonl(path)
    raise ValueError(f"Định dạng không hỗ trợ: {path} (dùng .csv hoặc .jsonl)")
