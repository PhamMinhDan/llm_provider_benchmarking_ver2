from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("merged_products_vi.csv")
OUTPUT_CSV = Path("merged_products_vi_with_searchable_text.csv")
OUTPUT_JSON = Path("merged_products_vi_with_searchable_text.json")

TEXT_FIELDS = ["title", "brand", "category", "description", "tags"]
LABELS = {
    "title": "Tên sản phẩm",
    "brand": "Thương hiệu",
    "category": "Danh mục",
    "description": "Mô tả",
    "tags": "Đặc điểm / tags",
}


def clean_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""

    cleaned = str(text).strip()
    if not cleaned:
        return ""

    # Normalize newlines and whitespace.
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)

    # Remove punctuation runs like "...,,,", "!!!", "???".
    cleaned = re.sub(r"([,.;:!?])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)

    return cleaned.strip(" \n\t,;")


def normalize_tags(tags: object) -> str:
    raw = clean_text(tags)
    if not raw:
        return ""

    parts = re.split(r"[|,;]+", raw)
    normalized: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = clean_text(part)
        if not token:
            continue
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(token)
    return ", ".join(normalized)


def _clip_field(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped + "..."


def build_searchable_text(row: pd.Series) -> str:
    values = {
        "title": clean_text(row.get("title")),
        "brand": clean_text(row.get("brand")),
        "category": clean_text(row.get("category")),
        "description": clean_text(row.get("description")),
        "tags": normalize_tags(row.get("tags")),
    }

    # Keep semantic richness while avoiding overly long content for embeddings.
    if values["description"]:
        values["description"] = _clip_field(values["description"], max_chars=900)

    lines: list[str] = []
    for key in TEXT_FIELDS:
        value = values[key]
        if value:
            lines.append(f"{LABELS[key]}: {value}")
    return "\n".join(lines).strip()


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)

    required_columns = [
        "product_id",
        "source",
        "title",
        "description",
        "category",
        "brand",
        "price",
        "rating",
        "reviews_count",
        "image_url",
        "tags",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong CSV: {missing}")

    df["searchable_text"] = df.apply(build_searchable_text, axis=1)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    total_rows = len(df)
    empty_count = int((df["searchable_text"].str.strip() == "").sum())
    avg_len = float(df["searchable_text"].str.len().mean()) if total_rows else 0.0

    print(f"Tổng số dòng ban đầu: {total_rows}")
    print(f"Số dòng có searchable_text rỗng: {empty_count}")
    print(f"Độ dài trung bình searchable_text: {avg_len:.2f} ký tự")
    print("\n5 dòng mẫu (title + searchable_text):")

    sample = df[["title", "searchable_text"]].head(5)
    for idx, row in sample.iterrows():
        title = clean_text(row["title"]) or "(không có title)"
        searchable_text = row["searchable_text"] or "(rỗng)"
        print(f"\n[{idx}] title: {title}")
        print(searchable_text)


if __name__ == "__main__":
    main()
