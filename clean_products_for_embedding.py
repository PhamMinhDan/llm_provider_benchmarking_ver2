from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("merged_products_vi_with_searchable_text.csv")
OUTPUT_CSV = Path("merged_products_vi_cleaned.csv")
OUTPUT_JSON = Path("merged_products_vi_cleaned.json")

OUTPUT_COLUMNS = [
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
    "color",
    "size",
    "searchable_text",
]

NOISY_TAG_PATTERNS = [
    r"100%\s*m(ới|oi)",
    r"ch(ất|at)\s*lượng\s*tuyệt\s*vời",
    r"dịch\s*vụ",
    r"hoàn\s*trả",
    r"bảo\s*hành",
    r"quà\s*tặng\s*tuyệt\s*vời",
    r"tiện\s*lợi\s*nhanh\s*chóng",
]

COLOR_PATTERNS = [
    ("vàng gold", r"\bvàng\s*gold\b|\bgold\b"),
    ("xanh lá", r"\bxanh\s*lá\b|\bgreen\b"),
    ("xanh dương", r"\bxanh\s*dương\b|\bblue\b"),
    ("đen", r"\bđen\b|\bblack\b"),
    ("trắng", r"\btrắng\b|\bwhite\b"),
    ("đỏ", r"\bđỏ\b|\bred\b"),
    ("vàng", r"\bvàng\b|\byellow\b"),
    ("hồng", r"\bhồng\b|\bpink\b"),
    ("tím", r"\btím\b|\bpurple\b"),
    ("cam", r"\bcam\b|\borange\b"),
    ("xám", r"\bxám\b|\bgray\b|\bgrey\b"),
    ("nâu", r"\bnâu\b|\bbrown\b"),
    ("bạc", r"\bbạc\b|\bsilver\b"),
    ("be", r"\bbe\b|\bbeige\b"),
    ("kem", r"\bkem\b|\bcream\b"),
    ("xanh", r"\bxanh\b"),
]

SIZE_PATTERNS = [
    r"\bxxxl\b",
    r"\bxxl\b",
    r"\bxl\b",
    r"\bl\b",
    r"\bm\b",
    r"\bs\b",
    r"\bxs\b",
    r"\bone[\s-]*size\b",
    r"\bmột[\s-]*size\b",
    r"\bsize\s*[a-z]{1,4}\b",
    r"\b(?:3[6-9]|4[0-4])\b",
    r"\b\d+(?:[.,]\d+)?\s*(?:ml|l|oz|inch|in)\b",
]


def remove_emoji(text: str) -> str:
    return "".join(ch for ch in text if ord(ch) <= 0xFFFF)


def clean_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""

    cleaned = str(text)
    cleaned = remove_emoji(cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\n+", "\n", cleaned)
    cleaned = re.sub(r"[^\w\s\-\.,;:!?\(\)/&>%+|°'\"áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]",
            " ",
            cleaned,
        )
    cleaned = re.sub(r"([,.;:!?])\1+", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", "\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip(" \n\t,;")


def _is_noisy_tag(tag: str) -> bool:
    value = tag.casefold()
    return any(re.search(pattern, value) for pattern in NOISY_TAG_PATTERNS)


def normalize_tags(tags: object) -> str:
    raw = clean_text(tags)
    if not raw:
        return ""

    parts = re.split(r"[|,;]+", raw)
    unique_tags: list[str] = []
    seen: set[str] = set()
    for part in parts:
        tag = clean_text(part)
        if not tag:
            continue
        if _is_noisy_tag(tag):
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_tags.append(tag)
        if len(unique_tags) >= 8:
            break
    return " | ".join(unique_tags)


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        v = clean_text(value)
        if not v:
            continue
        key = v.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def extract_color_size(row: pd.Series) -> tuple[str, str]:
    title = clean_text(row.get("title"))
    description = clean_text(row.get("description"))
    tags = normalize_tags(row.get("tags"))
    context = f"{title}\n{description}\n{tags}".casefold()

    colors: list[str] = []
    for label, pattern in COLOR_PATTERNS:
        if re.search(pattern, context, flags=re.I):
            colors.append(label)

    sizes: list[str] = []
    for pattern in SIZE_PATTERNS:
        matches = re.findall(pattern, context, flags=re.I)
        for match in matches:
            sizes.append(str(match).upper().replace("  ", " ").strip())

    # For numeric shoe/apparel sizes, keep only when product context is relevant.
    product_context = f"{title} {description} {clean_text(row.get('category'))}".casefold()
    is_wearable = bool(
        re.search(r"giày|dép|áo|quần|shirt|shoe|sandal|clothing|fashion", product_context)
    )
    filtered_sizes: list[str] = []
    for s in sizes:
        if re.fullmatch(r"(3[6-9]|4[0-4])", s) and not is_wearable:
            continue
        filtered_sizes.append(s)

    return " | ".join(_unique_preserve_order(colors)), " | ".join(
        _unique_preserve_order(filtered_sizes)
    )


def clean_price(value: object) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    normalized = re.sub(r"[^\d.,]", "", text)
    if not normalized:
        return None

    # Handle decimal separator heuristically.
    if normalized.count(",") > 0 and normalized.count(".") > 0:
        if normalized.rfind(",") > normalized.rfind("."):
            normalized = normalized.replace(".", "").replace(",", ".")
        else:
            normalized = normalized.replace(",", "")
    elif normalized.count(",") > 0 and normalized.count(".") == 0:
        normalized = normalized.replace(",", ".")

    try:
        return float(normalized)
    except ValueError:
        return None


def clean_rating(value: object) -> float:
    result = clean_price(value)
    return float(result) if result is not None else 0.0


def clean_reviews_count(value: object) -> int:
    text = clean_text(value)
    if not text:
        return 0
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return 0
    try:
        return int(digits)
    except ValueError:
        return 0


def truncate_words(text: str, max_words: int = 420) -> str:
    value = clean_text(text)
    if not value:
        return ""
    words = value.split()
    if len(words) <= max_words:
        return value
    return " ".join(words[:max_words]).rstrip(" ,;") + "..."


def build_searchable_text(row: pd.Series) -> str:
    pieces: list[str] = []
    mapping = [
        ("Tên sản phẩm", clean_text(row.get("title"))),
        ("Thương hiệu", clean_text(row.get("brand"))),
        ("Danh mục", clean_text(row.get("category"))),
        ("Màu sắc", clean_text(row.get("color"))),
        ("Kích thước", clean_text(row.get("size"))),
    ]

    description = truncate_words(clean_text(row.get("description")), max_words=300)
    tags = clean_text(row.get("tags"))

    for label, value in mapping:
        if value:
            pieces.append(f"{label}: {value}")
    if description:
        pieces.append(f"Mô tả: {description}")
    if tags:
        pieces.append(f"Đặc điểm: {tags}")

    return truncate_words("\n".join(pieces), max_words=420)


def _missing_ratio(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna("").astype(str).str.strip().eq("").mean())


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    total_before = len(df)

    for col in ["title", "description", "category", "brand", "tags", "searchable_text"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)

    df["tags"] = df["tags"].apply(normalize_tags)

    extracted = df.apply(extract_color_size, axis=1, result_type="expand")
    extracted.columns = ["color", "size"]
    df["color"] = extracted["color"]
    df["size"] = extracted["size"]

    df["price"] = df["price"].apply(clean_price)
    df["rating"] = df["rating"].apply(clean_rating)
    df["reviews_count"] = df["reviews_count"].apply(clean_reviews_count)

    df["title"] = df["title"].apply(clean_text)
    df["description"] = df["description"].apply(clean_text)

    # Remove invalid rows where both title and description are empty.
    df = df[~((df["title"] == "") & (df["description"] == ""))].copy()

    # Ensure product_id always exists.
    df["source"] = df["source"].apply(clean_text)
    missing_id_mask = df["product_id"].isna() | (df["product_id"].astype(str).str.strip() == "")
    for idx in df[missing_id_mask].index:
        source = clean_text(df.at[idx, "source"]) or "unknown"
        df.at[idx, "product_id"] = f"{source}_{idx}"
    df["product_id"] = df["product_id"].astype(str).str.strip()

    df["searchable_text"] = df.apply(build_searchable_text, axis=1)
    df = df[df["searchable_text"].str.strip() != ""].copy()

    df = df.drop_duplicates(subset=["product_id", "source"], keep="first").copy()

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLUMNS]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    total_after = len(df)
    removed = total_before - total_after
    has_color = int(df["color"].fillna("").astype(str).str.strip().ne("").sum())
    has_size = int(df["size"].fillna("").astype(str).str.strip().ne("").sum())
    word_lengths = df["searchable_text"].fillna("").astype(str).apply(lambda x: len(x.split()))

    print(f"Số dòng ban đầu: {total_before}")
    print(f"Số dòng sau khi clean: {total_after}")
    print(f"Số dòng bị loại: {removed}")
    print("\nSố sản phẩm theo source:")
    source_counts = df["source"].fillna("").astype(str).value_counts()
    for source, count in source_counts.items():
        print(f"- {source or '(trống)'}: {count}")

    print(f"\nSố sản phẩm có color: {has_color}")
    print(f"Số sản phẩm có size: {has_size}")

    print("\nTỷ lệ missing các cột chính:")
    for col in ["title", "description", "category", "brand", "tags", "searchable_text"]:
        print(f"- {col}: {_missing_ratio(df[col]) * 100:.2f}%")

    print(
        f"\nĐộ dài trung bình searchable_text: "
        f"{df['searchable_text'].fillna('').astype(str).str.len().mean():.2f} ký tự "
        f"({word_lengths.mean():.2f} từ)"
    )
    print(
        f"Độ dài lớn nhất searchable_text: "
        f"{df['searchable_text'].fillna('').astype(str).str.len().max()} ký tự "
        f"({word_lengths.max()} từ)"
    )

    print("\n5 mẫu dữ liệu (title, color, size, tags, searchable_text):")
    sample = df[["title", "color", "size", "tags", "searchable_text"]].head(5)
    for i, row in sample.iterrows():
        print(f"\n[{i}] title: {row['title']}")
        print(f"color: {row['color']}")
        print(f"size: {row['size']}")
        print(f"tags: {row['tags']}")
        print(f"searchable_text:\n{row['searchable_text']}")

    # File `merged_products_vi_cleaned.csv` sẽ dùng để:
    # - tạo embedding vector;
    # - upsert vào Qdrant;
    # - tạo tập query-positive để fine-tune embedding model;
    # - làm product corpus cho semantic search.


if __name__ == "__main__":
    main()
