from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

INPUT_CSV = Path("merged_products_vi_cleaned.csv")
OUTPUT_DIR = Path("data/training")
TRAIN_PATH = OUTPUT_DIR / "train.jsonl"
VALID_PATH = OUTPUT_DIR / "valid.jsonl"
TEST_PATH = OUTPUT_DIR / "test.jsonl"
LABELS_PATH = OUTPUT_DIR / "query_product_labels.json"
REPORT_PATH = OUTPUT_DIR / "labeling_report.txt"
SEED = 42

GENERIC_BAD_QUERIES = {
    "sản phẩm tốt",
    "chất lượng cao",
    "mua hàng online",
    "sản phẩm giá rẻ",
}

FEATURE_KEYWORDS = [
    "chống nước",
    "chống ồn",
    "giảm tiếng ồn",
    "thoáng khí",
    "giảm chấn",
    "nhẹ",
    "dưỡng ẩm",
    "không dây",
    "bluetooth",
    "an toàn",
    "chạy bộ",
    "năng lượng mặt trời",
]


def clean_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    value = str(text).strip()
    if not value:
        return ""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s*\n\s*", " ", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"([,.;:!?])\1+", r"\1", value)
    return value.strip(" \n\t,;")


def split_tags(tags: object) -> list[str]:
    raw = clean_text(tags)
    if not raw:
        return []
    parts = re.split(r"[|,;]+", raw)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        tag = clean_text(part)
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    return out


def get_leaf_category(category: object) -> str:
    raw = clean_text(category)
    if not raw:
        return ""
    parts = [clean_text(x) for x in raw.split(">")]
    parts = [p for p in parts if p]
    return parts[-1] if parts else ""


def extract_key_terms(row: pd.Series) -> dict:
    title = clean_text(row.get("title"))
    description = clean_text(row.get("description"))
    category = clean_text(row.get("category"))
    leaf_category = get_leaf_category(category)
    brand = clean_text(row.get("brand"))
    color = clean_text(row.get("color"))
    size = clean_text(row.get("size"))
    tags = split_tags(row.get("tags"))
    source = clean_text(row.get("source"))
    positive = clean_text(row.get("searchable_text"))
    product_id = clean_text(row.get("product_id"))

    context = f"{title} {description} {' '.join(tags)}".casefold()
    matched_features = [kw for kw in FEATURE_KEYWORDS if kw in context]

    return {
        "title": title,
        "description": description,
        "category": category,
        "leaf_category": leaf_category,
        "brand": brand,
        "color": color,
        "size": size,
        "tags": tags,
        "source": source,
        "positive": positive,
        "product_id": product_id,
        "features": matched_features,
    }


def _truncate_query(query: str, max_words: int = 12) -> str:
    words = query.split()
    if len(words) <= max_words:
        return query
    return " ".join(words[:max_words]).strip()


def generate_queries(row: pd.Series) -> list[str]:
    terms = extract_key_terms(row)
    leaf = terms["leaf_category"].lower()
    title = terms["title"]
    brand = terms["brand"]
    color = terms["color"]
    size = terms["size"]
    tags = terms["tags"]
    features = terms["features"]

    queries: list[str] = []

    # Base category query from title/category context.
    if leaf:
        if "giày" in leaf:
            queries.extend(
                [
                    "giày chạy bộ thoáng khí",
                    "giày thể thao giảm chấn tốt",
                ]
            )
        elif "áo" in leaf or "quần" in leaf or "vest" in leaf:
            queries.extend(
                [
                    f"{leaf} mặc thoải mái",
                    f"{leaf} chất liệu bền",
                ]
            )
        elif "tai nghe" in leaf:
            queries.extend(
                [
                    "tai nghe không dây chống ồn",
                    "tai nghe bluetooth dùng hằng ngày",
                ]
            )
        else:
            queries.append(f"{leaf} chất lượng dùng tốt")

    # Pull 1-2 product-specific queries from title.
    title_words = [w for w in re.findall(r"[A-Za-zÀ-ỹ0-9]+", title) if len(w) > 2]
    if title_words:
        base = " ".join(title_words[:6]).lower()
        queries.append(base)
        if len(title_words) >= 4:
            queries.append(" ".join(title_words[:4]).lower() + " chính hãng")

    # Feature-driven queries from tags/description.
    for feat in features[:2]:
        if leaf:
            queries.append(f"{leaf} {feat}")
        else:
            queries.append(f"sản phẩm {feat}")

    for tag in tags[:2]:
        tag_norm = tag.lower()
        if leaf and tag_norm not in leaf:
            queries.append(f"{leaf} {tag_norm}")
        else:
            queries.append(tag_norm)

    # Query with color/size if available.
    if color:
        first_color = clean_text(color.split("|")[0]).lower()
        color_q = f"{leaf or 'sản phẩm'} màu {first_color}"
        queries.append(color_q)

    if size:
        first_size = clean_text(size.split("|")[0]).upper()
        if re.search(r"\d+\s*(ML|L|OZ|INCH|IN)\b", first_size, flags=re.I):
            size_q = f"{leaf or 'sản phẩm'} {first_size.lower()}"
        else:
            size_q = f"{leaf or 'sản phẩm'} size {first_size}"
        queries.append(size_q)

    if brand:
        # Must not be brand only.
        if leaf:
            queries.append(f"{leaf} {brand} chính hãng".lower())
        else:
            queries.append(f"sản phẩm {brand} chất lượng".lower())

    # Normalize and select 3-5 strong queries.
    normalized: list[str] = []
    seen: set[str] = set()
    for query in queries:
        q = clean_text(query).lower()
        q = re.sub(r"\s{2,}", " ", q).strip()
        q = _truncate_query(q, max_words=12)
        if not q:
            continue
        if q in GENERIC_BAD_QUERIES:
            continue
        if q in seen:
            continue
        seen.add(q)
        normalized.append(q)

    # Prioritize richer queries first.
    normalized.sort(key=lambda q: (len(q.split()) < 3, len(q.split())))

    # Ensure at least 3 queries if possible.
    if len(normalized) < 3 and title_words:
        fallback = " ".join(title_words[:5]).lower()
        fallback = _truncate_query(clean_text(fallback), 12)
        if fallback and fallback not in seen:
            normalized.append(fallback)

    return normalized[:5]


def build_labeled_pairs(df: pd.DataFrame) -> list[dict]:
    pairs: list[dict] = []
    for _, row in df.iterrows():
        queries = generate_queries(row)
        positive = clean_text(row.get("searchable_text"))
        product_id = clean_text(row.get("product_id"))
        category = clean_text(row.get("category"))
        source = clean_text(row.get("source"))
        for query in queries:
            pairs.append(
                {
                    "query": query,
                    "positive": positive,
                    "product_id": product_id,
                    "category": category,
                    "source": source,
                }
            )
    return pairs


def remove_invalid_pairs(pairs: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in pairs:
        query = clean_text(item.get("query"))
        positive = clean_text(item.get("positive"))
        product_id = clean_text(item.get("product_id"))

        if not query or not positive or not product_id:
            continue
        wc = len(query.split())
        if wc < 3 or wc > 12:
            continue
        key = (query.casefold(), product_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "query": query,
                "positive": positive,
                "product_id": product_id,
                "category": clean_text(item.get("category")),
                "source": clean_text(item.get("source")),
            }
        )
    return out


def split_train_valid_test(pairs: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * 0.8)
    valid_end = train_end + int(n * 0.1)
    train = shuffled[:train_end]
    valid = shuffled[train_end:valid_end]
    test = shuffled[valid_end:]
    return train, valid, test


def build_query_product_labels(pairs: list[dict]) -> list[dict]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for item in pairs:
        mapping[item["query"]].add(item["product_id"])

    result = [
        {"query": query, "relevant_product_ids": sorted(product_ids)}
        for query, product_ids in sorted(mapping.items(), key=lambda x: x[0])
    ]
    return result


def save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_report(
    *,
    report_path: Path,
    total_products: int,
    total_pairs: int,
    avg_queries_per_product: float,
    train_count: int,
    valid_count: int,
    test_count: int,
    unique_query_count: int,
    category_counter: Counter,
    sample_pairs: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("BÁO CÁO GÁN NHÃN DỮ LIỆU EMBEDDING")
    lines.append("=" * 50)
    lines.append(f"Tổng số sản phẩm đầu vào: {total_products}")
    lines.append(f"Tổng số query-positive pairs tạo được: {total_pairs}")
    lines.append(f"Số query trung bình trên mỗi sản phẩm: {avg_queries_per_product:.2f}")
    lines.append(f"Số dòng train: {train_count}")
    lines.append(f"Số dòng valid: {valid_count}")
    lines.append(f"Số dòng test: {test_count}")
    lines.append(f"Số query unique: {unique_query_count}")
    lines.append("")
    lines.append("Top 10 category có nhiều query nhất:")
    for category, count in category_counter.most_common(10):
        lines.append(f"- {category or '(trống)'}: {count}")
    lines.append("")
    lines.append("20 mẫu query-product để kiểm tra thủ công:")
    for idx, item in enumerate(sample_pairs[:20], start=1):
        lines.append(
            f"{idx:02d}. query='{item['query']}' | product_id={item['product_id']} | "
            f"source={item['source']}"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(f"Không tìm thấy file input: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    total_products = len(df)

    pairs_raw = build_labeled_pairs(df)
    pairs = remove_invalid_pairs(pairs_raw)
    train, valid, test = split_train_valid_test(pairs)
    labels = build_query_product_labels(pairs)

    save_jsonl(train, TRAIN_PATH)
    save_jsonl(valid, VALID_PATH)
    save_jsonl(test, TEST_PATH)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    category_counter = Counter(item["category"] for item in pairs)
    unique_query_count = len({item["query"] for item in pairs})
    avg_queries = (len(pairs) / total_products) if total_products else 0.0

    rng = random.Random(SEED)
    sample_pairs = pairs[:]
    rng.shuffle(sample_pairs)

    save_report(
        report_path=REPORT_PATH,
        total_products=total_products,
        total_pairs=len(pairs),
        avg_queries_per_product=avg_queries,
        train_count=len(train),
        valid_count=len(valid),
        test_count=len(test),
        unique_query_count=unique_query_count,
        category_counter=category_counter,
        sample_pairs=sample_pairs,
    )

    print("Đã tạo xong các file:")
    print(f"- {TRAIN_PATH.resolve()}")
    print(f"- {VALID_PATH.resolve()}")
    print(f"- {TEST_PATH.resolve()}")
    print(f"- {LABELS_PATH.resolve()}")
    print(f"- {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
