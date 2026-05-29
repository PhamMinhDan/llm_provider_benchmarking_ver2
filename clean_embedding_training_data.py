from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path


INPUT_DIR = Path("data/training")
OUTPUT_DIR = Path("data/training_cleaned")
TRAIN_IN = INPUT_DIR / "train.jsonl"
VALID_IN = INPUT_DIR / "valid.jsonl"
TEST_IN = INPUT_DIR / "test.jsonl"

TRAIN_OUT = OUTPUT_DIR / "train_cleaned.jsonl"
VALID_OUT = OUTPUT_DIR / "valid_cleaned.jsonl"
TEST_OUT = OUTPUT_DIR / "test_cleaned.jsonl"
LABELS_OUT = OUTPUT_DIR / "query_product_labels_cleaned.json"
REPORT_OUT = OUTPUT_DIR / "cleaning_report.txt"

SEED = 42
MAX_QUERIES_PER_PRODUCT = 5

NOISY_PHRASES = [
    "chính hãng",
    "chất lượng dùng tốt",
    "sản phẩm tốt",
    "mua online",
    "giá rẻ",
    "hot sale",
    "2024 new",
    "new",
]

BAD_QUERY_PATTERNS = [
    r"\bb09[a-z0-9]+\b",
    r"\b\d{3,}[a-z]\d+\b",
    r"\b\d{4}new\b",
]

SENSITIVE_WORDS = ["sex", "khiêu dâm", "ma túy", "vũ khí", "lừa đảo"]
PRODUCT_HINTS = [
    "giày",
    "áo",
    "quần",
    "tai nghe",
    "đèn",
    "dầu",
    "ốp lưng",
    "ba lô",
    "dụng cụ",
    "máy",
    "sạc",
    "loa",
    "bàn phím",
    "chuột",
    "kem",
    "serum",
    "nước hoa",
]
STOPWORDS_VI = {
    "và",
    "của",
    "cho",
    "với",
    "có",
    "dùng",
    "nam",
    "nữ",
    "loại",
    "màu",
    "size",
    "tên",
    "sản",
    "phẩm",
    "mô",
    "tả",
    "đặc",
    "điểm",
    "thương",
    "hiệu",
    "danh",
    "mục",
    "kích",
    "thước",
    "sắc",
    "loại",
}

GENERIC_QUERY_PHRASES = {
    "sản phẩm chất lượng",
    "hàng chính hãng",
    "mua sản phẩm tốt",
    "thương hiệu nổi tiếng",
    "sản phẩm",
    "hàng",
    "thương hiệu",
}

GARBAGE_TOKENS = {
    "ten",
    "san",
    "pham",
    "mo",
    "ta",
    "dac",
    "diem",
    "thuong",
    "hieu",
    "danh",
    "muc",
}


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zà-ỹ0-9]+", text.lower())


def clean_query(query: str) -> str:
    q = (query or "").strip()
    q = q.replace("\r", " ").replace("\n", " ")
    q = re.sub(r"[^\w\sà-ỹÀ-Ỹ]", " ", q)
    q = re.sub(r"_+", " ", q)
    q = re.sub(r"\s{2,}", " ", q).strip().lower()

    for phrase in NOISY_PHRASES:
        q = re.sub(rf"\b{re.escape(phrase)}\b", " ", q, flags=re.I)

    # Remove standalone "premium" if weak semantic value.
    q = re.sub(r"\bpremium\b", " ", q, flags=re.I)
    q = re.sub(r"\b(tên|sản|phẩm|mô|tả|đặc|điểm)\b", " ", q, flags=re.I)
    q = re.sub(r"\b(ten|san|pham|mo|ta|dac|diem)\b", " ", q, flags=re.I)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def english_token_ratio(query: str) -> float:
    toks = _tokens(query)
    if not toks:
        return 1.0
    en = 0
    for t in toks:
        if re.fullmatch(r"[a-z0-9]+", t) and not re.search(r"[à-ỹ]", t):
            en += 1
    return en / len(toks)


def numeric_token_ratio(query: str) -> float:
    toks = _tokens(query)
    if not toks:
        return 1.0
    num = sum(1 for t in toks if re.fullmatch(r"\d+(?:[a-z]+)?", t))
    return num / len(toks)


def _looks_like_product_code(token: str) -> bool:
    return bool(
        re.fullmatch(r"[a-z]*\d[a-z0-9-]{4,}", token.lower())
        or re.fullmatch(r"[a-z0-9]{8,}", token.lower())
    )


def is_invalid_query(query: str) -> bool:
    q = clean_query(query)
    if not q:
        return True

    toks = _tokens(q)
    if len(toks) < 3 or len(toks) > 12:
        return True

    if numeric_token_ratio(q) > 0.5:
        return True
    if english_token_ratio(q) > 0.6:
        return True

    if any(sw in q for sw in SENSITIVE_WORDS):
        return True

    if any(re.search(p, q, flags=re.I) for p in BAD_QUERY_PATTERNS):
        return True

    # Too many code-like tokens.
    code_like = sum(1 for t in toks if _looks_like_product_code(t))
    if code_like >= 2:
        return True

    # 1pc/10pcs without product noun context.
    if re.search(r"\b\d+\s*pcs?\b|\b1pc\b", q):
        if not any(hint in q for hint in PRODUCT_HINTS):
            return True

    # brand-only / code-only / weak generic query
    if q in GENERIC_QUERY_PHRASES:
        return True
    if len(toks) <= 2:
        return True
    if sum(1 for t in toks if t in GARBAGE_TOKENS) >= 2:
        return True

    return False


def extract_title_from_positive(positive: str) -> str:
    text = positive or ""
    m = re.search(r"Tên sản phẩm:\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip()
    return ""


def extract_category_terms(category: str) -> list[str]:
    cat = (category or "").strip()
    if not cat:
        return []
    parts = [p.strip().lower() for p in cat.split(">") if p.strip()]
    return parts[-2:] if len(parts) >= 2 else parts


def _parse_positive_fields(positive: str) -> dict[str, str]:
    fields = {
        "title": "",
        "brand": "",
        "category": "",
        "color": "",
        "size": "",
        "description": "",
        "features": "",
    }
    text = positive or ""
    pattern_map = {
        "title": r"Tên sản phẩm:\s*(.+)",
        "brand": r"Thương hiệu:\s*(.+)",
        "category": r"Danh mục:\s*(.+)",
        "color": r"Màu sắc:\s*(.+)",
        "size": r"Kích thước:\s*(.+)",
        "description": r"Mô tả:\s*(.+)",
        "features": r"(Đặc điểm(?:\s*/\s*tags)?|Đặc điểm):\s*(.+)",
    }
    for key, pattern in pattern_map.items():
        m = re.search(pattern, text, flags=re.I)
        if not m:
            continue
        if key == "features":
            fields[key] = m.group(2).strip()
        else:
            fields[key] = m.group(1).strip()
    return fields


def _pick_keywords(text: str, max_terms: int = 4) -> list[str]:
    toks = _tokens(text)
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if len(t) < 3:
            continue
        if t in STOPWORDS_VI:
            continue
        if _looks_like_product_code(t):
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def generate_natural_queries(record: dict) -> list[str]:
    positive = record.get("positive", "") or ""
    category = record.get("category", "") or ""
    parsed = _parse_positive_fields(positive)
    title = parsed["title"] or extract_title_from_positive(positive)
    brand = parsed["brand"]
    color = parsed["color"]
    size = parsed["size"]
    features_text = parsed["features"]
    category_from_positive = parsed["category"]
    category = category_from_positive or category
    category_terms = extract_category_terms(category)
    title_terms = _pick_keywords(title, max_terms=5)
    pos_terms = _pick_keywords(f"{features_text} {parsed['description']}", max_terms=8)

    text_low = positive.lower()
    domain_context = f"{category} {title}".lower()
    queries: list[str] = []

    domain = "general"
    if any(k in domain_context for k in ["giày", "áo", "quần", "váy", "ba lô", "túi", "thời trang"]):
        domain = "fashion"
    elif any(
        k in domain_context
        for k in ["tai nghe", "bluetooth", "loa", "bàn phím", "chuột", "điện thoại", "điện tử"]
    ):
        domain = "electronics"
    elif any(k in domain_context for k in ["dầu", "serum", "kem", "tóc", "da", "makeup", "mỹ phẩm", "làm đẹp"]):
        domain = "beauty"
    elif any(k in domain_context for k in ["đèn", "năng lượng mặt trời", "nhà", "bếp", "vườn", "gia dụng"]):
        domain = "home"
    elif any(k in domain_context for k in ["lốp", "ô tô", "xe", "phanh", "sửa chữa", "automotive"]):
        domain = "automotive"
    elif any(k in domain_context for k in ["chạy bộ", "thể thao", "gym", "fitness", "training"]):
        domain = "sports"

    # Template theo ngành hàng nhưng gắn với danh mục cụ thể.
    leaf = category_terms[-1] if category_terms else ""
    if domain == "fashion" and leaf:
        queries.extend([f"{leaf} mặc thoải mái", f"{leaf} thời trang dễ phối đồ"])
    elif domain == "electronics" and leaf:
        queries.extend([f"{leaf} dùng hằng ngày", f"{leaf} tiện dụng ổn định"])
    elif domain == "beauty" and leaf:
        queries.extend([f"{leaf} chăm sóc dịu nhẹ", f"{leaf} phục hồi hiệu quả"])
    elif domain == "home" and leaf:
        queries.extend([f"{leaf} dùng ngoài trời", f"{leaf} bền dễ sử dụng"])
    elif domain == "automotive" and leaf:
        queries.extend([f"{leaf} cho xe bền bỉ", f"{leaf} sửa chữa tiện lợi"])
    elif domain == "sports" and leaf:
        queries.extend([f"{leaf} tập luyện hằng ngày", f"{leaf} nhẹ thoải mái"])

    # Build from category + title terms.
    if category_terms:
        head = category_terms[-1]
        if title_terms:
            queries.append(
                f"{head} {title_terms[0]} {title_terms[1] if len(title_terms) > 1 else ''}".strip()
            )
        if len(pos_terms) >= 2:
            queries.append(f"{head} {pos_terms[0]} {pos_terms[1]}".strip())

    # Build from title key terms.
    if len(title_terms) >= 3:
        queries.append(" ".join(title_terms[:3]))
    if len(title_terms) >= 4:
        queries.append(" ".join(title_terms[:4]))

    if brand and category_terms:
        queries.append(f"{category_terms[-1]} {brand} {title_terms[0] if title_terms else 'chất lượng'}")
    elif brand and len(title_terms) >= 2:
        queries.append(f"{title_terms[0]} {title_terms[1]} {brand}")

    if color and category_terms:
        primary_color = clean_query(color.split("|")[0].strip())
        if primary_color:
            queries.append(f"{category_terms[-1]} màu {primary_color}")

    if size and category_terms:
        primary_size = clean_query(size.split("|")[0].strip())
        if primary_size:
            if re.search(r"\d+(ml|l|oz|inch|in)\b", primary_size):
                queries.append(f"{category_terms[-1]} {primary_size}")
            else:
                queries.append(f"{category_terms[-1]} size {primary_size.upper()}")

    # Feature-based generation.
    for feat in [
        "chống nước",
        "chống ồn",
        "thoáng khí",
        "giảm chấn",
        "không dây",
        "chống thấm",
        "dưỡng ẩm",
        "giảm rụng",
    ]:
        if feat in text_low and category_terms:
            queries.append(f"{category_terms[-1]} {feat}")

    # Normalize + validate
    normalized: list[str] = []
    seen: set[str] = set()
    for q in queries:
        qc = clean_query(q)
        if not qc or qc in seen:
            continue
        seen.add(qc)
        if is_invalid_query(qc):
            continue
        if "tên sản" in qc or "sản phẩm tên" in qc:
            continue
        normalized.append(qc)
        if len(normalized) >= 5:
            break

    return normalized[:5]


def _similarity_key(query: str) -> str:
    toks = sorted(set(_tokens(query)))
    return " ".join(toks)


def clean_and_regenerate(records: list[dict]) -> tuple[list[dict], int, list[tuple[str, str]]]:
    cleaned: list[dict] = []
    regenerated_count = 0
    before_after_samples: list[tuple[str, str]] = []

    grouped_by_product: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        pid = (rec.get("product_id") or "").strip()
        grouped_by_product[pid].append(rec)

    for pid, group in grouped_by_product.items():
        kept_queries: set[tuple[str, str]] = set()
        product_records: list[dict] = []

        for rec in group:
            old_q = rec.get("query", "") or ""
            new_q = clean_query(old_q)
            positive = (rec.get("positive") or "").strip()
            if not positive or not pid:
                continue

            if is_invalid_query(new_q):
                generated = generate_natural_queries(rec)
                if not generated:
                    before_after_samples.append((old_q, "REMOVED"))
                    continue
                regenerated_count += 1
                for gq in generated:
                    key = (gq, pid)
                    if key in kept_queries:
                        continue
                    kept_queries.add(key)
                    item = {
                        "query": gq,
                        "positive": positive,
                        "product_id": pid,
                        "category": (rec.get("category") or "").strip(),
                        "source": (rec.get("source") or "").strip(),
                    }
                    product_records.append(item)
                before_after_samples.append((old_q, generated[0]))
            else:
                key = (new_q, pid)
                if key in kept_queries:
                    continue
                kept_queries.add(key)
                item = {
                    "query": new_q,
                    "positive": positive,
                    "product_id": pid,
                    "category": (rec.get("category") or "").strip(),
                    "source": (rec.get("source") or "").strip(),
                }
                product_records.append(item)
                before_after_samples.append((old_q, new_q))

        cleaned.extend(product_records)

    return cleaned, regenerated_count, before_after_samples[:30]


def limit_queries_per_product(records: list[dict], max_queries: int = 5) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["product_id"]].append(rec)

    out: list[dict] = []
    for pid, items in grouped.items():
        unique_items: list[dict] = []
        seen_pairs: set[tuple[str, str]] = set()
        seen_sim: set[str] = set()
        for it in items:
            pair = (it["query"], pid)
            if pair in seen_pairs:
                continue
            sim = _similarity_key(it["query"])
            if sim in seen_sim:
                continue
            seen_pairs.add(pair)
            seen_sim.add(sim)
            unique_items.append(it)
        unique_items.sort(key=lambda x: (len(_tokens(x["query"])), x["query"]), reverse=True)
        out.extend(unique_items[:max_queries])
    return out


def split_train_valid_test(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(SEED)
    data = records[:]
    rng.shuffle(data)
    n = len(data)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    train = data[:n_train]
    valid = data[n_train : n_train + n_valid]
    test = data[n_train + n_valid :]
    return train, valid, test


def build_query_product_labels(records: list[dict]) -> list[dict]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        mapping[rec["query"]].add(rec["product_id"])
    return [
        {"query": q, "relevant_product_ids": sorted(list(pids))}
        for q, pids in sorted(mapping.items(), key=lambda x: x[0])
    ]


def _contains_phrase_count(records: list[dict], phrase: str) -> int:
    p = phrase.lower()
    return sum(1 for r in records if p in (r.get("query", "") or "").lower())


def _ratio_over_threshold(records: list[dict], ratio_fn, threshold: float) -> float:
    if not records:
        return 0.0
    count = 0
    for r in records:
        if ratio_fn(r.get("query", "")) > threshold:
            count += 1
    return count / len(records)


def save_report(
    *,
    report_path: Path,
    total_before: int,
    total_after: int,
    regenerated_count: int,
    before_records: list[dict],
    after_records: list[dict],
    train_count: int,
    valid_count: int,
    test_count: int,
    samples_before_after: list[tuple[str, str]],
) -> None:
    removed = total_before - total_after
    ch_before = _contains_phrase_count(before_records, "chính hãng")
    ch_after = _contains_phrase_count(after_records, "chính hãng")
    cl_before = _contains_phrase_count(before_records, "chất lượng dùng tốt")
    cl_after = _contains_phrase_count(after_records, "chất lượng dùng tốt")
    en_before = _ratio_over_threshold(before_records, english_token_ratio, 0.6)
    en_after = _ratio_over_threshold(after_records, english_token_ratio, 0.6)
    num_before = _ratio_over_threshold(before_records, numeric_token_ratio, 0.5)
    num_after = _ratio_over_threshold(after_records, numeric_token_ratio, 0.5)

    lines: list[str] = [
        "CLEANING REPORT - EMBEDDING TRAINING DATA",
        "=" * 60,
        f"Tổng số dòng ban đầu: {total_before}",
        f"Tổng số dòng sau khi clean: {total_after}",
        f"Số dòng bị loại: {removed}",
        f"Số query được tạo lại: {regenerated_count}",
        f"Số query chứa 'chính hãng' trước/sau: {ch_before}/{ch_after}",
        f"Số query chứa 'chất lượng dùng tốt' trước/sau: {cl_before}/{cl_after}",
        f"Tỷ lệ query nhiều tiếng Anh trước/sau: {en_before:.2%}/{en_after:.2%}",
        f"Tỷ lệ query nhiều số/ký hiệu trước/sau: {num_before:.2%}/{num_after:.2%}",
        f"Số dòng train/valid/test mới: {train_count}/{valid_count}/{test_count}",
        "",
        "30 mẫu query trước/sau:",
    ]
    for i, (b, a) in enumerate(samples_before_after[:30], start=1):
        lines.append(f"{i:02d}. BEFORE: {b}")
        lines.append(f"    AFTER : {a}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    train = load_jsonl(TRAIN_IN)
    valid = load_jsonl(VALID_IN)
    test = load_jsonl(TEST_IN)
    all_records = train + valid + test
    if not all_records:
        raise FileNotFoundError("Không tìm thấy dữ liệu đầu vào train/valid/test JSONL.")

    total_before = len(all_records)
    cleaned, regenerated_count, samples = clean_and_regenerate(all_records)
    cleaned = limit_queries_per_product(cleaned, max_queries=MAX_QUERIES_PER_PRODUCT)

    # Remove final duplicate query-product pairs
    seen: set[tuple[str, str]] = set()
    final_records: list[dict] = []
    for r in cleaned:
        key = ((r.get("query") or "").strip().lower(), (r.get("product_id") or "").strip())
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        final_records.append(r)

    train_new, valid_new, test_new = split_train_valid_test(final_records)
    labels = build_query_product_labels(final_records)

    save_jsonl(train_new, TRAIN_OUT)
    save_jsonl(valid_new, VALID_OUT)
    save_jsonl(test_new, TEST_OUT)
    LABELS_OUT.parent.mkdir(parents=True, exist_ok=True)
    LABELS_OUT.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")

    save_report(
        report_path=REPORT_OUT,
        total_before=total_before,
        total_after=len(final_records),
        regenerated_count=regenerated_count,
        before_records=all_records,
        after_records=final_records,
        train_count=len(train_new),
        valid_count=len(valid_new),
        test_count=len(test_new),
        samples_before_after=samples,
    )

    print("Đã tạo xong các file output:")
    print(f"- {TRAIN_OUT.resolve()}")
    print(f"- {VALID_OUT.resolve()}")
    print(f"- {TEST_OUT.resolve()}")
    print(f"- {LABELS_OUT.resolve()}")
    print(f"- {REPORT_OUT.resolve()}")
    print(f"Tổng records cuối cùng: {len(final_records)}")
    print("Mẫu query sạch:")
    for rec in final_records[:8]:
        print(f"- {rec['query']} (product_id={rec['product_id']})")


if __name__ == "__main__":
    main()
