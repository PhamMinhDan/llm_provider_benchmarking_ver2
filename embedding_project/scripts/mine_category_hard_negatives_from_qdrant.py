"""
Mine hard negatives từ Qdrant (v2 — lọc chất lượng).

Điều kiện negative (tất cả phải thỏa):
  1. product_id != positive
  2. Khác category leaf (sau normalize mạnh); L1 khác hoặc leaf khác hẳn loại SP
  3. Negative KHÔNG cùng nhóm từ khóa sản phẩm với query (giày/sneaker/chạy bộ/tai nghe...)
  4. Không match cặp loại dễ nhầm (tai nghe vs bông tai...)
  5. Score cosine trong khoảng [min_score, max_score] — vừa khó, không quá dễ / quá giống
  6. Bỏ qua top-(min_rank-1) kết quả (thường là SP đúng khác)

Output: JSONL triplets cho TripletLoss.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Intent cụ thể (không dùng nhóm rộng "giày") — query VÀ negative cùng match → false negative.
SPECIFIC_INTENT_PATTERNS: list[str] = [
    r"giày sneaker|sneaker thời trang|\bsneaker\b",
    r"chạy bộ|giày chạy|giày running",
    r"giày thể thao",
    r"giày mưa",
    r"giày công nghiệp|mũi thép|giày ủng công",
    r"giày trượt|trượt inline|trượt patin",
    r"lót giày|đệm.*giày",
    r"\boxford\b",
    r"\btai nghe\b",
    r"\bearbuds?\b",
    r"\bheadphone",
    r"dầu dưỡng tóc|dầu gội",
    r"máy sấy tóc",
    r"kem chống nắng",
    r"\bson môi\b",
    r"đồng hồ quartz",
    r"micro thu|microphone",
    r"\bloa bluetooth\b",
    r"ốp lưng",
    r"\bchuột\b",
    r"bàn phím",
    r"pin dự phòng|sạc dự phòng",
    r"phụ kiện giày|trang trí giày|khóa giày",
]

# Cặp loại dễ nhầm (chỉ check title + category leaf, tránh match "Quần áo" trong path).
CONFUSION_REJECT: list[tuple[str, list[str]]] = [
    (r"\btai nghe\b", [r"\bbông tai\b", r"\bhoa tai\b", r"\bkhuyên tai\b"]),
    (r"\bearbuds?\b", [r"\bbông tai\b", r"\bhoa tai\b"]),
    (r"giày trượt|trượt inline", [r"giày sneaker thời trang", r"\bsneaker\b", r"giày lười"]),
]

PROBLEM_QUERY_PATTERNS: list[str] = [
    r"\btai nghe\b",
    r"\bgiày\b",
    r"máy sấy tóc",
    r"kem chống nắng",
    r"dầu gội|dầu dưỡng tóc|tinh dầu mọc tóc|giảm rụng",
    r"\bson môi\b",
    r"micro thu",
    r"giày trượt",
]

_CAT_ALIASES: list[tuple[str, str]] = [
    (r"quần áo,\s*giày dép\s*&\s*trang sức", "quần áo, giày & trang sức"),
    (r"giày dép", "giày"),
]

_LEAF_STOPWORDS = frozenset(
    {
        "nam",
        "nữ",
        "unisex",
        "trẻ",
        "em",
        "cho",
        "giày",
        "quần",
        "áo",
        "thể",
        "thao",
        "ngoài",
        "trời",
        "thời",
        "trang",
        "và",
        "&",
    }
)

DEFAULT_MINE_MODEL = Path("embedding_project/models/e5_base_finetuned_final")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_category_full(cat: str) -> str:
    c = (cat or "").strip().lower()
    for pat, repl in _CAT_ALIASES:
        c = re.sub(pat, repl, c)
    return re.sub(r"\s+", " ", c)


def category_parts(cat: str) -> list[str]:
    return [p.strip() for p in normalize_category_full(cat).split(">") if p.strip()]


def category_l1(cat: str) -> str:
    parts = category_parts(cat)
    return parts[0] if parts else ""


def category_leaf(cat: str) -> str:
    parts = category_parts(cat)
    return parts[-1] if parts else ""


def leaf_type_tokens(leaf: str) -> set[str]:
    tokens = set(re.findall(r"[\wàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]+", leaf.lower()))
    return {t for t in tokens if t not in _LEAF_STOPWORDS and len(t) > 2}


def leaves_too_similar(pos_cat: str, neg_cat: str) -> bool:
    pos_leaf = category_leaf(pos_cat)
    neg_leaf = category_leaf(neg_cat)
    if not pos_leaf or not neg_leaf:
        return False
    if pos_leaf == neg_leaf:
        return True
    if pos_leaf in neg_leaf or neg_leaf in pos_leaf:
        return True
    pos_t = leaf_type_tokens(pos_leaf)
    neg_t = leaf_type_tokens(neg_leaf)
    if not pos_t or not neg_t:
        return False
    overlap = pos_t & neg_t
    if len(overlap) >= 2:
        return True
    if len(overlap) == 1 and len(pos_t) <= 2 and len(neg_t) <= 2:
        return True
    return False


FOOTWEAR_SUBTYPES: list[tuple[str, str]] = [
    ("running", r"chạy bộ|giày chạy|giày running"),
    ("sneaker", r"sneaker|giày sneaker"),
    ("athletic", r"giày thể thao"),
    ("slip_on", r"slip[\s-]?on|giày lười|loafer"),
    ("rain", r"giày mưa"),
    ("inline", r"giày trượt|trượt inline|trượt patin"),
    ("boot", r"\bboot\b|\bbốt\b"),
]

_ATHLETIC_FAMILY = frozenset({"running", "sneaker", "athletic", "slip_on"})


def _footwear_subtypes(text: str) -> set[str]:
    t = (text or "").lower()
    return {name for name, pat in FOOTWEAR_SUBTYPES if re.search(pat, t)}


def _is_work_safety_shoe(text: str) -> bool:
    return bool(re.search(r"mũi thép|toe thép|giày ủng công|công nghiệp|an toàn lao động", text.lower()))


def footwear_subtype_false_negative(anchor: str, neg_title: str, neg_category: str) -> bool:
    """Query tìm loại giày X, negative vẫn là giày cùng họ (chạy/sneaker/thể thao/slip-on)."""
    if not re.search(r"\bgiày\b", (anchor or "").lower()):
        return False
    neg_blob = f"{neg_title} {category_leaf(neg_category)}".lower()
    if _is_work_safety_shoe(neg_blob):
        return False

    q_sub = _footwear_subtypes(anchor)
    if not q_sub:
        return False

    n_sub = _footwear_subtypes(neg_blob)
    if not n_sub:
        return False
    if q_sub & n_sub:
        return True
    if (q_sub & _ATHLETIC_FAMILY) and (n_sub & _ATHLETIC_FAMILY):
        return True
    return False


def shared_specific_intent(anchor: str, neg_title: str, neg_category: str) -> bool:
    """Query và negative cùng intent cụ thể → false negative."""
    anchor_l = (anchor or "").lower()
    neg_blob = f"{neg_title} {neg_category}".lower()
    for pattern in SPECIFIC_INTENT_PATTERNS:
        if re.search(pattern, anchor_l) and re.search(pattern, neg_blob):
            return True
    return False


def matches_confusion_reject(anchor: str, neg_title: str, neg_category: str) -> bool:
    anchor_l = (anchor or "").lower()
    neg_blob = f"{neg_title} {category_leaf(neg_category)}".lower()
    for query_pat, neg_pats in CONFUSION_REJECT:
        if re.search(query_pat, anchor_l):
            if any(re.search(np, neg_blob) for np in neg_pats):
                return True
    return False


def looks_problem_query(q: str, patterns: list[str]) -> bool:
    qq = (q or "").lower()
    return any(re.search(p, qq) for p in patterns)


def neg_text_from_candidate(c: dict[str, Any]) -> str:
    neg_text = str(c.get("searchable_text", "")).strip()
    if not neg_text:
        neg_text = (str(c.get("title", "")).strip() + " " + str(c.get("category", "")).strip()).strip()
    return neg_text


def is_valid_negative(
    anchor: str,
    pos_pid: str,
    pos_cat: str,
    candidate: dict[str, Any],
    seen_neg_pid: set[str],
    *,
    min_score: float,
    max_score: float,
    rank: int,
    min_rank: int,
) -> tuple[bool, str]:
    if rank < min_rank:
        return False, "rank_too_high"

    neg_pid = str(candidate.get("product_id", "")).strip()
    if not neg_pid or neg_pid == pos_pid:
        return False, "same_or_empty_pid"
    if neg_pid in seen_neg_pid:
        return False, "duplicate_pid"

    neg_cat_raw = str(candidate.get("category", "")).strip()
    neg_cat_norm = normalize_category_full(neg_cat_raw)
    pos_cat_norm = normalize_category_full(pos_cat)

    if neg_cat_norm == pos_cat_norm:
        return False, "same_category"

    pos_l1 = category_l1(pos_cat)
    neg_l1 = category_l1(neg_cat_raw)
    if pos_l1 == neg_l1 and leaves_too_similar(pos_cat, neg_cat_raw):
        return False, "same_l1_similar_leaf"

    if leaves_too_similar(pos_cat, neg_cat_raw):
        return False, "similar_leaf"

    neg_title = str(candidate.get("title", "")).strip()
    if shared_specific_intent(anchor, neg_title, neg_cat_raw):
        return False, "shared_intent"

    if footwear_subtype_false_negative(anchor, neg_title, neg_cat_raw):
        return False, "footwear_subtype"

    if matches_confusion_reject(anchor, neg_title, neg_cat_raw):
        return False, "confusion_pair"

    score = float(candidate.get("score", 0.0))
    if score < min_score:
        return False, "score_too_low"
    if score > max_score:
        return False, "score_too_high"

    if not neg_text_from_candidate(candidate):
        return False, "empty_text"

    return True, ""


def mine_for_rows(
    rows: list[dict[str, Any]],
    emb: Any,
    qdr: Any,
    *,
    problem_only: bool,
    max_anchors: int,
    top_k: int,
    neg_per_anchor: int,
    min_score: float,
    max_score: float,
    min_rank: int,
    max_neg_reuse: int,
    label: str = "",
    global_neg_use: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mined: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "anchors_processed": 0,
        "anchors_no_neg": 0,
        "skipped_rank_too_high": 0,
        "skipped_same_category": 0,
        "skipped_similar_leaf": 0,
        "skipped_same_l1_similar_leaf": 0,
        "skipped_shared_intent": 0,
        "skipped_footwear_subtype": 0,
        "skipped_confusion_pair": 0,
        "skipped_score_too_low": 0,
        "skipped_score_too_high": 0,
        "skipped_other": 0,
    }

    stat_key = {
        "rank_too_high": "skipped_rank_too_high",
        "same_category": "skipped_same_category",
        "similar_leaf": "skipped_similar_leaf",
        "same_l1_similar_leaf": "skipped_same_l1_similar_leaf",
        "shared_intent": "skipped_shared_intent",
        "footwear_subtype": "skipped_footwear_subtype",
        "confusion_pair": "skipped_confusion_pair",
        "score_too_low": "skipped_score_too_low",
        "score_too_high": "skipped_score_too_high",
    }

    for r in rows:
        if problem_only and not looks_problem_query(r.get("query", ""), PROBLEM_QUERY_PATTERNS):
            continue
        if stats["anchors_processed"] >= max_anchors:
            break

        anchor = str(r.get("query", "")).strip()
        positive = str(r.get("positive", "")).strip()
        pos_pid = str(r.get("product_id", "")).strip()
        pos_cat = str(r.get("category", "")).strip()

        if not anchor or not positive or not pos_pid:
            continue

        qvec = emb.encode_query(anchor)
        candidates = qdr.search(qvec, top_k=top_k)

        passing: list[tuple[dict[str, Any], float, int]] = []
        seen_neg_pid: set[str] = set()
        neg_use = global_neg_use if global_neg_use is not None else {}

        for rank, c in enumerate(candidates, start=1):
            ok, reason = is_valid_negative(
                anchor,
                pos_pid,
                pos_cat,
                c,
                seen_neg_pid,
                min_score=min_score,
                max_score=max_score,
                rank=rank,
                min_rank=min_rank,
            )
            if not ok:
                key = stat_key.get(reason)
                if key:
                    stats[key] += 1
                elif reason not in ("same_or_empty_pid", "duplicate_pid"):
                    stats["skipped_other"] += 1
                continue

            neg_pid = str(c.get("product_id", "")).strip()
            if neg_use.get(neg_pid, 0) >= max_neg_reuse:
                stats["skipped_other"] += 1
                continue
            seen_neg_pid.add(neg_pid)
            passing.append((c, float(c.get("score", 0.0)), rank))

        # Ưu tiên: score cao, ít bị reuse, rank thấp hơn.
        passing.sort(key=lambda x: (-x[1], neg_use.get(str(x[0].get("product_id", "")).strip(), 0), x[2]))

        negs: list[dict[str, Any]] = []
        for c, score, rank in passing[:neg_per_anchor]:
            neg_pid = str(c.get("product_id", "")).strip()
            neg_use[neg_pid] = neg_use.get(neg_pid, 0) + 1
            negs.append(
                {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": neg_text_from_candidate(c),
                    "positive_product_id": pos_pid,
                    "negative_product_id": neg_pid,
                    "positive_category": pos_cat,
                    "negative_category": str(c.get("category", "")).strip(),
                    "negative_score": round(score, 4),
                    "negative_rank": rank,
                }
            )

        if not negs:
            stats["anchors_no_neg"] += 1

        mined.extend(negs)
        stats["anchors_processed"] += 1

        if stats["anchors_processed"] % 50 == 0:
            prefix = f"[{label}] " if label else ""
            print(f"{prefix}Processed={stats['anchors_processed']} | triplets={len(mined)}")

    return mined, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine category hard negatives from Qdrant (v2).")
    parser.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("embedding_project/data/train_cleaned_v2.jsonl"),
    )
    parser.add_argument("--valid-jsonl", type=Path, default=None)
    parser.add_argument("--mine-valid", action="store_true")

    parser.add_argument(
        "--out-train",
        type=Path,
        default=Path("embedding_project/data/train_triplets_category_hardneg_v2.jsonl"),
    )
    parser.add_argument(
        "--out-valid",
        type=Path,
        default=Path("embedding_project/data/valid_triplets_category_hardneg_v2.jsonl"),
    )

    parser.add_argument("--collection", type=str, default="products_vi_e5_2ep")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MINE_MODEL,
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--neg-per-anchor", type=int, default=1)
    parser.add_argument("--max-anchors", type=int, default=2000)
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.48,
        help="Score tối thiểu — loại negative quá dễ (vd. bông tai cho query tai nghe).",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=0.84,
        help="Score tối đa — loại negative quá giống intent (false negative).",
    )
    parser.add_argument(
        "--min-rank",
        type=int,
        default=2,
        help="Bỏ qua top-(min_rank-1) kết quả search (thường là SP đúng khác).",
    )
    parser.add_argument("--problem-only", action="store_true", default=False)
    parser.add_argument(
        "--max-neg-reuse",
        type=int,
        default=5,
        help="Mỗi negative_product_id tối đa xuất hiện N lần trong file output.",
    )
    args = parser.parse_args()

    import sys

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from vector_db.config import apply_runtime_config
    from vector_db.embedding_service import EmbeddingService
    from vector_db.qdrant_service import QdrantService

    apply_runtime_config(
        collection=args.collection,
        model_path=str(args.model_path),
        use_e5_prefix=True,
        encode_batch_size=8,
    )
    print(
        f"Mine v2 | model={args.model_path} | collection={args.collection}\n"
        f"Score band [{args.min_score}, {args.max_score}] | min_rank={args.min_rank}\n"
        "Lần đầu load thư viện trên Windows có thể ~20–30s.",
        flush=True,
    )
    emb = EmbeddingService()
    qdr = QdrantService()
    print("Sẵn sàng mine.", flush=True)

    rows = load_jsonl(args.train_jsonl)
    global_neg_use: dict[str, int] = {}
    mined, stats = mine_for_rows(
        rows,
        emb,
        qdr,
        problem_only=args.problem_only,
        max_anchors=args.max_anchors,
        top_k=args.top_k,
        neg_per_anchor=args.neg_per_anchor,
        min_score=args.min_score,
        max_score=args.max_score,
        min_rank=args.min_rank,
        max_neg_reuse=args.max_neg_reuse,
        global_neg_use=global_neg_use,
    )
    save_jsonl(args.out_train, mined)

    print(f"\nSaved: {args.out_train}")
    print(f"  anchors={stats['anchors_processed']} | triplets={len(mined)}")
    print(f"  anchors_no_neg={stats['anchors_no_neg']}")
    for k in (
        "skipped_similar_leaf",
        "skipped_shared_intent",
        "skipped_footwear_subtype",
        "skipped_confusion_pair",
        "skipped_score_too_low",
        "skipped_score_too_high",
        "skipped_rank_too_high",
    ):
        print(f"  {k}={stats[k]}")

    if args.mine_valid:
        if not args.valid_jsonl:
            raise ValueError("--mine-valid bật nhưng thiếu --valid-jsonl")
        rows_valid = load_jsonl(args.valid_jsonl)
        mined_valid, _ = mine_for_rows(
            rows_valid,
            emb,
            qdr,
            problem_only=args.problem_only,
            max_anchors=args.max_anchors,
            top_k=args.top_k,
            neg_per_anchor=args.neg_per_anchor,
            min_score=args.min_score,
            max_score=args.max_score,
            min_rank=args.min_rank,
            max_neg_reuse=args.max_neg_reuse,
            label="valid",
            global_neg_use=global_neg_use,
        )
        save_jsonl(args.out_valid, mined_valid)
        print(f"Saved: {args.out_valid} | triplets={len(mined_valid)}")


if __name__ == "__main__":
    main()
