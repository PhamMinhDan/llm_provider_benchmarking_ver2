"""
Làm sạch train/valid: loại cặp query–positive sai intent (rule-based, không cần chấm tay).

Usage:
  python embedding_project/scripts/clean_train_intent_v2.py
  python embedding_project/scripts/clean_train_intent_v2.py --inplace
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "embedding_project/data"
DEFAULT_INPUTS = [
    DATA / "train_cleaned.jsonl",
    DATA / "valid_cleaned.jsonl",
]
DEFAULT_OUTPUTS = [
    DATA / "train_cleaned_v2.jsonl",
    DATA / "valid_cleaned_v2.jsonl",
]
REPORT = DATA / "train_cleaning_report_v2.txt"
REMOVED_LOG = DATA / "train_removed_pairs_v2.jsonl"


def norm(s: str) -> str:
    return (s or "").lower().strip()


def text_blob(row: dict) -> str:
    return norm(row.get("query", "")) + " " + norm(row.get("category", "")) + " " + norm(
        row.get("positive", "")
    )[:500]


def query_has_any(query: str, patterns: list[str]) -> bool:
    q = norm(query)
    return any(re.search(p, q) for p in patterns)


def category_has_any(category: str, needles: list[str]) -> bool:
    c = norm(category)
    return any(n in c for n in needles)


def title_positive_has_any(row: dict, needles: list[str]) -> bool:
    t = norm(row.get("positive", ""))[:400]
    return any(n in t for n in needles)


def check_rules(row: dict) -> str | None:
    q = row.get("query", "")
    cat = row.get("category", "")
    pid = str(row.get("product_id", ""))
    blob = text_blob(row)

    # --- Thú cưng: query người, SP pet ---
    pet_cat = category_has_any(
        cat,
        ["thú cưng", "cho chó", "cho mèo", " > chó >", " > mèo >"],
    )
    pet_query = query_has_any(q, [r"\bchó\b", r"\bmèo\b", r"thú cưng", r"chăm sóc tai"])
    if pet_cat and not pet_query:
        return "pet_product_human_query"

    # --- Giày chạy bộ / chạy bộ: loại SP không phải giày chạy ---
    if query_has_any(q, [r"giày chạy", r"chạy bộ", r"chay bo", r"chạy đường"]):
        running_reject_cats = [
            "an toàn lao động",
            "an toàn lao độn",
            "công cụ làm việc",
            "công cụ lao động",
            "công việc",
            "công nghiệp",
            "giày sneaker thời trang",
            "giày thể thao thời trang",
            "bảo hộ",
            "phụ kiện giày",
            "lót giày",
            "trang trí giày",
            "giày cao gót",
            "sandal",
            "trượt inline",
            "trượt patin",
            "giày leo núi",
            "giày lười & slip",
        ]
        if category_has_any(cat, running_reject_cats):
            return "running_shoe_wrong_category"
        if category_has_any(cat, ["giày sneaker", "sneaker thời trang"]) and "chạy bộ" not in norm(
            cat
        ):
            return "running_shoe_sneaker_fashion"
        # Query chạy bộ nhưng positive không phải giày (lót, phụ kiện, trang trí)
        if title_positive_has_any(
            row,
            [
                "lót giày",
                "đệm giày",
                "insole",
                "phụ kiện giày",
                "trang trí giày",
                "kẹp giày",
                "giày cao gót",
                "sandal",
                "giày trượt",
                "rollerblade",
                "giày sandal",
            ],
        ):
            return "running_shoe_not_footwear"
        # Phải có dấu hiệu giày chạy trong category/title; nếu chỉ "giày thể thao" chung → loại
        has_running_signal = category_has_any(
            cat, ["chạy bộ", "chạy đường", "running"]
        ) or title_positive_has_any(row, ["giày chạy", "running shoe", "chạy bộ"])
        if not has_running_signal and category_has_any(
            cat, ["giày thể thao", "> giày >"]
        ):
            if not title_positive_has_any(row, ["giày chạy", "run ", "runner"]):
                return "running_shoe_weak_match"

    # --- Giày nam / giày thể thao (không nhắc công trường): loại giày bảo hộ ---
    if query_has_any(
        q, [r"^giày nam", r"^giày nữ", r"giày thể thao", r"giay the thao", r"^giày size", r"^giày "]
    ):
        if not query_has_any(q, [r"công trường", r"công nghiệp", r"bảo hộ", r"làm việc", r"an toàn"]):
            if category_has_any(
                cat,
                [
                    "an toàn lao động",
                    "công cụ làm việc",
                    "công việc",
                    "bảo hộ mũi thép",
                    "công nghiệp & xây dựng",
                ],
            ):
                return "generic_shoe_work_boot"
            if category_has_any(cat, ["tất nam", "tất nữ", "> tất "]):
                return "shoe_query_sock_product"

    # --- Tai nghe consumer: loại SP chó, tai nghe xe tải ---
    if query_has_any(
        q,
        [
            r"tai nghe",
            r"tai nghe bluetooth",
            r"chống ồn",
            r"không dây",
            r"earbud",
            r"^bluetooth",
        ],
    ):
        if title_positive_has_any(row, ["cho chó", "thú cưng", "bảo vệ tai cho chó"]):
            return "headphone_query_pet_product"
        if pid == "B0BK8QY1JB" or title_positive_has_any(row, ["tài xế xe tải", "xe tải"]):
            if not query_has_any(q, [r"xe tải", r"tài xế", r"lái xe"]):
                return "headphone_query_truck_driver"

    # --- Máy sấy tóc: loại móng tay, adapter điện áp ---
    if query_has_any(q, [r"máy sấy tóc", r"may say toc"]):
        if category_has_any(cat, ["móng", "nghệ thuật móng", "nail"]):
            return "hair_dryer_nail_device"
        if title_positive_has_any(row, ["chuyển đổi điện áp", "travel plug", "110v", "220v"]):
            return "hair_dryer_voltage_adapter"

    # --- Kem chống nắng: loại khăn cổ, kem dưỡng đêm ---
    if query_has_any(q, [r"kem chống nắng", r"chống nắng"]):
        if not title_positive_has_any(row, ["chống nắng", "sunscreen", "spf", "uv"]):
            if title_positive_has_any(row, ["khăn cổ", "đạp xe", "ban đêm", "night cream", "dưỡng ẩm ban đêm"]):
                return "sunscreen_query_wrong_product"

    # --- Dầu gội / dầu dưỡng tóc: loại thiết bị laser iGrow ---
    if query_has_any(q, [r"dầu gội", r"dầu dưỡng tóc", r"giảm rụng", r"mọc tóc", r"tinh dầu.*tóc"]):
        if title_positive_has_any(row, ["laser", "mũ đầu", "igrow", "tăng trưởng tóc:"]):
            if not query_has_any(q, [r"laser", r"mũ", r"thiết bị"]):
                return "hair_oil_query_laser_device"

    # --- Son môi: loại sơn móng ---
    if query_has_any(q, [r"son môi", r"son moi"]):
        if category_has_any(cat, ["sơn móng", "móng tay"]) or title_positive_has_any(
            row, ["sơn móng", "nail polish"]
        ):
            return "lipstick_query_nail_polish"

    # --- Micro thu âm: loại tai nghe, dù ---
    if query_has_any(q, [r"micro thu", r"microphone", r"^micro "]):
        if category_has_any(cat, ["tai nghe", "dù", "ô"]) or title_positive_has_any(
            row, ["tài xế xe tải", "dù mini", "umbrella"]
        ):
            return "microphone_query_wrong_product"

    # --- Chuột không dây: loại bàn chuột ---
    if query_has_any(q, [r"chuột không dây", r"chuột bluetooth", r"^chuột "]):
        if category_has_any(cat, ["bàn chuột", "mouse pad"]):
            return "mouse_query_mousepad"

    # --- Pin dự phòng: loại đèn pin, pin thay thế iphone ---
    if query_has_any(q, [r"pin dự phòng", r"sạc dự phòng", r"power bank"]):
        if title_positive_has_any(row, ["đèn pin", "flashlight", "pin thay thế", "replacement battery"]):
            return "powerbank_query_wrong_product"

    # --- Máy rửa mặt: phải là thiết bị, không chỉ sữa rửa mặt (giữ sữa rửa mặt OK) ---
    if query_has_any(q, [r"máy rửa mặt"]):
        if not title_positive_has_any(
            row, ["máy rửa mặt", "facial cleansing brush", "face brush", "tẩy lông", "cleansing device"]
        ):
            if title_positive_has_any(row, ["sữa rửa mặt", "face wash"]):
                return "face_device_query_face_wash_only"

    # --- Áo / quần người: loại áo chó ---
    if query_has_any(q, [r"^áo ", r"^quần ", r"áo khoác", r"áo hoodie", r"váy "]):
        if not query_has_any(q, [r"chó", r"mèo", r"thú cưng"]):
            if title_positive_has_any(row, ["cho chó", "áo khoác chó", "chó kích thước"]):
                return "apparel_query_pet_clothing"

    # --- SP cụ thể gây overfit (nhiều query consumer gán 1 SP lạ) ---
    OVERFIT_BLOCK = {
        "B00U4ADYCM": r"giày chạy|chạy bộ",  # Wolverine work boot
        "B07BL2HT5N": r"giày chạy|chạy bộ|giày nam|giày an toàn",  # NB industrial
        "B09TWW1N6J": r"giày chạy|chạy bộ",  # SeaVees sneaker
        "B07RGW5GRY": r"giày chạy|chạy bộ",  # NB 997 lifestyle for running query
    }
    if pid in OVERFIT_BLOCK and re.search(OVERFIT_BLOCK[pid], norm(q)):
        return f"blocked_product_{pid}"

    return None


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_file(in_path: Path, out_path: Path) -> tuple[list[dict], list[dict], Counter]:
    rows = load_jsonl(in_path)
    kept: list[dict] = []
    removed: list[dict] = []
    reasons: Counter = Counter()

    for row in rows:
        reason = check_rules(row)
        if reason:
            removed.append({**row, "remove_reason": reason, "source_file": in_path.name})
            reasons[reason] += 1
        else:
            kept.append(row)

    save_jsonl(out_path, kept)
    return kept, removed, reasons


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inplace", action="store_true", help="Ghi đè train_cleaned.jsonl / valid_cleaned.jsonl")
    args = p.parse_args()

    if args.inplace:
        pairs = list(zip(DEFAULT_INPUTS, DEFAULT_INPUTS))
    else:
        pairs = list(zip(DEFAULT_INPUTS, DEFAULT_OUTPUTS))

    all_removed: list[dict] = []
    report_lines: list[str] = []

    for in_path, out_path in pairs:
        if not in_path.is_file():
            report_lines.append(f"SKIP (missing): {in_path}")
            continue
        before = len(load_jsonl(in_path))
        kept, removed, reasons = clean_file(in_path, out_path)
        all_removed.extend(removed)
        report_lines.append(f"\n=== {in_path.name} -> {out_path.name} ===")
        report_lines.append(f"Before: {before} | After: {len(kept)} | Removed: {len(removed)}")
        for reason, cnt in reasons.most_common():
            report_lines.append(f"  {reason}: {cnt}")

    save_jsonl(REMOVED_LOG, all_removed)
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n".join(report_lines))
    print(f"\nRemoved log: {REMOVED_LOG}")
    print(f"Report: {REPORT}")
    if not args.inplace:
        print("\nFile train mới: embedding_project/data/train_cleaned_v2.jsonl")
        print("Train notebook: đổi path sang train_cleaned_v2.jsonl")


if __name__ == "__main__":
    main()
