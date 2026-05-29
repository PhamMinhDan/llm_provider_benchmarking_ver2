"""
Tạo bản chấm tay nhanh từ manual_ab_results.csv.

Giảm từ ~780 dòng xuống ~30–40 dòng (chỉ top-1, so sánh cạnh nhau).

Usage:
  python embedding_project/scripts/create_manual_ab_fast_sample.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create fast manual A/B sample (top-1 pairwise).")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("embedding_project/outputs/evaluation/manual_ab_results.csv"),
    )
    p.add_argument(
        "--queries",
        type=Path,
        default=Path("embedding_project/data/manual_eval_queries.csv"),
    )
    p.add_argument("--sample-size", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_project/outputs/evaluation/manual_ab_fast_top1.csv"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    qmeta = pd.read_csv(args.queries)

    # Stratified sample by group if possible
    merged = df.merge(qmeta[["query_id", "group"]], on="query_id", how="left", suffixes=("", "_meta"))
    group_col = "group_meta" if "group_meta" in merged.columns else "group"
    groups = merged[group_col].fillna("").unique().tolist()
    per_group = max(1, args.sample_size // max(1, len(groups)))

    picked_ids: list[str] = []
    for g in groups:
        ids = (
            merged[merged[group_col].fillna("") == g]["query_id"]
            .drop_duplicates()
            .tolist()
        )
        picked_ids.extend(ids[:per_group])

    # Fill remaining randomly
    all_ids = merged["query_id"].drop_duplicates().tolist()
    remaining = [x for x in all_ids if x not in picked_ids]
    need = args.sample_size - len(picked_ids)
    if need > 0 and remaining:
        extra = (
            pd.Series(remaining)
            .sample(n=min(need, len(remaining)), random_state=args.seed)
            .tolist()
        )
        picked_ids.extend(extra)

    picked_ids = picked_ids[: args.sample_size]

    top1 = df[df["rank"] == 1].copy()
    pre = top1[top1["model"] == "pretrained"].set_index("query_id")
    ft = top1[top1["model"] == "finetuned"].set_index("query_id")

    rows = []
    for qid in picked_ids:
        if qid not in pre.index or qid not in ft.index:
            continue
        p_row, f_row = pre.loc[qid], ft.loc[qid]
        group_val = ""
        if qid in qmeta.set_index("query_id").index:
            group_val = str(qmeta.set_index("query_id").loc[qid].get("group", ""))
        rows.append(
            {
                "query_id": qid,
                "query": p_row["query"],
                "group": group_val,
                "pretrained_product_id": p_row["product_id"],
                "pretrained_title": p_row["title"],
                "finetuned_product_id": f_row["product_id"],
                "finetuned_title": f_row["title"],
                # Chấm nhanh:
                # pretrained_label: 0/1/2
                # finetuned_label: 0/1/2
                # winner: pretrained | finetuned | tie
                "pretrained_label": "",
                "finetuned_label": "",
                "winner": "",
                "notes": "",
            }
        )

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved fast sample: {args.output} ({len(out)} queries)")
    print("Chấm 3 cột: pretrained_label, finetuned_label (0/1/2), winner (pretrained/finetuned/tie)")


if __name__ == "__main__":
    main()
