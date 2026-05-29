"""
Tính metric thủ công từ file A/B đã chấm label.

Usage:
  python embedding_project/scripts/score_manual_ab.py \
    --input embedding_project/outputs/evaluation/manual_ab_scored.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def precision_at_k(labels: list[int], k: int) -> float:
    top = labels[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x >= 2) / k


def mrr_at_k(labels: list[int], k: int) -> float:
    for i, x in enumerate(labels[:k], start=1):
        if x >= 2:
            return 1.0 / i
    return 0.0


def hit_at_1(labels: list[int]) -> float:
    return 1.0 if labels and labels[0] >= 2 else 0.0


def bad_at_1(labels: list[int]) -> float:
    return 1.0 if labels and labels[0] == 0 else 0.0


def score_model(df: pd.DataFrame, model: str, k: int = 5) -> dict[str, float]:
    sub = df[df["model"] == model].copy()
    if sub.empty:
        return {}

    p_list, mrr_list, hit1_list, bad1_list = [], [], [], []
    for _, g in sub.groupby("query_id"):
        g = g.sort_values("rank")
        labels = g["label"].astype(int).tolist()
        p_list.append(precision_at_k(labels, k))
        mrr_list.append(mrr_at_k(labels, k))
        hit1_list.append(hit_at_1(labels))
        bad1_list.append(bad_at_1(labels))

    n = len(p_list)
    return {
        f"Precision@{k}": sum(p_list) / n,
        f"MRR@{k}": sum(mrr_list) / n,
        "Hit@1": sum(hit1_list) / n,
        "Bad@1": sum(bad1_list) / n,
        "queries_scored": n,
    }


def paired_win_rate(df: pd.DataFrame, k: int = 5) -> dict[str, float]:
    """% query mà model này thắng model kia theo MRR@k."""
    pretrained = df[df["model"] == "pretrained"]
    finetuned = df[df["model"] == "finetuned"]
    wins_pre, wins_ft, ties = 0, 0, 0

    for qid in sorted(set(df["query_id"])):
        p = pretrained[pretrained["query_id"] == qid].sort_values("rank")
        f = finetuned[finetuned["query_id"] == qid].sort_values("rank")
        if p.empty or f.empty:
            continue
        mrr_p = mrr_at_k(p["label"].astype(int).tolist(), k)
        mrr_f = mrr_at_k(f["label"].astype(int).tolist(), k)
        if mrr_f > mrr_p:
            wins_ft += 1
        elif mrr_p > mrr_f:
            wins_pre += 1
        else:
            ties += 1

    total = wins_pre + wins_ft + ties
    if total == 0:
        return {}
    return {
        "finetuned_win_rate": wins_ft / total,
        "pretrained_win_rate": wins_pre / total,
        "tie_rate": ties / total,
        "paired_queries": total,
    }


def score_fast_top1(df: pd.DataFrame) -> dict:
    """Format manual_ab_fast_top1.csv: pretrained_label, finetuned_label, winner."""
    required = {"query_id", "pretrained_label", "finetuned_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột (fast top-1 format): {missing}")

    work = df.copy()
    work["pretrained_label"] = pd.to_numeric(work["pretrained_label"], errors="coerce")
    work["finetuned_label"] = pd.to_numeric(work["finetuned_label"], errors="coerce")
    work = work.dropna(subset=["pretrained_label", "finetuned_label"])
    work["pretrained_label"] = work["pretrained_label"].astype(int)
    work["finetuned_label"] = work["finetuned_label"].astype(int)

    for col in ("pretrained_label", "finetuned_label"):
        if not set(work[col].unique()).issubset({0, 1, 2}):
            raise ValueError(f"{col} chỉ được là 0, 1, hoặc 2.")

    n = len(work)
    pre = work["pretrained_label"].tolist()
    ft = work["finetuned_label"].tolist()

    def side_metrics(labels: list[int]) -> dict[str, float]:
        return {
            "Hit@1": sum(1 for x in labels if x >= 2) / n,
            "Bad@1": sum(1 for x in labels if x == 0) / n,
            "AvgLabel@1": sum(labels) / n,
            "PartialOrBetter@1": sum(1 for x in labels if x >= 1) / n,
        }

    # Winner: use column if filled, else infer from labels
    wins_pre, wins_ft, ties = 0, 0, 0
    if "winner" in work.columns and work["winner"].fillna("").astype(str).str.strip().ne("").any():
        for _, row in work.iterrows():
            w = str(row.get("winner", "")).strip().lower()
            if w == "pretrained":
                wins_pre += 1
            elif w == "finetuned":
                wins_ft += 1
            else:
                ties += 1
    else:
        for p, f in zip(pre, ft):
            if f > p:
                wins_ft += 1
            elif p > f:
                wins_pre += 1
            else:
                ties += 1

    return {
        "format": "fast_top1",
        "queries_scored": n,
        "pretrained": side_metrics(pre),
        "finetuned": side_metrics(ft),
        "paired_comparison": {
            "pretrained_wins": wins_pre,
            "finetuned_wins": wins_ft,
            "ties": ties,
            "finetuned_win_rate": wins_ft / n if n else 0.0,
            "pretrained_win_rate": wins_pre / n if n else 0.0,
            "tie_rate": ties / n if n else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score manual A/B labeled CSV.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("embedding_project/outputs/evaluation/manual_ab_scored.csv"),
    )
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_project/outputs/evaluation/manual_ab_report.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    # Fast top-1 pairwise format (manual_ab_fast_top1.csv)
    if {"pretrained_label", "finetuned_label"}.issubset(df.columns):
        report = score_fast_top1(df)
    else:
        required = {"query_id", "query", "model", "rank", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Thiếu cột: {missing}. "
                "Dùng manual_ab_scored.csv (format dài) hoặc manual_ab_fast_top1.csv (format nhanh)."
            )

        # Map blind labels back if user scored blind file
        if set(df["model"].unique()).issubset({"model_a", "model_b"}):
            df = df.copy()
            df["model"] = df["model"].replace({"model_a": "pretrained", "model_b": "finetuned"})

        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
        if not set(df["label"].unique()).issubset({0, 1, 2}):
            raise ValueError("label chỉ được là 0, 1, hoặc 2.")

        report = {
            "format": "full_topk",
            "pretrained": score_model(df, "pretrained", k=args.top_k),
            "finetuned": score_model(df, "finetuned", k=args.top_k),
            "paired_comparison": paired_win_rate(df, k=args.top_k),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
