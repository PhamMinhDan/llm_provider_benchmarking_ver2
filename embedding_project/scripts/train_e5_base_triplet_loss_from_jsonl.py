"""
Train E5-base bằng TripletLoss trên triplets đã mine (category hard negative).

Mặc định fine-tune tiếp từ model 1 epoch (e5_base_finetuned_final), không từ pretrained.
"""

from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Any

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from model_presets import get_preset

DEFAULT_BASE_MODEL = Path("embedding_project/models/e5_base_finetuned_final")


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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train E5-base with TripletLoss from mined category hard negatives.",
    )
    p.add_argument(
        "--triplets",
        type=Path,
        default=Path("embedding_project/data/train_triplets_category_hardneg_v2.jsonl"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("embedding_project/models"))
    p.add_argument("--final-subdir", type=str, default="e5_base_finetuned_triplet_hardneg_final")
    p.add_argument(
        "--base-model",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Checkpoint khởi đầu (mặc định: e5_base_finetuned_final = 1 epoch).",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--triplet-margin", type=float, default=0.2)
    p.add_argument("--max-triplets", type=int, default=None, help="Giới hạn để test nhanh.")
    args = p.parse_args()

    preset = get_preset("e5-base")

    rows = load_jsonl(args.triplets)
    if args.max_triplets is not None:
        rows = rows[: args.max_triplets]
    if not rows:
        raise ValueError("No triplets found.")

    examples = [
        InputExample(texts=[r["anchor"], r["positive"], r["negative"]])
        for r in rows
        if r.get("anchor") and r.get("positive") and r.get("negative")
    ]
    if not examples:
        raise ValueError("Triplets invalid: missing anchor/positive/negative.")

    print(f"Base model: {args.base_model}")
    print(f"Triplets: {len(examples)} | epochs={args.epochs} | margin={args.triplet_margin}")

    model = SentenceTransformer(args.base_model, trust_remote_code=preset.trust_remote_code)
    model.max_seq_length = preset.max_seq_length

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(model=model, triplet_margin=args.triplet_margin)

    total_steps = ceil(len(train_dataloader)) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    out_dir = args.output_dir / args.final_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(out_dir),
        show_progress_bar=True,
        use_amp=False,
    )

    model.save(str(out_dir))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
