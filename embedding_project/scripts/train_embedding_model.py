from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

from model_presets import get_preset, load_sentence_transformer


LOGGER = logging.getLogger("train_embedding_model")


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_hf_dataset(records: list[dict]) -> Dataset:
    pairs = [{"anchor": r["query"], "positive": r["positive"]} for r in records]
    return Dataset.from_list(pairs)


def train(
    train_path: Path,
    valid_path: Path,
    output_dir: Path,
    preset_name: str,
    base_model: str | None,
    final_subdir: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_ratio: float,
    max_seq_length: int,
    fp16: bool | None,
    run_name: str | None,
) -> Path:
    preset = get_preset(preset_name)
    resolved_base = base_model or preset.base_model
    resolved_final = final_subdir or preset.final_subdir
    resolved_run = run_name or preset.run_name
    use_fp16 = preset.fp16_default if fp16 is None else fp16

    LOGGER.info("Preset: %s", preset.name)
    LOGGER.info("Loading datasets...")
    train_records = load_jsonl(train_path)
    valid_records = load_jsonl(valid_path)
    if not train_records or not valid_records:
        raise ValueError("Train/valid dataset is empty.")

    train_ds = build_hf_dataset(train_records)
    valid_ds = build_hf_dataset(valid_records)

    LOGGER.info("Loading model: %s", resolved_base)
    if resolved_base == preset.base_model:
        model = load_sentence_transformer(preset)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(resolved_base, trust_remote_code=preset.trust_remote_code)

    model.max_seq_length = max_seq_length
    loss = MultipleNegativesRankingLoss(model)

    use_fp16_runtime = use_fp16 and torch.cuda.is_available()
    if use_fp16 and not torch.cuda.is_available():
        LOGGER.warning("fp16=True nhưng không có CUDA. Tự động chuyển fp16=False để chạy CPU.")

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir / preset.name),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        fp16=use_fp16_runtime,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        run_name=resolved_run,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        loss=loss,
    )

    LOGGER.info("Start training...")
    trainer.train()

    final_dir = output_dir / resolved_final
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_dir))
    LOGGER.info("Saved fine-tuned model to: %s", final_dir)
    return final_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune embedding model for Vietnamese product search.",
    )
    parser.add_argument(
        "--preset",
        choices=["minilm", "bge-m3"],
        default="minilm",
        help="minilm = model 1 (MiniLM). bge-m3 = model 2 (BAAI/bge-m3).",
    )
    parser.add_argument("--train", type=Path, default=Path("embedding_project/data/train_cleaned.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("embedding_project/data/valid_cleaned.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("embedding_project/models"))
    parser.add_argument("--base-model", default=None, help="Override base model from preset.")
    parser.add_argument("--final-subdir", default=None, help="Override output folder name.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    preset = get_preset(args.preset)

    train(
        train_path=args.train,
        valid_path=args.valid,
        output_dir=args.output_dir,
        preset_name=args.preset,
        base_model=args.base_model,
        final_subdir=args.final_subdir,
        epochs=args.epochs if args.epochs is not None else preset.epochs,
        batch_size=args.batch_size if args.batch_size is not None else preset.batch_size,
        lr=args.learning_rate if args.learning_rate is not None else preset.learning_rate,
        warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else preset.warmup_ratio,
        max_seq_length=args.max_seq_length if args.max_seq_length is not None else preset.max_seq_length,
        fp16=args.fp16,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
