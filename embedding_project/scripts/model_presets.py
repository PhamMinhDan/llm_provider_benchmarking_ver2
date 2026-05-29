"""Preset cấu hình fine-tune / evaluate cho từng base model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EmbeddingModelPreset:
    name: str
    base_model: str
    final_subdir: str
    run_name: str
    max_seq_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    fp16_default: bool
    trust_remote_code: bool
    metrics_filename: str

    @property
    def finetuned_rel_path(self) -> str:
        return f"embedding_project/models/{self.final_subdir}"


PRESETS: dict[str, EmbeddingModelPreset] = {
    "minilm": EmbeddingModelPreset(
        name="minilm",
        base_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        final_subdir="minilm_finetuned_final",
        run_name="minilm-vi-semantic-search",
        max_seq_length=256,
        epochs=2,
        batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16_default=False,
        trust_remote_code=False,
        metrics_filename="metrics_minilm.json",
    ),
    "bge-m3": EmbeddingModelPreset(
        name="bge-m3",
        base_model="BAAI/bge-m3",
        final_subdir="bge_m3_finetuned_final",
        run_name="bge-m3-vi-semantic-search",
        max_seq_length=512,
        epochs=1,
        batch_size=2,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        fp16_default=True,
        trust_remote_code=True,
        metrics_filename="metrics_bge_m3.json",
    ),
}


def get_preset(name: str) -> EmbeddingModelPreset:
    key = name.strip().lower()
    if key not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Choose from: {', '.join(PRESETS)}")
    return PRESETS[key]


def load_sentence_transformer(preset: EmbeddingModelPreset) -> Any:
    from sentence_transformers import SentenceTransformer

    kwargs: dict[str, Any] = {}
    if preset.trust_remote_code:
        kwargs["trust_remote_code"] = True
    return SentenceTransformer(preset.base_model, **kwargs)
