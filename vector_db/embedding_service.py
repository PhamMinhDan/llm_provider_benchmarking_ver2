"""Encode searchable_text bằng SentenceTransformer (BGE-M3 / E5 fine-tuned)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vector_db.config import get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


def _prefix_passage(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("passage:"):
        return t
    return f"passage: {t}"


def _prefix_query(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("query:"):
        return t
    return f"query: {t}"


class EmbeddingService:
    _instance: "EmbeddingService | None" = None

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        cfg = get_settings()
        self.model_path = Path(cfg.EMBEDDING_MODEL_PATH)
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy model: {self.model_path}\n"
                "Đặt model fine-tuned vào embedding_project/models/"
            )
        self.use_e5_prefix = cfg.EMBEDDING_USE_E5_PREFIX
        self._model: SentenceTransformer | None = None
        self._vector_size: int | None = None
        self._initialized = True

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading embedding model: %s", self.model_path)
            if self.use_e5_prefix:
                LOGGER.info("E5 prefix: passage: (corpus) / query: (search)")
            kwargs = {}
            if get_settings().EMBEDDING_TRUST_REMOTE_CODE:
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(str(self.model_path), **kwargs)
            probe_text = "test"
            if self.use_e5_prefix:
                probe_text = _prefix_passage(probe_text)
            probe = self._model.encode([probe_text], convert_to_numpy=True)
            self._vector_size = int(probe.shape[1])
            LOGGER.info("Vector size: %d", self._vector_size)
        return self._model

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            _ = self.model
        assert self._vector_size is not None
        return self._vector_size

    def _prepare_corpus_texts(self, texts: list[str]) -> list[str]:
        if not self.use_e5_prefix:
            return texts
        return [_prefix_passage(t) for t in texts]

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        batch_size = batch_size or get_settings().EMBEDDING_BATCH_SIZE
        inputs = self._prepare_corpus_texts(texts)
        vectors = self.model.encode(
            inputs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms

    def encode_query(self, query: str) -> list[float]:
        text = _prefix_query(query) if self.use_e5_prefix else query
        return self.encode([text], show_progress=False)[0].tolist()
