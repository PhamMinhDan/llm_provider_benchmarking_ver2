"""Encode searchable_text bằng BGE-M3 fine-tuned."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vector_db.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


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
        self.model_path = Path(settings.EMBEDDING_MODEL_PATH)
        if not self.model_path.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy model: {self.model_path}\n"
                "Đặt bge_m3_finetuned_final vào embedding_project/models/"
            )
        self._model: SentenceTransformer | None = None
        self._vector_size: int | None = None
        self._initialized = True

    @property
    def model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            LOGGER.info("Loading embedding model: %s", self.model_path)
            kwargs = {}
            if settings.EMBEDDING_TRUST_REMOTE_CODE:
                kwargs["trust_remote_code"] = True
            self._model = SentenceTransformer(str(self.model_path), **kwargs)
            probe = self._model.encode(["test"], convert_to_numpy=True)
            self._vector_size = int(probe.shape[1])
            LOGGER.info("Vector size: %d", self._vector_size)
        return self._model

    @property
    def vector_size(self) -> int:
        if self._vector_size is None:
            _ = self.model
        assert self._vector_size is not None
        return self._vector_size

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        vectors = self.model.encode(
            texts,
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
        return self.encode([query], show_progress=False)[0].tolist()


embedding_service = EmbeddingService()
