"""Cấu hình Qdrant + embedding — đọc từ .env hoặc biến môi trường."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = Path(__file__).resolve().parent
_ENV_CANDIDATES = [REPO_ROOT / ".env", VECTOR_DB_DIR / ".env"]

DEFAULT_QDRANT_PORT = 6333


def normalize_qdrant_url(url: str, use_https: bool) -> str:
    """Chuẩn hóa URL Qdrant.

    - https://host (không port) → nginx reverse proxy, port 443 mặc định
    - http://localhost → thêm :6333
    """
    from urllib.parse import urlparse, urlunparse

    raw = url.strip().rstrip("/")
    if use_https and raw.startswith("http://"):
        raw = "https://" + raw[len("http://") :]
    if not use_https and raw.startswith("https://"):
        raw = "http://" + raw[len("https://") :]

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    scheme = parsed.scheme or ("https" if use_https else "http")
    host = parsed.hostname or parsed.path.split("/")[0]

    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    elif scheme == "https":
        netloc = f"{host}:443"  # qdrant-client mặc định 6333 nếu thiếu port → lỗi SSL
    elif host in ("localhost", "127.0.0.1"):
        netloc = f"{host}:{DEFAULT_QDRANT_PORT}"
    else:
        netloc = host

    return urlunparse((scheme, netloc, "", "", "", ""))


def _load_dotenv() -> None:
    """Load .env: repo root trước, vector_db/.env sau (ghi đè Qdrant config)."""
    for env_path in _ENV_CANDIDATES:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION: str = "products_vi_bge_m3"
    QDRANT_HTTPS: bool = False

    EMBEDDING_MODEL_PATH: str = ""
    EMBEDDING_BATCH_SIZE: int = 4
    EMBEDDING_TRUST_REMOTE_CODE: bool = True

    PRODUCTS_CSV: str = ""
    PRODUCTS_JSONL: str = ""
    DEFAULT_TOP_K: int = 10

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        default_model = str(
            REPO_ROOT / "embedding_project" / "models" / "bge_m3_finetuned_final"
        )
        default_csv = str(
            REPO_ROOT / "embedding_project" / "data" / "merged_products_vi_cleaned.csv"
        )
        default_jsonl = str(VECTOR_DB_DIR / "products_with_documents.jsonl")
        raw_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        use_https = os.getenv("QDRANT_HTTPS", "false").lower() in ("1", "true", "yes")
        qdrant_url = normalize_qdrant_url(raw_url, use_https)

        return cls(
            QDRANT_URL=qdrant_url,
            QDRANT_API_KEY=os.getenv("QDRANT_API_KEY", ""),
            QDRANT_COLLECTION=os.getenv("QDRANT_COLLECTION", "products_vi_bge_m3"),
            QDRANT_HTTPS=use_https,
            EMBEDDING_MODEL_PATH=os.getenv("EMBEDDING_MODEL_PATH", default_model),
            EMBEDDING_BATCH_SIZE=int(os.getenv("EMBEDDING_BATCH_SIZE", "4")),
            EMBEDDING_TRUST_REMOTE_CODE=os.getenv(
                "EMBEDDING_TRUST_REMOTE_CODE", "true"
            ).lower()
            in ("1", "true", "yes"),
            PRODUCTS_CSV=os.getenv("PRODUCTS_CSV", default_csv),
            PRODUCTS_JSONL=os.getenv("PRODUCTS_JSONL", default_jsonl),
            DEFAULT_TOP_K=int(os.getenv("DEFAULT_TOP_K", "10")),
        )


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


settings = get_settings()
