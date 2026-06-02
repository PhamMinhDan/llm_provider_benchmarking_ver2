"""Kiểm tra kết nối Qdrant trước khi index."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from vector_db.config import settings
from vector_db.qdrant_service import qdrant_service


def main() -> None:
    print("QDRANT_URL:", settings.QDRANT_URL)
    print("QDRANT_COLLECTION:", settings.QDRANT_COLLECTION)
    client = qdrant_service._ensure_client()
    cols = client.get_collections().collections
    print("OK — collections:", [c.name for c in cols] or "(empty)")


if __name__ == "__main__":
    main()
