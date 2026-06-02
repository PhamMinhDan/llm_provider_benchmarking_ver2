# Vector DB — Qdrant + BGE-M3 fine-tuned

Index từ **`vector_db/products_with_documents.jsonl`** → embed trường **`search_document`** (đã có prefix `passage:`) → Qdrant.

Không tạo thêm `searchable_text` — đó là tên trường ở CSV project cũ (`merged_products_vi_cleaned.csv`).

## Cài đặt

```bash
pip install -r vector_db/requirements.txt
pip install -r embedding_project/requirements.txt
```

## Cấu hình

Copy `vector_db/.env.example` → `.env` ở **repo root**:

```env
QDRANT_URL=http://your-qdrant-host:6333
QDRANT_API_KEY=...
QDRANT_COLLECTION=products_vi_bge_m3
QDRANT_HTTPS=true
```

Model mặc định: `embedding_project/models/bge_m3_finetuned_final/`

## Index (encode + upsert)

```bash
cd c:\llm_provider_benchmarking

# Encode + đẩy Qdrant (mặc định products_with_documents.jsonl)
python vector_db/03_index_to_qdrant.py --recreate

# Chỉ định file JSONL khác
python vector_db/03_index_to_qdrant.py --products-jsonl vector_db/products_with_documents.jsonl --recreate

# Hoặc dùng vector đã tạo sẵn (create_product_vectors.py --preset bge-m3)
python vector_db/03_index_to_qdrant.py --from-files --recreate
```

## Thử search

```bash
python vector_db/04_search_test.py --query "tai nghe bluetooth chống ồn" --top-k 5
```

## Cấu trúc

| File | Vai trò |
|------|---------|
| `config.py` | URL Qdrant, path model, CSV |
| `embedding_service.py` | SentenceTransformer BGE-M3 fine-tuned |
| `qdrant_service.py` | Collection, upsert, search |
| `03_index_to_qdrant.py` | Pipeline index |
| `04_search_test.py` | Test query |

`vector_id` = uuid5(`source_product_id`) — khớp `create_product_vectors.py`.
