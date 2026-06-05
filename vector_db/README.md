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

# E5-base fine-tuned 2 epoch — collection MỚI (không ghi đè BGE)
python vector_db/03_index_to_qdrant.py --preset-e5-2ep --recreate --encode-batch-size 8

# Hoặc chỉ định rõ tên collection + model
python vector_db/03_index_to_qdrant.py --recreate \
  --collection products_vi_e5_2ep \
  --model-path embedding_project/models/e5_base_finetuned_2ep_final \
  --e5-prefix --encode-batch-size 8

# BGE-M3 (collection cũ products_vi_bge_m3)
python vector_db/03_index_to_qdrant.py --recreate
```

`.env` (tùy chọn):

```env
QDRANT_COLLECTION=products_vi_e5_2ep
EMBEDDING_MODEL_PATH=embedding_project/models/e5_base_finetuned_2ep_final
EMBEDDING_USE_E5_PREFIX=true
```

## Thử search

```bash
python vector_db/04_search_test.py --preset-e5-2ep --query "giày chạy bộ nam" --top-k 5
```

## Cấu trúc

| File | Vai trò |
|------|---------|
| `config.py` | URL Qdrant, path model, CSV |
| `embedding_service.py` | SentenceTransformer (BGE-M3 / E5, prefix E5 tùy chọn) |
| `qdrant_service.py` | Collection, upsert, search |
| `03_index_to_qdrant.py` | Pipeline index |
| `04_search_test.py` | Test query |

`vector_id` = uuid5(`source_product_id`) — khớp `create_product_vectors.py`.
