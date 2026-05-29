# Embedding Project - Vietnamese Semantic Product Search

Project fine-tune embedding cho semantic product search tiếng Việt. Hỗ trợ **2 base model**:

| Preset | Base model | Output folder |
|--------|------------|---------------|
| `minilm` (model 1) | `paraphrase-multilingual-MiniLM-L12-v2` | `models/minilm_finetuned_final/` |
| `bge-m3` (model 2) | `BAAI/bge-m3` | `models/bge_m3_finetuned_final/` |

## 1) Cấu trúc thư mục

```text
embedding_project/
├── data/
├── notebooks/
│   ├── train_embedding_model.ipynb          # Model 1 — MiniLM
│   └── train_embedding_model_bge_m3.ipynb   # Model 2 — BGE-M3 (GPU khuyến nghị)
├── scripts/
│   ├── model_presets.py
│   ├── train_embedding_model.py
│   └── evaluate_embedding_model.py
├── models/
├── outputs/
│   └── evaluation/
├── requirements.txt
└── README.md
```

## 2) Chuẩn bị dữ liệu

Đặt các file sau vào `embedding_project/data/`:

- `train_cleaned.jsonl`
- `valid_cleaned.jsonl`
- `test_cleaned.jsonl`
- `query_product_labels_cleaned.json`
- `merged_products_vi_cleaned.csv`

## 3) Cài thư viện

```bash
pip install -r embedding_project/requirements.txt
```

## 4) Train

### Model 1 — MiniLM (CPU-friendly)

```bash
python embedding_project/scripts/train_embedding_model.py --preset minilm
```

### Model 2 — BGE-M3 (cần GPU / RAM lớn)

```bash
python embedding_project/scripts/train_embedding_model.py --preset bge-m3 --fp16
```

Override hyperparams (ví dụ Colab T4):

```bash
python embedding_project/scripts/train_embedding_model.py \
  --preset bge-m3 \
  --epochs 1 \
  --batch-size 4 \
  --max-seq-length 512 \
  --fp16
```

Output:

- `embedding_project/models/minilm_finetuned_final/`
- `embedding_project/models/bge_m3_finetuned_final/`

## 5) Evaluate

```bash
# MiniLM
python embedding_project/scripts/evaluate_embedding_model.py --preset minilm

# BGE-M3
python embedding_project/scripts/evaluate_embedding_model.py --preset bge-m3
```

Metrics: `Recall@10`, `Precision@10`, `MRR@10`, `NDCG@10`

Output:

- `outputs/evaluation/metrics_minilm.json`
- `outputs/evaluation/metrics_bge_m3.json`

## 6) Colab

1. **MiniLM:** `notebooks/train_embedding_model.ipynb`
2. **BGE-M3:** `notebooks/train_embedding_model_bge_m3.ipynb` — đổi `GITHUB_REPO_URL`, bật GPU

Notebook BGE-M3 import `model_presets.py` từ repo sau khi clone.

### Push model lên GitHub sau train

File `model.safetensors` ~450MB → dùng **Git LFS**. Xem: [`docs/PUSH_MODEL_TO_GITHUB.md`](docs/PUSH_MODEL_TO_GITHUB.md)

```bash
git lfs install
bash embedding_project/scripts/push_models_after_train.sh minilm   # hoặc bge-m3 / all
git commit -m "Add fine-tuned embedding model"
git push origin main
```

## 7) A/B thực tế (query do người viết)

Xem: `embedding_project/docs/MANUAL_AB_GUIDE.md`

```bash
# MiniLM (mặc định)
python embedding_project/scripts/run_manual_ab_test.py

# BGE-M3 sau khi train xong
python embedding_project/scripts/run_manual_ab_test.py --preset bge-m3
```

## 8) Tạo product vectors (local)

```bash
python create_product_vectors.py --preset minilm
python create_product_vectors.py --preset bge-m3   # sau khi train BGE-M3
```

Output: `outputs/embeddings/product_vectors_{minilm|bge_m3}.npy` và `product_payloads_*.json`
