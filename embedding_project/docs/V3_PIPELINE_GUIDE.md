# Hướng dẫn chạy V3 Pipeline

## Tổng quan thay đổi V2 → V3

| Issue | V2 | V3 |
|-------|----|----|
| Pos text format | Clean `title \| description` | **Corpus `searchable_text`** (đồng nhất với retrieval) |
| TripletLoss margin | 0.5 | **1.0** |
| Evaluator | TripletEvaluator | **InformationRetrievalEvaluator** (NDCG@10) |
| Loss weights | Both 1.0 | **Triplet=0.3, MNRL=0.7** |
| Epochs | 2 | **3** |
| LR | 1e-5 | 2e-5 |
| Easy negs | Không có | **24k added** cho triplets n_hard=0 |
| Avg negs/row | 3.27 | **4.39** |

## Step-by-step

### 1. Build train_v3 (đã chạy ✓)
```bash
python embedding_project/scripts/build_train_v3.py
```
Output:
- `data/train_v3.jsonl` (34,466 rows)
- `data/valid_v3.jsonl` (6,151 rows)
- `data/test_v3.jsonl` (5,538 rows)

### 2. Train V3 trên Colab

1. Upload notebook: `embedding_project/notebooks/finetune_e5_base_tripletloss_v3.ipynb` lên Colab
2. **GPU runtime**: Runtime → Change runtime type → T4 GPU
3. Update `GITHUB_REPO` nếu cần (default: `PhamMinhDan/llm_provider_benchmarking_ver2`)
4. Run all cells
5. Download zip model → extract vào `embedding_project/models/e5_base_v3_finetuned/final/`

### 3. Benchmark (sau khi có V3 model)

```bash
# Local
python embedding_project/scripts/benchmark_v3.py

# Hoặc trên Colab với GPU (nhanh hơn với corpus embedding)
```

Output: `embedding_project/outputs/evaluation/comparison_v3.json`

### 4. So sánh kết quả

Script in bảng so sánh các models trên:
- Test set overall
- Test set stratified by query_type (specific vs vague)
- Benchmark 200 (queries vague thực sự)

## Expected improvements

- **Specific queries**: V3 ≈ V2 ≈ E5 pretrained (BM25 mạnh ở đây)
- **Vague queries (test)**: V3 > V2 > E5 pretrained (semantic search cải thiện)
- **Benchmark 200 (real vague)**: V3 > V2 > E5 pretrained >> BM25 (semantic mạnh nhất)

Nếu V3 không vượt V2 → check:
1. Có IR evaluator đang tracking đúng metric không
2. Train có converge không (check loss curve)
3. Hard negs có quality tốt không (sample 50 random xem có positive ẩn không)