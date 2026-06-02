# Hướng dẫn A/B thực tế: Pretrained vs Fine-tuned

## Mục tiêu

Chọn model deploy dựa trên **query thật do người viết**, không chỉ metric JSON auto-label.

---

## Bước 1 — Chuẩn bị 50–100 query thật

Sửa file: `embedding_project/data/manual_eval_queries.csv`

Cột bắt buộc:
- `query_id`
- `query`
- `group` (A/B/C/D)
- `notes` (tùy chọn)

Gợi ý phân bổ:
- A (20–30): query tự nhiên
- B (10–20): query mơ hồ
- C (10–20): có màu/size
- D (10–20): typo / không dấu / Anh-Việt

---

## Bước 2 — Chạy A/B export kết quả search

### MiniLM (model 1)

```bash
python embedding_project/scripts/run_manual_ab_test.py --preset minilm --top-k 5
```

Output: `manual_ab_results.csv`, `manual_ab_blind.csv`

### BGE-M3 (model 2)

```bash
python embedding_project/scripts/run_manual_ab_test.py --preset bge-m3 --top-k 5 --batch-size 4
```

Output:
- `manual_ab_results_bge_m3.csv`
- `manual_ab_blind_bge_m3.csv`
- `manual_ab_model_mapping_bge_m3.txt`

*(BGE-M3 trên CPU rất chậm — ưu tiên GPU hoặc chạy qua đêm.)*

---

## Bước 3 — Chấm tay trên Google Sheets

Mở `manual_ab_blind.csv`, điền cột `label` cho **top 5** mỗi query:

- `0` = Irrelevant (sai)
- `1` = Partial (gần đúng)
- `2` = Relevant (đúng)

Chỉ chấm theo **ý định query**, không vì title dài hay brand nổi tiếng.

Lưu file đã chấm: `manual_ab_scored.csv`

---

## Bước 3b — Chấm nhanh top-1 (30 query, khuyên dùng)

```bash
# BGE-M3
python embedding_project/scripts/create_manual_ab_fast_sample.py \
  --input embedding_project/outputs/evaluation/manual_ab_results_bge_m3.csv \
  --output embedding_project/outputs/evaluation/manual_ab_fast_top1_bge_m3.csv

# MiniLM
python embedding_project/scripts/create_manual_ab_fast_sample.py
```

Chấm cột: `pretrained_label`, `finetuned_label` (0/1/2), `winner` (pretrained/finetuned/tie)

## Bước 4 — Tính điểm và chọn model

```bash
# Full top-5
python embedding_project/scripts/score_manual_ab.py \
  --input embedding_project/outputs/evaluation/manual_ab_scored_bge_m3.csv \
  --top-k 5

# Fast top-1
python embedding_project/scripts/score_manual_ab.py \
  --input embedding_project/outputs/evaluation/manual_ab_fast_top1_bge_m3.csv
```

Xem report: `embedding_project/outputs/evaluation/manual_ab_report.json`

---

## Tiêu chí chọn model (MVP)

Chọn **fine-tuned** nếu đồng thời:
- `Hit@1` cao hơn pretrained rõ rệt
- `Precision@5` cao hơn
- `Bad@1` thấp hơn
- `finetuned_win_rate` >= 60% (paired theo MRR@5)

Nếu fine-tuned chỉ thắng trên query auto-like (không thắng nhóm A/B) → **không deploy**.

---

## Quyết định triển khai

| Kết quả A/B | Hành động |
|---|---|
| Fine-tuned thắng rõ trên query thật | Dùng fine-tuned cho search |
| Pretrained thắng hoặc hòa | Giữ pretrained, thu thập query log thật rồi train lại |
| Cả hai đều kém (Hit@1 < 50%) | Sửa dữ liệu label + hybrid BM25 + embedding |

---

## Lưu ý quan trọng

- Không dùng `test_cleaned.jsonl` làm nguồn query chính cho A/B thủ công.
- Checkpoint train (`checkpoint-778`, `checkpoint-1556`) **không cần** cho bước này; chỉ cần `minilm_finetuned_final`.
- Chấm ít nhất 2 người trên 30 query mẫu để giảm bias cá nhân.
