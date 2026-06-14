# BÁO CÁO THỬ NGHIỆM: E5 Bi-encoder + Cross-Encoder Reranker — Corpus 5000 SP

**Dự án:** `embedding_project` — Tìm kiếm ngữ nghĩa sản phẩm tiếng Việt  
**Ngày báo cáo:** 13/06/2026  
**Dữ liệu:** `embedding_project/data/ecommerce.csv` (5000 sản phẩm)  
**Model:** `e5_base_finetuned_5000` + `reranker` (XLM-RoBERTa cross-encoder)

---

## 1. Tóm tắt

Báo cáo tổng hợp: (1) khám phá dữ liệu `ecommerce.csv`, (2) fine-tune E5-base trên tập mới, (3) đánh giá pipeline 2 giai đoạn Bi-encoder + Reranker.

| Chỉ số | Chỉ Bi-encoder @10 | + Reranker @10 (n=20) |
|--------|-------------------|------------------------|
| Precision@10 | 0,1114 | 0,1115 |
| Recall@10 | 0,9979 | 0,9986 |
| F1@10 | 0,2005 | 0,2006 |
| MRR@10 | 0,9902 | — |
| NDCG@10 | 0,9926 | — |

**Precision@10 ~11% không phải do model kém** — đây là giới hạn toán học của cách đánh giá (1 nhãn đúng/query, top-10 cố định → trần P@10 ≈ 10%). Recall@10 ≈ 99,8% cho thấy model gần như luôn đưa đúng sản phẩm vào top-10.

---

## 2. Khám phá dữ liệu mới (`ecommerce.csv`)

### 2.1 Nguồn và quy mô

5000 sản phẩm từ 5 sàn, mỗi nguồn **1000 SP** — cân bằng tốt:

![Phân phối số lượng sản phẩm theo nguồn](figures/c__llm_provider_benchmarking_source_top.png)

| Cột | Mô tả |
|-----|--------|
| `product_id` | ID duy nhất |
| `source` | Amazon, Lazada, Shein, Shopee, Walmart |
| `title` | Tiêu đề — dùng làm **query** khi đánh giá |
| `searchable_text` | Văn bản corpus (title + mô tả + danh mục + thương hiệu) |
| `rating`, `reviews_count`, `price` | Metadata |

### 2.2 Giá trị thiếu

![Tỷ lệ giá trị thiếu theo cột](figures/c__llm_provider_benchmarking_missing_values.png)

| Cột | Tỷ lệ thiếu (ước lượng) |
|-----|-------------------------|
| `tags` | ~84% |
| `size` | ~60% |
| `color_vi` | ~51% |
| `brand` | ~15% |
| `description`, `price`, … | 0% |

Model chủ yếu học từ `title` và `searchable_text`; các trường cấu trúc thưa.

### 2.3 Số lượng review

Phân phối lệch phải, nhiều SP có **0 review**:

![Phân phối số review log1p](figures/c__llm_provider_benchmarking_reviews_count_hist.png)

![Boxplot số review](figures/c__llm_provider_benchmarking_reviews_count_box.png)

### 2.4 Rating

Hai đỉnh: rating = 0 và 4–5 sao:

![Phân phối rating](figures/c__llm_provider_benchmarking_rating_hist.png)

![Boxplot rating](figures/c__llm_provider_benchmarking_rating_box.png)

### 2.5 Giá sản phẩm

Khoảng giá rất rộng, cần log để nhìn phân phối:

![Phân phối giá log1p](figures/c__llm_provider_benchmarking_price_num_hist.png)

![Boxplot giá](figures/c__llm_provider_benchmarking_price_num_box.png)

Có outlier cực đoan (có thể lỗi đơn vị/nhập liệu).

### 2.6 Rating vs số review

SP nhiều review thường có rating cao hơn (độ phổ biến):

![Mối quan hệ rating và số review](figures/c__llm_provider_benchmarking_rating_vs_reviews.png)

---

## 3. Công việc gần đây: Fine-tune trên tập mới

### 3.1 Chia dữ liệu

| File | Số mẫu | Mục đích |
|------|--------|----------|
| `train_5000.jsonl` | 3500 | Train |
| `valid_5000.jsonl` | 750 | Validation |
| `test_5000.jsonl` | 750 | Test (tùy chọn) |

- **Query:** `title` hoặc câu query tự nhiên trong jsonl  
- **Passage:** `searchable_text`  
- **Loss:** `MultipleNegativesRankingLoss`

### 3.2 Fine-tune E5-base → `e5_base_finetuned_5000`

| Tham số | Giá trị |
|---------|---------|
| Base | `intfloat/multilingual-e5-base` |
| Epoch | 1 |
| Batch | 8 |
| LR | 1e-5 |
| FP16 | Có |
| Thời gian train | ~4,8 phút (Colab GPU) |

Notebook: `train_e5_base_5000_1epoch_colab.ipynb`

### 3.3 Pipeline đánh giá E5 + Reranker

```
Query (title)
  → E5 encode → cosine → top-n ứng viên
  → Cross-encoder reranker → xếp hạng lại → top-k
```

Script: `evaluate_reranker_pipeline.py`  
Đánh giá trên **toàn bộ** `ecommerce.csv`: corpus 5000 SP, query = `title`, nhãn = `product_id` cùng dòng.

---

## 4. Kết quả thực nghiệm (Colab T4)

### Bi-encoder only @10
```
P@10=0,1114 | R@10=0,9979 | F1@10=0,2005 | MRR@10=0,9902 | NDCG@10=0,9926
```

### + Reranker (n=20) @10
```
P@10=0,1115 | R@10=0,9986 | F1@10=0,2006
```

Reranker với n=20 chưa cải thiện rõ P/R/F1; bi-encoder đã đạt MRR/NDCG rất cao.

---

## 5. Giải thích: Vì sao Precision@10 chỉ ~10%?

### 5.1 Giới hạn toán học

Cách đánh giá hiện tại:
- Mỗi query (`title`) có **đúng 1** `product_id` là ground truth
- Precision@10 = (số hit trong top-10) / **10**

Nếu mỗi query chỉ có **1** SP đúng:

$$\text{Precision@10} \leq \frac{1}{10} = 10\%$$

Kết quả **11,14%** đã gần trần lý thuyết → **model đang hoạt động tốt**, không phải “chỉ đúng 10%”.

### 5.2 Số liệu từ `ecommerce.csv`

| Thống kê | Giá trị |
|----------|---------|
| Tổng SP | 5000 |
| Title unique | 4350 |
| Dòng trùng title | 650 |
| Query đánh giá | 4350 |
| Query 1 nhãn | 4123 (94,8%) |
| Query nhiều nhãn (cùng title, khác SKU) | 227 |

### 5.3 Đọc kết quả đúng cách

| Chỉ số | Ý nghĩa |
|--------|---------|
| **Recall@10 ≈ 99,8%** | Gần như mọi query đều tìm được SP đúng trong top-10 |
| **Precision@10 ≈ 11%** | Top-10 có ~1,1 SP “đúng nhãn” / 10 vị trí — do định nghĩa metric |
| **MRR@10 ≈ 0,99** | SP đúng thường nằm rất cao (vị trí 1–2) |

**Khuyến nghị báo cáo GVHD:** nhấn mạnh **Recall, MRR, NDCG, F1** và so sánh trước/sau rerank; không dùng P@10 đơn lẻ để kết luận model yếu.

Muốn P@10 phân biệt hơn → mở rộng tập nhãn (cùng danh mục = relevant) hoặc dùng `test_5000.jsonl` với query tự nhiên.

---

## 6. Hạn chế & hướng phát triển

- Title trùng (650 dòng) → 227 query đa nhãn  
- Metadata thưa (`tags`, `size`, `color`)  
- Giá có outlier → cần làm sạch  
- Reranker chậm (không flash_attn) → grid n lớn mất nhiều giờ  
- P@10 bão hòa với setup 1 nhãn/query → ưu tiên MRR/NDCG

---

## 7. Lệnh chạy lại

```bash
python embedding_project/scripts/evaluate_reranker_pipeline.py \
  --embedding-model embedding_project/models/e5_base_finetuned_5000 \
  --reranker-model embedding_project/models/reranker \
  --eval-csv embedding_project/data/ecommerce.csv \
  --query-col title \
  --n-values 20 50 100 \
  --k-values 5 10 20
```

Kết quả: `embedding_project/outputs/evaluation/reranker_pipeline_eval.json`

---

## 8. Phụ lục — Danh mục hình

Tất cả hình nằm trong `figures/` cùng thư mục với file báo cáo này.

---

*Báo cáo phục vụ trình bày GVHD — pipeline embedding + reranker trên corpus 5000 SP.*
