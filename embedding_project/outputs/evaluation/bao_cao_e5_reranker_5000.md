# 实验报告：E5 双编码器 + Cross-Encoder 重排序 — 5000 商品语料

**项目：** `embedding_project` — 越南语电商语义检索  
**报告日期：** 2026-06-13  
**数据：** `embedding_project/data/ecommerce.csv`（5000 SKU）  
**模型：** `e5_base_finetuned_5000` + `reranker`（XLM-RoBERTa cross-encoder）

---

## 1. 摘要

本报告汇总了在新数据集 `ecommerce.csv` 上的数据探索、E5-base 微调流程，以及 **Bi-encoder + Reranker** 两阶段检索管道的评估结果。

核心结论：

| 指标 | Bi-encoder @10 | Reranker @10 (n=20) |
|------|----------------|---------------------|
| Precision@10 | 0.1114 | 0.1115 |
| Recall@10 | 0.9979 | 0.9986 |
| F1@10 | 0.2005 | 0.2006 |
| MRR@10 | 0.9902 | — |
| NDCG@10 | 0.9926 | — |

**Precision@10 约 11% 并非模型失效**，而是由评估设定（每 query 仅 1 个正样本 + top-10 固定长度）决定的数学上限约为 **10%**；同时 Recall@10 ≈ 99.8% 说明模型几乎总能把正确商品召回到 top-10。

---

## 2. 新数据集概览

### 2.1 数据来源与规模

`ecommerce.csv` 包含 **5000 条商品**，来自 5 个电商平台，各 **1000 条**，分布均衡：

![各数据源商品数量分布](figures/c__llm_provider_benchmarking_source_top.png)

| 字段 | 说明 |
|------|------|
| `product_id` | 商品唯一 ID |
| `source` | 平台（Amazon / Lazada / Shein / Shopee / Walmart） |
| `title` | 商品标题（评估时作 query） |
| `searchable_text` | 检索语料（标题 + 描述 + 类目 + 品牌等拼接） |
| `rating`, `reviews_count`, `price` | 元数据 |

### 2.2 缺失值分析

部分结构化字段缺失率较高，模型主要依赖 `title` 与 `searchable_text`：

![各列缺失率](figures/c__llm_provider_benchmarking_missing_values.png)

| 列 | 缺失率（约） |
|----|-------------|
| `tags` | ~84% |
| `size` | ~60% |
| `color_vi` | ~51% |
| `brand` | ~15% |
| `description`, `image_url`, `price`, `price_num` | 0% |

**影响：** 训练与检索几乎完全依赖文本字段；`tags`/`size`/`color` 稀疏，难以作为强特征。

### 2.3 评论数分布

大量商品评论数为 0，呈长尾分布：

![评论数分布 log1p](figures/c__llm_provider_benchmarking_reviews_count_hist.png)

![评论数箱线图](figures/c__llm_provider_benchmarking_reviews_count_box.png)

- 峰值在 **reviews_count = 0**（约 2250+ 商品）
- 少数爆款商品评论数达数万，存在极端离群点

### 2.4 评分分布

评分呈双峰：0 分与 4–5 分聚集：

![评分分布](figures/c__llm_provider_benchmarking_rating_hist.png)

![评分箱线图](figures/c__llm_provider_benchmarking_rating_box.png)

- 约 **1800** 条 `rating = 0`（可能为未评分默认值）
- 中位数约 **4.3**，75% 分位约 **4.8**

### 2.5 价格分布

价格跨度极大，需 log 变换才能可视化：

![价格分布 log1p](figures/c__llm_provider_benchmarking_price_num_hist.png)

![价格箱线图](figures/c__llm_provider_benchmarking_price_num_box.png)

- 主体价格集中在 log 尺度 10–16
- 存在极端离群值（疑似单位/录入错误），箱线图几乎压缩在 0 附近

### 2.6 评分 vs 评论数

评论越多，评分越向 4–5 分收敛（流行度偏差）：

![评分与评论数关系](figures/c__llm_provider_benchmarking_rating_vs_reviews.png)

---

## 3. 近期工作：新数据上的模型微调

### 3.1 数据划分（`train_5000` / `valid_5000` / `test_5000`）

从 `ecommerce.csv` 构建 query–passage 对，按 `source` 分层划分：

| 划分 | 样本数 | 用途 |
|------|--------|------|
| `train_5000.jsonl` | 3500 | 训练 |
| `valid_5000.jsonl` | 750 | 验证 |
| `test_5000.jsonl` | 750 | 测试（可选） |

- **Query：** 商品 `title`（或训练 jsonl 中的自然语言 query）
- **Passage：** `searchable_text`
- **Loss：** `MultipleNegativesRankingLoss`（in-batch negatives）

### 3.2 E5-base 微调配置

| 项 | 值 |
|----|-----|
| 基座模型 | `intfloat/multilingual-e5-base` |
| 输出模型 | `e5_base_finetuned_5000` |
| Epoch | 1 |
| Batch size | 8（per device） |
| Learning rate | 1e-5 |
| FP16 | 是 |
| 训练时长 | ~4.8 分钟（Colab GPU） |
| 训练 loss（末步） | ~0.0001 |

训练脚本 / Notebook：
- `embedding_project/notebooks/train_e5_base_5000_1epoch_colab.ipynb`
- `embedding_project/scripts/train_embedding_model.py`

### 3.3 Reranker 管道

两阶段检索：

```
Query (title)
  → [1] E5 bi-encoder 编码 → cosine 相似度 → 取 top-n 候选
  → [2] Cross-encoder reranker 对 (query, passage) 打分 → 重排 → 取 top-k
```

评估脚本：`embedding_project/scripts/evaluate_reranker_pipeline.py`  
Colab Notebook：`embedding_project/notebooks/evaluate_e5_reranker_5000_colab.ipynb`

| 参数 | 默认值 |
|------|--------|
| Corpus | 5000 商品 `searchable_text` |
| Query | `title`（4350 条唯一标题） |
| Ground truth | 同行 `product_id` |
| n（候选数） | [20, 50, 100, 200, 500] |
| k（最终 top-k） | [5, 10, 20] |
| eval-k | 10 |

---

## 4. 评估结果（Colab T4，全量 4350 query）

### 4.1 Bi-encoder only @10

```
Precision@10: 0.1114
Recall@10:    0.9979
F1@10:        0.2005
MRR@10:       0.9902
NDCG@10:      0.9926
n_queries:    4350
```

### 4.2 Bi-encoder + Reranker @10（n=20，初步结果）

```
Precision@10: 0.1115
Recall@10:    0.9986
F1@10:        0.2006
```

Reranker 在 n=20 时对 P/R/F1 提升极小，但 MRR/NDCG 层面 bi-encoder 已接近上限（MRR@10 ≈ 0.99）。

---

## 5. 为什么 Precision@10 天花板约 10%？

这是 **评估指标定义 + 标注方式** 共同造成的，不是模型“只找对 10%”。

### 5.1 数学上限：每 query 只有 1 个正样本

评估逻辑（`build_eval_from_ecommerce`）：

- **Query** = 商品 `title`
- **Ground truth** = 该商品自己的 `product_id`（每行 1 个标签）
- **Corpus** = 全部 5000 商品

对绝大多数 query，相关商品 **恰好 1 个**。在 top-10 中：

$$\text{Precision@10} = \frac{\text{命中数}}{10} \leq \frac{1}{10} = 0.1 = 10\%$$

实测 **0.1114 ≈ 11%**，已非常接近理论上界。

### 5.2 数据佐证

| 统计项 | 数值 |
|--------|------|
| 商品总数 | 5000 |
| 唯一 title 数 | 4350 |
| 重复 title 行数 | 650 |
| 评估 query 数 | 4350 |
| 仅 1 个标签的 query | 4123（94.8%） |
| 多标签 query（同 title 多 SKU） | 227 |

### 5.3 高 Recall、低 Precision 的组合说明什么？

| 现象 | 含义 |
|------|------|
| Recall@10 ≈ **99.8%** | 几乎每个 query 的正确商品都在 top-10 里 |
| Precision@10 ≈ **11%** | top-10 中平均只有 ~1.1 个是“标注为正”的商品 |
| MRR@10 ≈ **0.99** | 正确商品通常排在很靠前（常在前 1–2 位） |

**结论：** 模型检索能力已经很强；Precision@10 低是因为 **分母固定为 10、分子最多为 1**，不能单独用 P@10 判断好坏。

### 5.4 更合理的汇报方式

向导师汇报时建议强调：

1. **Recall@10、MRR@10、NDCG@10** — 衡量“是否找到、排得多靠前”
2. **F1@10** — 在单标签设定下与 P@10 高度相关，约 0.20 是预期范围
3. **Reranker 前后对比** — 看 n/k 网格搜索与 MRR 提升
4. **阈值分析（FPR/FNR/EER）** — 业务可用的相关性分数截断

若希望 P@10 更有区分度，可改用：
- 多相关标签评估（同类目/同品牌均为正样本）
- 或通过 `test_5000.jsonl` 的自然语言 query 评估

---

## 6. 局限性与后续方向

| 问题 | 说明 | 建议 |
|------|------|------|
| 标题重复 | 650 行共享 title，227 个 query 多标签 | 评估时去重或扩展正样本集 |
| 元数据稀疏 | tags/size/color 缺失高 | 增强 `searchable_text` 构建 |
| 价格离群 | 极端值影响统计 | 清洗 `price_num` |
| Reranker 慢 | cross-encoder 无 flash_attn，全网格耗时数小时 | 缩小 n 网格或增大 batch |
| P@10 不敏感 | 单标签 + k=10 饱和 | 主报 MRR/NDCG/Recall |

---

## 7. 复现命令

```bash
# 仅 bi-encoder + reranker 全管道评估
python embedding_project/scripts/evaluate_reranker_pipeline.py \
  --embedding-model embedding_project/models/e5_base_finetuned_5000 \
  --reranker-model embedding_project/models/reranker \
  --eval-csv embedding_project/data/ecommerce.csv \
  --query-col title \
  --n-values 20 50 100 \
  --k-values 5 10 20 \
  --rerank-batch-size 64
```

输出：`embedding_project/outputs/evaluation/reranker_pipeline_eval.json`

---

## 8. 附录：图表索引

| 图 | 文件 |
|----|------|
| 数据源分布 | `figures/c__llm_provider_benchmarking_source_top.png` |
| 评论数直方图 | `figures/c__llm_provider_benchmarking_reviews_count_hist.png` |
| 评论数箱线图 | `figures/c__llm_provider_benchmarking_reviews_count_box.png` |
| 评分 vs 评论 | `figures/c__llm_provider_benchmarking_rating_vs_reviews.png` |
| 评分直方图 | `figures/c__llm_provider_benchmarking_rating_hist.png` |
| 评分箱线图 | `figures/c__llm_provider_benchmarking_rating_box.png` |
| 价格直方图 | `figures/c__llm_provider_benchmarking_price_num_hist.png` |
| 价格箱线图 | `figures/c__llm_provider_benchmarking_price_num_box.png` |
| 缺失值 | `figures/c__llm_provider_benchmarking_missing_values.png` |

---

*Báo cáo được tạo tự động từ pipeline `evaluate_reranker_pipeline.py` và EDA trên `ecommerce.csv`.*
