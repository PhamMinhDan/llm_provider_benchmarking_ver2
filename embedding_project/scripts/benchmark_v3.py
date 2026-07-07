"""
Benchmark V3 — so sánh E5 pretrained vs E5-FT V3

Test set chia rõ 2 phần (từ test_v3_fair.jsonl):
  • Part A: SPECIFIC queries (3,229) → single GT, exact-match (product_id)
  • Part B: VAGUE queries    (2,309) → multi-GT (cùng category + BM25 candidates)

Metric: Recall@K (multi-GT aware: hits / len(gt_set))
        MRR@K    (reciprocal rank of first relevant)
        NDCG@K   (binary relevance, position-discounted)
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path('embedding_project/data')
OUTPUT_DIR = Path('embedding_project/outputs/evaluation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== LOAD CORPUS ==============
print('Loading corpus...')
corpus_df = pd.read_csv(DATA_DIR / 'Dataset_DATN_28k.csv', usecols=['product_id', 'searchable_text'])
corpus_df = corpus_df.dropna(subset=['product_id', 'searchable_text'])
corpus_df['product_id'] = corpus_df['product_id'].astype(str)
corpus_df = corpus_df.drop_duplicates('product_id', keep='first')
corpus_ids = corpus_df['product_id'].tolist()
corpus_texts = corpus_df['searchable_text'].tolist()
print(f'Corpus: {len(corpus_ids):,} products')

# ============== LOAD TEST ==============
test_items = []
with open(DATA_DIR / 'test_v3_fair_filtered.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        test_items.append(json.loads(line))
test_queries = [t['query'] for t in test_items]
test_gt = [t['relevant_pids'] for t in test_items]  # multi-GT (list of relevant pids)
test_qtype = [t.get('query_type', '') for t in test_items]
print(f'Test: {len(test_items):,} queries (specific={test_qtype.count("specific")}, vague={test_qtype.count("vague")})')

# ============== LOAD BENCHMARK 200 ==============
bench_df = pd.read_csv(DATA_DIR / 'benchmark_queries_200.csv')
bench_queries = bench_df['query'].tolist()
bench_gt = bench_df['product_id'].astype(str).tolist()
print(f'Benchmark 200: {len(bench_queries):,} vague queries')

# ============== METRICS ==============
def recall_at_k(retrieved, gt_set, k):
    """Recall@K: tỷ lệ ground truth relevant nằm trong top-k retrieval (multi-GT aware)"""
    if not gt_set:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if pid in gt_set)
    return hits / len(gt_set)

def mrr_at_k(retrieved, gt_set, k):
    """MRR@K: reciprocal rank của ground truth relevant đầu tiên"""
    for i, pid in enumerate(retrieved[:k], 1):
        if pid in gt_set:
            return 1.0 / i
    return 0.0

def ndcg_at_k(retrieved, gt_set, k):
    """NDCG@K: binary relevance dựa trên relevant set"""
    k = min(k, len(retrieved))
    actual = [1.0 if pid in gt_set else 0.0 for pid in retrieved[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(actual))
    # IDCG = perfect ranking với tất cả relevant ở top
    n_rel = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(retrieved_lists, gt_lists, k_values=[1, 5, 10, 20]):
    """Evaluate với multi-GT (relevant_pids là list/set)"""
    # Convert GT lists thành sets
    gt_sets = [set(gt) if isinstance(gt, list) else {gt} for gt in gt_lists]
    results = {}
    for k in k_values:
        rec  = [recall_at_k(r, g, k) for r, g in zip(retrieved_lists, gt_sets)]
        mrr  = [mrr_at_k(r, g, k)    for r, g in zip(retrieved_lists, gt_sets)]
        ndcg = [ndcg_at_k(r, g, k)   for r, g in zip(retrieved_lists, gt_sets)]
        results[f'Recall@{k}'] = round(np.mean(rec)  * 100, 2)
        results[f'MRR@{k}']    = round(np.mean(mrr)  * 100, 2)
        results[f'NDCG@{k}']   = round(np.mean(ndcg) * 100, 2)
    return results

# ============== MODEL LOADING ==============
from sentence_transformers import SentenceTransformer

def load_model(path_or_name):
    if Path(path_or_name).exists():
        return SentenceTransformer(str(path_or_name))
    return SentenceTransformer(path_or_name)

# ============== EMBED & RETRIEVE ==============
def encode_passages(model, texts, batch_size=128):
    return model.encode(
        [f'passage: {t}' for t in texts],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

def encode_queries(model, queries, batch_size=128):
    return model.encode(
        [f'query: {q}' for q in queries],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

def retrieve_topk(query_emb, corpus_emb, corpus_ids, k=50):
    scores = np.dot(query_emb, corpus_emb.T)
    topk_idx = np.argpartition(-scores, kth=k, axis=1)[:, :k]
    sorted_idx = np.argsort(-np.take_along_axis(scores, topk_idx, axis=1), axis=1)
    topk_idx = np.take_along_axis(topk_idx, sorted_idx, axis=1)
    return [[corpus_ids[j] for j in row] for row in topk_idx]

# ============== RUN BENCHMARK ==============
all_results = {}

# --- E5 Pretrained ---
print('\n' + '='*60)
print('E5 Pretrained (multilingual-e5-base)')
print('='*60)
t0 = time.time()
e5_pre = load_model('intfloat/multilingual-e5-base')
print(f'Loaded in {time.time()-t0:.1f}s')

corpus_emb_pre = encode_passages(e5_pre, corpus_texts)
test_emb_pre = encode_queries(e5_pre, test_queries)
bench_emb_pre = encode_queries(e5_pre, bench_queries)

e5_pre_test = retrieve_topk(test_emb_pre, corpus_emb_pre, corpus_ids, k=50)
e5_pre_bench = retrieve_topk(bench_emb_pre, corpus_emb_pre, corpus_ids, k=20)

all_results['E5_pretrained'] = {
    'test': evaluate(e5_pre_test, test_gt),
    'test_specific': evaluate(
        [r for r, q in zip(e5_pre_test, test_qtype) if q == 'specific'],
        [g for g, q in zip(test_gt, test_qtype) if q == 'specific']
    ),
    'test_vague': evaluate(
        [r for r, q in zip(e5_pre_test, test_qtype) if q == 'vague'],
        [g for g, q in zip(test_gt, test_qtype) if q == 'vague']
    ),
    'benchmark_200': evaluate(e5_pre_bench, bench_gt, k_values=[1, 5, 10, 20]),
}

del e5_pre
del corpus_emb_pre
del test_emb_pre
del bench_emb_pre

# --- E5-FT V3 ---
v3_path = Path('embedding_project/models/e5_base_v3_finetuned/final')
if v3_path.exists():
    print('\n' + '='*60)
    print('E5-FT V3 (NEW)')
    print('='*60)
    t0 = time.time()
    e5_v3 = load_model(v3_path)
    print(f'Loaded in {time.time()-t0:.1f}s')

    corpus_emb_v3 = encode_passages(e5_v3, corpus_texts)
    test_emb_v3 = encode_queries(e5_v3, test_queries)
    bench_emb_v3 = encode_queries(e5_v3, bench_queries)

    e5_v3_test = retrieve_topk(test_emb_v3, corpus_emb_v3, corpus_ids, k=50)
    e5_v3_bench = retrieve_topk(bench_emb_v3, corpus_emb_v3, corpus_ids, k=20)

    all_results['E5_FT_v3'] = {
        'test': evaluate(e5_v3_test, test_gt),
        'test_specific': evaluate(
            [r for r, q in zip(e5_v3_test, test_qtype) if q == 'specific'],
            [g for g, q in zip(test_gt, test_qtype) if q == 'specific']
        ),
        'test_vague': evaluate(
            [r for r, q in zip(e5_v3_test, test_qtype) if q == 'vague'],
            [g for g, q in zip(test_gt, test_qtype) if q == 'vague']
        ),
        'benchmark_200': evaluate(e5_v3_bench, bench_gt, k_values=[1, 5, 10, 20]),
    }

    del e5_v3
    del corpus_emb_v3
    del test_emb_v3
    del bench_emb_v3
else:
    print(f'\nE5-FT V3 NOT found at {v3_path}')
    print('  → Train V3 model first, then re-run benchmark')

# ============== SAVE & PRINT ==============
out_file = OUTPUT_DIR / 'comparison_v3.json'
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f'\nResults saved: {out_file}')

# Pretty print comparison
print('\n' + '='*70)
print(' COMPARISON: E5 Pretrained vs E5-FT V3')
print('='*70)
for model_name, res in all_results.items():
    print(f'\n{model_name}:')
    print(f'  Test overall:    R@10={res["test"]["Recall@10"]:.2f}  MRR@10={res["test"]["MRR@10"]:.2f}  NDCG@10={res["test"]["NDCG@10"]:.2f}')
    if 'test_specific' in res:
        print(f'  Test specific:   R@10={res["test_specific"]["Recall@10"]:.2f}  MRR@10={res["test_specific"]["MRR@10"]:.2f}')
    if 'test_vague' in res:
        print(f'  Test vague:      R@10={res["test_vague"]["Recall@10"]:.2f}  MRR@10={res["test_vague"]["MRR@10"]:.2f}')
    if 'benchmark_200' in res:
        print(f'  Bench 200 (vague): R@10={res["benchmark_200"]["Recall@10"]:.2f}  MRR@10={res["benchmark_200"]["MRR@10"]:.2f}')

# Diff table
if 'E5_pretrained' in all_results and 'E5_FT_v3' in all_results:
    print('\n' + '='*70)
    print(' DELTA (V3 - Pretrained)')
    print('='*70)
    pre = all_results['E5_pretrained']
    v3 = all_results['E5_FT_v3']
    for split_name in ['test', 'test_specific', 'test_vague', 'benchmark_200']:
        if split_name not in pre or split_name not in v3:
            continue
        print(f'\n  {split_name}:')
        for metric in ['Recall@10', 'MRR@10', 'NDCG@10']:
            if metric in pre[split_name] and metric in v3[split_name]:
                diff = v3[split_name][metric] - pre[split_name][metric]
                sign = '+' if diff >= 0 else ''
                print(f'    {metric:20s}: {pre[split_name][metric]:6.2f} → {v3[split_name][metric]:6.2f} ({sign}{diff:.2f})')
