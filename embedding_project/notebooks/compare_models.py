"""
Compare v1 vs v2 models on sample queries
Build corpus from Dataset_DATN_28k.csv
"""
import json
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy as np

# Paths
MODEL_V1 = Path("embedding_project/models/e5_base_finetune_v1")
MODEL_V2 = Path("embedding_project/models/e5_base_finetune_v2")
CORPUS_CSV = Path("embedding_project/data/Dataset_DATN_28k.csv")
TEST_PATH = Path("embedding_project/data/test_v2.jsonl")

print("Loading models...")
model_v1 = SentenceTransformer(str(MODEL_V1))
model_v2 = SentenceTransformer(str(MODEL_V2))
print("Models loaded!")

# Build corpus from CSV
print("\nBuilding corpus from CSV...")
df = pd.read_csv(CORPUS_CSV)
print(f"Loaded {len(df)} products")

# Build corpus text (same as training)
def build_corpus_text(row):
    parts = []
    if pd.notna(row.get('product_name')):
        parts.append(str(row['product_name']))
    if pd.notna(row.get('description')):
        parts.append(str(row['description'])[:200])
    if pd.notna(row.get('category_name')):
        parts.append(str(row['category_name']))
    if pd.notna(row.get('brand')):
        parts.append(str(row['brand']))
    if pd.notna(row.get('price')):
        parts.append(f"Giá: {row['price']}")
    return " | ".join(parts)

corpus_ids = df['product_id'].tolist()
corpus_texts = [build_corpus_text(row) for _, row in df.iterrows()]
print(f"Built {len(corpus_texts)} corpus items")

# Load test queries
print("\nLoading test queries...")
test_queries = []
with open(TEST_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data.get('query_type') == 'specific':
            test_queries.append(data)

print(f"Loaded {len(test_queries)} specific queries")

# Encode corpus with both models
print("\nEncoding corpus with v1...")
corpus_emb_v1 = model_v1.encode(corpus_texts, show_progress_bar=True, convert_to_tensor=True)
print("Encoding corpus with v2...")
corpus_emb_v2 = model_v2.encode(corpus_texts, show_progress_bar=True, convert_to_tensor=True)

# Search function
def search(query_emb, corpus_emb, corpus_ids, top_k=10):
    scores = torch.cosine_similarity(query_emb.unsqueeze(0), corpus_emb)
    top_indices = torch.argsort(scores, descending=True)[:top_k]
    return [(corpus_ids[i], scores[i].item()) for i in top_indices]

# Show sample comparisons
print("\n" + "="*80)
print("SAMPLE SEARCH RESULTS: v1 vs v2")
print("="*80)

for i, q in enumerate(test_queries[:5]):
    print(f"\n{'='*80}")
    print(f"Query: {q['query']}")
    print(f"Ground truth: {q['product_id']}")
    
    # Encode query
    query_emb_v1 = model_v1.encode(q['query'])
    query_emb_v2 = model_v2.encode(q['query'])
    
    # Search with v1
    results_v1 = search(torch.tensor(query_emb_v1), corpus_emb_v1, corpus_ids, top_k=5)
    # Search with v2
    results_v2 = search(torch.tensor(query_emb_v2), corpus_emb_v2, corpus_ids, top_k=5)
    
    print(f"\n  v1 (TripletLoss only):")
    for j, (pid, score) in enumerate(results_v1, 1):
        marker = "✓" if pid == q['product_id'] else " "
        gt_marker = " [GT]" if pid == q['product_id'] else ""
        print(f"    {marker} {j}. [{pid}] {score:.4f}{gt_marker}")
    
    print(f"\n  v2 (TripletLoss + MNRL):")
    for j, (pid, score) in enumerate(results_v2, 1):
        marker = "✓" if pid == q['product_id'] else " "
        gt_marker = " [GT]" if pid == q['product_id'] else ""
        print(f"    {marker} {j}. [{pid}] {score:.4f}{gt_marker}")

# Calculate hit rate
print("\n" + "="*80)
print("RECALL@10 COMPARISON")
print("="*80)

sample_queries = test_queries[:100]  # Sample 100 queries

hits_v1 = {1: 0, 5: 0, 10: 0}
hits_v2 = {1: 0, 5: 0, 10: 0}

for q in sample_queries:
    query_emb_v1 = model_v1.encode(q['query'])
    query_emb_v2 = model_v2.encode(q['query'])
    
    results_v1 = search(torch.tensor(query_emb_v1), corpus_emb_v1, corpus_ids, top_k=10)
    results_v2 = search(torch.tensor(query_emb_v2), corpus_emb_v2, corpus_ids, top_k=10)
    
    top10_v1 = [r[0] for r in results_v1]
    top10_v2 = [r[0] for r in results_v2]
    
    if q['product_id'] in top10_v1[:1]:
        hits_v1[1] += 1
    if q['product_id'] in top10_v1[:5]:
        hits_v1[5] += 1
    if q['product_id'] in top10_v1:
        hits_v1[10] += 1
        
    if q['product_id'] in top10_v2[:1]:
        hits_v2[1] += 1
    if q['product_id'] in top10_v2[:5]:
        hits_v2[5] += 1
    if q['product_id'] in top10_v2:
        hits_v2[10] += 1

n = len(sample_queries)
print(f"\nSample size: n={n} queries (SPECIFIC only)")
print(f"\n              v1           v2           Improvement")
print(f"-" * 55)
print(f"Recall@1     {hits_v1[1]/n*100:6.2f}%     {hits_v2[1]/n*100:6.2f}%     {(hits_v2[1]-hits_v1[1])/max(hits_v1[1],1)*100:+6.1f}%")
print(f"Recall@5     {hits_v1[5]/n*100:6.2f}%     {hits_v2[5]/n*100:6.2f}%     {(hits_v2[5]-hits_v1[5])/max(hits_v1[5],1)*100:+6.1f}%")
print(f"Recall@10    {hits_v1[10]/n*100:6.2f}%     {hits_v2[10]/n*100:6.2f}%     {(hits_v2[10]-hits_v1[10])/max(hits_v1[10],1)*100:+6.1f}%")

print("\n" + "="*80)
