import json
import random
import pandas as pd
import ast

random.seed(42)

# Template query tự nhiên
QUERY_TEMPLATES = {
    'keyword': [
        '{product_type}',
        '{brand} {product_type}',
        'mua {product_type}',
        '{product_type} giá rẻ',
        '{product_type} tốt nhất',
    ],
    'question': [
        '{product_type} nào tốt',
        'tìm {product_type} cho {use_case}',
        '{product_type} cho {use_case}',
        'nên mua {product_type} nào',
        'gợi ý {product_type}',
    ],
    'brand_type': [
        '{brand} {product_type}',
        '{brand} {color} {product_type}',
    ],
    'use_case': [
        '{product_type} cho {use_case}',
        '{product_type} dành cho {use_case}',
        '{product_type} {use_case}',
    ],
    'shopping': [
        'mua {product_type} online',
        'đặt mua {product_type}',
        'shop bán {product_type}',
    ],
}

USE_CASES = ['nam', 'nữ', 'du lịch', 'đi làm', 'người lớn tuổi']
COLORS = ['đen', 'trắng', 'đỏ', 'xanh', 'hồng', 'vàng']

def parse_category(cat_str):
    if pd.isna(cat_str):
        return []
    try:
        return ast.literal_eval(cat_str)
    except:
        return []

def extract_product_type(title, category_leaf):
    if category_leaf and category_leaf != 'nan':
        return category_leaf.lower()
    words = str(title).split()
    if len(words) > 3:
        return ' '.join(words[:3]).lower()
    return str(title).lower()

def generate_natural_query(row):
    brand = str(row.get('brand', '')).strip() if pd.notna(row.get('brand')) else ''
    if brand in ['nan', 'No Brand', 'No brands', '']:
        brand = ''
    
    category = str(row.get('category_leaf', '')).strip() if pd.notna(row.get('category_leaf')) else ''
    title = str(row.get('title', ''))
    product_type = extract_product_type(title, category)
    
    query_type = random.choice(list(QUERY_TEMPLATES.keys()))
    template = random.choice(QUERY_TEMPLATES[query_type])
    use_case = random.choice(USE_CASES)
    color = random.choice(COLORS)
    use_brand = brand and random.random() > 0.3
    query_brand = brand if use_brand else ''
    
    try:
        query = template.format(
            product_type=product_type,
            brand=query_brand,
            use_case=use_case,
            color=color,
        )
    except:
        query = product_type
    
    query = ' '.join(query.split())
    return {
        'query': query,
        'query_type': query_type,
        'product_id': str(row['product_id']),
        'title': title,
        'brand': brand,
        'category': category,
    }

# Load corpus
df = pd.read_csv('embedding_project/data/ecommerce.csv')
df = df.dropna(subset=['product_id', 'searchable_text'])
df = df.drop_duplicates(subset=['product_id'])
df['category_list'] = df['category'].apply(parse_category)
df['category_leaf'] = df['category_list'].apply(lambda x: x[-1] if x else '')

# Generate
NUM_QUERIES = 500
sampled_indices = random.sample(range(len(df)), min(NUM_QUERIES, len(df)))
sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

queries = []
for idx, row in sampled_df.iterrows():
    q = generate_natural_query(row)
    queries.append(q)

# Save
with open('embedding_project/outputs/natural_queries_500.jsonl', 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps(q, ensure_ascii=False) + '\n')

print('Generated', len(queries), 'queries')
print()
print('Sample queries:')
for i in range(10):
    q = queries[i]
    print(f'  [{q["query_type"]}] "{q["query"]}"')
    print(f'         -> {q["title"][:50]}...')
