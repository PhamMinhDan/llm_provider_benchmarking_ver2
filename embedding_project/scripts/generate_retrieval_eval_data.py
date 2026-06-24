"""Tạo queries tự nhiên cho retrieval evaluation - Final version.

- Hỗ trợ cả tiếng Việt và tiếng Anh
- Query phải có nghĩa, có hành động rõ ràng
- Product type phải rõ ràng
"""

import json
import random
import pandas as pd
import re
from pathlib import Path

random.seed(42)

# ===== CONFIG =====
NUM_QUERIES = 500
OUTPUT_DIR = Path("embedding_project/outputs/retrieval_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== PRODUCT KEYWORDS (tiếng Việt + English) =====
PRODUCT_KEYWORDS = [
    # Vietnamese
    'áo phông', 'áo thun', 'áo sơ mi', 'áo len', 'áo khoác', 'áo hoodie', 'áo nỉ',
    'áo thể thao', 'áo cổ lọ', 'áo ba lỗ', 'áo dài', 'áo đôi', 'áo vest',
    'quần jeans', 'quần short', 'quần dài', 'quần thể thao', 'quần jogger',
    'váy', 'chân váy', 'đầm', 'đầm maxi', 'đầm suông',
    'giày', 'giày thể thao', 'giày cao gót', 'giày bốt', 'giày sandal', 'giày chạy bộ',
    'tất', 'vớ', 'mũ', 'nón', 'khăn', 'khăn quàng', 'khăn tắm',
    'túi xách', 'túi đeo chéo', 'túi tote', 'balo', 'ba lô',
    'kính mát', 'gọng kính', 'mắt kính',
    'ga trải giường', 'vỏ gối', 'vỏ nệm', 'chăn', 'chăn bông', 'gối', 'đệm',
    'thảm', 'rèm', 'rèm cửa', 'bàn', 'ghế', 'kệ', 'tủ',
    'điện thoại', 'tai nghe', 'loa', 'sạc', 'cáp sạc',
    'ốp điện thoại', 'dán màn hình', 'máy tính', 'laptop', 'máy tính bảng',
    'tivi', 'camera', 'máy chiếu',
    'son môi', 'son', 'kem dưỡng', 'sữa rửa mặt', 'mặt nạ', 'nước hoa',
    'kem chống nắng', 'phấn', 'mascara', 'kem nền',
    'thức ăn cho mèo', 'thức ăn cho chó', 'đồ chơi cho mèo', 'xương cho chó',
    'bóng', 'vợt', 'máy chạy bộ', 'xe đạp', 'vali',
    'sữa', 'bánh', 'cà phê', 'trà', 'mì', 'gạo',
    'đèn', 'bóng đèn', 'dụng cụ', 'máy khoan', 'bộ cờ lê',
    'hộp', 'hộp đựng', 'túi', '包', '包',
    
    # English
    'shirt', 't-shirt', 'tshirt', 'blouse', 'sweater', 'jacket', 'hoodie',
    'pants', 'jeans', 'shorts', 'leggings', 'skirt', 'dress',
    'shoes', 'sneakers', 'boots', 'sandals', 'slippers',
    'socks', 'hat', 'cap', 'scarf', 'gloves',
    'bag', 'backpack', 'purse', 'wallet', 'belt',
    'sunglasses', 'glasses', 'watch', 'jewelry',
    'bedding', 'pillow', 'blanket', 'sheet', 'comforter',
    'towel', 'rug', 'curtain', 'lamp', 'furniture',
    'phone', 'tablet', 'laptop', 'computer', 'charger', 'cable',
    'headphone', 'earphone', 'speaker', 'camera', 'projector',
    'lipstick', 'makeup', 'skincare', 'perfume', 'cream',
    'food', 'snack', 'drink', 'coffee', 'tea',
    'toy', 'game', 'book', 'tool', 'battery',
    'cable', 'adapter', 'case', 'cover', 'stand',
    'plant', 'pot', 'vase', 'candle', 'decor',
]

# ===== TEMPLATES =====
TEMPLATES_ACTION = [
    'buy {p}', 'order {p}', 'shop {p}', 'get {p}', 'find {p}',
    'mua {p}', 'đặt mua {p}', 'tìm {p}', 'mua {p} online',
]

TEMPLATES_QUESTION = [
    '{p} nào tốt', 'recommend {p}', 'best {p}',
    'tìm {p} cho {u}', '{p} cho {u}', 'nên mua {p} nào',
    '{p} giá tốt', '{p} tốt nhất', '{p} chất lượng',
]

USE_CASES = ['nam', 'nữ', 'trẻ em', 'du lịch', 'đi làm', 'người lớn tuổi', 'unisex']
COLORS = ['đen', 'trắng', 'đỏ', 'xanh', 'hồng', 'vàng', 'tím', 'cam', 'nâu', 'xám', 'đen']

# ===== LOAD =====
print("Loading corpus...")
df = pd.read_csv("embedding_project/data/ecommerce.csv")
df = df.dropna(subset=['product_id', 'title'])
df = df.drop_duplicates(subset=['product_id'])
print(f"Loaded {len(df)} products")

def has_meaningful_brand(brand):
    if pd.isna(brand) or not brand:
        return False
    brand_str = str(brand).strip()
    if len(brand_str) < 3 or len(brand_str) > 25:
        return False
    brand_lower = brand_str.lower()
    invalid = ['nan', 'no brand', 'no brands', 'null', 'none', '', '-', 'unknown']
    if brand_lower in invalid:
        return False
    if re.match(r'^[\d\s\-]+$', brand_lower):
        return False
    return True

def clean_text(text):
    """Remove special chars but keep Vietnamese."""
    return re.sub(r'[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹđ]', ' ', text.lower())

def extract_product(title):
    """Extract meaningful product name from title."""
    title_clean = clean_text(title)
    
    # Find known keywords
    for kw in sorted(PRODUCT_KEYWORDS, key=lambda x: -len(x)):
        if kw in title_clean:
            return kw
    
    # Fallback: first 2-3 words that look like product type
    words = title_clean.split()
    product_words = []
    skip_words = {'the', 'a', 'an', 'for', 'with', 'and', 'or', 'in', 'on', 'to', 'of', 'pack', 'set', 'pcs', 'piece', 'size', 'color', 'inch', 'cm', 'ml', 'g', 'oz', 'oz', 'new', 'sale', 'best', 'top', 'hot'}
    
    for w in words:
        if w in skip_words or len(w) < 3:
            continue
        # Skip pure numbers
        if re.match(r'^[\d\.]+$', w):
            continue
        product_words.append(w)
        if len(product_words) >= 3:
            break
    
    if product_words:
        result = ' '.join(product_words)
        if len(result) >= 4:
            return result
    
    return ' '.join(words[:2]) if words else 'sản phẩm'

def is_valid_query(query):
    """Check if query is meaningful."""
    words = query.split()
    if len(words) < 2:
        return False
    if len(query) < 6:
        return False
    if re.match(r'^[\d\s\-]+$', query):
        return False
    # Must have at least one meaningful word (not just numbers)
    meaningful = any(len(w) >= 3 for w in words)
    return meaningful

def generate_query(row):
    title = str(row.get('title', ''))
    brand = str(row.get('brand', '')).strip() if pd.notna(row.get('brand')) else ''
    
    product = extract_product(title)
    
    # Random template selection
    use_question = random.random() < 0.4
    use_vn = random.random() < 0.7  # 70% Vietnamese
    
    if use_question:
        if use_vn:
            template = random.choice([
                '{p} nào tốt',
                'tìm {p} cho {u}',
                '{p} cho {u}',
                'nên mua {p} nào',
                'gợi ý {p}',
            ])
        else:
            template = random.choice([
                'best {p}',
                'recommend {p}',
                '{p} for {u}',
            ])
    else:
        if use_vn:
            template = random.choice([
                'mua {p}',
                'đặt mua {p}',
                'tìm mua {p}',
                'mua {p} online',
            ])
        else:
            template = random.choice([
                'buy {p}',
                'order {p}',
                'shop {p}',
                'find {p}',
            ])
    
    # Format template
    use_case = random.choice(USE_CASES) if '{u}' in template else None
    try:
        query = template.format(p=product, u=use_case)
    except:
        query = f"mua {product}"
    
    # Clean up
    query = ' '.join(query.split())
    
    # Validate
    if not is_valid_query(query):
        query = f"mua {product}"
    
    if not is_valid_query(query):
        query = "tìm sản phẩm"
    
    return {
        'qid': None,
        'query': query,
        'product_id': str(row['product_id']),
    }

# ===== GENERATE =====
print(f"Generating {NUM_QUERIES} queries...")

sampled_indices = random.sample(range(len(df)), min(NUM_QUERIES, len(df)))
sampled_df = df.iloc[sampled_indices].reset_index(drop=True)

queries = []
for idx, row in sampled_df.iterrows():
    q = generate_query(row)
    q['qid'] = f"q{idx:04d}"
    queries.append(q)

# Validate and fix bad queries
bad_count = 0
for q in queries:
    if not is_valid_query(q['query']):
        idx = int(q['qid'][1:])
        if idx < len(sampled_df):
            row = sampled_df.iloc[idx]
            product = extract_product(str(row.get('title', '')))
            q['query'] = f"mua {product}"
            if not is_valid_query(q['query']):
                q['query'] = "tìm sản phẩm tốt"
            bad_count += 1

print(f"Fixed {bad_count} bad queries")

# ===== SAVE =====
queries_file = OUTPUT_DIR / "queries.jsonl"
with open(queries_file, 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps({'qid': q['qid'], 'query': q['query']}, ensure_ascii=False) + '\n')

qrels_file = OUTPUT_DIR / "qrels.jsonl"
with open(qrels_file, 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps({'qid': q['qid'], 'product_id': q['product_id'], 'relevance': 1}, ensure_ascii=False) + '\n')

labels_file = OUTPUT_DIR / "labels_with_metadata.jsonl"
with open(labels_file, 'w', encoding='utf-8') as f:
    for q in queries:
        idx = int(q['qid'][1:])
        row = sampled_df.iloc[idx]
        f.write(json.dumps({
            'qid': q['qid'],
            'query': q['query'],
            'product_id': q['product_id'],
            'title': str(row['title'])[:150],
        }, ensure_ascii=False) + '\n')

# ===== OUTPUT =====
print("\n" + "="*60)
print("SAMPLE (First 20 queries)")
print("="*60)
for i in range(min(20, len(queries))):
    print(json.dumps({'qid': queries[i]['qid'], 'query': queries[i]['query']}, ensure_ascii=False))

print(f"\nTotal: {len(queries)} queries")
print(f"Saved to: {OUTPUT_DIR}")
