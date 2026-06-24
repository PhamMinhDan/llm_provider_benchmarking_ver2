"""Sửa queries - Phiên bản 2."""

import json
import re
from pathlib import Path

# ===== RULES =====

# 1. Chuyển English verb -> Vietnamese
ENGLISH_VERBS = {
    'find': 'tìm',
    'find': 'tìm',
    'buy': 'mua',
    'order': 'đặt mua',
    'shop': 'shop bán',
    'recommend': 'gợi ý',
    'best': 'tốt nhất nên mua',
}

# 2. Thay thế English words -> Vietnamese
ENGLISH_TO_VN = {
    'phone': 'điện thoại',
    'cap': 'nón',
    'hat': 'mũ',
    'bag': 'túi',
    'toy': 'đồ chơi',
    'book': 'sách',
    'lamp': 'đèn',
    'pot': 'chậu cây',
    'for ': 'cho ',
    'for,': 'cho,',
    'for.': 'cho.',
    'for?': 'cho?',
}

# 3. Bổ sung từ cho product quá ngắn
PRODUCT_EXPANSION = {
    'vớ': 'vớ nam',
    'tất': 'tất chân',
    'mũ': 'mũ nón',
    'nón': 'nón',
    'váy': 'váy',
    'túi': 'túi xách',
    'bóng': 'bóng',
    'ghế': 'ghế',
    'tủ': 'tủ',
    'bàn': 'bàn',
    'đèn': 'đèn',
    'gối': 'gối',
    'chăn': 'chăn',
    'khăn': 'khăn',
    'son': 'son môi',
    'hộp': 'hộp đựng',
    'sữa': 'sữa',
    'trà': 'trà',
    'bánh': 'bánh',
    'cáp': 'cáp sạc',
    'vali': 'vali',
    'áo': 'áo',
    'quần': 'quần',
    'giày': 'giày',
    'điện thoại': 'điện thoại',
    'mascara': 'mascara',
}

# 4. Bổ sung adjective cho query ngắn
SHORT_QUERY_TEMPLATES = {
    'shop': 'shop bán {p} chất lượng',
    'mua': 'mua {p} online',
    'đặt mua': 'đặt mua {p}',
    'tìm': 'tìm mua {p}',
    'tìm mua': 'tìm mua {p}',
    'gợi ý': 'gợi ý {p} tốt nhất',
    'nên mua': 'nên mua {p} nào',
}

def fix_english_words(query):
    """Replace English words with Vietnamese."""
    q_lower = query.lower()
    for en, vn in ENGLISH_TO_VN.items():
        q_lower = q_lower.replace(en.lower(), vn)
    return q_lower

def fix_short_query(query):
    """Fix queries that are too short."""
    q_lower = query.lower().strip()
    words = q_lower.split()
    
    # If only 1-2 words, try to expand
    if len(words) <= 2:
        # Check if starts with known verb
        for prefix, template in SHORT_QUERY_TEMPLATES.items():
            if q_lower.startswith(prefix):
                rest = q_lower[len(prefix):].strip()
                # Get product
                product = rest.split()[0] if rest.split() else ''
                if product in PRODUCT_EXPANSION:
                    product = PRODUCT_EXPANSION[product]
                return template.format(p=product)
        
        # Single product word
        if words[0] in PRODUCT_EXPANSION:
            return f"tìm mua {PRODUCT_EXPANSION[words[0]]}"
        if words[0] in ['vớ', 'tất', 'mũ', 'nón', 'váy', 'túi', 'bóng', 'ghế', 'tủ', 'bàn', 'đèn', 'gối', 'chăn', 'khăn', 'son', 'hộp', 'sữa', 'trà', 'bánh', 'cáp']:
            return f"tìm mua {words[0]}"
    
    return query

def fix_query(query):
    """Apply all fixes to a query."""
    # Step 1: Replace English words
    query = fix_english_words(query)
    
    # Step 2: Title case fix for common words
    query = query.replace('phone', 'điện thoại')
    
    # Step 3: Fix query starting patterns
    q_lower = query.lower().strip()
    
    # Fix "shop X" -> "shop bán X"
    if q_lower.startswith('shop ') and not q_lower.startswith('shop bán'):
        query = 'shop bán ' + query[5:]
    
    # Fix "buy X" -> "mua X"
    if q_lower.startswith('buy '):
        query = 'mua ' + query[4:]
    
    # Fix "find X" -> "tìm X"
    if q_lower.startswith('find '):
        query = 'tìm ' + query[5:]
    
    # Fix "order X" -> "đặt mua X"
    if q_lower.startswith('order '):
        query = 'đặt mua ' + query[6:]
    
    # Fix "recommend X" -> "gợi ý X"
    if q_lower.startswith('recommend '):
        query = 'gợi ý ' + query[10:]
    
    # Fix "best X" -> "X tốt nhất"
    if q_lower.startswith('best '):
        query = query[5:] + ' tốt nhất'
    
    # Fix "for X" in middle
    query = re.sub(r'\sfor\s', ' cho ', query)
    
    # Step 4: Fix short queries
    words = query.split()
    if len(words) <= 2:
        # Check for single product word
        if q_lower in ['vớ', 'tất', 'mũ', 'nón', 'váy', 'túi', 'bóng', 'ghế', 'tủ', 'bàn', 'đèn', 'gối', 'chăn', 'khăn', 'son', 'hộp', 'sữa', 'trà', 'bánh', 'cáp', 'vali', 'áo', 'quần', 'giày']:
            product = PRODUCT_EXPANSION.get(q_lower, q_lower)
            # Check what verb precedes it
            for prefix in ['shop bán ', 'mua ', 'đặt mua ', 'tìm mua ', 'gợi ý ', 'nên mua ', 'tìm ', 'tìm mua ']:
                if q_lower.startswith(prefix):
                    rest = q_lower[len(prefix):]
                    if rest == q_lower:  # prefix not found
                        continue
                    return prefix + product
            return f"tìm mua {product}"
        
        # Single "phone"
        if 'phone' in q_lower:
            query = query.replace('phone', 'điện thoại')
    
    return query


# ===== MAIN =====

INPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries.jsonl")
OUTPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries_fixed_v2.jsonl")

# Load queries
queries = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            queries.append(json.loads(line))

print(f"Loaded {len(queries)} queries")

# Fix queries
fixed_count = 0
for q in queries:
    original = q['query']
    fixed = fix_query(original)
    if fixed != original:
        fixed_count += 1
        q['query'] = fixed

print(f"Fixed {fixed_count} queries")

# Save fixed queries
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps({'qid': q['qid'], 'query': q['query']}, ensure_ascii=False) + '\n')

print(f"\nSaved to: {OUTPUT_FILE}")

# Show samples
print("\n" + "="*60)
print("ALL CHANGES")
print("="*60)

# Load original for comparison
originals = {}
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            q = json.loads(line)
            originals[q['qid']] = q['query']

changes = 0
for q in queries:
    qid = q['qid']
    orig = originals.get(qid, '')
    fixed = q['query']
    if orig != fixed:
        changes += 1
        print(f"{qid}:")
        print(f"  OLD: {orig}")
        print(f"  NEW: {fixed}")

print(f"\nTotal changes: {changes}")
