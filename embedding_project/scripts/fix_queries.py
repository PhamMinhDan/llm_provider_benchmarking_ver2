"""Sửa queries tự nhiên cho retrieval evaluation."""

import json
import re
from pathlib import Path

# ===== RULES =====

# 1. Chuyển English verb -> Vietnamese
ENGLISH_VERBS = {
    'find': 'tìm',
    'buy': 'mua',
    'order': 'đặt mua',
    'shop': 'shop bán',
    'recommend': 'gợi ý',
    'best': 'tốt nhất nên mua',
}

# 2. Sửa "for X" -> "cho X"
def fix_for(query):
    """Replace English 'for' with Vietnamese 'cho'."""
    return re.sub(r'\sfor\s', ' cho ', query)

# 3. Bổ sung từ cho query quá ngắn
SHORT_PRODUCTS = {
    'vớ': 'vớ nam',
    'tất': 'tất',
    'mũ': 'mũ',
    'nón': 'nón',
    'váy': 'váy',
    'túi': 'túi',
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
    'phone': 'điện thoại',
    'cap': 'nón',
    'hat': 'mũ',
    'bag': 'túi',
    'toy': 'đồ chơi',
    'book': 'sách',
    'lamp': 'đèn',
    'pot': 'chậu cây',
}

# 4. Templates sửa query ngắn
def fix_short_query(query):
    """Fix queries that are too short or vague."""
    q_lower = query.lower()
    
    # Check if starts with known verbs
    for en_verb, vn_verb in ENGLISH_VERBS.items():
        if q_lower.startswith(en_verb):
            # Extract product after verb
            rest = query[len(en_verb):].strip()
            if rest:
                # Fix 'for' in rest
                rest = fix_for(rest)
                # Check if product needs fix
                rest_words = rest.split()
                if rest_words:
                    first_word = rest_words[0].lower()
                    if first_word in SHORT_PRODUCTS:
                        rest_words[0] = SHORT_PRODUCTS[first_word]
                        rest = ' '.join(rest_words)
                # Return fixed query
                if en_verb == 'shop':
                    return f"shop bán {rest}"
                elif en_verb == 'recommend':
                    return f"gợi ý {rest}"
                elif en_verb == 'best':
                    return f"{rest} tốt nhất nên mua"
                elif en_verb == 'find':
                    return f"tìm {rest}"
                elif en_verb == 'buy':
                    return f"mua {rest}"
                elif en_verb == 'order':
                    return f"đặt mua {rest}"
    
    return query

# 5. Xử lý từng query
def fix_query(query):
    """Apply all fixes to a query."""
    original = query
    
    # Step 1: Fix "for X" -> "cho X"
    query = fix_for(query)
    
    # Step 2: Fix English verbs at start
    q_lower = query.lower().strip()
    
    for en_verb, vn_verb in ENGLISH_VERBS.items():
        if q_lower.startswith(en_verb + ' '):
            rest = query[len(en_verb):].strip()
            
            if en_verb == 'shop':
                # "shop X" -> "shop bán X" (if not already)
                if not rest.startswith('bán'):
                    return f"shop bán {rest}"
                return query
            elif en_verb == 'find':
                return f"tìm {rest}"
            elif en_verb == 'buy':
                return f"mua {rest}"
            elif en_verb == 'order':
                return f"đặt mua {rest}"
            elif en_verb == 'recommend':
                return f"gợi ý {rest}"
            elif en_verb == 'best':
                return f"{rest} tốt nhất nên mua"
    
    # Step 3: Fix queries starting with Vietnamese verbs
    if q_lower.startswith('shop '):
        rest = query[5:].strip()
        if not rest.startswith('bán'):
            return f"shop bán {rest}"
    
    if q_lower.startswith('mua ') or q_lower.startswith('đặt mua ') or q_lower.startswith('tìm mua '):
        words = query.split()
        if len(words) <= 2:
            # Too short, add context
            for short, full in SHORT_PRODUCTS.items():
                if short in q_lower:
                    if q_lower.startswith('mua '):
                        return f"mua {full}"
                    elif q_lower.startswith('đặt mua '):
                        return f"đặt mua {full}"
                    elif q_lower.startswith('tìm mua '):
                        return f"tìm mua {full}"
    
    # Step 4: Fix single product word queries
    single_words = ['vớ', 'tất', 'mũ', 'nón', 'váy', 'túi', 'bóng', 'ghế', 'tủ', 'bàn', 'đèn', 'gối', 'chăn', 'khăn', 'son', 'hộp', 'sữa', 'trà', 'bánh']
    if q_lower in single_words:
        return f"tìm {SHORT_PRODUCTS.get(q_lower, q_lower)}"
    
    # Step 5: Fix English single words
    en_single = ['phone', 'cap', 'hat', 'bag', 'toy', 'book', 'lamp', 'pot']
    if q_lower in en_single:
        return f"mua {SHORT_PRODUCTS.get(q_lower, q_lower)}"
    
    return query


# ===== MAIN =====

INPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries.jsonl")
OUTPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries_fixed.jsonl")

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
print("SAMPLE FIXES")
print("="*60)

# Load original for comparison
originals = {}
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            q = json.loads(line)
            originals[q['qid']] = q['query']

for q in queries[:30]:
    qid = q['qid']
    orig = originals.get(qid, '')
    fixed = q['query']
    if orig != fixed:
        print(f"{qid}:")
        print(f"  OLD: {orig}")
        print(f"  NEW: {fixed}")
        print()
