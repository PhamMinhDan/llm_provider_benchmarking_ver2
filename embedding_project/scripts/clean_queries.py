"""Làm sạch queries - Chỉ loại bỏ từ quảng cáo và SKU rõ ràng."""

import json
import re
from pathlib import Path

# ===== DANH SÁCH TỪ CẦN LOẠI BỎ =====

# 1. Từ quảng cáo - Chỉ loại bỏ khi đứng một mình
AD_WORDS = [
    'unbeatablesale', 'bestseller', 'hotsale', 'flashsale', 'flash sale',
    'discount', 'sale', 'khuyến mãi', 'giảm giá', 'promotion',
    'hot sale', 'best sale', 'special offer', 'deal', 'deals',
    'free ship', 'freeship', 'mua 1 tặng 1', 'giam gia', 'khuyen mai',
    'new arrival', 'newest', 'latest',
    'bán chạy', 'bán chạy nhất', 'top seller', 'hot item',
    'hàng mới', 'hàng hot', 'trend',
    'shopee', 'lazada', 'tiki', 'sendo',
]

# 2. SKU/Model codes rõ ràng - looser pattern
SKU_CODES = [
    r'\b[A-Z]{2,}[0-9]{3,}[A-Z0-9]*\b',  # P11456R, ABC123
    r'\b[0-9]{3,}[A-Z0-9]+\b',  # 123ABC
    r'\bco\d+\b',  # co3, co2
    r'\bcm\d+\b',  # cm3
]

# 3. Platform/seller names trong ad context
PLATFORM_NAMES = [
    'shopee', 'lazada', 'tiki', 'sendo', 'aday', 'taka', 
]

# 4. Ký tự đặc biệt ở cuối từ
SPECIAL_CHARS = r'[,!?;:\.\-\+\*\/\\\'\"\(\)\[\]\{\}]+'


def is_ad_word(word):
    """Check if word is an advertisement word."""
    w_lower = word.lower().strip('.,!?;:')
    return w_lower in AD_WORDS


def is_platform(word):
    """Check if word is a platform name."""
    w_lower = word.lower().strip('.,!?;:')
    return w_lower in PLATFORM_NAMES


# Known legitimate product model patterns (DON'T remove these)
KNOWN_MODELS = [
    's21', 's22', 's23', 's24', 's25',  # Samsung Galaxy S series
    'a01', 'a02', 'a03', 'a04', 'a05', 'a10', 'a11', 'a12', 'a13', 'a50', 'a51', 'a52', 'a60', 'a70', 'a71', 'a72',  # Samsung/Oppo A series
    'y21', 'y22', 'y23',  # Vivo Y series
    'mi10', 'mi11', 'mi12',  # Xiaomi
]

def is_known_model(word):
    """Check if word is a known legitimate product model."""
    return word.lower() in KNOWN_MODELS


def is_sku_code(word):
    """Check if word looks like SKU/model code."""
    w = word.strip('.,!?;:')
    
    # Skip known legitimate models
    if is_known_model(w):
        return False
    
    # Pattern: uppercase letters + 3+ digits (like P11456R)
    if re.match(r'^[A-Z]{2,}\d{3,}[A-Z0-9]*$', w):
        return True
    
    # Pattern: 3+ digits + letters (like 123ABC)
    if re.match(r'^\d{3,}[A-Z0-9]+$', w):
        return True
        
    # Pattern: co3, cm3, etc (short letter + digits)
    if re.match(r'^[a-z]{1,3}\d{1,3}$', w, re.I) and len(w) <= 5:
        return True
    
    # Pattern: d2498, c6090 (single letter + 4+ digits)
    if re.match(r'^[a-z]\d{4,}$', w, re.I):
        return True
    
    # Pattern: wp52 (2 letters + 2 digits)
    if re.match(r'^[a-z]{2,3}\d{2,}$', w, re.I):
        return True
        
    # Pattern: p11456r (lowercase + 5+ digits + lowercase)
    if re.match(r'^[a-z]\d{5,}[a-z]*$', w):
        return True
    
    # Too many uppercase + digits = likely SKU
    if len(w) >= 6 and any(c.isdigit() for c in w) and sum(1 for c in w if c.isupper()) >= 2:
        return True
    
    return False


def remove_suffix_punctuation(word):
    """Remove trailing special characters."""
    return re.sub(r'[,!?;:\.\-\+\*\/\\\'\"\(\)\[\]\{\}]+$', '', word)


def clean_query(query):
    """Clean a single query."""
    # Keep original for comparison
    original = query
    
    # Split into words
    words = query.split()
    
    cleaned_words = []
    for word in words:
        # Remove trailing punctuation
        clean_word = remove_suffix_punctuation(word)
        
        # Skip empty
        if not clean_word:
            continue
            
        # Skip ad words
        if is_ad_word(clean_word):
            continue
            
        # Skip platform names
        if is_platform(clean_word):
            continue
            
        # Skip SKU codes
        if is_sku_code(clean_word):
            continue
        
        # Keep everything else
        cleaned_words.append(word)
    
    # Join back
    result = ' '.join(cleaned_words)
    
    # Apply regex patterns to remove remaining SKU-like strings
    # Only remove clear garbage/seller codes, NOT legitimate product model names
    # NOTE: Keep s21, s24, a60, y21, etc - these are legitimate Samsung/Oppo/Vivo model names
    
    # Specific garbage patterns (seller codes, random strings)
    result = re.sub(r'\bp\d{5,}\b', '', result, flags=re.I)  # p11456r, p12345
    result = re.sub(r'\b[A-Z]{2,}\d{5,}\b', '', result)  # AB12345
    result = re.sub(r'\bco\d+\b', '', result, flags=re.I)  # co3 (keep s21, s24)
    result = re.sub(r'\bd\d{4,}\b', '', result, flags=re.I)  # d2498, d1234 (generic component codes)
    result = re.sub(r'\bc\d{4,}\b', '', result, flags=re.I)  # c6090
    result = re.sub(r'\bwp\d{2,}\b', '', result, flags=re.I)  # wp52
    result = re.sub(r'\bps\d+\b', '', result, flags=re.I)  # ps3, ps4 (not real skincare codes)
    result = re.sub(r'\bsuntime\s+co\d+', '', result, flags=re.I)  # suntime co3
    result = re.sub(r'\biz\d+\b', '', result, flags=re.I)  # ez8
    
    # DON'T remove these legitimate model patterns:
    # - s21, s22, s23, s24, s25 (Samsung Galaxy S)
    # - a01, a02, ..., a60, a70, a71, a72 (Samsung/Oppo A series)
    # - y21, y22, y23 (Vivo Y series)
    
    # Clean up extra spaces
    result = re.sub(r'\s+', ' ', result).strip()
    
    # If became empty, try to extract meaningful parts
    if not result.strip():
        # Try to keep first 2-3 meaningful words
        meaningful = []
        for word in original.split()[:3]:
            w = clean_word = remove_suffix_punctuation(word)
            if len(w) >= 3 and not w.isdigit() and not is_sku_code(w):
                meaningful.append(word)
        result = ' '.join(meaningful) if meaningful else original
    
    return result


# ===== MAIN =====

INPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries.jsonl")
OUTPUT_FILE = Path("embedding_project/outputs/retrieval_eval/queries_cleaned.jsonl")

# Load
queries = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            queries.append(json.loads(line))

print(f"Loaded {len(queries)} queries")

# Clean
cleaned_count = 0
changed_queries = []
for q in queries:
    original = q['query']
    cleaned = clean_query(original)
    if cleaned != original:
        cleaned_count += 1
        changed_queries.append({
            'qid': q['qid'],
            'before': original,
            'after': cleaned
        })
    q['query'] = cleaned

print(f"Cleaned {cleaned_count} queries")

# Save
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for q in queries:
        f.write(json.dumps({'qid': q['qid'], 'query': q['query']}, ensure_ascii=False) + '\n')

print(f"\nSaved to: {OUTPUT_FILE}")

# Show all changes
print("\n" + "="*70)
print(f"ALL {len(changed_queries)} CHANGES")
print("="*70)

for item in changed_queries:
    print(f"\n{item['qid']}:")
    print(f"  BEFORE: {item['before']}")
    print(f"  AFTER:  {item['after']}")
