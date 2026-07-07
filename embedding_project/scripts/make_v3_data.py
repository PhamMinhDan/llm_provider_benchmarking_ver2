"""Generate V3 train/valid/test data từ upstream files.

Run 1 lần trên local HOẶC trên Colab. Output 3 file JSONL:
  - train_v3_filtered.jsonl    (no 0-neg rows, pos = corpus searchable_text)
  - valid_v3_filtered.jsonl    (no 0-neg rows)
  - test_v3_fair_filtered.jsonl (no 0-neg rows, multi-GT for vague queries)

Inputs (cần có trên máy hoặc drive):
  - embedding_project/data/train_v3.jsonl
  - embedding_project/data/valid_v3.jsonl
  - embedding_project/data/test_v3_fair.jsonl
  - embedding_project/data/Dataset_DATN_28k.csv

Usage:
  python embedding_project/scripts/make_v3_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'embedding_project' / 'data'

CORPUS_CSV = DATA_DIR / 'Dataset_DATN_28k.csv'

# Input files
INPUT_FILES = {
    'train': DATA_DIR / 'train_v3.jsonl',
    'valid': DATA_DIR / 'valid_v3.jsonl',
    'test':  DATA_DIR / 'test_v3_fair.jsonl',
}

# Output files (post-filter)
OUTPUT_FILES = {
    'train': DATA_DIR / 'train_v3_filtered.jsonl',
    'valid': DATA_DIR / 'valid_v3_filtered.jsonl',
    'test':  DATA_DIR / 'test_v3_fair_filtered.jsonl',
}


def filter_zero_neg(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Remove rows where neg is empty. Return (before, after)."""
    items = []
    with input_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    before = len(items)
    filtered = [r for r in items if len(r.get('neg', [])) > 0]
    after = len(filtered)
    with output_path.open('w', encoding='utf-8') as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return before, after


def main():
    print('=' * 70)
    print('MAKE V3 DATA — filter 0-neg rows')
    print('=' * 70)

    if not CORPUS_CSV.exists():
        print(f'ERROR: corpus not found: {CORPUS_CSV}')
        return 1

    print(f'Corpus: {CORPUS_CSV.name}')
    for name, src in INPUT_FILES.items():
        if not src.exists():
            print(f'SKIP {name}: {src.name} not found')
            continue
        dst = OUTPUT_FILES[name]
        before, after = filter_zero_neg(src, dst)
        print(f'{name:5s}: {before:>7,} → {after:>7,} (removed {before-after:>5,} 0-neg rows) → {dst.name}')

    print('\nDone. Files ready for V3 training.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())