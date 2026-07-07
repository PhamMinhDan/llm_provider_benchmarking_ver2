"""Upload V3 data files lên Google Drive (chạy trên Colab).

Yêu cầu:
  - Drive đã mount
  - DATN/data/ folder đã tồn tại trên Google Drive root

Chạy 1 cell trên Colab, hoặc:
  !python embedding_project/scripts/upload_v3_to_drive.py
"""
from pathlib import Path
import shutil, os, sys


# === CONFIG — chỉnh nếu cần ===
DRIVE_TARGET = Path("/content/drive/MyDrive/DATN/data")
LOCAL_DATA   = Path("/content/llm_provider_benchmarking_ver2/embedding_project/data")

FILES = [
    "train_v3_filtered.jsonl",
    "valid_v3_filtered.jsonl",
    "test_v3_fair_filtered.jsonl",
]


def main():
    print("=" * 70)
    print("UPLOAD V3 FILES → GOOGLE DRIVE")
    print("=" * 70)

    if not DRIVE_TARGET.exists():
        print(f"ERROR: {DRIVE_TARGET} not found")
        print("Create folder DATN/data in Google Drive root first.")
        sys.exit(1)

    for fname in FILES:
        src = LOCAL_DATA / fname
        if not src.exists():
            print(f"  MISSING: {src}")
            continue
        dst = DRIVE_TARGET / fname
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"\n→ {fname} ({size_mb:.1f} MB)")
        print(f"  src: {src}")
        print(f"  dst: {dst}")
        # shutil.copyfile copies to /content/drive path, fully controlled
        shutil.copyfile(src, dst)
        print(f"  ✓ uploaded ({(dst.stat().st_size/(1024*1024)):.1f} MB)")

    print("\n" + "=" * 70)
    print("DONE. Files available at:")
    print(f"  {DRIVE_TARGET}")
    print("=" * 70)


if __name__ == "__main__":
    main()