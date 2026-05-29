#!/usr/bin/env bash
# Stage fine-tuned model folders for Git LFS push.
# Usage:
#   bash embedding_project/scripts/push_models_after_train.sh minilm
#   bash embedding_project/scripts/push_models_after_train.sh bge-m3
#   bash embedding_project/scripts/push_models_after_train.sh all
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PRESET="${1:-}"
if [[ -z "$PRESET" ]]; then
  echo "Usage: $0 {minilm|bge-m3|all}"
  exit 1
fi

git lfs install

add_model() {
  local dir="$1"
  local name="$2"
  if [[ ! -d "$dir" ]]; then
    echo "SKIP: $name — không tìm thấy $dir"
    return 1
  fi
  if [[ ! -f "$dir/model.safetensors" ]]; then
    echo "SKIP: $name — thiếu model.safetensors trong $dir"
    return 1
  fi
  echo "ADD: $dir"
  git add "$dir"
  return 0
}

STAGED=0
case "$PRESET" in
  minilm)
    add_model "embedding_project/models/minilm_finetuned_final" "MiniLM" && STAGED=1
    ;;
  bge-m3)
    add_model "embedding_project/models/bge_m3_finetuned_final" "BGE-M3" && STAGED=1
    ;;
  all)
    add_model "embedding_project/models/minilm_finetuned_final" "MiniLM" && STAGED=1 || true
    add_model "embedding_project/models/bge_m3_finetuned_final" "BGE-M3" && STAGED=1 || true
    ;;
  *)
    echo "Unknown preset: $PRESET"
    exit 1
    ;;
esac

git add .gitattributes .gitignore 2>/dev/null || true

if [[ "$STAGED" -eq 0 ]]; then
  echo "Không có model nào được stage. Train xong rồi chạy lại."
  exit 1
fi

echo ""
echo "Đã stage. Kiểm tra:"
git status -s
echo ""
echo "Tiếp theo:"
echo "  git commit -m \"Add fine-tuned embedding model ($PRESET)\""
echo "  git push origin main"
