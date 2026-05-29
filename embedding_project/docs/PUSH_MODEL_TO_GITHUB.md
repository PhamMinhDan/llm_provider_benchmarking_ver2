# Đẩy model fine-tune lên GitHub (sau khi train)

Repo: `https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git`

File model (`model.safetensors`) ~450MB → **phải dùng Git LFS**, không push trực tiếp bằng git thường.

## 1) Chuẩn bị (một lần)

```bash
cd /path/to/llm_provider_benchmarking

# Cài Git LFS (nếu chưa có): https://git-lfs.com
git lfs install
```

Repo đã có `.gitattributes` track `*.safetensors` qua LFS.

## 2) Sau khi train xong — thư mục cần có

| Model | Thư mục |
|-------|---------|
| MiniLM | `embedding_project/models/minilm_finetuned_final/` |
| BGE-M3 | `embedding_project/models/bge_m3_finetuned_final/` |

Chỉ push thư mục `*_finetuned_final/`, **không** push checkpoint trung gian (`models/minilm/`, `models/bge-m3/`).

## 3) Push từ máy local (Windows / Linux)

```bash
git status
git lfs track   # xác nhận LFS đã bật

# Thêm model (chọn một hoặc cả hai)
git add embedding_project/models/minilm_finetuned_final/
git add embedding_project/models/bge_m3_finetuned_final/

# (Tuỳ chọn) code + metrics
git add embedding_project/scripts/ embedding_project/notebooks/
git add embedding_project/outputs/evaluation/metrics_*.json

git commit -m "$(cat <<'EOF'
Add fine-tuned embedding models (MiniLM / BGE-M3).

EOF
)"

git push origin main
```

Lần đầu push LFS có thể mất vài phút (upload ~450MB/model).

## 4) Push từ Google Colab (sau train)

```python
from google.colab import userdata
import os

GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')  # Colab Secrets
REPO = 'PhamMinhDan/llm_provider_benchmarking_ver2.git'
BRANCH = 'main'

os.chdir('/content/llm_provider_benchmarking')

# Cài LFS
!apt-get update -qq && apt-get install -qq git-lfs
!git lfs install

# Cấu hình git (dùng token, không lưu password)
!git config user.email "you@example.com"
!git config user.name "Your Name"

!git add embedding_project/models/bge_m3_finetuned_final/
!git add embedding_project/outputs/evaluation/metrics_bge_m3.json
!git commit -m "Add BGE-M3 fine-tuned model from Colab"

# Push qua HTTPS + token
remote_url = f"https://{GITHUB_TOKEN}@github.com/{REPO}"
!git push {remote_url} {BRANCH}
```

Tạo **Personal Access Token** (GitHub → Settings → Developer settings → Fine-grained hoặc classic) với quyền `contents: write`. Lưu vào Colab Secrets tên `GITHUB_TOKEN`.

**Lưu ý:** Colab clone thường shallow — nếu `git push` báo lỗi, clone full repo trước:

```bash
!rm -rf /content/llm_provider_benchmarking
!git clone https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git /content/llm_provider_benchmarking
# copy model từ Drive hoặc train lại vào đúng path
```

## 5) Clone model trên máy khác / Colab mới

```bash
git lfs install
git clone https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git
cd llm_provider_benchmarking_ver2
git lfs pull
```

## 6) Giới hạn GitHub

- LFS free: ~1GB storage + 1GB bandwidth/tháng (đủ 1–2 model ~450MB nếu ít clone).
- Nếu vượt quota: dùng [Hugging Face Hub](https://huggingface.co) + `model.save_to_hub()` hoặc release artifact.

## 7) Script nhanh (local)

```bash
bash embedding_project/scripts/push_models_after_train.sh minilm
bash embedding_project/scripts/push_models_after_train.sh bge-m3
bash embedding_project/scripts/push_models_after_train.sh all
```
Script **không** tự `push` — chỉ `git add` + `commit`; bạn chạy `git push` sau khi kiểm tra.

