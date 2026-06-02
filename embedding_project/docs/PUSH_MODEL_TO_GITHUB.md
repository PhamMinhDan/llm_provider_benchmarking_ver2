# Đẩy model fine-tune lên GitHub (sau khi train)

## Tên repo vs tên folder

| | Tên |
|--|-----|
| **Repo trên GitHub** | `llm_provider_benchmarking_ver2` |
| **Folder local / Colab** | `llm_provider_benchmarking` |

URL clone:

```text
https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git
```

Clone vào folder tùy chọn (Colab/local dùng tên ngắn):

```bash
git clone https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git llm_provider_benchmarking
```

Colab: `/content/llm_provider_benchmarking`

File model (`model.safetensors`) ~450MB → **phải dùng Git LFS**.

---

## 1) Chuẩn bị (một lần)

```bash
cd /path/to/llm_provider_benchmarking   # folder local, không phải _ver2

git lfs install
```

---

## 2) Thư mục model sau train

| Model | Path trong project |
|-------|------------------|
| MiniLM | `embedding_project/models/minilm_finetuned_final/` |
| BGE-M3 | `embedding_project/models/bge_m3_finetuned_final/` |

Không push checkpoint trung gian (`models/minilm/`, `models/bge-m3/`).

---

## 3) Push từ máy local

```bash
cd c:/llm_provider_benchmarking   # hoặc path folder của bạn

git lfs install
git add embedding_project/models/bge_m3_finetuned_final/
git commit -m "Add BGE-M3 fine-tuned model"
git push origin main
```

---

## 4) Push từ Google Colab

1. **Secrets** → Name: `GITHUB_TOKEN` → token GitHub (quyền ghi repo).
2. Cell clone (§2): `REPO_DIR = "/content/llm_provider_benchmarking"`.
3. Train §5 → Push §7 (notebook tự tìm folder có model).

Clone đúng:

```python
REPO_DIR = "/content/llm_provider_benchmarking"
!git clone https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git "$REPO_DIR"
```

**Lỗi `FileNotFoundError ... _ver2`:** đừng `chdir` vào `_ver2` — folder Colab là `llm_provider_benchmarking`, repo GitHub mới là `_ver2`.

---

## 5) Clone trên máy / Colab mới

```bash
git lfs install
git clone https://github.com/PhamMinhDan/llm_provider_benchmarking_ver2.git llm_provider_benchmarking
cd llm_provider_benchmarking
git lfs pull
```

---

## 6) Script local

```bash
bash embedding_project/scripts/push_models_after_train.sh bge-m3
git push origin main
```
