"""Helper module cho Colab notebook — giữ nguyên 100% logic từ build_hard_negatives.py.

Sự khác biệt duy nhất với bản gốc:
  - Đường dẫn được trỏ vào thư mục Google Drive (/content/drive/MyDrive/...)
  - Embedding model chạy trên GPU (cuda) thay vì CPU
  - QdrantClient trỏ vào URL HTTP công khai (đã định trong .env)
"""

import os, json, time, re, logging, threading, codecs
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("hardneg")

# ---------------------------------------------------------------------------
# Colab: set QDRANT credentials here hoặc dùng Colab Secrets
# ---------------------------------------------------------------------------
# Cách 1: Colab Secrets (khuyên dùng)
# from google.colab import userdata
# os.environ["QDRANT_API_KEY"] = userdata.get("QDRANT_API_KEY")

# Cách 2: Set trực tiếp (thay bằng key thật của bạn)
# os.environ["QDRANT_API_KEY"] = "your-qdrant-api-key-here"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config (giữ nguyên như bản local)
# ---------------------------------------------------------------------------
CORPUS_CSV      = Path(os.getenv("CORPUS_CSV", "/content/drive/MyDrive/DATN/data/Dataset_DATN_28k.csv"))
QUERIES_JSONL   = Path(os.getenv("QUERIES_JSONL", "/content/drive/MyDrive/DATN/data/llm_queries_all_28k.jsonl"))
OUT_JSONL       = Path(os.getenv("OUT_JSONL", "/content/drive/MyDrive/DATN/data/training_data.jsonl"))

QDRANT_URL      = os.getenv("QDRANT_URL", "http://qdrant.datn-nextgen-suggest.site")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = "products_corpus"
EMBED_MODEL     = "intfloat/multilingual-e5-base"
EMBED_DIM       = 768
EMBED_BATCH     = 128                # tăng batch vì GPU
EMBED_PREFIX    = "passage: "
QUERY_PREFIX    = "query: "
FORCE_RECREATE  = True               # True = xóa collection cũ và upsert lại (đúng dữ liệu)

TOP_K           = 20
MAX_HARD_NEG    = 5
N_EASY_NEG      = 2

NVIDIA_URL      = "https://integrate.api.nvidia.com/v1"
LLM_JUDGE_1     = "meta/llama-4-maverick-17b-128e-instruct"
LLM_JUDGE_2     = "nvidia/llama-3.3-nemotron-super-49b-v1"
RPM_PER_KEY     = 40
EFFECTIVE_RPM   = int(RPM_PER_KEY * 0.8)
MIN_INTERVAL    = 60.0 / EFFECTIVE_RPM
JUDGE_WORKERS   = max(1, EFFECTIVE_RPM // 4)

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class _RateLimiter:
    def __init__(self, interval: float):
        self._lock = threading.Lock()
        self._last: float = 0.0
        self._interval = interval

    def sleep(self):
        with self._lock:
            wait = self._last + self._interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()

_limiter_1 = _RateLimiter(MIN_INTERVAL)
_limiter_2 = _RateLimiter(MIN_INTERVAL)

# Lazy init — gọi _get_client() bên trong hàm dùng, không phải lúc import
CLIENT_1: OpenAI = None  # type: ignore[assignment]
CLIENT_2: OpenAI = None  # type: ignore[assignment]


def _get_client_1() -> OpenAI:
    global CLIENT_1
    if CLIENT_1 is None:
        CLIENT_1 = OpenAI(base_url=NVIDIA_URL, api_key=os.environ["NVIDIA_API_KEY_1"])
    return CLIENT_1


def _get_client_2() -> OpenAI:
    global CLIENT_2
    if CLIENT_2 is None:
        CLIENT_2 = OpenAI(base_url=NVIDIA_URL, api_key=os.environ["NVIDIA_API_KEY_2"])
    return CLIENT_2

# ---------------------------------------------------------------------------
# Judge prompt & parsing (giữ nguyên)
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """\
Bạn là TRỌNG TÀI đánh giá mức độ liên quan cho hệ thống tìm kiếm thương mại điện tử tiếng Việt.
Cho một CÂU TRUY VẤN và DANH SÁCH sản phẩm ứng viên, hãy chấm điểm 0–3 cho MỖI sản phẩm:

  3 = ĐÚNG thứ người dùng tìm: khớp hoàn toàn ý định query.
  2 = KHỚP / mua thay thế được: cùng loại, thoả thuộc tính quan trọng query nêu.
  1 = CÙNG LOẠI NHƯNG KHÔNG KHỚP: cùng nhóm nhưng SAI thuộc tính/mục đích/đối tượng query yêu cầu.
  0 = KHÔNG LIÊN QUAN: khác loại sản phẩm hoàn toàn.

Nguyên tắc bắt buộc:
- Nếu query nêu thuộc tính cụ thể (màu, kích cỡ, chất liệu, công dụng, đối tượng dùng...)
  mà sản phẩm KHÔNG đáp ứng → cho điểm 1, KHÔNG cho 2.
- Nghiêm khắc: chỉ cho 2 hoặc 3 khi thực sự khớp; phân vân giữa 1 và 2 thì chọn 1.
- Chỉ dựa vào thông tin cho sẵn, KHÔNG suy diễn.
- Chấm ĐỦ tất cả sản phẩm, dùng đúng "id" đã cho.

Trả về DUY NHẤT một JSON (không markdown, không giải thích):
{"labels": [{"id": 1, "score": 0}, {"id": 2, "score": 1}, ...]}"""

JUDGE_USER_TEMPLATE = """\
CÂU TRUY VẤN: "{query}"

DANH SÁCH SẢN PHẨM ỨNG VIÊN:
{candidates_block}

Trả về JSON với đủ {n} nhãn:"""

# ---------------------------------------------------------------------------
# Step 1: build corpus
# ---------------------------------------------------------------------------
def build_corpus(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["product_id"] = df["product_id"].astype(str)
    df = df.drop_duplicates(subset=["product_id"], keep="first")
    df = df[df["product_name"].astype(str).str.len() > 3]
    for col in ("description", "brand", "category_name"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    def _unescape_double_encoded(text: str) -> str:
        if r"\u" not in text and r"\r" not in text:
            return text
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text

    def _build(row):
        parts = [_unescape_double_encoded(str(row.get("product_name", "")).strip())]
        desc = _unescape_double_encoded(str(row.get("description", "")).strip())
        if desc:
            parts.append(desc[:300])
        cat = str(row.get("category_name", "")).strip()
        if cat:
            parts.append(f"Danh mục: {cat}")
        brand = _unescape_double_encoded(str(row.get("brand", "")).strip())
        if brand and brand.lower() not in ("nan", "no brand", "unknown", "-", ""):
            parts.append(f"Thương hiệu: {brand}")
        price = row.get("price")
        if pd.notna(price):
            try:
                parts.append(f"Giá: {int(float(price)):,} VNĐ")
            except Exception:
                pass
        return " | ".join(p for p in parts if p)

    df["corpus_text"] = df.apply(_build, axis=1)
    log.info("Corpus built: %d sản phẩm", len(df))
    return df

# ---------------------------------------------------------------------------
# Step 2: embed + upsert
# ---------------------------------------------------------------------------
def embed_and_upsert(df: pd.DataFrame, model: SentenceTransformer, client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if FORCE_RECREATE and COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        log.info("Qdrant: đã xóa collection cũ (FORCE_RECREATE=True)")
        existing = []
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("Qdrant: tạo collection mới")
    else:
        count = client.count(COLLECTION_NAME).count
        if count == len(df):
            log.info("Qdrant: đã có %d points, bỏ qua upsert", count)
            return
        log.info("Qdrant: %d points ≠ %d sp → re-upsert", count, len(df))

    log.info("Encoding %d texts (batch=%d) trên GPU...", len(df), EMBED_BATCH)
    texts = [EMBED_PREFIX + t for t in df["corpus_text"].tolist()]
    vectors = model.encode(
        texts, batch_size=EMBED_BATCH,
        show_progress_bar=True, normalize_embeddings=True,
        convert_to_numpy=True,
    )

    log.info("Upserting...")
    buf = []
    for i, (_, row) in enumerate(df.iterrows()):
        buf.append(PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "product_id":    row["product_id"],
                "corpus_text":   row["corpus_text"],
                "category_name": None if pd.isna(row.get("category_name")) or str(row.get("category_name","")).strip() == ""
                                 else str(row["category_name"]).strip(),
            },
        ))
        if len(buf) == 256:
            client.upsert(COLLECTION_NAME, points=buf); buf = []
    if buf:
        client.upsert(COLLECTION_NAME, points=buf)
    log.info("Upsert xong %d points", len(df))

# ---------------------------------------------------------------------------
# Step 3: retrieve candidates
# ---------------------------------------------------------------------------
def retrieve_candidates(
    query: str, pos_pid: str,
    model: SentenceTransformer, client: QdrantClient,
) -> List[Dict]:
    q_vec = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_vec,
            limit=TOP_K + 5,
            with_payload=True,
        )
    except AttributeError:
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=TOP_K + 5,
            with_payload=True,
        ).points
    candidates = []
    for r in results:
        pid = r.payload.get("product_id", "")
        if pid == pos_pid:
            continue
        candidates.append({
            "product_id":    pid,
            "corpus_text":   r.payload.get("corpus_text", ""),
            "category_name": r.payload.get("category_name", ""),
            "embed_score": r.score,
        })
        if len(candidates) >= TOP_K:
            break
    return candidates

# ---------------------------------------------------------------------------
# Step 4 + 5: batch judge
# ---------------------------------------------------------------------------
def _build_candidates_block(candidates: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        text = c["corpus_text"][:200].replace("\n", " ")
        lines.append(f"[id={i}] {text}")
    return "\n".join(lines)


def _parse_judge_response(raw: str, n_candidates: int) -> Dict[int, int]:
    raw = raw.strip()
    try:
        m = re.search(r'\{.*"labels".*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return {
                int(item["id"]): int(item["score"])
                for item in data.get("labels", [])
                if 1 <= int(item.get("id", 0)) <= n_candidates
                and item.get("score") in (0, 1, 2, 3)
            }
    except Exception:
        pass
    scores = {}
    for m in re.finditer(r'"id"\s*:\s*(\d+).*?"score"\s*:\s*([0-3])', raw, re.DOTALL):
        cid, score = int(m.group(1)), int(m.group(2))
        if 1 <= cid <= n_candidates:
            scores[cid] = score
    return scores


def batch_judge(
    query: str,
    candidates: List[Dict],
    model_name: str,
    client: OpenAI,
    limiter: _RateLimiter,
    max_retries: int = 4,
) -> Dict[int, int]:
    if not candidates:
        return {}

    candidates_block = _build_candidates_block(candidates)
    user_msg = JUDGE_USER_TEMPLATE.format(
        query=query,
        candidates_block=candidates_block,
        n=len(candidates),
    )

    for attempt in range(max_retries):
        limiter.sleep()
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=700,
                timeout=90,
            )
            raw = (resp.choices[0].message.content or "").strip()
            scores = _parse_judge_response(raw, len(candidates))
            if scores:
                return scores
        except Exception as e:
            wait = (10 + attempt * 10) if "429" in str(e) else (2 + attempt * 2)
            log.debug("Judge %s attempt %d: %s (sleep %ds)",
                      model_name, attempt + 1, str(e)[:80], wait)
            time.sleep(wait)

    log.warning("batch_judge thất bại hoàn toàn cho query: %s", query[:60])
    return {}


def judge_candidates(
    query: str,
    candidates: List[Dict],
    pos_category: Optional[str],
) -> Tuple[List[Dict], List[Dict]]:
    if not candidates:
        return [], []

    scores_v1 = batch_judge(query, candidates, LLM_JUDGE_1, _get_client_1(), _limiter_1)
    if not scores_v1:
        return [], []

    passed_v1, easy_pool = [], []
    for i, cand in enumerate(candidates, 1):
        s = scores_v1.get(i)
        if s is None:
            continue
        if s == 0:
            easy_pool.append(cand)
        elif s == 1:
            passed_v1.append(cand)

    hard_negatives = []
    if passed_v1:
        scores_v2 = batch_judge(query, passed_v1, LLM_JUDGE_2, _get_client_2(), _limiter_2)
        for i, cand in enumerate(passed_v1, 1):
            s = scores_v2.get(i)
            if s is not None and s <= 1:
                cand["score_llm1"] = scores_v1.get(
                    candidates.index(cand) + 1 if cand in candidates else 0
                )
                cand["score_llm2"] = s
                hard_negatives.append(cand)
            if len(hard_negatives) >= MAX_HARD_NEG:
                break

    easy_negatives = [
        c for c in easy_pool
        if c.get("category_name") is not None or pos_category != ""
    ][:N_EASY_NEG]
    if len(easy_negatives) < N_EASY_NEG:
        extra = [c for c in easy_pool if c not in easy_negatives]
        easy_negatives += extra[:N_EASY_NEG - len(easy_negatives)]

    return hard_negatives, easy_negatives

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_queries(jsonl_path: Path) -> List[Dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pid = str(item["product_id"])
            for anchor in item.get("anchors", []):
                a = anchor.strip()
                if a:
                    rows.append({"query": a, "product_id": pid, "query_type": "specific"})
            for vq in item.get("vague_anchors", []):
                v = vq.strip()
                if v:
                    rows.append({"query": v, "product_id": pid, "query_type": "vague"})
    log.info("Loaded %d queries", len(rows))
    return rows


def load_done_queries(out_path: Path) -> set:
    done = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                done.add((item.get("query", ""), item.get("product_id", "")))
            except Exception:
                pass
    return done

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(limit: Optional[int] = None):
    log.info("=== START build_hard_negatives trên Colab GPU ===")

    df_corpus = build_corpus(CORPUS_CSV)
    pid_to_corpus   = {str(r["product_id"]): r["corpus_text"] for _, r in df_corpus.iterrows()}
    pid_to_category = {
        str(r["product_id"]): (None if pd.isna(r.get("category_name")) or str(r.get("category_name","")).strip() == ""
                               else str(r["category_name"]).strip())
        for _, r in df_corpus.iterrows()
    }

    log.info("Loading embedding model: %s", EMBED_MODEL)
    device = "cuda" if _has_cuda() else "cpu"
    log.info("Device: %s", device)
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    embed_model.max_seq_length = 512

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    embed_and_upsert(df_corpus, embed_model, qdrant)

    queries  = load_queries(QUERIES_JSONL)
    done_set = load_done_queries(OUT_JSONL)
    pending  = [q for q in queries if (q["query"], q["product_id"]) not in done_set]
    if limit:
        pending = pending[:limit]
        log.info("Smoke-test mode: chỉ xử lý %d queries đầu tiên", limit)
    log.info("Pending: %d queries | done: %d", len(pending), len(done_set))

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    success = fail = 0
    t0 = time.time()

    def process_query(item: Dict) -> Optional[Dict]:
        query  = item["query"]
        pid    = item["product_id"]
        qtype  = item["query_type"]
        pos    = pid_to_corpus.get(pid)
        if not pos:
            return None

        if qtype == "vague":
            return {"query": query, "pos": [pos], "neg": [],
                    "n_hard": 0, "n_easy": 0,
                    "product_id": pid, "query_type": "vague"}

        candidates = retrieve_candidates(query, pid, embed_model, qdrant)
        if not candidates:
            return {"query": query, "pos": [pos], "neg": [],
                    "n_hard": 0, "n_easy": 0,
                    "product_id": pid, "query_type": "specific"}

        pos_category = pid_to_category.get(pid, "")
        hard_negs, easy_negs = judge_candidates(query, candidates, pos_category)
        neg_texts = [c["corpus_text"] for c in hard_negs] + \
                    [c["corpus_text"] for c in easy_negs]

        return {
            "query":      query,
            "pos":        [pos],
            "neg":        neg_texts,
            "n_hard":     len(hard_negs),
            "n_easy":     len(easy_negs),
            "product_id": pid,
            "query_type": "specific",
        }

    log.info("JUDGE_WORKERS=%d | batch_judge=True | RPM/key=%d→eff=%d",
             JUDGE_WORKERS, RPM_PER_KEY, EFFECTIVE_RPM)

    with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as ex:
        futures = {ex.submit(process_query, item): item for item in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Mining"):
            try:
                record = fut.result()
            except Exception as e:
                log.error("Future error: %s", e)
                fail += 1
                continue
            if record is None:
                fail += 1
                continue
            with write_lock:
                with OUT_JSONL.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            success += 1
            if success % 100 == 0:
                elapsed = time.time() - t0
                log.info("✓ %d/%d | %.1f q/min | fail=%d",
                         success, len(pending),
                         success / max(1, elapsed) * 60, fail)

    elapsed = time.time() - t0
    log.info("DONE. success=%d | fail=%d | %.1f min", success, fail, elapsed / 60)

    if OUT_JSONL.exists():
        from collections import Counter
        with OUT_JSONL.open() as f:
            recs = [json.loads(l) for l in f if l.strip()]
        nh = Counter(r["n_hard"] for r in recs)
        qt = Counter(r.get("query_type","specific") for r in recs)
        log.info("Stats: total=%d | query_type=%s | n_hard=%s | avg_neg=%.2f",
                 len(recs), dict(qt), dict(sorted(nh.items())),
                 sum(len(r["neg"]) for r in recs) / max(1, len(recs)))


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
