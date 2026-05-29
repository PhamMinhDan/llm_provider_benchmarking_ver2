"""
Dịch merged_products.csv sang tiếng Việt qua NVIDIA API.

Ghi đè các cột văn bản: title, description, category, tags (tên cột giữ nguyên).
Giữ nguyên: product_id, source, brand, price, rating, reviews_count, image_url.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nvidia_provider.llm import (  # noqa: E402
    MODELS,
    ModelConfig,
    chat_complete,
    create_client,
    probe_models,
)
from nvidia_provider.prompts import (  # noqa: E402
    TRANSLATE_FIELD_NAMES,
    PRODUCT_ROW_SYSTEM,
    PRODUCT_ROW_USER,
)
from nvidia_provider.text_utils import prepare_description_for_llm  # noqa: E402

PRESERVE_COLUMNS = frozenset(
    {
        "product_id",
        "source",
        "brand",
        "price",
        "rating",
        "reviews_count",
        "image_url",
    }
)


def _checkpoint_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".checkpoint")


def build_messages(row: dict) -> list[dict]:
    brand = (row.get("brand") or "").strip() or "(không rõ)"
    title = (row.get("title") or "").strip()
    category = (row.get("category") or "").strip()
    tags = (row.get("tags") or "").strip()
    raw_desc = row.get("description") or ""
    description = prepare_description_for_llm(raw_desc)

    if not description.strip() and title:
        description = (
            "(Mô tả gốc không đọc được hoặc rỗng. "
            "Dựa trên title, brand, category — viết description tiếng Việt ngắn, "
            "không bịa thông số kỹ thuật.)"
        )

    user_content = PRODUCT_ROW_USER.format(
        brand=brand,
        title=title,
        description=description,
        category=category or "(trống)",
        tags=tags or "(trống)",
    )
    system_content = PRODUCT_ROW_SYSTEM
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _normalize_text(s: str) -> str:
    # Normalize whitespace and casing for "unchanged" detection.
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _needs_retry_fields(src_row: dict, translated: dict[str, str]) -> list[str]:
    """Return fields that appear empty or unchanged (likely translation failure)."""
    retry_fields: list[str] = []
    for key in ("title", "category", "tags"):
        src = _normalize_text(src_row.get(key) or "")
        tr = _normalize_text(translated.get(key) or "")
        if not tr or tr == src:
            retry_fields.append(key)
    return retry_fields


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("JSON phải là object.")
    return data


def parse_translation(raw: str) -> dict[str, str]:
    data = _extract_json(raw)
    out: dict[str, str] = {}
    for key in TRANSLATE_FIELD_NAMES:
        val = data.get(key)
        if val is None:
            raise ValueError(f"Thiếu khóa JSON: {key}")
        out[key] = str(val).strip()
    return out


def apply_translation(row: dict, translated: dict[str, str]) -> None:
    for key in TRANSLATE_FIELD_NAMES:
        if key in translated and translated[key]:
            row[key] = translated[key]


def load_done_ids(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.is_file():
        return set()
    return {
        line.strip()
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_done(checkpoint_path: Path, product_id: str) -> None:
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(product_id + "\n")


def row_needs_translation(row: dict) -> bool:
    return any((row.get(k) or "").strip() for k in TRANSLATE_FIELD_NAMES)


def _is_rate_limited_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        " 429 " in f" {msg} "
        or "too many request" in msg
        or "rate limit" in msg
        or "ratelimit" in msg
    )


def _complete_with_backoff(
    client,
    candidates: list[ModelConfig],
    messages: list[dict],
    *,
    max_retries: int,
    backoff_base_s: float,
    pid: str,
    verbose: bool,
    adaptive_state: AdaptiveState | None = None,
) -> tuple[str, str]:
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            ordered = candidates
            for cfg in ordered:
                try:
                    text = chat_complete(client, cfg, messages, max_tokens=2048)
                    if text:
                        return text, cfg.name
                except Exception as model_err:
                    if adaptive_state is not None and _is_rate_limited_error(model_err):
                        adaptive_state.mark_rate_limited(cfg.name)
                    continue
            raise RuntimeError("không có model trả về nội dung")
        except Exception as e:
            last_error = e
            if not _is_rate_limited_error(e) or attempt >= attempts:
                raise
            sleep_s = min(30.0, backoff_base_s * (2 ** (attempt - 1)))
            if verbose:
                print(
                    f"  ⏳ {pid}: rate-limit, retry {attempt}/{attempts - 1} sau {sleep_s:.1f}s",
                    file=sys.stderr,
                )
            time.sleep(sleep_s)
    raise RuntimeError(str(last_error) if last_error else "Dịch thất bại không rõ nguyên nhân")


@dataclass
class AdaptiveState:
    primary_model: str
    fallback_models: list[str] = field(default_factory=list)
    cooldown_seconds: float = 20.0
    _lock: Lock = field(default_factory=Lock)
    _cooldown_until: dict[str, float] = field(default_factory=dict)

    def model_order(self, all_candidates: list[ModelConfig]) -> list[ModelConfig]:
        by_name = {c.name: c for c in all_candidates}
        preferred = [self.primary_model, *self.fallback_models]
        ordered_names = [n for n in preferred if n in by_name]
        # Append any remaining probed candidates as last-resort fallback.
        ordered_names.extend([c.name for c in all_candidates if c.name not in ordered_names])

        now = time.time()
        with self._lock:
            active = [n for n in ordered_names if self._cooldown_until.get(n, 0.0) <= now]
            cooling = [n for n in ordered_names if n not in active]
        final_names = active + cooling
        return [by_name[n] for n in final_names]

    def mark_rate_limited(self, model_name: str) -> None:
        with self._lock:
            self._cooldown_until[model_name] = time.time() + max(0.0, self.cooldown_seconds)


def translate_one_row(
    row: dict,
    *,
    client,
    candidates: list[ModelConfig],
    verbose: bool,
    max_retries: int,
    backoff_base_s: float,
    adaptive_state: AdaptiveState | None,
) -> tuple[dict, str, bool, str | None]:
    """Translate one row and return (out_row, pid, has_error, error_msg)."""
    pid = (row.get("product_id") or "").strip()
    out_row = dict(row)
    for col in PRESERVE_COLUMNS:
        if col in row:
            out_row[col] = row[col]

    if not row_needs_translation(row):
        return out_row, pid, False, None

    try:
        messages = build_messages(row)
        ordered_candidates = (
            adaptive_state.model_order(candidates) if adaptive_state is not None else candidates
        )
        raw, used_model = _complete_with_backoff(
            client,
            ordered_candidates,
            messages,
            max_retries=max_retries,
            backoff_base_s=backoff_base_s,
            pid=pid,
            verbose=verbose,
            adaptive_state=adaptive_state,
        )
        translated = parse_translation(raw)

        retry_fields = _needs_retry_fields(row, translated)
        if retry_fields:
            strict_note = (
                "BẠN CHƯA DỊCH ĐÚNG. Hãy dịch SANG TIẾNG VIỆT cho các trường: "
                f"{', '.join(retry_fields)}. Tuyệt đối KHÔNG giữ nguyên tiếng Anh đầu vào "
                "(trừ thương hiệu/tên riêng/size/SKU). Không để chuỗi rỗng. "
                "Trả lại JSON đúng 4 khóa: title, description, category, tags."
            )
            messages = build_messages(row)
            messages[0]["content"] = messages[0]["content"] + "\n\n" + strict_note
            ordered_candidates = (
                adaptive_state.model_order(candidates)
                if adaptive_state is not None
                else candidates
            )
            raw, used_model = _complete_with_backoff(
                client,
                ordered_candidates,
                messages,
                max_retries=max_retries,
                backoff_base_s=backoff_base_s,
                pid=pid,
                verbose=verbose,
                adaptive_state=adaptive_state,
            )
            translated = parse_translation(raw)

        apply_translation(out_row, translated)
        if verbose:
            title_preview = (out_row.get("title") or "")[:55]
            retry_note = f" retry_fields={retry_fields}" if retry_fields else ""
            print(f"{pid} ({used_model}) {title_preview}…{retry_note}")
        return out_row, pid, False, None
    except Exception as e:
        return out_row, pid, True, str(e)


def resolve_candidates(
    client,
    *,
    model_name: str | None,
    skip_probe: bool,
    verbose: bool,
) -> list[ModelConfig]:
    if model_name:
        cfg = next((m for m in MODELS if m.name == model_name), None)
        if cfg is None:
            raise ValueError(f"Model không có trong MODELS: {model_name}")
        if verbose:
            print(f"Dùng model cố định: {model_name}\n")
        return [cfg]

    if skip_probe:
        if verbose:
            print("Bỏ probe — dùng toàn bộ MODELS theo thứ tự ưu tiên.\n")
        return list(MODELS)

    candidates, responded = probe_models(client, verbose=verbose)
    if not candidates:
        raise RuntimeError("Không có model nào phản hồi probe.")
    if verbose:
        winner = candidates[0]
        print(
            f"\n→ Model ưu tiên: {winner.name} "
            f"(TTFT: {responded[winner.name]:.3f}s, fallback: {len(candidates) - 1})\n"
        )
    return candidates


def translate_csv(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    resume: bool = True,
    delay_s: float = 0.0,
    model_name: str | None = None,
    skip_probe: bool = False,
    workers: int = 1,
    max_retries: int = 3,
    backoff_base_s: float = 1.5,
    fallback_models: list[str] | None = None,
    cooldown_seconds: float = 20.0,
    verbose: bool = True,
) -> None:
    client = create_client()
    candidates = resolve_candidates(
        client,
        model_name=model_name,
        skip_probe=skip_probe,
        verbose=verbose,
    )

    checkpoint = _checkpoint_path(output_path)
    done_ids = load_done_ids(checkpoint) if resume else set()
    if verbose and done_ids:
        print(f"Resume: bỏ qua {len(done_ids)} sản phẩm (checkpoint).\n")

    with input_path.open(encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("CSV không có header.")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    missing = [k for k in TRANSLATE_FIELD_NAMES if k not in fieldnames]
    if missing:
        raise ValueError(f"CSV thiếu cột: {missing}")

    write_header = not output_path.is_file() or not resume or not done_ids
    out_mode = "w" if write_header else "a"

    processed = 0
    skipped = 0
    errors = 0
    adaptive_state = None
    if model_name:
        adaptive_state = AdaptiveState(
            primary_model=model_name,
            fallback_models=fallback_models or [],
            cooldown_seconds=cooldown_seconds,
        )
        if verbose and fallback_models:
            print(
                "Adaptive fallback: "
                f"primary={model_name}, fallback={fallback_models}, cooldown={cooldown_seconds}s"
            )

    rows_to_process: list[dict] = []
    for row in rows:
        if limit is not None and len(rows_to_process) >= limit:
            break
        pid = (row.get("product_id") or "").strip()
        if resume and pid and pid in done_ids:
            skipped += 1
            continue
        rows_to_process.append(row)

    with output_path.open(out_mode, encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        workers = max(1, int(workers))
        if workers == 1:
            for row in rows_to_process:
                out_row, pid, has_error, error_msg = translate_one_row(
                    row,
                    client=client,
                    candidates=candidates,
                    verbose=verbose,
                    max_retries=max_retries,
                    backoff_base_s=backoff_base_s,
                    adaptive_state=adaptive_state,
                )
                if has_error:
                    errors += 1
                    if verbose:
                        print(f"  ✗ {pid}: {error_msg}", file=sys.stderr)
                writer.writerow(out_row)
                fout.flush()
                if pid:
                    mark_done(checkpoint, pid)
                processed += 1
                if delay_s > 0:
                    time.sleep(delay_s)
        else:
            if verbose:
                print(f"Chạy song song với {workers} workers")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(
                        translate_one_row,
                        row,
                        client=client,
                        candidates=candidates,
                        verbose=False,
                        max_retries=max_retries,
                        backoff_base_s=backoff_base_s,
                        adaptive_state=adaptive_state,
                    ): idx
                    for idx, row in enumerate(rows_to_process)
                }
                pending_results: dict[int, tuple[dict, str, bool, str | None]] = {}
                next_write_idx = 0
                for future in as_completed(future_map):
                    row_idx = future_map[future]
                    pending_results[row_idx] = future.result()

                    while next_write_idx in pending_results:
                        out_row, pid, has_error, error_msg = pending_results.pop(next_write_idx)
                        if has_error:
                            errors += 1
                            if verbose:
                                print(f"  ✗ {pid}: {error_msg}", file=sys.stderr)
                        elif verbose:
                            title_preview = (out_row.get("title") or "")[:55]
                            print(f"[{next_write_idx + 1}] {pid} {title_preview}…")

                        writer.writerow(out_row)
                        fout.flush()
                        if pid:
                            mark_done(checkpoint, pid)
                        processed += 1
                        if delay_s > 0:
                            time.sleep(delay_s)
                        next_write_idx += 1

    if verbose:
        print(
            f"\nXong. Dịch mới: {processed}, bỏ qua (resume): {skipped}, lỗi: {errors}."
        )
        print(f"File: {output_path}")
        print(
            f"Cột dịch: {', '.join(TRANSLATE_FIELD_NAMES)}. "
            f"Giữ nguyên: {', '.join(sorted(PRESERVE_COLUMNS))}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dịch title, description, category, tags sang tiếng Việt (cùng tên cột)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=_ROOT / "merged_products.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=_ROOT / "merged_products_vi.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-base", type=float, default=1.5)
    parser.add_argument(
        "--fallback-models",
        default="openai/gpt-oss-20b,nvidia/nvidia-nemotron-nano-9b-v2",
        help="Danh sách model fallback phân tách bởi dấu phẩy.",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=20.0)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Không tìm thấy file input: {args.input}")

    if args.no_resume:
        if args.output.is_file():
            args.output.unlink()
        cp = _checkpoint_path(args.output)
        if cp.is_file():
            cp.unlink()

    fallback_models = [
        x.strip() for x in (args.fallback_models or "").split(",") if x.strip()
    ]

    translate_csv(
        args.input,
        args.output,
        limit=args.limit,
        resume=not args.no_resume,
        delay_s=args.delay,
        model_name=args.model,
        skip_probe=args.skip_probe,
        workers=args.workers,
        max_retries=args.max_retries,
        backoff_base_s=args.backoff_base,
        fallback_models=fallback_models,
        cooldown_seconds=args.cooldown_seconds,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
