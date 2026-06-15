"""Xuất báo cáo E5 + Reranker (corpus 5000 SP) ra file Word."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

REPO = Path(__file__).resolve().parent.parent
FIGURES = REPO / "embedding_project/outputs/evaluation/figures"
REPORT_OUT = REPO / "embedding_project/outputs/evaluation/bao_cao_e5_reranker_5000.docx"

# Kết quả Colab 14/06/2026 — smoke test 500 query (MAX_QUERIES=500)
EVAL_N_QUERIES = 500
EVAL_CORPUS = 5000
SELECTED_N = 10
TARGET_RECALL = 0.98
RERANK_PAIRS = 5000

BI_ENCODER = {
    "P@10": "0,1002",
    "R@10": "1,0000",
    "F1@10": "0,1821",
    "MRR@10": "0,9990",
    "NDCG@10": "0,9993",
}
RERANKER = {
    "P@10": "0,1002",
    "R@10": "1,0000",
    "F1@10": "0,1821",
    "MRR@10": "0,9990",
    "NDCG@10": "0,9993",
}
RECALL_BY_N = {"10": 1.0, "20": 1.0, "30": 1.0, "50": 1.0, "75": 1.0, "100": 1.0}
K_GRID = [
    ["5", "0,2004", "1,0000", "0,3339", "0,9990", "0,9993"],
    ["10", "0,1002", "1,0000", "0,1821", "0,9990", "0,9993"],
]
OPT_K_NDCG = 5
THRESHOLD = {
    "tau_eer": 0.005,
    "tau_min_err": 0.005,
    "fpr": 0.0,
    "fnr": 0.0,
    "error_rate": 0.0,
    "n_pairs": 2004,
    "n_pos": 501,
    "n_neg": 1503,
}


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val


def add_figure(doc: Document, path: Path, caption: str, width: float = 5.8) -> None:
    if path.is_file():
        doc.add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        doc.add_paragraph(f"[Thiếu hình: {path.name}]")


def main() -> None:
    doc = Document()

    title = doc.add_heading(
        "BÁO CÁO THỬ NGHIỆM: E5 BI-ENCODER + CROSS-ENCODER RERANKER", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Dự án: embedding_project — Tìm kiếm ngữ nghĩa sản phẩm tiếng Việt")
    doc.add_paragraph(f"Ngày báo cáo: {date.today().strftime('%d/%m/%Y')}")
    doc.add_paragraph(
        "Dữ liệu: ecommerce.csv (5000 SP) | Model: e5_base_finetuned_5000 + reranker (XLM-RoBERTa)"
    )

    doc.add_heading("1. Tóm tắt", level=1)
    add_table(
        doc,
        ["Chỉ số", "Chỉ Bi-encoder @10", f"+ Reranker @10 (n={SELECTED_N})"],
        [
            ["Precision@10", BI_ENCODER["P@10"], RERANKER["P@10"]],
            ["Recall@10", BI_ENCODER["R@10"], RERANKER["R@10"]],
            ["F1@10", BI_ENCODER["F1@10"], RERANKER["F1@10"]],
            ["MRR@10", BI_ENCODER["MRR@10"], RERANKER["MRR@10"]],
            ["NDCG@10", BI_ENCODER["NDCG@10"], RERANKER["NDCG@10"]],
        ],
    )
    doc.add_paragraph(
        f"Thí nghiệm mới (Colab T4, {EVAL_N_QUERIES} query / corpus {EVAL_CORPUS}): "
        f"dò Recall@n trước → chọn n={SELECTED_N} (nhỏ nhất đạt target recall {TARGET_RECALL}) "
        f"→ chỉ rerank {RERANK_PAIRS} cặp. Bi-encoder và reranker cho metric giống nhau vì E5 đã xếp đúng từ đầu."
    )
    doc.add_paragraph(
        f"n tối ưu (recall): {SELECTED_N}. k tối ưu (max NDCG): {OPT_K_NDCG}. "
        f"Ngưỡng triển khai τ={THRESHOLD['tau_eer']:.3f} (EER, FPR=FNR=0 trên {THRESHOLD['n_pairs']} cặp mẫu)."
    )
    doc.add_paragraph(
        "Precision@10 ≈ 10% là giới hạn toán học khi mỗi query có 1 nhãn và top-k=10 cố định; "
        "Recall@10 = 100% cho thấy model luôn tìm đúng SP trong top-10."
    )

    doc.add_heading("2. Khám phá dữ liệu ecommerce.csv", level=1)

    doc.add_heading("2.1. Nguồn và quy mô", level=2)
    doc.add_paragraph(
        "Corpus gồm 5000 sản phẩm từ 5 sàn, mỗi nguồn 1000 SP. "
        "Query đánh giá: cột title; corpus: searchable_text."
    )
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_source_top.png",
               "Hình 1. Phân phối số lượng sản phẩm theo nguồn dữ liệu")

    doc.add_heading("2.2. Danh mục (category)", level=2)
    add_figure(doc, FIGURES / "category_top.png",
               "Hình 2. Top category xuất hiện nhiều nhất")
    add_figure(doc, FIGURES / "category_depth.png",
               "Hình 3. Phân phối độ sâu category")
    doc.add_paragraph(
        "Điện thoại thông minh và phụ kiện điện tử chiếm tỷ trọng lớn; "
        "độ sâu category chủ yếu 3–5 cấp."
    )

    doc.add_heading("2.3. Thương hiệu (brand)", level=2)
    add_figure(doc, FIGURES / "brand_top.png",
               "Hình 4. Top thương hiệu xuất hiện nhiều nhất")
    doc.add_paragraph(
        "Nhiều SP thiếu brand (Missing / No Brand); model phụ thuộc title và searchable_text."
    )

    doc.add_heading("2.4. Giá trị thiếu", level=2)
    add_table(
        doc,
        ["Cột", "Tỷ lệ thiếu (ước lượng)"],
        [
            ["tags", "~85%"],
            ["size", "~60%"],
            ["color_vi", "~51%"],
            ["brand", "~15%"],
            ["description, price, image_url", "0%"],
        ],
    )
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_missing_values.png",
               "Hình 5. Tỷ lệ giá trị thiếu theo cột")

    doc.add_heading("2.5. Số lượng review", level=2)
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_reviews_count_hist.png",
               "Hình 6. Phân phối số review (log1p)")
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_reviews_count_box.png",
               "Hình 7. Boxplot số review")

    doc.add_heading("2.6. Rating", level=2)
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_rating_hist.png",
               "Hình 8. Phân phối rating")
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_rating_box.png",
               "Hình 9. Boxplot rating")

    doc.add_heading("2.7. Giá sản phẩm", level=2)
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_price_num_hist.png",
               "Hình 10. Phân phối giá sản phẩm (log1p)")
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_price_num_box.png",
               "Hình 11. Boxplot giá sản phẩm")

    doc.add_heading("2.8. Rating vs số review", level=2)
    add_figure(doc, FIGURES / "c__llm_provider_benchmarking_rating_vs_reviews.png",
               "Hình 12. Mối quan hệ giữa rating và số lượng review")

    doc.add_heading("3. Pipeline và cấu hình thí nghiệm", level=1)
    doc.add_paragraph(
        "Giai đoạn 1: E5 bi-encoder → cosine → Recall@n trên các n ∈ {10,20,30,50,75,100}.\n"
        f"Giai đoạn 2: Chọn n nhỏ nhất đạt Recall ≥ {TARGET_RECALL} → load reranker → chỉ rerank tại n đó.\n"
        "Giai đoạn 3: Grid k ∈ {5,10,20}; phân tích ngưỡng FPR/FNR/EER riêng (--threshold-only)."
    )
    add_table(
        doc,
        ["Tham số", "Giá trị"],
        [
            ["Embedding model", "e5_base_finetuned_5000"],
            ["Reranker", "XLM-RoBERTa cross-encoder"],
            ["target_recall", str(TARGET_RECALL)],
            ["n_search_values", "10, 20, 30, 50, 75, 100"],
            ["k_values", "5, 10, 20"],
            ["eval-k", "10"],
            ["max_queries (smoke test)", str(EVAL_N_QUERIES)],
            ["rerank_batch_size", "32"],
        ],
    )

    doc.add_heading("4. Kết quả đánh giá retrieval", level=1)

    doc.add_heading("4.1. So sánh Bi-encoder vs Reranker @10", level=2)
    doc.add_paragraph(
        f"P@10={BI_ENCODER['P@10']} | R@10={BI_ENCODER['R@10']} | F1@10={BI_ENCODER['F1@10']} | "
        f"MRR@10={BI_ENCODER['MRR@10']} | NDCG@10={BI_ENCODER['NDCG@10']}"
    )
    doc.add_paragraph(
        "Sau rerank (n=10): metric trùng bi-encoder — reranker không cải thiện thêm "
        "vì E5 đã đạt Recall/MRR/NDCG ≈ 1,0 trên mẫu này."
    )

    doc.add_heading("4.2. Chọn n theo Recall@n", level=2)
    recall_rows = [[str(n), f"{v:.4f}".replace(".", ",")] for n, v in RECALL_BY_N.items()]
    add_table(doc, ["n", f"Recall@n"], recall_rows)
    doc.add_paragraph(
        f"→ selected_n = {SELECTED_N} (nhỏ nhất đạt target). "
        f"Rerank {EVAL_N_QUERIES} × {SELECTED_N} = {RERANK_PAIRS} cặp (thay vì 50.000 nếu rerank n=100)."
    )

    doc.add_heading("4.3. Tìm k tối ưu (grid tại n=10)", level=2)
    add_table(
        doc,
        ["k", "P", "R", "F1", "MRR", "NDCG"],
        K_GRID,
    )
    doc.add_paragraph(
        f"k tối ưu theo NDCG: k={OPT_K_NDCG} (P@5=0,20, F1@5=0,33, NDCG≈0,999). "
        "Báo cáo chính @10: k=10."
    )

    doc.add_heading("5. Ngưỡng tối ưu trước triển khai", level=1)
    doc.add_paragraph(
        "Phân tích trên cặp (query, passage) positive/negative: "
        f"{THRESHOLD['n_pairs']} cặp ({THRESHOLD['n_pos']} pos, {THRESHOLD['n_neg']} neg). "
        "Mỗi query: 1 positive (searchable_text đúng) + 3 negative ngẫu nhiên."
    )
    add_table(
        doc,
        ["Loại ngưỡng", "τ", "FPR", "FNR", "Error rate"],
        [
            ["EER (FPR ≈ FNR)", f"{THRESHOLD['tau_eer']:.4f}", "0,0000", "0,0000", "0,0000"],
            ["Min error rate", f"{THRESHOLD['tau_min_err']:.4f}", "0,0000", "0,0000", "0,0000"],
        ],
    )
    doc.add_paragraph(
        f"Triển khai: lọc kết quả reranker với score ≥ {THRESHOLD['tau_eer']:.3f}. "
        "Vùng τ ∈ [0,01; 0,99] đều cho error rate = 0 trên tập mẫu này. "
        "Lưu ý: negative ngẫu nhiên dễ tách hơn negative cứng từ top-n E5."
    )

    doc.add_heading("6. Giải thích Precision@10", level=1)
    doc.add_paragraph(
        "Mỗi query có 1 nhãn đúng, Precision@10 = hit/10 ≤ 10%. "
        "Kết quả ~10% là gần trần lý thuyết khi Recall = 100%. "
        "Nên báo cáo thêm MRR, NDCG, F1 và so sánh trước/sau rerank."
    )
    add_table(
        doc,
        ["Thống kê corpus", "Giá trị"],
        [
            ["Tổng SP", "5000"],
            ["Title unique", "4350"],
            ["Query đánh giá (full)", "4350"],
            ["Query smoke test", str(EVAL_N_QUERIES)],
        ],
    )

    doc.add_heading("7. Hạn chế", level=1)
    for item in [
        f"Smoke test {EVAL_N_QUERIES} query — cần chạy MAX_QUERIES=None để chốt full 4350 query.",
        "Reranker không cải thiện khi bi-encoder đã Recall/MRR ≈ 1.",
        "Ngưỡng τ đo trên negative ngẫu nhiên — nên bổ sung hard negative từ top-n E5.",
        "Metadata thưa (tags, brand); giá có outlier.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("8. Tái lập", level=1)
    doc.add_paragraph(
        "Notebook: embedding_project/notebooks/evaluate_e5_reranker_5000_colab.ipynb\n"
        "Script: embedding_project/scripts/evaluate_reranker_pipeline.py"
    )
    doc.add_paragraph(
        "Eval: --target-recall 0.98 --n-search-values 10 20 30 50 75 100 --skip-threshold\n"
        "Ngưỡng: run_threshold_only (cell 5 trong notebook)"
    )

    doc.add_heading("Phụ lục — Danh mục hình", level=1)
    figures = [
        ("Hình 1", "source_top", "c__llm_provider_benchmarking_source_top.png"),
        ("Hình 2", "category_top", "category_top.png"),
        ("Hình 3", "category_depth", "category_depth.png"),
        ("Hình 4", "brand_top", "brand_top.png"),
        ("Hình 5", "missing_values", "c__llm_provider_benchmarking_missing_values.png"),
        ("Hình 6–7", "reviews_count", "reviews_count_hist/box"),
        ("Hình 8–9", "rating", "rating_hist/box"),
        ("Hình 10–11", "price", "price_num_hist/box"),
        ("Hình 12", "rating_vs_reviews", "c__llm_provider_benchmarking_rating_vs_reviews.png"),
    ]
    add_table(doc, ["STT", "Nội dung", "File"], [[a, b, c] for a, b, c in figures])

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(REPORT_OUT))
    print(f"Đã lưu: {REPORT_OUT}")


if __name__ == "__main__":
    main()
