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
        "Dữ liệu: ecommerce.csv (5000 SP) | Model: e5_base_finetuned_5000 + reranker"
    )

    doc.add_heading("1. Tóm tắt", level=1)
    add_table(
        doc,
        ["Chỉ số", "Chỉ Bi-encoder @10", "+ Reranker @10 (n=20)"],
        [
            ["Precision@10", "0,1114", "0,1115"],
            ["Recall@10", "0,9979", "0,9986"],
            ["F1@10", "0,2005", "0,2006"],
            ["MRR@10", "0,9902", "—"],
            ["NDCG@10", "0,9926", "—"],
        ],
    )
    doc.add_paragraph(
        "Precision@10 khoảng 11% không phải do model kém. Đây là giới hạn toán học "
        "của cách đánh giá: mỗi query chỉ có 1 nhãn đúng, top-10 cố định → trần P@10 ≈ 10%. "
        "Recall@10 ≈ 99,8% cho thấy model gần như luôn đưa đúng sản phẩm vào top-10."
    )

    doc.add_heading("2. Khám phá dữ liệu ecommerce.csv", level=1)

    doc.add_heading("2.1. Nguồn và quy mô", level=2)
    doc.add_paragraph(
        "Corpus gồm 5000 sản phẩm từ 5 sàn thương mại điện tử, mỗi nguồn 1000 SP, "
        "phân bố cân bằng. Query đánh giá lấy từ cột title; corpus dùng searchable_text."
    )
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_source_top.png",
        "Hình 1. Phân phối số lượng sản phẩm theo nguồn dữ liệu",
    )

    doc.add_heading("2.2. Giá trị thiếu", level=2)
    add_table(
        doc,
        ["Cột", "Tỷ lệ thiếu (ước lượng)"],
        [
            ["tags", "~84%"],
            ["size", "~60%"],
            ["color_vi", "~51%"],
            ["brand", "~15%"],
            ["description, price, image_url", "0%"],
        ],
    )
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_missing_values.png",
        "Hình 2. Tỷ lệ giá trị thiếu theo cột",
    )
    doc.add_paragraph(
        "Model chủ yếu học từ title và searchable_text; các trường cấu trúc (tags, size, color) thưa."
    )

    doc.add_heading("2.3. Số lượng review", level=2)
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_reviews_count_hist.png",
        "Hình 3. Phân phối số review (log1p)",
    )
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_reviews_count_box.png",
        "Hình 4. Boxplot số review",
    )
    doc.add_paragraph(
        "Phân phối lệch phải: nhiều sản phẩm có 0 review; một số SP có số review cực đoan."
    )

    doc.add_heading("2.4. Rating", level=2)
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_rating_hist.png",
        "Hình 5. Phân phối rating",
    )
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_rating_box.png",
        "Hình 6. Boxplot rating",
    )
    doc.add_paragraph(
        "Hai đỉnh ở rating = 0 và 4–5 sao; median khoảng 4,3."
    )

    doc.add_heading("2.5. Giá sản phẩm", level=2)
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_price_num_hist.png",
        "Hình 7. Phân phối giá sản phẩm (log1p)",
    )
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_price_num_box.png",
        "Hình 8. Boxplot giá sản phẩm",
    )
    doc.add_paragraph(
        "Khoảng giá rất rộng; có outlier cực đoan (có thể do lỗi đơn vị hoặc nhập liệu)."
    )

    doc.add_heading("2.6. Rating vs số review", level=2)
    add_figure(
        doc,
        FIGURES / "c__llm_provider_benchmarking_rating_vs_reviews.png",
        "Hình 9. Mối quan hệ giữa rating và số lượng review",
    )
    doc.add_paragraph(
        "Sản phẩm nhiều review thường có rating cao hơn (hiệu ứng độ phổ biến)."
    )

    doc.add_heading("3. Công việc gần đây: Fine-tune trên tập mới", level=1)

    doc.add_heading("3.1. Chia dữ liệu", level=2)
    add_table(
        doc,
        ["File", "Số mẫu", "Mục đích"],
        [
            ["train_5000.jsonl", "3500", "Train"],
            ["valid_5000.jsonl", "750", "Validation"],
            ["test_5000.jsonl", "750", "Test (tùy chọn)"],
        ],
    )
    doc.add_paragraph(
        "Query: title hoặc câu query tự nhiên trong jsonl. Passage: searchable_text. "
        "Loss: MultipleNegativesRankingLoss (in-batch negatives)."
    )

    doc.add_heading("3.2. Fine-tune E5-base", level=2)
    add_table(
        doc,
        ["Tham số", "Giá trị"],
        [
            ["Base model", "intfloat/multilingual-e5-base"],
            ["Output", "e5_base_finetuned_5000"],
            ["Epoch", "1"],
            ["Batch size", "8"],
            ["Learning rate", "1e-5"],
            ["FP16", "Có"],
            ["Thời gian train", "~4,8 phút (Colab GPU)"],
        ],
    )

    doc.add_heading("3.3. Pipeline đánh giá E5 + Reranker", level=2)
    doc.add_paragraph(
        "Giai đoạn 1: E5 bi-encoder encode query và corpus → cosine similarity → lấy top-n ứng viên.\n"
        "Giai đoạn 2: Cross-encoder reranker chấm điểm cặp (query, passage) → xếp hạng lại → top-k.\n"
        "Script: evaluate_reranker_pipeline.py. Đánh giá trên toàn bộ ecommerce.csv: "
        "corpus 5000 SP, query = title, ground truth = product_id cùng dòng."
    )

    doc.add_heading("4. Kết quả thực nghiệm (Colab T4)", level=1)
    doc.add_paragraph("Bi-encoder only @10:")
    doc.add_paragraph(
        "P@10=0,1114 | R@10=0,9979 | F1@10=0,2005 | MRR@10=0,9902 | NDCG@10=0,9926 | n_queries=4350"
    )
    doc.add_paragraph("+ Reranker (n=20) @10:")
    doc.add_paragraph("P@10=0,1115 | R@10=0,9986 | F1@10=0,2006")
    doc.add_paragraph(
        "Reranker với n=20 chưa cải thiện rõ P/R/F1 so với chỉ bi-encoder; "
        "bi-encoder đã đạt MRR/NDCG rất cao (≈0,99)."
    )

    doc.add_heading("5. Giải thích: Vì sao Precision@10 chỉ ~10%?", level=1)

    doc.add_heading("5.1. Giới hạn toán học", level=2)
    doc.add_paragraph(
        "Cách đánh giá: mỗi query (title) có đúng 1 product_id là ground truth. "
        "Precision@10 = (số hit trong top-10) / 10. "
        "Nếu mỗi query chỉ có 1 SP đúng thì P@10 tối đa = 1/10 = 10%. "
        "Kết quả 11,14% đã gần trần lý thuyết — model hoạt động tốt, không phải chỉ đúng 10%."
    )

    doc.add_heading("5.2. Số liệu từ ecommerce.csv", level=2)
    add_table(
        doc,
        ["Thống kê", "Giá trị"],
        [
            ["Tổng sản phẩm", "5000"],
            ["Title unique", "4350"],
            ["Dòng trùng title", "650"],
            ["Query đánh giá", "4350"],
            ["Query 1 nhãn", "4123 (94,8%)"],
            ["Query nhiều nhãn (cùng title, khác SKU)", "227"],
        ],
    )

    doc.add_heading("5.3. Cách đọc kết quả", level=2)
    add_table(
        doc,
        ["Chỉ số", "Ý nghĩa"],
        [
            ["Recall@10 ≈ 99,8%", "Gần như mọi query đều tìm được SP đúng trong top-10"],
            ["Precision@10 ≈ 11%", "Top-10 có ~1,1 SP đúng nhãn / 10 vị trí (do định nghĩa metric)"],
            ["MRR@10 ≈ 0,99", "SP đúng thường nằm rất cao (vị trí 1–2)"],
        ],
    )
    doc.add_paragraph(
        "Khuyến nghị trình bày GVHD: nhấn mạnh Recall, MRR, NDCG, F1 và so sánh trước/sau rerank; "
        "không dùng P@10 đơn lẻ để kết luận model yếu. "
        "Muốn P@10 phân biệt hơn: mở rộng tập nhãn (cùng danh mục = relevant) "
        "hoặc dùng test_5000.jsonl với query tự nhiên."
    )

    doc.add_heading("6. Hạn chế và hướng phát triển", level=1)
    for item in [
        "650 dòng trùng title → 227 query đa nhãn.",
        "Metadata thưa (tags, size, color_vi).",
        "Giá có outlier → cần làm sạch price_num.",
        "Reranker chậm khi không có flash_attn; grid n lớn mất nhiều giờ.",
        "P@10 bão hòa với setup 1 nhãn/query → ưu tiên báo cáo MRR/NDCG.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("7. Lệnh tái lập thí nghiệm", level=1)
    doc.add_paragraph(
        "python embedding_project/scripts/evaluate_reranker_pipeline.py "
        "--embedding-model embedding_project/models/e5_base_finetuned_5000 "
        "--reranker-model embedding_project/models/reranker "
        "--eval-csv embedding_project/data/ecommerce.csv "
        "--query-col title --n-values 20 50 100 --k-values 5 10 20"
    )
    doc.add_paragraph("Kết quả JSON: embedding_project/outputs/evaluation/reranker_pipeline_eval.json")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(REPORT_OUT))
    print(f"Đã lưu: {REPORT_OUT}")


if __name__ == "__main__":
    main()
