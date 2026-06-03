"""Tạo lại bao_cao_thu_nghiem_e5_base_v2.docx: pretrained (mục 3) + fine-tuned (mục 5)."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH

METRICS_PATH = Path("embedding_project/outputs/evaluation/metrics_e5_base.json")
REPORT_PATH = Path("embedding_project/outputs/evaluation/bao_cao_thu_nghiem_e5_base_v2.docx")


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def pct_delta(before: float, after: float) -> str:
    if before == 0:
        return "—"
    return f"{((after - before) / before) * 100:+.1f}%"


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        table.rows[0].cells[i].text = h
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = val


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def main() -> None:
    data = load_metrics()
    pre = data["pretrained"]
    ft = data["finetuned"]

    doc = Document()

    title = doc.add_heading("BÁO CÁO THỬ NGHIỆM MÔ HÌNH intfloat/multilingual-e5-base", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("Dự án: embedding_project")
    doc.add_paragraph("Ngày cập nhật: 03/06/2026")

    doc.add_heading("1. Mục tiêu", level=1)
    doc.add_paragraph(
        "Đánh giá chất lượng truy hồi ngữ nghĩa của mô hình intfloat/multilingual-e5-base trên tập test đã làm sạch."
    )
    doc.add_paragraph(
        "Ghi nhận kết quả pretrained làm baseline và kết quả sau fine-tune (e5_base_finetuned_final)."
    )

    doc.add_heading("2. Cấu hình thử nghiệm", level=1)
    for item in [
        "Model gốc: intfloat/multilingual-e5-base",
        "Model fine-tuned: embedding_project/models/e5_base_finetuned_final/",
        "Preset: e5-base",
        "Quy ước truy hồi: query: cho câu truy vấn và passage: cho văn bản sản phẩm",
        "Chỉ số: Precision@10, Recall@10, MRR@10, NDCG@10 (K = 10)",
        "Dữ liệu: test_cleaned.jsonl, query_product_labels_cleaned.json, merged_products_vi_cleaned.csv",
        "Pipeline: encode → L2 normalize → cosine (dot product) → top-10 product_id",
        "Script đánh giá: embedding_project/scripts/evaluate_embedding_model.py",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("3. Kết quả đánh giá pretrained", level=1)
    doc.add_paragraph(
        "Kết quả model gốc (chưa fine-tune) trên tập test — làm mốc so sánh."
    )
    add_table(
        doc,
        ["Chỉ số", "Pretrained"],
        [
            ["Precision@10", fmt(pre["Precision@10"], 6)],
            ["Recall@10", fmt(pre["Recall@10"], 6)],
            ["MRR@10", fmt(pre["MRR@10"], 6)],
            ["NDCG@10", fmt(pre["NDCG@10"], 6)],
        ],
    )

    doc.add_heading("4. Nhận xét pretrained", level=1)
    for item in [
        f"Recall@10 = {pre['Recall@10']:.1%} — bao phủ relevant trong top-10 khá tốt.",
        f"MRR@10 = {pre['MRR@10']:.3f}, NDCG@10 = {pre['NDCG@10']:.3f} — xếp hạng ở mức khá.",
        f"Precision@10 = {pre['Precision@10']:.1%} — top-10 vẫn có nhiễu (corpus ~2.000 sản phẩm).",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading(
        "5. Kết quả đánh giá fine-tuned (e5_base_finetuned_final)",
        level=1,
    )
    doc.add_paragraph(
        "Đánh giá model sau fine-tune trên cùng tập test và pipeline (MultipleNegativesRankingLoss, "
        "1 epoch, huấn luyện trên train_cleaned.jsonl / valid_cleaned.jsonl)."
    )

    doc.add_heading("5.1. Bảng metric fine-tuned", level=2)
    add_table(
        doc,
        ["Metric", "Fine-tuned", "Δ (so với pretrained)"],
        [
            ["NDCG@10", fmt(ft["NDCG@10"]), pct_delta(pre["NDCG@10"], ft["NDCG@10"])],
            ["Precision@10", fmt(ft["Precision@10"]), pct_delta(pre["Precision@10"], ft["Precision@10"])],
            ["Recall@10", fmt(ft["Recall@10"]), pct_delta(pre["Recall@10"], ft["Recall@10"])],
            ["MRR@10", fmt(ft["MRR@10"]), pct_delta(pre["MRR@10"], ft["MRR@10"])],
        ],
    )

    doc.add_heading("5.2. Nhận xét fine-tuned", level=2)
    for item in [
        f"NDCG@10: {pre['NDCG@10']:.3f} → {ft['NDCG@10']:.3f} ({pct_delta(pre['NDCG@10'], ft['NDCG@10'])}).",
        f"Recall@10: {pre['Recall@10']:.1%} → {ft['Recall@10']:.1%} — gần như toàn bộ relevant trong top-10.",
        f"MRR@10: {pre['MRR@10']:.3f} → {ft['MRR@10']:.3f} — relevant đầu tiên lên vị trí cao hơn.",
        f"Precision@10: {pre['Precision@10']:.1%} → {ft['Precision@10']:.1%} — cải thiện nhẹ; top-10 vẫn có nhiễu.",
        "Fine-tune giúp metric retrieval tốt hơn pretrained trên cùng pipeline; phù hợp làm model embedding chính cho semantic search.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_paragraph(
        "Nguồn số liệu: embedding_project/outputs/evaluation/metrics_e5_base.json "
        "(đánh giá local, --only-finetuned cho fine-tuned; file JSON có cả pretrained để đối chiếu)."
    )

    doc.add_heading("6. Kết luận", level=1)
    for item in [
        "E5-base pretrained là baseline mạnh cho semantic search đa ngôn ngữ.",
        "Fine-tune trên dữ liệu tiếng Việt cải thiện rõ NDCG, Recall và MRR so với pretrained.",
        "Model triển khai: embedding_project/models/e5_base_finetuned_final/.",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.save(str(REPORT_PATH))
    print(f"Đã ghi: {REPORT_PATH}")


if __name__ == "__main__":
    main()
