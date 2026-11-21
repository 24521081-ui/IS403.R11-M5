"""
Script biên soạn file Word (DOCX) giải thích quy trình, bản chất và source code.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from src.benchmarks.config import REPORTS_DOCS_DIR

OUTPUT_DOCX = REPORTS_DOCS_DIR / "model-guide.docx"


def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("w:top", top), ("w:bottom", bottom), ("w:left", left), ("w:right", right)]:
        node = OxmlElement(m)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def add_formatted_table(doc, headers, data_rows, col_widths=None, align_rights=None):
    table = doc.add_table(rows=len(data_rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    hdr_cells = table.rows[0].cells
    for i, header_text in enumerate(headers):
        cell = hdr_cells[i]
        cell.text = header_text
        set_cell_background(cell, "1F4E79")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Times New Roman"
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row_data in enumerate(data_rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F2F5F8" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell_value in enumerate(row_data):
            cell = row_cells[c_idx]
            cell.text = str(cell_value)
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
            p = cell.paragraphs[0]
            if align_rights and c_idx in align_rights:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
                if "HOÀN THÀNH" in str(cell_value) or "Croston_SBA" in str(cell_value) or "ES_bu" in str(cell_value):
                    run.font.bold = True

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()


def build_explanation_docx():
    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(6)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = title_p.add_run("TÀI LIỆU BÁO CÁO TIẾN ĐỘ & GIẢI THÍCH BẢN CHẤT MÃ NGUỒN")
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(16)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(14)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub_p.add_run("Đề tài: Dự báo bán lẻ phân cấp Walmart M5 | Statistical Benchmarks")
    r_sub.font.name = "Times New Roman"
    r_sub.font.size = Pt(10.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(89, 89, 89)

    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(5)
        r = h.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 78, 121)

    def add_p(text, bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Times New Roman"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
        r.font.italic = italic

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Times New Roman"
            rb.font.size = Pt(10)
            rb.font.bold = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)

    add_h1("1. TỔNG KẾT TÌNH TRẠNG TIẾN ĐỘ: ĐÃ HOÀN THÀNH 100%")
    headers_progress = ["Nhiệm vụ được giao", "Tình trạng", "Minh chứng thực tế"]
    data_progress = [
        ["1. Triển khai Croston & Croston-SBA", "HOÀN THÀNH 100%", "src/models/croston.py: Numba JIT khử độ chệch dương."],
        ["2. Triển khai thuật toán ES_bu", "HOÀN THÀNH 100%", "src/models/exponential_smoothing.py: Chu kỳ 7 ngày + Bottom-Up."],
        ["3. Triển khai mô hình Seasonal Naive", "HOÀN THÀNH 100%", "src/models/seasonal_naive.py: Chuẩn sàn tối thiểu."],
        ["4. Kiểm thử đơn vị trên chuỗi mẫu", "HOÀN THÀNH 100%", "scripts/test_models.py: Kiểm thử thành công."],
        ["5. Chạy trên phạm vi chung 10 siêu thị", "HOÀN THÀNH 100%", "scripts/run_benchmark.py: Đủ 30.490 chuỗi x 2 giai đoạn."],
        ["6. Xuất dữ liệu dự báo chuẩn hóa", "HOÀN THÀNH 100%", "outputs/predictions/predictions_validation.csv & test.csv."],
        ["7. Báo cáo đánh giá và tài liệu", "HOÀN THÀNH VƯỢT MỨC", "docs/ (Word & Markdown hoàn chỉnh)."],
    ]
    add_formatted_table(doc, headers_progress, data_progress, col_widths=[2.2, 1.4, 2.9])

    add_h1("2. QUY TRÌNH THỰC HIỆN 5 BƯỚC ĐỜI THƯỜNG")
    add_bullet(" Nạp sales_train_evaluation.csv và tách Validation (d_1886-1913) / Test (d_1914-1941).", bold_prefix="Bước 1 - Chuẩn bị nguyên liệu:")
    add_bullet(" Chạy thử trên 4 chuỗi nhân tạo và 10 chuỗi thực tế để kiểm tra lỗi NaN/âm.", bold_prefix="Bước 2 - Kiểm thử chuỗi mẫu:")
    add_bullet(" Cho 4 mô hình chạy song song bằng Numba JIT (xong 30.490 chuỗi trong vài giây).", bold_prefix="Bước 3 - Dự báo song song:")
    add_bullet(" Chấm điểm bằng MAE, RMSE và RMSSE theo ngành hàng, cửa hàng, mức độ gián đoạn.", bold_prefix="Bước 4 - Đánh giá sai số:")
    add_bullet(" Lưu tệp dự báo 8 cột vào outputs/predictions/ và 4 bảng sai số vào outputs/metrics/.", bold_prefix="Bước 5 - Đóng gói bàn giao:")

    doc.save(OUTPUT_DOCX)
    print(f"Đã xuất thành công: {OUTPUT_DOCX}")


if __name__ == "__main__":
    build_explanation_docx()

