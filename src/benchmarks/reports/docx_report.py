"""
Script biên soạn báo cáo Word research-report.docx
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

from src.benchmarks.config import DOCS_DIR

OUTPUT_DOCX = DOCS_DIR / "research-report.docx"


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
                if "Croston_SBA" in str(cell_value) or "ES_bu" in str(cell_value):
                    run.font.bold = True

    if col_widths:
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    doc.add_paragraph()


def build_docx_report():
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
    run_title = title_p.add_run("BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM CÁC PHƯƠNG PHÁP ĐỐI CHỨNG THỐNG KÊ TRONG DỰ BÁO NHU CẦU BÁN LẺ DỮ LIỆU M5")
    run_title.font.name = "Times New Roman"
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(31, 78, 121)

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(18)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Đánh giá định lượng trên 30.490 chuỗi thời gian phân cấp Walmart: Seasonal Naïve, ES_bu, Croston và Croston–SBA")
    run_sub.font.name = "Times New Roman"
    run_sub.font.size = Pt(11.5)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(89, 89, 89)

    def add_h1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        r = h.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = RGBColor(31, 78, 121)

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)

    add_h1("1. TỔNG QUAN VÀ KẾT QUẢ THỰC NGHIỆM")
    add_body("Bảng tổng hợp sai số trung bình trên toàn bộ 30.490 chuỗi sản phẩm–cửa hàng của 10 siêu thị Walmart:")

    res_headers = ["Giai đoạn", "Mô hình dự báo", "MAE trung bình", "RMSE trung bình", "RMSSE trung bình", "Số chuỗi"]
    res_rows = [
        ["Validation", "Seasonal_Naive", "1.2324", "1.7400", "0.9756", "30.490"],
        ["Validation", "Croston", "1.1026", "1.3986", "0.7616", "30.490"],
        ["Validation", "Croston_SBA", "1.0862", "1.3909", "0.7589", "30.490"],
        ["Validation", "ES_bu", "1.0608", "1.3983", "0.7775", "30.490"],
        ["Test", "Seasonal_Naive", "1.2440", "1.7519", "0.9972", "30.490"],
        ["Test", "Croston", "1.0898", "1.3969", "0.7704", "30.490"],
        ["Test", "Croston_SBA", "1.0759", "1.3926", "0.7687", "30.490"],
        ["Test", "ES_bu", "1.0659", "1.4121", "0.7972", "30.490"],
    ]
    add_formatted_table(doc, res_headers, res_rows, col_widths=[1.2, 1.5, 1.1, 1.1, 1.1, 0.9], align_rights=[2, 3, 4])

    try:
        doc.save(str(OUTPUT_DOCX))
        print(f"Đã xuất thành công: {OUTPUT_DOCX}")
    except PermissionError:
        print(f"-> Lưu ý: {OUTPUT_DOCX.name} đang được mở trong Word. Bỏ qua ghi đè.")


if __name__ == "__main__":
    build_docx_report()

