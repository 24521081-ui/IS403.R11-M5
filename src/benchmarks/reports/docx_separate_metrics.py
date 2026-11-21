"""
Biên soạn 2 file Word tách biệt:
- docs/mae-guide.docx
- docs/rmse-guide.docx
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

from src.benchmarks.config import REPORTS_DOCS_DIR


def set_font(run, size=11, bold=False, italic=False, color=None):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color


def create_base_doc():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(0.9)
        s.bottom_margin = Inches(0.9)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
    return doc


def add_title(doc, title_text, subtitle_text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title_text)
    set_font(r, size=16, bold=True, color=RGBColor(31, 78, 121))

    sub = doc.add_paragraph()
    sub.paragraph_format.space_after = Pt(14)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = sub.add_run(subtitle_text)
    set_font(r_sub, size=10.5, italic=True, color=RGBColor(89, 89, 89))


def add_h1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(text)
    set_font(r, size=12.5, bold=True, color=RGBColor(31, 78, 121))


def add_p(doc, text, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        set_font(rb, size=10.5, bold=True)
    r = p.add_run(text)
    set_font(r, size=10.5, italic=italic)
    return p


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        set_font(rb, size=10.5, bold=True)
    r = p.add_run(text)
    set_font(r, size=10.5)
    return p


def add_table(doc, headers, rows, col_widths, align_rights=None):
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="1F4E79"/>')
        cell._tc.get_or_add_tcPr().append(shading)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            set_font(run, size=10, bold=True, color=RGBColor(255, 255, 255))

    for r_idx, r_data in enumerate(rows):
        bg = "F2F5F8" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(r_data):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg}"/>')
            cell._tc.get_or_add_tcPr().append(shading)
            p = cell.paragraphs[0]
            if align_rights and c_idx in align_rights:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in p.runs:
                set_font(run, size=9.5)
                if "ES_bu" in str(val) or "Croston_SBA" in str(val) or "Thắng" in str(val):
                    run.font.bold = True

    for row in table.rows:
        for idx, w in enumerate(col_widths):
            row.cells[idx].width = Inches(w)
    doc.add_paragraph()


def build_mae_doc():
    doc = create_base_doc()
    add_title(
        doc,
        "MAE: MEAN ABSOLUTE ERROR (SAI SỐ TUYỆT ĐỐI TRUNG BÌNH)",
        "Tài liệu Chuyên sâu: Bản chất Toán học, Tối ưu hóa Trung vị và Thực nghiệm M5",
    )

    add_h1(doc, "1. Khái niệm và Nguồn gốc")
    add_p(
        doc,
        "Trong bài toán dự báo chuỗi thời gian bán lẻ M5, tại mỗi thời điểm ngày t, sai số dự báo là e_t = y_t - ŷ_t. "
        "Nếu chỉ cộng dồn sai số đơn thuần, các lỗi đoán thiếu (+5) và đoán thừa (-5) sẽ tự triệt tiêu lẫn nhau thành 0. "
        "Do đó, MAE sử dụng phép lấy trị tuyệt đối |e_t| để biến mọi độ lệch thành số dương trước khi tính trung bình qua h ngày.",
    )
    add_bullet(doc, " Trả lời câu hỏi: \"Trung bình mỗi ngày mô hình đoán lệch khoảng bao nhiêu món?\"", bold_prefix="Ý nghĩa đời thường:")
    add_bullet(doc, " Giữ nguyên đơn vị gốc của chuỗi (sản phẩm/ngày), cực kỳ trực quan với người quản trị kho.", bold_prefix="Đơn vị đo:")
    add_bullet(doc, " Gắn liền với hàm mất mát L1 (Laplace loss, 1774), mang tính chất kháng nhiễu ngoại lai cao.", bold_prefix="Nguồn gốc:")

    add_h1(doc, "2. Vì sao MAE lại tối ưu hóa Trung vị (Median)?")
    add_p(
        doc,
        "Mô hình được tối ưu bằng MAE luôn có xu hướng dự báo tiến về Trung vị (Median) thay vì Trung bình (Mean). "
        "Xét ví dụ một mặt hàng bán ngắt quãng trong 5 ngày với lượng bán thực tế: [0, 0, 0, 0, 10] món. "
        "Trung vị là 0 món, Trung bình là 2 món. Dưới đây là bảng tính điểm khi thử các mức dự báo cố định c:",
    )

    headers_med = ["Giá trị dự báo c", "Độ lệch tuyệt đối 5 ngày", "Tổng sai số", "Điểm MAE", "Kết luận"]
    rows_med = [
        ["c = 0 (Trung vị)", "0 + 0 + 0 + 0 + 10", "10", "2.00 món/ngày", "MAE thấp nhất (Thắng!)"],
        ["c = 1", "1 + 1 + 1 + 1 + 9", "13", "2.60 món/ngày", "Kém hơn"],
        ["c = 2 (Trung bình)", "2 + 2 + 2 + 2 + 8", "16", "3.20 món/ngày", "Kém hơn nhiều"],
        ["c = 3", "3 + 3 + 3 + 3 + 7", "19", "3.80 món/ngày", "Rất kém"],
        ["c = 10 (Cực đại)", "10 + 10 + 10 + 10 + 0", "40", "8.00 món/ngày", "Tệ nhất"],
    ]
    add_table(doc, headers_med, rows_med, [1.5, 2.0, 0.9, 1.1, 1.5])

    add_p(
        doc,
        "=> Nhận xét: Với các chuỗi có trên 70% số ngày bằng 0, mô hình nào dự báo bám sát về 0 sẽ đạt điểm MAE rất tốt, "
        "dù nó hoàn toàn bỏ lỡ ngày có khách mua 10 món.",
    )

    add_h1(doc, "3. Công thức toán học và Giải thích ký hiệu")
    add_p(doc, "MAE = (1 / h) × Tổng |y_t - ŷ_t|  (với t chạy từ T+1 đến T+h, h = 28 ngày)", bold_prefix="• Công thức cho 1 chuỗi: ")
    add_p(doc, "MAE_TB = (1 / N) × Tổng MAE_i  (với N = 30.490 chuỗi sản phẩm trong M5)", bold_prefix="• Công thức trung bình hệ thống: ")
    add_p(
        doc,
        "Trong đó: y_t là lượng bán thực tế; ŷ_t là lượng bán dự báo; h = 28 ngày; |y_t - ŷ_t| là sai số tuyệt đối tại ngày t.",
    )

    add_h1(doc, "4. Kết quả thực nghiệm đối chứng trên dữ liệu Walmart M5")
    add_p(doc, "Bảng sai số MAE trung bình trên 30.490 chuỗi (Trích từ outputs/metrics/summary_metrics_overall.csv):")

    headers_exp = ["Giai đoạn", "Mô hình", "MAE trung bình", "So với Seasonal Naïve", "Đánh giá vai trò"]
    rows_exp = [
        ["Validation", "Seasonal_Naive", "1.2324", "Mốc sàn", "Chuẩn sàn cơ sở"],
        ["Validation", "Croston", "1.1026", "Giảm 10.5%", "Xử lý tốt chuỗi gián đoạn"],
        ["Validation", "Croston_SBA", "1.0862", "Giảm 11.9%", "Khử độ chệch dương"],
        ["Validation", "ES_bu", "1.0608", "Giảm 13.9%", "Tốt nhất toàn hệ thống (Thắng!)"],
        ["Test", "Seasonal_Naive", "1.2440", "Mốc sàn", "Chuẩn sàn cơ sở"],
        ["Test", "Croston", "1.0898", "Giảm 12.4%", "Ổn định"],
        ["Test", "Croston_SBA", "1.0759", "Giảm 13.5%", "Rất sát thực tế"],
        ["Test", "ES_bu", "1.0659", "Giảm 14.3%", "Giữ vững vị trí số 1 về MAE"],
    ]
    add_table(doc, headers_exp, rows_exp, [1.1, 1.4, 1.1, 1.4, 2.0], align_rights=[2])

    add_p(
        doc,
        "Tại sao ES_bu dẫn đầu về MAE? Nhờ bóc tách được thành phần mùa vụ tuần (Seasonal) ở ngành Thực phẩm (FOODS), "
        "nơi diễn ra lượng mua sắm lớn nhất và bám sát nhịp cuối tuần (MAE của ES_bu tại ngành FOODS đạt 1.3794, giảm 14.7% so với Naïve).",
    )

    add_h1(doc, "5. Ứng dụng trong Quản trị Kinh doanh và Báo cáo")
    add_bullet(doc, " Tính toán chi phí lưu kho hoặc hụt hàng trực tiếp theo ngày: Giảm MAE từ 1.23 xuống 1.06 giúp siêu thị cắt giảm hàng nghìn sản phẩm tồn ứ mỗi ngày.", bold_prefix="Quản lý chi phí:")
    add_bullet(doc, " Đặt mốc benchmark cụ thể cho nhóm Machine Learning: Mô hình LightGBM muốn đạt yêu cầu bắt buộc phải có MAE < 1.0608.", bold_prefix="Thước đo chuẩn:")

    out_file = REPORTS_DOCS_DIR / "mae-guide.docx"
    try:
        doc.save(out_file)
        print(f"-> Đã lưu: {out_file}")
    except PermissionError:
        print(f"-> Lưu ý: {out_file.name} đang được mở trong Word. Bỏ qua ghi đè.")


def build_rmse_doc():
    doc = create_base_doc()
    add_title(
        doc,
        "RMSE: ROOT MEAN SQUARED ERROR (CĂN BẬC HAI BÌNH PHƯƠNG SAI SỐ)",
        "Tài liệu Chuyên sâu: Bản chất Toán học, Kiểm soát Rủi ro và Thực nghiệm M5",
    )

    add_h1(doc, "1. Khái niệm và Nguồn gốc")
    add_p(
        doc,
        "RMSE là căn bậc hai của trung bình các sai số bình phương: RMSE = Căn[(1/h) × Tổng (y_t - ŷ_t)²]. "
        "Khác với MAE đối xử công bằng với mọi độ lệch, RMSE phạt theo cấp số nhân các sai số lớn. "
        "Phép bình phương bắt nguồn từ phương pháp Bình phương Bé nhất (Gauss, 1795 & Legendre, 1805) và định lý Gauss-Markov kinh điển.",
    )
    add_bullet(doc, " Trả lời câu hỏi: \"Mô hình có hay đoán sai cú lớn vào những ngày cao điểm không?\"", bold_prefix="Ý nghĩa đời thường:")
    add_bullet(doc, " Phạt nặng sai số lớn (lệch 1 món phạt 1, lệch 10 món phạt 100).", bold_prefix="Đặc tính:")
    add_bullet(doc, " Là tử số trong công thức tính chỉ số chuẩn RMSSE của cuộc thi Walmart M5.", bold_prefix="Cầu nối M5:")

    add_h1(doc, "2. Vì sao Bình phương lại phạt nặng sai số lớn?")
    add_p(
        doc,
        "So sánh hai mô hình cùng có MAE = 2.00 món/ngày trong 3 ngày:\n"
        "• Mô hình A (lệch đều): [2, 2, 2] món => RMSE = Căn[(4+4+4)/3] = 2.00 món/ngày.\n"
        "• Mô hình B (lệch đột biến): [1, 1, 4] món => RMSE = Căn[(1+1+16)/3] = Căn(6) ≈ 2.45 món/ngày.\n"
        "=> Dù MAE bằng nhau, RMSE của mô hình B cao hơn 22.5% do bị phạt nặng ở cú lệch 4 món.",
    )

    add_h1(doc, "3. Bản chất thống kê: RMSE tối ưu hóa Kỳ vọng toán học (Mean)")
    add_p(
        doc,
        "Xét lại ví dụ chuỗi bán ngắt quãng 5 ngày: [0, 0, 0, 0, 10] món. "
        "Dưới đây là bảng tính điểm RMSE khi thử các mức dự báo hằng số c:",
    )

    headers_mean = ["Dự báo hằng số c", "Bình phương sai số 5 ngày", "Tổng MSE", "Điểm RMSE", "Kết luận"]
    rows_mean = [
        ["c = 0 (Trung vị)", "0 + 0 + 0 + 0 + 100", "20.00", "4.47 món/ngày", "Phạt rất nặng cú hụt 10 món"],
        ["c = 1", "4 × 1 + 81 = 85", "17.00", "4.12 món/ngày", "Cải thiện"],
        ["c = 2 (Trung bình)", "4 × 4 + 64 = 80", "16.00", "4.00 món/ngày", "RMSE thấp nhất (Thắng!)"],
        ["c = 3", "4 × 9 + 49 = 85", "17.00", "4.12 món/ngày", "Bắt đầu tăng lại"],
    ]
    add_table(doc, headers_mean, rows_mean, [1.5, 2.0, 0.9, 1.1, 1.5])

    add_p(
        doc,
        "Ý nghĩa quản trị: Trong chuỗi cung ứng, doanh nghiệp thà chấp nhận mỗi ngày tồn kho dư nhẹ 2 món "
        "(chi phí đệm được) còn hơn một ngày bị hụt tới 10 món (cháy hàng, mất khách vĩnh viễn). RMSE đại diện cho tư duy phòng ngừa rủi ro đứt hàng.",
    )

    add_h1(doc, "4. Cầu nối giữa RMSE và RMSSE trong M5")
    add_p(
        doc,
        "Do RMSE phụ thuộc vào quy mô bán (món bán 100 cái có RMSE lớn hơn món bán 1 cái), "
        "M5 chuẩn hóa RMSE bằng cách chia cho độ biến động lịch sử của chính sản phẩm đó:\n"
        "RMSSE = RMSE / Căn(Scale)\n"
        "Trong đó Scale là sai số bình phương trung bình của phương pháp Naive 1 bước trên lịch sử. "
        "Hiểu được RMSE chính là nắm trọn bản chất của RMSSE.",
    )

    add_h1(doc, "5. Kết quả thực nghiệm đối chứng trên dữ liệu Walmart M5")
    add_p(doc, "Bảng sai số RMSE trung bình trên 30.490 chuỗi (Trích từ outputs/metrics/summary_metrics_overall.csv):")

    headers_exp_rmse = ["Giai đoạn", "Mô hình", "RMSE trung bình", "MAE trung bình", "Chênh lệch RMSE - MAE", "Đánh giá an toàn"]
    rows_exp_rmse = [
        ["Validation", "Seasonal_Naive", "1.7400", "1.2324", "+0.5076 (+41.2%)", "Kém nhất (Nhiều ngoại lai)"],
        ["Validation", "Croston", "1.3986", "1.1026", "+0.2960 (+26.8%)", "Ổn định"],
        ["Validation", "Croston_SBA", "1.3909", "1.0862", "+0.3047 (+28.1%)", "Thấp nhất toàn diện (Thắng!)"],
        ["Validation", "ES_bu", "1.3983", "1.0608", "+0.3375 (+31.8%)", "Nhạy cảm ở hàng ngắt quãng"],
        ["Test", "Seasonal_Naive", "1.7519", "1.2440", "+0.5079 (+40.8%)", "Kém nhất"],
        ["Test", "Croston", "1.3969", "1.0898", "+0.3071 (+28.2%)", "Ổn định"],
        ["Test", "Croston_SBA", "1.3926", "1.0759", "+0.3167 (+29.4%)", "Giữ vị trí số 1 về RMSE"],
        ["Test", "ES_bu", "1.4121", "1.0659", "+0.3462 (+32.5%)", "Độ phân tán lớn hơn SBA"],
    ]
    add_table(doc, headers_exp_rmse, rows_exp_rmse, [1.0, 1.3, 1.0, 1.0, 1.5, 1.2], align_rights=[2, 3])

    add_p(
        doc,
        "Tại sao Croston-SBA thắng ở RMSE? Vì mô hình giữ đường dự báo phẳng bám sát kỳ vọng toán học và nhân hệ số hiệu chỉnh 0.95. "
        "Mô hình không bị dao động theo các số 0 ngẫu nhiên, nhờ đó triệt tiêu triệt để các sai số cực đại.",
    )

    add_h1(doc, "6. Ứng dụng Quản trị: Thiết lập Tồn kho An toàn (Safety Stock)")
    add_p(
        doc,
        "Trong vận hành bán lẻ, RMSE là tham số trực tiếp trong công thức tính mức tồn kho an toàn: "
        "Safety Stock = Z × RMSE × Căn(L) (với Z là hệ số dịch vụ, L là thời gian chờ hàng). "
        "Mô hình có RMSE thấp như Croston-SBA giúp siêu thị giảm bớt vốn lưu kho đọng mà vẫn đảm bảo 95% không bị đứt hàng.",
    )

    out_file = REPORTS_DOCS_DIR / "rmse-guide.docx"
    try:
        doc.save(out_file)
        print(f"-> Đã lưu: {out_file}")
    except PermissionError:
        print(f"-> Lưu ý: {out_file.name} đang được mở trong Word. Bỏ qua ghi đè.")


if __name__ == "__main__":
    build_mae_doc()
    build_rmse_doc()

