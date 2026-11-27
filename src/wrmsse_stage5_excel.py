"""Stream the official workbook and verify it against the underlying tables."""

from datetime import date, datetime
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import openpyxl
import pandas as pd
import pyarrow.parquet as pq
import xlsxwriter

from .wrmsse_stage4 import ROWS, read, verify_files
from .wrmsse_stage5 import (
    DETAIL, DETAIL_COLUMNS, REPORTS, ROOT, SELECTED, STAGE4, TABLES, components, csv, rel,
)

WORKBOOK = REPORTS / "RandomForest_Forecast_Actual.xlsx"
SHEETS = (
    ("Summary", "Tổng quan mô hình và kết quả", "summary"),
    ("Model_Comparison", "Chỉ số Test theo mô hình", "model_comparison"),
    ("Daily_Sales", "Doanh số và sai số theo 28 ngày Test", "daily_sales"),
    ("State_Analysis", "Phân tích theo bang", "state_analysis"),
    ("Store_Analysis", "Phân tích theo cửa hàng", "store_analysis"),
    ("Category_Analysis", "Phân tích theo ngành hàng", "category_analysis"),
    ("Department_Analysis", "Phân tích theo bộ phận", "department_analysis"),
    ("Weekly_Analysis", "Phân tích theo tuần Test", "weekly_analysis"),
    ("Weekday_Analysis", "Phân tích theo thứ trong tuần", "weekday_analysis"),
    ("SNAP_Analysis", "Sai số theo trạng thái SNAP", "snap_analysis"),
    ("SNAP_By_State", "Sai số theo bang và trạng thái SNAP", "snap_by_state"),
    ("Store_Department", "Sai số theo cửa hàng × bộ phận", "store_department"),
    ("WRMSSE_12_Levels", "WRMSSE và đóng góp theo 12 cấp", "wrmsse_12_levels"),
    ("Top_Underforecast", "Top 20 chuỗi dự báo thiếu theo lượng bán", "top_underforecast"),
    ("Top_Overforecast", "Top 20 chuỗi dự báo thừa theo lượng bán", "top_overforecast"),
    ("Top_WRMSSE_Series", "Top 20 chuỗi item × store đóng góp WRMSSE", "top_wrmsse_contribution"),
    ("Top_Items_Sales", "Top 20 sản phẩm theo lượng bán thực tế", "top_items_sales"),
    ("Top_Series_Sales", "Top 20 chuỗi theo lượng bán thực tế", "top_series_sales"),
    ("Representative_Series", "Các chuỗi đại diện theo tiêu chí đã công bố", "representative_series"),
    ("Feature_Importance", "Importance theo 15 đặc trưng logic", "feature_importance"),
    ("Experiment_Results", "36 lượt thực nghiệm Validation", "experiment_results"),
    ("Forecast_Detail", "Test d_1914–d_1941 · chi tiết item–store–day", "detail"),
    ("Data_Dictionary", "Từ điển dữ liệu và cách đọc các chỉ số", "dictionary"),
)


def summary():
    metrics = read(STAGE4 / "test_metrics.json")
    model = read(STAGE4 / "final_model_config.json")
    sample = read(STAGE4 / "final_training_sample_manifest.json")
    selector = read(SELECTED)
    levels = csv(TABLES / "wrmsse_12_levels.csv")
    return pd.DataFrame([
        ("Mô hình chính thức", model["candidate"], "Cấu hình khóa từ Validation"),
        ("Train", "d_1–d_1885", "Lịch sử trước Validation"),
        ("Validation", "d_1886–d_1913", "Chọn theo WRMSSE đầy đủ 12 cấp"),
        ("Train + Validation", "d_1–d_1913", "Phạm vi lấy mẫu huấn luyện cuối"),
        ("Test", "d_1914–d_1941", "Gốc d_1913, dự báo cố định 28 ngày"),
        ("Ngày bắt đầu Test", metrics["test_dates"][0], ""),
        ("Ngày kết thúc Test", metrics["test_dates"][1], ""),
        ("Chuỗi sản phẩm × cửa hàng", metrics["bottom_series"], ""),
        ("Quan sát Test", metrics["observations"], "item–store–day"),
        ("Mẫu fit cuối", sample["rows"], "Lấy mẫu phân tầng cửa hàng × bộ phận × quý"),
        ("Mẫu Train gốc", sample["train_rows"], ""),
        ("Mẫu Validation bổ sung", sample["validation_rows"], ""),
        ("Đặc trưng logic", model["logical_features"], ""),
        ("Cột mã hóa", model["encoded_features"], "CSR float32"),
        ("WRMSSE Validation RF", selector["wrmsse"], "Chọn RF-15 + FE-04"),
        ("WRMSSE Validation Seasonal Naive", selector["seasonal_naive_wrmsse"], ""),
        ("WRMSSE Test RF", metrics["models"]["random_forest"]["wrmsse"], "12 cấp M5"),
        ("WRMSSE Test Seasonal Naive", metrics["models"]["seasonal_naive"]["wrmsse"], "12 cấp M5"),
        ("Giảm WRMSSE tuyệt đối", metrics["wrmsse_reduction"], "Seasonal Naive trừ Random Forest"),
        ("Cải thiện WRMSSE Test (%)", metrics["improvement_percent"], "So với Seasonal Naive"),
        ("Số cấp RF có WRMSSE thấp hơn", int((levels.rf_wrmsse_within_level < levels.sn_wrmsse_within_level).sum()), ""),
        ("Tổng bán thực tế", metrics["models"]["random_forest"]["actual_sales"], "Đơn vị bán"),
        ("WAPE RF (%)", metrics["models"]["random_forest"]["wape_percent"], "item–store–day"),
        ("Bias RF (%)", metrics["models"]["random_forest"]["forecast_bias_percent"], "Âm = dự báo thiếu"),
        ("Giới hạn", "Test từng là Validation của nghiên cứu cũ", ""),
        ("Nguồn Test", rel(DETAIL), "Artifact chính thức Stage 4 đã khóa SHA-256"),
    ], columns=["chỉ_tiêu", "giá_trị", "ghi_chú"])


def dictionary():
    rows = [
        ("Quy ước", "rf_ / sn_", "RF-15 + FE-04 / Seasonal Naive chu kỳ bảy ngày."),
        ("Phạm vi", "Test", "d_1914–d_1941, 25/04/2016–22/05/2016."),
        ("Kiểu số", "%", "Cột phần trăm lưu theo thang 0–100. weight và importance dạng tỷ trọng lưu theo thang 0–1."),
        ("Khóa", "series_id", "Mã chuỗi sản phẩm × cửa hàng trong lưới nguồn, bắt đầu từ 0."),
        ("Khóa", "item_id / store_id", "Mã sản phẩm / mã cửa hàng."),
        ("Khóa", "dept_id / cat_id / state_id", "Mã bộ phận / ngành hàng / bang."),
        ("Thời gian", "day / date", "Chỉ số ngày M5 / ngày dương lịch."),
        ("Thời gian", "week / week_start / week_end", "Tuần trong Test, gồm bảy ngày liên tiếp tính từ ngày đầu tiên."),
        ("Thời gian", "weekday / weekday_no", "Thứ trong tuần. Thứ Hai = 1, Chủ nhật = 7."),
        ("Thời gian", "actual_change / actual_change_pct", "Tổng lượng bán thay đổi so với ngày trước, theo đơn vị và %."),
        ("Doanh số", "actual_sales", "Lượng bán thực tế theo đơn vị. Đây là lượng bán quan sát được."),
        ("Doanh số", "random_forest_forecast / seasonal_naive_forecast", "Dự báo dạng số thực nguồn Stage 4, định dạng hiển thị hai chữ số thập phân."),
        ("Sai lệch", "error / sales_difference", "Forecast trừ Actual. Trên nhóm, là tổng sai lệch có dấu."),
        ("Sai lệch", "absolute_error", "Tổng abs(Forecast − Actual) trên các quan sát trong nhóm."),
        ("Sai lệch", "net_absolute_error / abs_aggregate_error", "Độ lớn của tổng sai lệch có dấu. Có thể nhỏ do bù trừ."),
        ("Sai lệch", "squared_error", "Tổng bình phương sai lệch, thành phần tính RMSE."),
        ("Chỉ số", "mae / rmse", "Trung bình sai số tuyệt đối / căn trung bình sai số bình phương. Đơn vị bán."),
        ("Chỉ số", "wape_pct / wape_percent", "100 × tổng tuyệt đối sai số / tổng Actual. Ô trống khi tổng Actual bằng 0."),
        ("Chỉ số", "bias_pct / forecast_bias_percent", "100 × tổng sai lệch / tổng Actual. Âm là thiếu dự báo, dương là thừa dự báo."),
        ("Chỉ số", "overforecast_percent / underforecast_percent", "Phần dương của Bias / độ lớn phần âm của Bias."),
        ("Chỉ số", "improvement_pct", "100 × (chỉ số Seasonal Naive − chỉ số RF) / chỉ số Seasonal Naive."),
        ("Chỉ số", "observations / metric_grain", "Số quan sát và cấp tính chỉ số. Mặc định item–store–day."),
        ("Chỉ số", "actual_share_pct", "Tỷ trọng lượng bán thực tế của nhóm trong toàn phạm vi đang tổng hợp."),
        ("Cửa hàng", "store_day_*", "Cộng lượng bán và dự báo từng cửa hàng–ngày trước khi tính sai số."),
        ("Phân cấp", "rmsse", "Điểm của một chuỗi. Căn MSE Test chia scale, ở bên trong căn."),
        ("Phân cấp", "scale / scale_status", "Thang đo chuỗi từ sai phân lịch sử sau lần bán đầu. Trạng thái do evaluator trả về."),
        ("Phân cấp", "weight", "Tỷ trọng doanh thu lịch sử đã chia đều 12 cấp. Mỗi cấp tổng trọng số 1/12."),
        ("Phân cấp", "contribution", "weight × RMSSE. Bằng 0 cho chuỗi trọng số 0."),
        ("Phân cấp", "wrmsse_within_level", "12 × tổng contribution của cấp. WRMSSE chung là tổng contribution của 12 cấp."),
        ("Phân cấp", "level / hierarchy / n_series", "Số hiệu, tên cấp và số chuỗi ở cấp đó."),
        ("Lịch", "snap_active", "Cờ lịch được xác định từ bang và ngày trong tháng theo quy tắc đã khóa."),
        ("Đặc trưng", "importance / importance_pct", "Mức giảm impurity gộp theo đặc trưng logic, không mang nghĩa nhân quả."),
        ("Thực nghiệm", "reuse / eligible_for_selection", "Dùng lại lượt fit tương ứng / đủ điều kiện trên Validation."),
        ("Thực nghiệm", "validation_wrmsse / selection_status", "Điểm WRMSSE và kết luận hậu kiểm Validation."),
        ("Nguồn", "artifact_path", "Thư mục artifact lưu request, model và điểm của lượt fit được đối chiếu."),
    ]
    return pd.DataFrame(rows, columns=["nhóm", "trường", "diễn_giải"])


def small_tables():
    tables = {name: csv(TABLES / f"{key}.csv") for name, _, key in SHEETS if key not in ("summary", "detail", "dictionary")}
    tables["Summary"] = summary()
    tables["Data_Dictionary"] = dictionary()
    return tables


def formats(book):
    return {
        "title": book.add_format({"bold": True, "font_size": 15, "font_color": "white", "bg_color": "#183349", "valign": "vcenter"}),
        "header": book.add_format({"bold": True, "font_color": "white", "bg_color": "#17675E", "valign": "vcenter"}),
        "numeric": book.add_format({"num_format": "#,##0.00;[Red](#,##0.00);0.00"}),
        "integer": book.add_format({"num_format": "#,##0"}),
        "precise": book.add_format({"num_format": "0.0000000000"}),
        "percent": book.add_format({"num_format": '0.00"%"'}),
        "fraction": book.add_format({"num_format": "0.00%"}),
        "risk": book.add_format({"bg_color": "#FCE6D7", "font_color": "#812A09"}),
        "surplus": book.add_format({"bg_color": "#DEF1ED", "font_color": "#145A51"}),
    }


def header(sheet, title, columns, count, style):
    sheet.set_tab_color("#17675E")
    sheet.set_row(0, 27)
    sheet.merge_range(0, 0, 0, len(columns) - 1, title, style["title"])
    sheet.set_row(1, 22)
    sheet.write_row(1, 0, list(columns), style["header"])
    sheet.freeze_panes(2, 1)
    sheet.autofilter(1, 0, count + 1, len(columns) - 1)


def value_format(col, style):
    if "pct" in col or col.endswith("_percent"):
        return style["percent"]
    if col.endswith("weight") or col == "importance":
        return style["fraction"]
    if any(key in col for key in ("wrmsse", "rmsse", "contribution", "scale")) and not col.endswith("status"):
        return style["precise"]
    if col in {"series_id", "day", "week", "level", "observations", "n_series", "actual_sales", "logical", "encoded", "n_jobs"}:
        return style["integer"]
    return style["numeric"]


def write_small(sheet, title, table, style):
    header(sheet, title, table.columns, len(table), style)
    for i, col in enumerate(table.columns):
        longest = max(table[col].fillna("").astype(str).map(len), default=0)
        max_width = 100 if col in ("diễn_giải", "ghi_chú", "giá_trị") else 48
        sheet.set_column(i, i, min(max_width, max(15, len(col) + 3, min(longest + 2, max_width))),
                         value_format(col, style) if pd.api.types.is_numeric_dtype(table[col]) else None)
        if col.endswith(("bias_pct", "bias_percent")):
            for operator, color in (("<", "risk"), (">", "surplus")):
                sheet.conditional_format(2, i, len(table) + 1, i, {
                    "type": "cell", "criteria": operator, "value": 0, "format": style[color],
                })
    for row, values in enumerate(table.itertuples(index=False, name=None), 2):
        for col, value in enumerate(values):
            if pd.isna(value):
                continue
            direct_format = None
            if sheet.name == "Summary" and col == 1 and isinstance(value, (int, float)):
                label = values[0]
                direct_format = style["percent" if "(%)" in label else "precise" if "WRMSSE" in label else "integer"]
            sheet.write(row, col, value, direct_format)


def detail_batches():
    source = pq.ParquetFile(DETAIL)
    for batch in source.iter_batches(batch_size=16384):
        yield components(batch.to_pandas())[list(DETAIL_COLUMNS)]


def write_detail(sheet, title, style):
    assert ROWS + 2 < 1_048_576
    header(sheet, title, DETAIL_COLUMNS, ROWS, style)
    for position, column in enumerate(DETAIL_COLUMNS):
        sheet.set_column(position, position, max(13, len(column) + 2),
                         value_format(column, style) if position != 7 else None)
    dates = {value.date().isoformat(): value.to_pydatetime()
             for value in pd.date_range("2016-04-25", "2016-05-22")}
    current = 2
    for batch in detail_batches():
        for row in batch.itertuples(index=False, name=None):
            sheet.write_row(current, 0, (*row[:7], dates[row[7]], *row[8:]))
            current += 1
    assert current == ROWS + 2


def create_workbook():
    verify_files(ROOT, read(REPORTS / "stage5_inputs.json")["table_sha256"])
    if WORKBOOK.exists():
        raise FileExistsError(WORKBOOK)
    tables = small_tables()
    with TemporaryDirectory(prefix="stage5_xlsx_", dir=REPORTS) as directory:
        path = REPORTS / (WORKBOOK.stem + ".partial.xlsx")
        if path.exists():
            raise FileExistsError(path)
        with xlsxwriter.Workbook(str(path), {
            "constant_memory": True, "tmpdir": directory, "default_date_format": "dd/mm/yyyy",
            "strings_to_urls": False, "strings_to_formulas": False,
        }) as book:
            book.set_properties({"title": "Dự báo và lượng bán thực tế M5", "subject": "RF-15 + FE-04 · Test"})
            style = formats(book)
            for name, title, key in SHEETS:
                sheet = book.add_worksheet(name)
                if key == "detail":
                    write_detail(sheet, title, style)
                else:
                    write_small(sheet, title, tables[name], style)
        path.rename(WORKBOOK)


def xml_rows(stream):
    pending = b""
    while block := stream.read(2**20):
        parts = (pending + block).split(b"</row>")
        pending = parts.pop()
        yield from parts
    assert b"</worksheet>" in pending


NUMS = re.compile(rb'<c r="([AGHI-O])\d+"[^>]*><v>([^<]+)</v></c>')
STRINGS = re.compile(rb'<c r="([B-F])\d+"[^>]*><is><t>([^<]+)</t></is></c>')


def check_detail_xml():
    """Compare every full-grid row with Parquet, including numeric Excel round-trip."""
    sheet = 1 + [name for name, _, _ in SHEETS].index("Forecast_Detail")
    with ZipFile(WORKBOOK) as archive, archive.open(f"xl/worksheets/sheet{sheet}.xml") as stream:
        rows = xml_rows(stream)
        assert b"chi ti" in next(rows) and b"series_id" in next(rows)
        count = 0
        for batch in detail_batches():
            numeric = np.empty((len(batch), 10), dtype=float)
            text = []
            for i in range(len(batch)):
                row = next(rows)
                values = NUMS.findall(row)
                strings = STRINGS.findall(row)
                assert b"".join(key for key, _ in values) == b"AGHIJKLMNO"
                assert b"".join(key for key, _ in strings) == b"BCDEF"
                numeric[i] = [float(value) for _, value in values]
                text.append([value.decode("ascii") for _, value in strings])
            expected_dates = (pd.to_datetime(batch.date) - pd.Timestamp("1899-12-30")).dt.days
            expected = np.column_stack([batch.series_id, batch.day, expected_dates, batch.iloc[:, 8:].to_numpy()])
            np.testing.assert_allclose(numeric, expected, atol=1e-12, rtol=1e-14)
            np.testing.assert_array_equal(text, batch.iloc[:, 1:6].to_numpy())
            count += len(batch)
        assert next(rows, None) is None and count == ROWS
    return count


def verify_workbook():
    expected = small_tables()
    counts = {}
    workbook = openpyxl.load_workbook(WORKBOOK, read_only=True, data_only=True)
    try:
        assert workbook.sheetnames == [name for name, _, _ in SHEETS]
        for name, title, key in SHEETS:
            sheet = workbook[name]
            assert sheet["A1"].value == title
            if key == "detail":
                assert sheet.max_row == ROWS + 2 and sheet.max_column == len(DETAIL_COLUMNS)
                assert tuple(cell.value for cell in sheet[2]) == DETAIL_COLUMNS
                assert isinstance(sheet["H3"].value, (date, datetime))
                counts[name] = ROWS
                continue
            table = expected[name]
            assert sheet.max_row == len(table) + 2 and sheet.max_column == len(table.columns)
            assert tuple(cell.value for cell in sheet[2]) == tuple(table.columns)
            for row, expected_row in zip(sheet.iter_rows(min_row=3, values_only=True), table.itertuples(index=False, name=None)):
                for value, target in zip(row, expected_row):
                    if pd.isna(target) or target == "":
                        assert value is None or value == ""
                    elif isinstance(target, (float, np.floating)):
                        np.testing.assert_allclose(value, target, rtol=1e-14, atol=1e-12)
                    else:
                        assert value == target, (name, value, target)
            counts[name] = len(table)
    finally:
        workbook.close()
    assert check_detail_xml() == counts["Forecast_Detail"]
    return counts
