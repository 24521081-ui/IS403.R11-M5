"""Extend the business workbook and reconcile it before atomic publication."""

from contextlib import ExitStack
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import openpyxl
import pandas as pd
import xlsxwriter

from .wrmsse_demand_groups import GROUP
from .wrmsse_demand_figures import TIME_TITLES
from .wrmsse_stage5 import DETAIL_COLUMNS, REPORTS, components
from .wrmsse_stage5_excel import SHEETS, WORKBOOK, formats, header, small_tables, value_format, write_small, xml_rows

EXTRA = {"summary": "Cơ cấu và kết quả theo gốc dự báo", "metrics": "Sai số và đóng góp cấp 12",
         "store": "Cửa hàng × nhóm nhu cầu · Test", "category": "Ngành hàng × nhóm nhu cầu · Test",
         "adi": "Sai số theo khoảng ADI · Test",
         "daily": "Lượng bán, sai số và lũy kế từng ngày · Test",
         "weekly": "Chỉ số theo tuần · Test", "weekday": "Chỉ số theo thứ · Test"}
FIGURE_SHEET = "Demand_Group_Figures"
FIGURES = ["metrics", *TIME_TITLES, "examples"]
DICT = [
    ("Phân nhóm", "ADI", "Số ngày từ lần bán dương đầu tiên đến gốc (kể cả hai đầu) chia số ngày bán dương. Đơn vị: ngày."),
    ("Phân nhóm", "demand_group", "frequent khi ADI < 1,32, intermittent khi ADI ≥ 1,32. Gốc 1885 cho Validation, 1913 cho Test."),
    ("Doanh thu", "revenue_28d / revenue_share_pct", "Doanh thu và tỷ trọng (%) của nhóm trong 28 ngày kết thúc ở gốc dự báo."),
    ("Tần suất", "zero_day_pct / units_per_selling_day", "Tỷ lệ dòng có Actual bằng 0 (%) và số đơn vị bình quân trên dòng có bán."),
    ("Quy mô", "units_per_series_28d", "Tổng Actual trong kỳ đánh giá chia số chuỗi trong nhóm."),
    ("Sai số", "*_improvement_pct", "100 × (SN − RF) / SN theo từng chỉ số. Với Bias dùng trị tuyệt đối của Bias nhóm."),
    ("So sánh chuỗi", "rf_win_pct / tie_pct / rf_loss_pct", "So sánh MAE của từng chuỗi trên 28 ngày, hòa trong khoảng ±10⁻⁹ đơn vị."),
    ("WRMSSE", "L12_weight / *_L12_contribution", "Tổng trọng số / tổng (trọng số × RMSSE) của chuỗi cấp 12 trong nhóm. Tổng trọng số cấp 12 bằng 1/12."),
    ("RMSSE", "*_conditional_weighted_rmsse", "Đóng góp cấp 12 của nhóm chia tổng trọng số nhóm. Mẫu số 0 trả về ô trống."),
    ("Khoảng ADI", "ADI_bin", "Khoảng đóng trái, mở phải. +∞ và NA có dòng riêng. Ô chỉ số trống khi mẫu số bằng 0."),
    ("RMSSE", "*_mean_rmsse / *_valid_rmsse_series", "Trung bình không trọng số và số chuỗi có RMSSE xác định ở cấp 12. RMSSE = √(MSE chuỗi / scale lịch sử)."),
    ("Lũy kế", "*_cumulative", "Tổng từ 25/04/2016 đến hết ngày đang xét. rf_error_cumulative là tổng Forecast trừ tổng Actual, đơn vị bán."),
    ("Phạm vi", "Demand_Group_Daily / Weekly / Weekday", "Tính MAE, RMSE, WAPE và Bias từ các dòng chuỗi–ngày trong từng nhóm và kỳ. Sai lệch ròng có thể bù trừ giữa các chuỗi."),
]


def sheet_name(key):
    return "Demand_Group_" + ("ADI" if key == "adi" else key.title())


def extended_detail(sheet, title, style, detail):
    cols = [*DETAIL_COLUMNS, "ADI", GROUP]
    header(sheet, title, cols, len(detail), style)
    for i, name in enumerate(cols):
        sheet.set_column(i, i, max(13, len(name) + 2), value_format(name, style) if i != 7 else None)
    dates = {key: pd.Timestamp(key).to_pydatetime() for key in detail.date.unique()}
    for start in range(0, len(detail), 16384):
        chunk = components(detail.iloc[start:start + 16384])[cols]
        for row_id, row in enumerate(chunk.itertuples(index=False, name=None), start + 2):
            sheet.write_row(row_id, 0, (*row[:7], dates[row[7]], *row[8:]))


def verify_extension(previous, current, detail, tables):
    with ExitStack() as stack:
        old, new = [openpyxl.load_workbook(path, read_only=True, data_only=True) for path in (previous, current)]
        for book in (old, new):
            stack.callback(book.close)
        base_names = [s[0] for s in SHEETS]
        extended_names = base_names + [sheet_name(k) for k in EXTRA] + [FIGURE_SHEET]
        assert old.sheetnames in (base_names, extended_names[:len(base_names)+5], extended_names)
        assert new.sheetnames == extended_names
        already_extended = old.sheetnames != base_names
        for name in base_names:
            if name == "Forecast_Detail":
                assert new[name].max_row == old[name].max_row == len(detail) + 2
                assert tuple(c.value for c in new[name][2]) == (*DETAIL_COLUMNS, "ADI", GROUP)
                continue
            original = list(old[name].values)
            updated = list(new[name].values)
            if name == "Data_Dictionary":
                assert updated[:len(original)] == original and updated[-len(DICT):] == DICT
            else:
                assert updated == original, name
        for key in EXTRA:
            rows = list(new[sheet_name(key)].values)
            assert list(rows[1]) == list(tables[key].columns)
            stored = pd.DataFrame(rows[2:], columns=rows[1])
            pd.testing.assert_frame_equal(stored, tables[key].reset_index(drop=True),
                                          check_dtype=False, atol=1e-12, rtol=1e-14)
    with ZipFile(current) as package:
        embedded = [package.read(name) for name in package.namelist() if name.startswith("xl/media/")]
        assert sorted(embedded) == sorted((REPORTS / "figures" / f"demand_group_{slug}.png").read_bytes()
                                          for slug in FIGURES)
    member = f"xl/worksheets/sheet{1 + [s[0] for s in SHEETS].index('Forecast_Detail')}.xml"
    extra_cells = re.compile(rb'<c r="P\d+"[^>]*><v>([^<]+)</v></c><c r="Q\d+"[^>]*><is><t>([^<]+)</t></is></c>$')
    with ZipFile(previous) as orig, ZipFile(current) as updated, orig.open(member) as a, updated.open(member) as b:
        old_rows, new_rows = xml_rows(a), xml_rows(b)
        for _ in range(2):
            next(old_rows)
            next(new_rows)
        # Compare every original XML cell, including the full serialized forecast precision.
        for old_row, new_row, adi, label in zip(old_rows, new_rows, detail.ADI, detail[GROUP], strict=True):
            match = extra_cells.search(new_row)
            if already_extended:
                old_row = extra_cells.sub(b"", old_row)
            if not match or new_row[:match.start()] != old_row or match[2].decode() != label:
                raise ValueError("Workbook detail changed or label mismatched")
            if abs(float(match[1]) - adi) > abs(adi) * 1e-14:
                raise ValueError("Workbook ADI differs from the enriched Parquet")


def write_workbook(detail, tables, previous=WORKBOOK, destination=WORKBOOK):
    base = small_tables()
    dictionary = base["Data_Dictionary"]
    base["Data_Dictionary"] = pd.concat([dictionary, pd.DataFrame(DICT, columns=dictionary.columns)])
    with ExitStack() as stack:
        directory = stack.enter_context(TemporaryDirectory(dir=destination.parent, prefix="demand_xlsx_"))
        # Publish from the report directory so the final file inherits the report permissions on Windows.
        with NamedTemporaryFile(dir=destination.parent, suffix=".xlsx", prefix=".demand_", delete=False) as output:
            pending = Path(output.name)
        stack.callback(pending.unlink, missing_ok=True)
        with xlsxwriter.Workbook(pending, {"constant_memory": True, "tmpdir": directory,
                                           "default_date_format": "dd/mm/yyyy", "strings_to_urls": False,
                                           "strings_to_formulas": False}) as book:
            book.set_properties({"title": "Dự báo và lượng bán thực tế M5", "subject": "RF-15 + FE-04 · Test"})
            style = formats(book)
            for name, title, key in SHEETS:
                sheet = book.add_worksheet(name)
                if key == "detail":
                    extended_detail(sheet, title, style, detail)
                else:
                    write_small(sheet, title, base[name], style)
            for key, title in EXTRA.items():
                write_small(book.add_worksheet(sheet_name(key)), title, tables[key], style)
            gallery = book.add_worksheet(FIGURE_SHEET)
            gallery.set_column(0, 0, 115)
            gallery.hide_gridlines(2)
            for i, slug in enumerate(FIGURES):
                path = REPORTS / "figures" / f"demand_group_{slug}.png"
                gallery.insert_image(i * 28, 0, str(path), {"x_scale": .64, "y_scale": .64})
        verify_extension(previous, pending, detail, tables)
        pending.replace(destination)
