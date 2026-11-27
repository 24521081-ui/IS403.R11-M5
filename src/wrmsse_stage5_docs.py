"""Vietnamese business report, project guide, and delivery inventory."""

import pandas as pd

from .wrmsse_stage4 import read
from .wrmsse_stage5 import REPORTS, ROOT, SELECTED, STAGE4, TABLES, csv, rel
from .wrmsse_stage5_notebooks import cells

BUSINESS_REPORT = ROOT / "docs/STAGE5_BUSINESS_REPORT.md"
HANDOFF = REPORTS / "STAGE5_HANDOFF.md"


def fmt(value, decimals=2):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}".translate(str.maketrans(",.", ".,"))


def dates(value):
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def md_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        *["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows],
    ])


def sources(*names):
    return " · ".join(f"[{label}](../reports/tables/{name}.csv)" for label, name in names) + "."


def fig(filename, label):
    return f"[{label}](../reports/figures/{filename}.png)"


def groups(frame, key, sales=False):
    columns = (("Actual", "actual_sales", 0),)
    if sales:
        columns += (("RF", "random_forest_forecast", 2), ("Seasonal Naive", "seasonal_naive_forecast", 2))
    else:
        columns += (("Tỷ trọng (%)", "actual_share_pct", 2),)
    columns += (("Sai lệch RF", "rf_error", 2), ("WAPE RF (%)", "rf_wape_pct", 2),
                ("WAPE SN (%)", "sn_wape_pct", 2), ("Bias RF (%)", "rf_bias_pct", 2),
                ("Cải thiện WAPE (%)", "wape_pct_improvement_pct", 2))
    return md_table(("Nhóm", *[name for name, _, _ in columns]), [
        (getattr(r, key), *[fmt(getattr(r, col), n) for _, col, n in columns]) for r in frame.itertuples()
    ])


def business_report(t, metrics):
    win = "–".join(map(dates, metrics["test_dates"]))
    total = t["system"].iloc[0]
    rf, sn = [metrics["models"][key] for key in ("random_forest", "seasonal_naive")]
    overall = md_table(("Chỉ số", "Random Forest", "Seasonal Naive"), [
        (label, fmt(rf[key], n), fmt(sn[key], n)) for label, key, n in (
            ("WRMSSE M5", "wrmsse", 10), ("MAE (đơn vị)", "mae", 4), ("RMSE (đơn vị)", "rmse", 4),
            ("WAPE (%)", "wape_percent", 4), ("Bias (%)", "forecast_bias_percent", 4),
            ("Tổng lượng bán thực tế", "actual_sales", 0), ("Tổng lượng bán dự báo", "forecast_sales", 2),
            ("Sai lệch ròng (đơn vị)", "sales_difference", 2),
        )
    ] + [("Tổng sai số tuyệt đối", fmt(total.rf_absolute_error), fmt(total.sn_absolute_error))])
    daily = t["daily_sales"]
    day_rows = md_table(("Ngày", "Actual", "RF − Actual", "SN − Actual", "WAPE RF (%)"), [
        (dates(r.date), fmt(r.actual_sales, 0), fmt(r.rf_error), fmt(r.sn_error), fmt(r.rf_wape_pct))
        for r in daily.nlargest(5, "rf_abs_aggregate_error").itertuples()
    ])
    up, down = daily.loc[daily.actual_change.idxmax()], daily.loc[daily.actual_change.idxmin()]
    align = t["daily_alignment"].set_index("model")
    align_rf, align_sn = [align.loc[key] for key in ("random_forest", "seasonal_naive")]
    weekday = t["weekday_analysis"]
    peaks = weekday.nlargest(2, "mean_actual")
    states, stores = t["state_analysis"], t["store_analysis"]
    store_error = stores.loc[stores.rf_net_absolute_error.idxmax()]
    store_wape = stores.loc[stores.rf_wape_pct.idxmax()]
    store_level = t["wrmsse_12_levels"].query("level == 3").iloc[0]
    store_rmsse = md_table(("Cửa hàng", "RMSSE RF", "RMSSE SN", "Contribution RF", "Contribution SN"), [
        (r.store_id, fmt(r.rf_rmsse, 4), fmt(r.sn_rmsse, 4), fmt(r.rf_contribution, 6), fmt(r.sn_contribution, 6))
        for r in stores.itertuples()
    ])
    cats, depts = t["category_analysis"], t["department_analysis"]
    foods = cats.set_index("cat_id").loc["FOODS"]
    dept_watch = depts.loc[depts.rf_wape_pct.idxmax()]
    heat = t["store_department"].loc[t["store_department"].rf_wape_pct.idxmax()]
    product = t["top_items_sales"].iloc[0]
    rep_rows = md_table(("Tiêu chí", "Sản phẩm", "Cửa hàng", "Actual", "Sai lệch RF", "Contribution RF"), [
        (r.selection, r.item_id, r.store_id, fmt(r.actual_sales, 0), fmt(r.rf_error), fmt(r.rf_contribution, 8))
        for r in t["representative_series"].itertuples()
    ])
    week = t["weekly_analysis"]
    week_watch = week.loc[week.rf_bias_pct.idxmin()]
    snap = t["snap_analysis"].set_index("snap_active")
    week_rows = md_table(("Tuần", "Khoảng ngày", "Actual", "Sai lệch RF", "Bias RF (%)", "WAPE RF (%)", "WAPE SN (%)"), [
        (r.week, f"{dates(r.week_start)}–{dates(r.week_end)}", fmt(r.actual_sales, 0), fmt(r.rf_error),
         fmt(r.rf_bias_pct), fmt(r.rf_wape_pct), fmt(r.sn_wape_pct)) for r in week.itertuples()
    ])
    wday_rows = md_table(("Thứ", "Actual trung bình ngày", "Bias RF (%)", "WAPE RF (%)", "WAPE SN (%)"), [
        (r.weekday, fmt(r.mean_actual), fmt(r.rf_bias_pct), fmt(r.rf_wape_pct), fmt(r.sn_wape_pct))
        for r in weekday.itertuples()
    ])
    hierarchy_rows = md_table(("Cấp", "Số chuỗi", "WRMSSE RF trong cấp", "WRMSSE SN trong cấp", "Đóng góp RF", "Đóng góp SN"), [
        (f"{r.level}. {r.hierarchy}", fmt(r.n_series, 0), fmt(r.rf_wrmsse_within_level, 6),
         fmt(r.sn_wrmsse_within_level, 6), fmt(r.rf_contribution, 8), fmt(r.sn_contribution, 8))
        for r in t["wrmsse_12_levels"].itertuples()
    ])
    importance = t["feature_importance"]
    snap_imp = importance.set_index("feature").at["snap_state", "importance_pct"]
    rolling_imp = importance.loc[importance.feature.str.startswith("rolling_"), "importance_pct"].sum()
    importance_rows = md_table(("Đặc trưng", "Importance (%)"), [
        (r.feature, fmt(r.importance_pct, 4)) for r in importance.itertuples()
    ])
    return f"""# Phân tích kinh doanh từ dự báo Random Forest M5

Trên Test **{win}**, RF-15 + FE-04 giảm WRMSSE **{fmt(metrics['improvement_percent'])}%** so với Seasonal Naive và có điểm thấp hơn ở cả 12 cấp. WAPE **{fmt(rf['wape_percent'])}%** ở cấp item–store–day vẫn cho thấy sai số lớn ở mức chi tiết. [Kết quả Stage 4](STAGE4_REPORT.md) · [Excel](../reports/RandomForest_Forecast_Actual.xlsx) · [Danh mục hình](../reports/FIGURE_CATALOG.md).

## Cách đọc chỉ số

Error = Forecast − Actual. MAE = trung bình |Error|. RMSE = căn bậc hai của trung bình Error². WAPE = 100 × tổng |Error| chia tổng Actual. Bias = 100 × tổng Error chia tổng Actual. Bias âm biểu thị dự báo thiếu, Bias dương biểu thị dự báo thừa. Với tổng Actual bằng 0, WAPE và Bias để trống trong CSV/Excel và trình bày N/A.

Chỉ số mặc định đo trên từng dòng item–store–day trong nhóm. Các cột store_day_* gom sản phẩm theo cửa hàng–ngày trước khi tính. Độ lớn sai lệch ròng là |tổng Error| và chịu bù trừ giữa các dòng. Tổng sai số tuyệt đối là tổng |Error|. Cải thiện WAPE là tỷ lệ giảm so với Seasonal Naive, tính lại từ tổng thành phần của từng nhóm. [Từ điển chỉ số](../reports/RandomForest_Forecast_Actual.xlsx).

## Toàn hệ thống và theo ngày

{overall}

Random Forest dự báo thiếu **{fmt(abs(rf['sales_difference']))} đơn vị** trên tổng thực tế **{fmt(rf['actual_sales'], 0)}**. Bảng dưới xếp ngày theo độ lớn sai lệch ròng của Random Forest. [Chỉ số hai mô hình](../reports/tables/model_comparison.csv) · [Thành phần sai số](../reports/tables/system.csv).

{day_rows}

Lượng bán tăng mạnh nhất ngày **{dates(up.date)}**, thêm **{fmt(up.actual_change, 0)}** đơn vị. Lượng bán giảm mạnh nhất ngày **{dates(down.date)}**, giảm **{fmt(abs(down.actual_change), 0)}** đơn vị. Hai mô hình dự báo mức thay đổi tương ứng **{fmt(down.rf_change)}** và **{fmt(down.sn_change)}** đơn vị ở ngày giảm mạnh này. Tương quan Pearson của tổng theo ngày là **{fmt(align_rf.daily_total_pearson, 3)}** ở RF và **{fmt(align_sn.daily_total_pearson, 3)}** ở Seasonal Naive. Hai mô hình khớp hướng thay đổi lần lượt **{int(align_rf.direction_matches)}/{int(align_rf.day_to_day_comparisons)}** và **{int(align_sn.direction_matches)}/{int(align_sn.day_to_day_comparisons)}** lần. Seasonal Naive lặp nhịp bảy ngày từ lịch sử. RF thay đổi tổng dự báo giữa các tuần. {fig("14_doanh_so_hang_ngay", "Lượng bán theo ngày")} · {sources(("Dữ liệu theo ngày", "daily_sales"), ("Tương quan và hướng thay đổi", "daily_alignment"))}

## Bang và cửa hàng

{groups(states, "state_id", sales=True)}

CA có lượng bán lớn nhất. RF giảm WAPE ở cả ba bang, và cả ba có Bias âm. Bảng nguồn chứa MAE, RMSE, WAPE, Bias của hai mô hình và cải thiện theo từng chỉ số. {fig("15_doanh_so_theo_bang", "Lượng bán theo bang")} · {sources(("Bảng bang", "state_analysis"))}

{groups(stores, "store_id", sales=True)}

**{store_error.store_id}** có sai lệch ròng lớn nhất (**{fmt(store_error.rf_error)} đơn vị**, Bias **{fmt(store_error.rf_bias_pct)}%**). **{store_wape.store_id}** có WAPE item–store–day cao nhất (**{fmt(store_wape.rf_wape_pct)}%**). Hai tiêu chí phục vụ hai ưu tiên theo tổng lượng và theo sai số chi tiết. Điểm chính thức trong cấp cửa hàng là **{fmt(store_level.rf_wrmsse_within_level, 6)}** ở RF và **{fmt(store_level.sn_wrmsse_within_level, 6)}** ở Seasonal Naive. Mỗi cửa hàng có RMSSE riêng. Điểm trong cấp bằng 12 lần tổng contribution từ các chuỗi cửa hàng. {fig("16_doanh_so_theo_cua_hang", "Lượng bán theo cửa hàng")} · {fig("17_bias_cua_hang", "Bias theo cửa hàng")} · {sources(("Cửa hàng và độ đo store–day", "store_analysis"), ("Cấp cửa hàng trong WRMSSE", "wrmsse_12_levels"))}

{store_rmsse}

## Ngành hàng và bộ phận

{groups(cats, "cat_id")}

{groups(depts, "dept_id")}

FOODS chiếm **{fmt(foods.actual_share_pct)}%** lượng bán và sai lệch ròng RF **{fmt(foods.rf_error)}** đơn vị. **{dept_watch.dept_id}** có WAPE RF **{fmt(dept_watch.rf_wape_pct)}%**, so với **{fmt(dept_watch.sn_wape_pct)}%** của baseline. **{heat.store_id} × {heat.dept_id}** có WAPE RF cao nhất trong heatmap (**{fmt(heat.rf_wape_pct)}%**, Actual **{fmt(heat.actual_sales, 0)}**). Tỷ lệ cao cần được xem cùng quy mô bán và tổng sai số tuyệt đối. {fig("19_nganh_hang", "So sánh ngành hàng")} · {fig("18_heatmap_wape", "Heatmap cửa hàng × bộ phận")} · {sources(("Ngành hàng", "category_analysis"), ("Bộ phận", "department_analysis"), ("Cửa hàng × bộ phận", "store_department"))}

## Sản phẩm và chuỗi cần theo dõi

**{product.item_id}** đứng đầu về lượng bán toàn hệ thống với **{fmt(product.actual_sales, 0)}** đơn vị. Top thiếu và top thừa xếp theo tổng có dấu của Forecast − Actual trong 28 ngày. Danh sách đóng góp WRMSSE sử dụng contribution chính thức của chuỗi cấp 12. {sources(("Top sản phẩm theo Actual", "top_items_sales"), ("Top chuỗi theo Actual", "top_series_sales"), ("Top 20 thiếu", "top_underforecast"), ("Top 20 thừa", "top_overforecast"), ("Top 20 contribution RF", "top_wrmsse_contribution"))}

Bốn chuỗi minh họa lần lượt đứng đầu theo lượng bán, thiếu dự báo, thừa dự báo và contribution. Khi trùng, lấy chuỗi kế tiếp trong tiêu chí. Khi bằng điểm, ưu tiên series_id nhỏ hơn. {fig("21_top_du_bao_under", "Top thiếu theo lượng")} · {fig("22_top_du_bao_over", "Top thừa theo lượng")} · {fig("23_chuoi_dai_dien", "Bốn chuỗi minh họa")} · {sources(("Bảng đại diện", "representative_series"), ("Chi tiết 28 ngày", "representative_daily"))}

{rep_rows}

## Tuần, thứ trong tuần và SNAP

{week_rows}

{wday_rows}

**Tuần {int(week_watch.week)}** có Bias RF âm nhất (**{fmt(week_watch.rf_bias_pct)}%**), với WAPE **{fmt(week_watch.rf_wape_pct)}%**. {peaks.iloc[0].weekday} và {peaks.iloc[1].weekday.lower()} có lượng bán trung bình ngày cao nhất. {fig("20_tuan_va_thu", "Sai số theo tuần và thứ")} · {sources(("Tuần", "weekly_analysis"), ("Thứ", "weekday_analysis"))}

RF đạt WAPE **{fmt(snap.at[True, 'rf_wape_pct'])}%** khi SNAP bật và **{fmt(snap.at[False, 'rf_wape_pct'])}%** khi SNAP tắt. Bias tương ứng **{fmt(snap.at[True, 'rf_bias_pct'])}%** và **{fmt(snap.at[False, 'rf_bias_pct'])}%**. Sự khác biệt có thể phản ánh thành phần bang, nhóm ngày và cơ cấu bán. FE-04 giữ SNAP sau đánh giá và hậu kiểm Validation. So sánh Test mang tính mô tả. {fig("24_snap", "SNAP toàn hệ thống và theo bang")} · {sources(("Bảng SNAP", "snap_analysis"), ("SNAP theo bang", "snap_by_state"))} [Hậu kiểm SNAP](STAGE3_SNAP_REVIEW.md).

## Hệ phân cấp M5 và feature importance

{hierarchy_rows}

WRMSSE chung là tổng contribution của 12 cấp. Mỗi cấp có trọng số 1/12. Scale Test dùng d_1–d_1913, trọng số dùng doanh thu units × sell_price d_1886–d_1913. Validation dùng scale tới d_1885 và trọng số d_1858–d_1885. {fig("11_wrmsse_12_cap", "WRMSSE 12 cấp")} · {fig("10_validation_va_test", "Validation và Test")} · {sources(("Trọng số và contribution", "wrmsse_12_levels"))}

{importance_rows}

Hai rolling mean có tổng tỷ trọng **{fmt(rolling_imp)}%**. SNAP chiếm **{fmt(snap_imp)}%**. Importance dựa trên mức giảm impurity và được gộp từ các cột one-hot về 15 đặc trưng logic. Các đặc trưng tương quan có thể chia sẻ importance. {fig("13_feature_importance", "Feature importance")} · {sources(("15 đặc trưng logic", "feature_importance"))}

## Ứng dụng và ưu tiên vận hành

1. **Lập kế hoạch nhập hàng theo kỳ.** Dùng dự báo và Seasonal Naive song song trong bước rà soát. Ưu tiên FOODS và các cửa sổ có Bias âm lớn như tuần {int(week_watch.week)}. Xem sai lệch ròng cùng WAPE trước khi điều chỉnh kế hoạch. {sources(("Ngành hàng", "category_analysis"), ("Tuần Test", "weekly_analysis"))}
2. **Phân bổ hàng giữa cửa hàng.** Rà soát {store_error.store_id} theo sai lệch tổng và {store_wape.store_id} theo sai số chi tiết. Heatmap chỉ ra bộ phận cần kiểm tra ở mỗi địa điểm. Cần tồn kho khả dụng và lịch cung ứng trước khi chuyển hàng. {sources(("Cửa hàng", "store_analysis"), ("Cửa hàng × bộ phận", "store_department"))}
3. **Theo dõi ngoại lệ ở cấp sản phẩm.** Rà soát riêng top thiếu, top thừa và top contribution. Với Actual nhỏ, xem sai số tuyệt đối và bối cảnh danh mục. Lượng nhập cần dữ liệu tồn kho, lead time, chi phí thiếu hàng và chính sách tồn kho an toàn. {sources(("Top thiếu", "top_underforecast"), ("Top thừa", "top_overforecast"), ("Top contribution", "top_wrmsse_contribution"))}

## Giới hạn và hướng phát triển

- Mô hình dùng mẫu Train + Validation và tối ưu trên một cửa sổ Validation. Test d_1914–d_1941 từng là Validation của phiên bản nghiên cứu cũ.
- Dataset ghi lượng bán quan sát. Dự báo thấp hơn lượng bán thực tế là sai lệch dự báo. Đánh giá thiếu hàng và mức dịch vụ cần dữ liệu vận hành.
- Đánh giá tác động SNAP cần thiết kế kiểm soát theo cửa hàng, thứ trong tuần và danh mục. Nghiên cứu tiếp theo có thể kiểm tra nhiều forecast origin, đo bất định dự báo và đánh giá kết quả vận hành.

[Đặc tả phương pháp](WRMSSE_PREPARATION.md) · [Tối ưu theo Validation](STAGE3_REPORT.md) · [Manifest Giai đoạn 5](../reports/manifest_stage5.json) · [Biên bản bàn giao](../reports/STAGE5_HANDOFF.md).
"""


def readme(tables, metrics, selected, config, sample):
    win = "–".join(map(dates, metrics["test_dates"]))
    rf, sn = [metrics["models"][key] for key in ("random_forest", "seasonal_naive")]
    comparison = md_table(("Cửa sổ", "Random Forest", "Seasonal Naive"), [
        ("Validation d_1886–d_1913", fmt(selected["wrmsse"], 10), fmt(selected["seasonal_naive_wrmsse"], 10)),
        ("Test d_1914–d_1941", fmt(rf["wrmsse"], 10), fmt(sn["wrmsse"], 10)),
    ])
    params = ", ".join(f"{key}={value}" for key, value in selected["params"].items())
    features = ", ".join(read(STAGE4 / "final_feature_metadata.json")["logical_features"])
    return f"""# Dự báo lượng bán M5 bằng Random Forest

Dự án dự báo lượng bán theo sản phẩm × cửa hàng trong 28 ngày, so sánh Random Forest Regression với Seasonal Naive chu kỳ 7 và đánh giá bằng **WRMSSE đủ 12 cấp M5**. Random Forest lấy trung bình dự báo của các cây hồi quy được fit với mẫu bootstrap và tập con đặc trưng khi chia nút. Mục tiêu là đánh giá chất lượng ở nhiều cấp tổng hợp và tìm nơi cần theo dõi trong kế hoạch bán hàng.

Bộ dữ liệu M5 có lượng bán hằng ngày, lịch và giá tuần của Walmart. Lưới dưới cùng gồm **{fmt(metrics['bottom_series'], 0)}** chuỗi tại 10 cửa hàng thuộc CA, TX và WI.

| Tập | Ngày M5 | Cách dùng |
|---|---|---|
| Train | d_1–d_1885 | Lấy mẫu cho các vòng thử nghiệm |
| Validation | d_1886–d_1913 | Chọn siêu tham số và đặc trưng |
| Train + Validation | d_1–d_1913 | Lấy mẫu fit mô hình cuối |
| Test | d_1914–d_1941 | Đánh giá {win} từ gốc d_1913 |

Train dùng **{fmt(sample['train_rows'], 0)}** quan sát phân tầng cửa hàng × bộ phận × quý, không hoàn lại, seed 42. Fit cuối giữ mẫu Train và thêm **{fmt(sample['validation_rows'], 0)}** quan sát Validation ở mật độ tương đương, tổng **{fmt(sample['rows'], 0)}** dòng.

## Đặc trưng và đánh giá

FE-04 gồm {config['logical_features']} đặc trưng logic và {fmt(config['encoded_features'], 0)} cột CSR float32. Thứ tự logic là {features}. Feature doanh số sử dụng lịch sử đến t−28 và forecast origin. SNAP xác định từ bang và ngày trong tháng. Vocabulary one-hot của năm trường ID giữ nhất quán. [Hậu kiểm SNAP](docs/STAGE3_SNAP_REVIEW.md) · [Notebook đặc trưng](notebooks/03_feature_engineering.ipynb).

WRMSSE cộng contribution qua {fmt(metrics['hierarchical_series'], 0)} chuỗi. Contribution là RMSSE nhân trọng số doanh thu lịch sử. Mỗi cấp có tổng trọng số 1/12. RMSSE dùng scale sai phân lịch sử từ lần bán khác 0 đầu tiên. Điểm trong cấp bằng 12 lần tổng contribution cấp. [Đặc tả evaluator](docs/WRMSSE_PREPARATION.md) · [Notebook WRMSSE](notebooks/02_wrmsse_evaluation_method.ipynb).

## Thực nghiệm và mô hình khóa

Các vòng lần lượt chọn **RF-04 → RF-v1 → RF-15 → FE-04**. Có {len(tables['experiment_results'])} lượt gồm {len(tables['experiment_results']) - int(tables['experiment_results'].reuse.sum())} fit và {int(tables['experiment_results'].reuse.sum())} reuse. FE-09 và FE-10 kiểm tra All Features trên cùng Validation. [Báo cáo tối ưu](docs/STAGE3_REPORT.md) · [Bảng thực nghiệm](reports/tables/experiment_results.csv).

**{config['candidate']}** dùng {params}.

{comparison}

Trên Test, Random Forest giảm WRMSSE **{fmt(metrics['improvement_percent'])}%** và đạt WAPE **{fmt(rf['wape_percent'])}%** ở cấp item–store–day. [Báo cáo Test](docs/STAGE4_REPORT.md) · [Báo cáo kinh doanh](docs/STAGE5_BUSINESS_REPORT.md).

## Xem kết quả đã công bố

1. Mở [Excel dự báo và thực tế](reports/RandomForest_Forecast_Actual.xlsx). Bắt đầu từ Summary, các sheet theo nhóm và Data_Dictionary.
2. Dùng [danh mục biểu đồ](reports/FIGURE_CATALOG.md) để chọn hình và tìm CSV nguồn.
3. Mở [notebook tổng quan](notebooks/00_project_overview.ipynb) đến [notebook phân tích kinh doanh](notebooks/09_business_insights_and_visualization.ipynb). Chọn kernel của .venv. Run All đọc artifact đã công bố. Notebook mới có output thực thi đã lưu.
4. Xem [biên bản bàn giao](reports/STAGE5_HANDOFF.md) và [manifest](reports/manifest_stage5.json) để tra số dòng, đường dẫn và checksum.

## Thư mục

| Thư mục | Nội dung |
|---|---|
| data/raw/ | CSV gốc được bảo toàn |
| data/processed/ | Doanh số lịch sử, khóa và mẫu |
| data/features/ | Ma trận đặc trưng và vocabulary |
| src/ và scripts/ | Evaluator, đặc trưng, điều phối, báo cáo và kiểm tra |
| configs/ | Split, metric và stage gate |
| notebooks/ | Mười notebook trình bày ở chế độ chỉ đọc |
| results/wrmsse_v1/ | Manifest và artifact Giai đoạn 2–4 |
| reports/figures/ và reports/tables/ | Hình PNG và bảng CSV nguồn |
| reports/ | Excel, danh mục hình và biên bản bàn giao |
| docs/ | Đặc tả và báo cáo |
| tests/ | Unit test evaluator, pipeline, data leakage |
| preparation/ | Biên bản chuẩn bị và audit |

## Môi trường

Môi trường thực nghiệm là Python {read(STAGE4 / "resource_summary.json")["versions"]["python"]}. Mở workbook bằng Excel hỗ trợ XLSX. Dùng VS Code với Python và Jupyter extensions để mở notebook. Để cài từ thư mục dự án:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
```

Tệp requirements-lock.txt khóa các phiên bản thực nghiệm, Excel và notebook đang dùng. [Biên bản tinh gọn dự án](docs/PROJECT_CLEANUP_REPORT.md).

## Kiểm tra và tái hiện

Dùng Python của môi trường dự án với lệnh dưới đây để xác minh đầu ra và chạy unit test.

```powershell
.venv/Scripts/python.exe -B scripts/run_stage5.py --verify
.venv/Scripts/python.exe -B -m unittest discover -s tests -t . -p "test_wrmsse*.py"
```

Các entry point dưới đây dành cho việc tái hiện giai đoạn tương ứng theo stage gate đã thiết lập, với namespace đầu ra mới.

| Giai đoạn | Entry point và điều kiện |
|---|---|
| 1 | [Kiểm thử](scripts/run_preparation_checks.py) và [audit context lịch sử](scripts/audit_wrmsse_contexts.py). Thực hiện theo báo cáo chuẩn bị. |
| 2 | [Điều phối Stage 2](scripts/run_stage2.py) tạo processed/features, baseline và smoke fit qua gate kiểm thử. |
| 3 | [Điều phối Stage 3](scripts/run_stage3.py) và [All Features](scripts/run_stage3_all_features.py) dùng gate tuning, evidence và namespace đã thiết lập. Selection v2 đã khóa. |
| 4 | [Điều phối Stage 4](scripts/run_stage4.py) kiểm tra selection v2, tạo run mới, fit một lần và khóa forecast trước Test. |
| 5 | [Điều phối Stage 5](scripts/run_stage5.py) đọc run Stage 4 đã khóa rồi tạo lớp báo cáo. Dùng tham số --verify để kiểm tra bản hoàn thành. |

## Giới hạn

Mô hình học từ mẫu và được tối ưu trên một cửa sổ Validation. Test mới từng là Validation của nghiên cứu cũ. Validation và Test dùng các context scale và trọng số khác nhau. Quyết định tồn kho và nhập hàng cần thêm dữ liệu vận hành.
"""


def write_docs():
    tables = {path.stem: csv(path) for path in TABLES.glob("*.csv")}
    metrics = read(STAGE4 / "test_metrics.json")
    selected = read(SELECTED)
    config = read(STAGE4 / "final_model_config.json")
    sample = read(STAGE4 / "final_training_sample_manifest.json")
    BUSINESS_REPORT.write_text(business_report(tables, metrics), encoding="utf-8")
    (ROOT / "README.md").write_text(readme(tables, metrics, selected, config, sample), encoding="utf-8")


def write_handoff(counts, files, checks):
    catalog = csv(REPORTS / "figure_catalog.csv")
    figures = "\n".join(f"- [{row.file}](figures/{row.file})" for row in catalog.itertuples(index=False))
    notebooks = "\n".join(f"- [{name}.ipynb](../notebooks/{name}.ipynb) — {title}"
                          for name, title, _ in cells())
    inventory = "\n".join(f"- [{path.name}](../{rel(path)})" for path in files)
    HANDOFF.write_text(f"""# Bàn giao Giai đoạn 5

Bộ báo cáo dùng kết quả RF-15 + FE-04 đã khóa. Bảng và workbook được đối chiếu với artifact Stage 4. Mười notebook được thực thi ở chế độ đọc và lưu output. [Hướng dẫn mở dự án](../README.md) · [Phát hiện và ưu tiên kinh doanh](../docs/STAGE5_BUSINESS_REPORT.md).

## Excel và số dòng

[RandomForest_Forecast_Actual.xlsx](RandomForest_Forecast_Actual.xlsx) gồm {len(counts)} sheet. Bảng đếm dòng dữ liệu, cộng hai dòng tiêu đề và tên trường khi xem trong Excel.

{md_table(("Sheet", "Dòng dữ liệu"), [(name, fmt(count, 0)) for name, count in counts.items()])}

## Biểu đồ chính thức

[Danh mục nguồn và vị trí đề xuất](FIGURE_CATALOG.md).

{figures}

## Notebook theo thứ tự nghiên cứu

{notebooks}

## Kiểm tra và tính toàn vẹn

{md_table(("Kiểm tra", "Kết quả"), checks)}

## File mới và file chỉnh sửa

README là file hướng dẫn được cập nhật. Báo cáo, workbook, bảng, hình, notebook, mã điều phối và file khóa thư viện dưới đây được thêm trong Giai đoạn 5.

{inventory}

- [Biên bản bàn giao](STAGE5_HANDOFF.md)
- [Manifest Giai đoạn 5](manifest_stage5.json)

## Giới hạn cần lưu ý

WAPE cấp item–store–day vẫn cao dù WRMSSE thấp hơn Seasonal Naive. Cửa sổ Test từng là Validation ở phiên bản nghiên cứu cũ. Ứng dụng nhập hàng cần dữ liệu tồn kho, lead time, chi phí và chính sách tồn kho an toàn.
""", encoding="utf-8")
    return HANDOFF
