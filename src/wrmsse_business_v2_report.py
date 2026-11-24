"""Narrative, notebook and figure catalogue for business analysis version 2."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

import nbformat as nbf
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient
import pandas as pd

from .wrmsse_business_v2 import MAE_TOL, PREFIX, ROOT, WIN_LABELS
from .wrmsse_business_v2_figures import SPECS, figure_path, vi

VERSION = "business_v2"
REPLACEMENTS = {
    13: ("importance", "Chú thích cũ sát nhãn trục; bản mới chia theo đặc trưng và nhóm."),
    14: ("daily", "Đường rõ; bổ sung ngày cực trị, lệch có dấu và lũy kế."),
    15: ("regional", "Quy mô bang dễ đọc; thêm tỷ trọng và WAPE cửa hàng."),
    16: ("regional", "Cột lượng bán lặp với phân tích bang; thay bằng so sánh WAPE tại 10 cửa hàng, giữ lượng bán ở bảng."),
    17: ("store_priority", "Bias có đường 0 nhưng thiếu góc nhìn quy mô; thêm quadrant và số lệch ở bảng."),
    18: ("heatmaps", "Heatmap có giá trị; thêm panel Bias, đổi thứ tự theo mức độ ưu tiên."),
    19: ("category", "Giữ các trục riêng; thêm tỷ trọng và Bias."),
    20: ("weekly, weekday", "Chỉ có WAPE; tách nhịp tuần, lượng bán, Bias và bình quân theo thứ."),
    21: ("exceptions", "Nhãn dài và thiếu giá trị ở đầu cột; gom hai phía và thêm nhãn số."),
    22: ("exceptions", "Gộp vào panel lệch dương có màu riêng và trục 0."),
    23: ("representatives", "Mở rộng từ bốn lên sáu tiêu chí; thêm WAPE và Bias trực tiếp."),
    24: ("snap", "So sánh WAPE rõ nhưng thiếu lượng bán bình quân, Bias và số ngày."),
}


def day(value):
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def insights(t, b, m):
    model = m["models"]["random_forest"]
    baseline = m["models"]["seasonal_naive"]
    levels = b["wrmsse_12_levels"].assign(reduction=lambda x: x.sn_wrmsse_within_level - x.rf_wrmsse_within_level)
    top_level = levels.loc[levels.reduction.idxmax()]
    daily = t["daily"]
    peak = daily.loc[daily.actual_sales.idxmax()]
    worst_day = daily.loc[daily.rf_error.abs().idxmax()]
    last = daily.iloc[-1]
    weekly = b["weekly_analysis"]
    heavy = weekly.loc[weekly.actual_sales.idxmax()]
    weak = weekly.loc[weekly.rf_wape_pct.idxmax()]
    low = weekly.loc[weekly.rf_bias_pct.idxmin()]
    weekday = b["weekday_analysis"]
    dow = weekday.loc[weekday.mean_actual.idxmax()]
    corr = daily[["actual_sales", "random_forest_forecast", "seasonal_naive_forecast"]].corr().iloc[0, 1:]
    state = b["state_analysis"].sort_values("actual_sales", ascending=False).iloc[0]
    store = t["store_priority"]
    priority = ", ".join(store.loc[store.quadrant.eq("Cao / cao"), "store_id"])
    bias = store.loc[store.rf_error.abs().idxmax()]
    high = store.loc[store.rf_wape_pct.idxmax()]
    cat = b["category_analysis"].set_index("cat_id")
    dep = b["department_analysis"].sort_values("rf_wape_pct", ascending=False).iloc[0]
    heat = b["store_department"].sort_values("rf_wape_pct", ascending=False).iloc[0]
    lower = b["store_department"].sort_values("rf_bias_pct").iloc[0]
    p50, p80, p90 = list(t["pareto_thresholds"].itertuples())
    top = t["top_contribution"]
    abc = t["abc"].set_index("abc")
    velocity = t["volume_segments"].sort_values("volume")
    a, c = abc.loc["A"], abc.loc["C"]
    vol0, vol_high = velocity.iloc[0], velocity.iloc[-1]
    lo_cut, hi_cut = t["velocity_cutoffs"].historical_units
    under = t["exceptions"].query("direction == 'Thiếu'").iloc[0]
    over = t["exceptions"].query("direction == 'Thừa'").iloc[0]
    sample = t["representatives"].iloc[1]
    quantile = t["error_quantiles"].set_index("model")
    valid = t["series"].actual_sales.gt(0)
    wins = t["series_wins"].query("dimension == 'all_series'").set_index("outcome")
    win, tie, loss = [wins.loc[k] for k in WIN_LABELS]
    zero = t["zero_actual"].set_index("zero_actual")
    z, pos = zero.loc[True], zero.loc[False]
    snap = t["snap"].query("scope == 'Toàn bộ' and state_id == 'CA'").set_index("snap_active")
    on, off = snap.loc[True], snap.loc[False]
    foods = t["snap"].query("scope == 'FOODS' and state_id == 'CA'").set_index("snap_active")
    imp = t["importance"].groupby("feature_group").importance_pct.sum()
    fe = t["snap_validation"].set_index("experiment").validation_wrmsse
    return [
        ("Mục tiêu và phạm vi", "Kết quả nào có thể dùng để ưu tiên theo dõi hàng bán lẻ?", [
            f"Phạm vi Test {day(m['test_dates'][0])}–{day(m['test_dates'][1])}, {vi(m['bottom_series'])} chuỗi và {vi(m['observations'])} dòng. Forecast có gốc d_{m['origin']}.",
            "Đọc Actual, hai forecast và trọng số từ artifact niêm phong. Các nhóm biết trước dùng lịch sử đến forecast origin, còn xếp hạng ngoại lệ mô tả riêng cửa sổ Test.",
        ], "Dùng bảng tổng hợp để xác định điểm cần rà soát, sau đó mở bảng chi tiết theo chuỗi.", [], []),
        ("Tổng quan", "Mô hình mới đem lại mức cải thiện nào và còn lệch bao nhiêu?", [
            f"WRMSSE RF **{vi(model['wrmsse'], 6)}** so với **{vi(baseline['wrmsse'], 6)}** của Seasonal Naive, giảm **{vi(m['improvement_percent'], 2)}%**.",
            f"RF dự báo **{vi(model['forecast_sales'])}** trên **{vi(model['actual_sales'])}** đơn vị đã bán, lệch **{vi(model['sales_difference'])}** đơn vị.",
        ], "Dùng cả sai số chi tiết và độ lệch tổng lượng khi đánh giá phương án kế hoạch.", ["dashboard"], []),
        ("So sánh hiệu quả mô hình", "Mức cải thiện có nhất quán giữa các chỉ số và cấp tổng hợp?", [
            f"WAPE giảm từ **{vi(baseline['wape_percent'],2)}%** xuống **{vi(model['wape_percent'],2)}%**, giảm **{vi(baseline['wape_percent']-model['wape_percent'],2)} điểm %**. Bias RF là **{vi(model['forecast_bias_percent'],2)}%**.",
            f"RF có WRMSSE thấp hơn ở **{int((levels.reduction > 0).sum())}/{len(levels)} cấp**. Cấp {top_level.level} ({top_level.hierarchy}) giảm tuyệt đối nhiều nhất, **{vi(top_level.reduction,3)}** điểm chuẩn hóa.",
            "Điểm trong cấp bằng contribution cấp nhân 12. Tổng contribution của các cấp cho WRMSSE toàn hệ thống.",
        ], "Dùng WRMSSE M5 để đối chiếu thí nghiệm và WAPE/Bias để theo dõi sai số vận hành theo đúng độ phân giải.", ["model_metrics", "hierarchy"],
         [("model_comparison", ["model", "wrmsse", "mae", "rmse", "wape_percent", "forecast_bias_percent"], 2)]),
        ("Nhu cầu theo thời gian", "Khi nào chênh lệch tích lũy và nhịp tuần thay đổi?", [
            f"Đỉnh bán là **{day(peak.date)}** với **{vi(peak.actual_sales)}** đơn vị. Ngày có sai lệch RF lớn nhất là **{day(worst_day.date)}**, lệch **{vi(worst_day.rf_error)}** đơn vị.",
            f"Đến cuối kỳ, sai lệch tích lũy của RF là **{vi(last.rf_error_cumulative)}**, của SN là **{vi(last.sn_error_cumulative)}** đơn vị. Sai lệch trái dấu có thể bù trừ trong tổng tích lũy.",
            f"Tuần **{heavy.week}** bán nhiều nhất (**{vi(heavy.actual_sales)}**), tuần **{low.week}** có Bias RF âm nhất (**{vi(low.rf_bias_pct,2)}%**), tuần **{weak.week}** có WAPE RF cao nhất (**{vi(weak.rf_wape_pct,2)}%**).",
            f"**{dow.weekday}** dẫn đầu bình quân với **{vi(dow.mean_actual)}** đơn vị/ngày. Tương quan Pearson giữa tổng ngày dự báo và thực tế là **{vi(corr.iloc[0],3)}** (RF) và **{vi(corr.iloc[1],3)}** (SN).",
        ], "Rà soát lịch nhập hàng theo tuần và ngày bán cao, kiểm tra lượng giao trước khi điều chỉnh kế hoạch.", ["daily", "weekly", "weekday"], []),
        ("Bang và cửa hàng", "Nơi nào cần ưu tiên do vừa có quy mô bán lớn vừa khó dự báo?", [
            f"**{state.state_id}** chiếm **{vi(state.actual_share_pct,2)}%** lượng bán. RF tại bang này lệch **{vi(state.rf_error)}** đơn vị.",
            f"**{high.store_id}** có WAPE RF cao nhất (**{vi(high.rf_wape_pct,2)}%**). **{bias.store_id}** có độ lệch tổng lớn nhất (**{vi(bias.rf_error)}** đơn vị).",
            f"Nhóm quy mô cao và WAPE cao gồm **{priority or 'không có cửa hàng'}**, theo trung vị **{vi(store.median_actual.iloc[0])}** đơn vị và **{vi(store.median_wape.iloc[0],2)}%**. Giá trị bằng ngưỡng thuộc phía cao.",
        ], "Rà soát trước nhóm cao/cao, duy trì theo dõi nhóm cao/thấp, xem nguyên nhân cục bộ ở nhóm thấp/cao và kiểm tra định kỳ nhóm thấp/thấp. Phân nhóm mô tả Test theo lượng bán và WAPE.", ["regional", "store_priority"],
         [("state_analysis", ["state_id", "actual_sales", "rf_error", "sn_error", "rf_wape_pct", "rf_bias_pct"], 3),
          ("@store_priority", ["store_id", "actual_sales", "rf_error", "rf_wape_pct", "sn_wape_pct", "rf_bias_pct", "quadrant"], 10)]),
        ("Ngành hàng và bộ phận", "Ngành nào chi phối tổng lượng và bộ phận nào có ngoại lệ?", [
            f"FOODS chiếm **{vi(cat.at['FOODS','actual_share_pct'],2)}%** thực tế, với lệch RF **{vi(cat.at['FOODS','rf_error'])}** đơn vị.",
            f"**{dep.dept_id}** có WAPE RF **{vi(dep.rf_wape_pct,2)}%** so với **{vi(dep.sn_wape_pct,2)}%** của SN.",
            f"Ô cao nhất theo WAPE là **{heat.store_id} × {heat.dept_id}** (**{vi(heat.rf_wape_pct,2)}%**, {vi(heat.actual_sales)} đơn vị thực tế). Ô Bias âm nhất là **{lower.store_id} × {lower.dept_id}** (**{vi(lower.rf_bias_pct,2)}%**).",
        ], "Xem quy mô thực tế và sai số tuyệt đối cùng tỷ lệ, nhất là các tổ hợp có lượng bán ít.", ["category", "heatmaps"],
         [("department_analysis", ["dept_id", "actual_sales", "rf_error", "rf_wape_pct", "sn_wape_pct", "rf_bias_pct"], 7)]),
        ("Sản phẩm và danh mục ABC", "Sai số tập trung ở đâu và có thể phân nhóm từ lịch sử ra sao?", [
            f"**{vi(p50.rank)}**, **{vi(p80.rank)}**, **{vi(p90.rank)}** chuỗi đầu tiên tạo ít nhất 50%, 80% và 90% sai số tuyệt đối RF. Mốc 80% tương ứng **{vi(p80.series_pct,2)}%** số chuỗi.",
            f"Top {len(top)} contribution chiếm **{vi(top.share_of_level12_pct.sum(),2)}%** contribution cấp 12, theo RMSSE nhân trọng số doanh thu lịch sử.",
            f"Phân vị của chuỗi có bán lịch sử là **{vi(lo_cut,1)}** và **{vi(hi_cut,1)}** đơn vị. WAPE RF của nhóm chưa bán trong 28 ngày lịch sử là **{vi(vol0.rf_wape_pct,2)}%**, của nhóm cao là **{vi(vol_high.rf_wape_pct,2)}%**.",
            f"ABC A gồm **{vi(a.n_series)}** chuỗi và **{vi(a.historical_revenue_share*100,2)}%** tỷ trọng doanh thu lịch sử. WAPE RF của A là **{vi(a.rf_wape_pct,2)}%**. Ở C, WAPE RF **{vi(c.rf_wape_pct,2)}%** cao hơn SN **{vi(c.sn_wape_pct,2)}%**.",
        ], "Ưu tiên chuỗi doanh thu lịch sử cao hoặc contribution lớn. Giữ danh sách Pareto riêng khi mục tiêu là giảm tổng sai số đơn vị.", ["concentration", "segments"],
         [("@pareto_thresholds", ["target_pct", "rank", "series_pct", "cumulative_error_pct"], 3),
          ("@volume_segments", ["volume", "n_series", "actual_sales", "rf_wape_pct", "sn_wape_pct", "rf_mae", "sn_mae", "wape_pct_improvement_pct"], 4),
          ("@abc", ["abc", "n_series", "historical_revenue_share", "actual_sales", "rf_wape_pct", "sn_wape_pct", "rf_bias_pct", "sn_bias_pct", "wape_pct_improvement_pct"], 3)]),
        ("Ngoại lệ cấp chuỗi", "Sai lệch lớn nhất về số lượng nằm ở chuỗi nào?", [
            f"RF thiếu nhiều nhất ở **{under.item_id} × {under.store_id}** (**{vi(under.rf_error)}** đơn vị), và thừa nhiều nhất ở **{over.item_id} × {over.store_id}** (**+{vi(over.rf_error)}** đơn vị).",
            f"Ví dụ bán gián đoạn **{sample.item_id} × {sample.store_id}** có **{sample.zero_actual_days}** ngày không bán trong 28 ngày. Panel riêng cho thấy hình dạng chuỗi trước khi so sánh WAPE.",
        ], "Đối chiếu lịch bán, trạng thái mặt hàng và dữ liệu cửa hàng ở những chuỗi cực trị. Đây là tín hiệu kiểm tra dự báo.", ["exceptions", "representatives"],
         [("@exceptions", ["direction", "item_id", "store_id", "cat_id", "actual_sales", "random_forest_forecast", "rf_error", "rf_bias_pct", "rf_wape_pct"], 30),
          ("@representatives", ["criterion", "item_id", "store_id", "zero_actual_days", "rf_wape_pct", "rf_bias_pct"], 6)]),
        ("Sai số và độ ổn định", "Lợi thế RF có xuất hiện nhất quán ở từng chuỗi và từng kiểu ngày bán?", [
            f"RF có MAE thấp hơn ở **{vi(win.n_series)}** chuỗi (**{vi(win.share_pct,2)}%**); tương đương **{vi(tie.n_series)}**; SN thấp hơn **{vi(loss.n_series)}**. Tolerance tuyệt đối **{MAE_TOL:g}** đơn vị mỗi quan sát.",
            f"**{vi(valid.sum())}** chuỗi có WAPE xác định và **{vi((~valid).sum())}** chuỗi có tổng Actual bằng 0 nên để N/A. Phân vị 95 sai số tuyệt đối theo dòng lần lượt **{vi(quantile.at['rf','absolute_q95'],2)}** (RF) và **{vi(quantile.at['sn','absolute_q95'],2)}** đơn vị (SN).",
            f"Trên **{vi(z.observations)}** dòng Actual = 0, MAE RF **{vi(z.rf_mae,3)}** so với **{vi(z.sn_mae,3)}** của SN. Trên dòng có bán, MAE là **{vi(pos.rf_mae,3)}** và **{vi(pos.sn_mae,3)}**.",
        ], "Theo dõi riêng chuỗi bán gián đoạn và so sánh với baseline theo nhóm, nhất là ở ngày không bán.", ["stability"],
         [("@zero_actual", ["zero_actual", "observations", "rf_mae", "sn_mae", "rf_mean_forecast", "sn_mean_forecast", "rf_mean_error", "sn_mean_error", "rf_wape_pct", "rf_bias_pct"], 2),
          ("@series_wins", ["dimension", "group", "outcome", "n_series", "share_pct"], 30)]),
        ("SNAP", "Sau khi tính theo số ngày, lượng bán và chất lượng thay đổi ra sao?", [
            f"CA có **{on.n_days}** ngày SNAP bật và **{off.n_days}** ngày tắt. Lượng bán bình quân **{vi(on.actual_per_day)}** và **{vi(off.actual_per_day)}** đơn vị/ngày.",
            f"Ở CA, WAPE RF lần lượt **{vi(on.rf_wape_pct,2)}%** và **{vi(off.rf_wape_pct,2)}%**; trong FOODS bình quân là **{vi(foods.at[True,'actual_per_day'])}** và **{vi(foods.at[False,'actual_per_day'])}** đơn vị/ngày.",
        ], "Dùng lịch SNAP để phân lớp kiểm tra lịch bán. So sánh theo bang có thể chịu ảnh hưởng của thứ trong tuần và cơ cấu SKU.", ["snap"],
         [("@snap", ["scope", "state_id", "snap_active", "n_days", "observations", "actual_per_day", "rf_wape_pct", "sn_wape_pct", "rf_bias_pct", "sn_bias_pct"], 12)]),
        ("Giải thích mô hình", "Rừng sử dụng dữ liệu lịch sử và SNAP như thế nào?", [
            f"Rolling Mean chiếm **{vi(imp.at['Rolling Mean'],2)}%** importance, SNAP chiếm **{vi(imp.at['SNAP'],2)}%**.",
            f"Trên Validation, FE-01 đạt **{vi(fe.at['FE-01'],6)}**, FE-04 đạt **{vi(fe.at['FE-04'],6)}**, chênh **{vi(fe.at['FE-01']-fe.at['FE-04'],6)}** WRMSSE trong lần so sánh hai tập đặc trưng.",
        ], "Giữ kiểm tra chất lượng lịch và lags. Importance theo impurity nhạy với số điểm chia và tương quan, chỉ mô tả mức sử dụng đặc trưng.", ["importance"], []),
        ("Hàm ý ứng dụng", "Bắt đầu theo dõi ở đâu trong quy trình vận hành?", [
            f"Dùng danh sách **{priority or 'cửa hàng quy mô cao'}** để ưu tiên rà soát dự báo và dùng **{bias.store_id}** để kiểm tra chênh lệch tổng lượng.",
            f"Đối chiếu nhóm ABC A và top contribution trước khi chọn chuỗi cần theo dõi. Nhóm A tương ứng **{vi(a.historical_revenue_share*100,2)}%** doanh thu lịch sử trong cửa sổ trọng số.",
        ], "Kết hợp tồn kho khả dụng, lead time, chi phí và mức dịch vụ trước khi quyết định lượng nhập hoặc phân bổ.", [], []),
        ("Hạn chế và kết luận", "Kết quả dùng được đến đâu?", [
            f"Test gồm 28 ngày và WAPE RF ở độ chi tiết item–store–day vẫn là **{vi(model['wape_percent'],2)}%**. Cửa sổ này từng là Validation trong nghiên cứu Mean RMSSE cũ.",
            f"Lợi thế WRMSSE **{vi(m['improvement_percent'],2)}%** là bằng chứng so với Seasonal Naive trong cửa sổ này. Lượng bán, SNAP và các nhóm minh họa chỉ hỗ trợ phân tích mô tả.",
        ], "Kiểm tra lại trên các forecast origin khác và ghép dữ liệu vận hành trước khi quy đổi sai lệch dự báo thành tác động tài chính.", [], []),
    ]


METHOD = """**Phương pháp.** Error = Forecast − Actual. MAE và RMSE tính ở cấp item–store–day trừ khi ghi rõ. WAPE = 100 × tổng |Error| / tổng Actual. Bias = 100 × tổng Error / tổng Actual. Tổng Actual bằng 0 trả về N/A. Tổng sai số tuyệt đối và độ lớn lệch tổng là hai đại lượng riêng.

**Nhóm biết trước.** Quy mô bán dùng tổng đơn vị d_1886–d_1913. Hai ngưỡng tercile tính theo nội suy tuyến tính trên các chuỗi có bán. Giá trị bằng ngưỡng thuộc nhóm thấp hơn, chuỗi không bán có nhóm riêng. ABC dùng tỷ trọng từ trọng số cấp 12 của evaluator, tương đương tỷ trọng doanh thu units × sell_price ở cùng cửa sổ. Xếp giảm theo trọng số, hòa theo series_id tăng dần. Chuỗi làm vượt 80% hoặc 95% vẫn ở lớp trước khi vượt. Khi đã chạm ngưỡng trong sai số số học 10⁻¹², chuỗi kế tiếp thuộc lớp sau. Doanh thu bằng 0 vào C. Evaluator gán giá 0 khi thiếu giá và lượng bán bằng 0, đồng thời dừng nếu thiếu giá tại dòng có bán.

**Các ví dụ.** Chọn theo thứ tự trình bày và loại chuỗi đã dùng. Ví dụ gián đoạn có ít nhất một ngày bán, ưu tiên số ngày 0 và sau đó là lượng bán. Ví dụ WAPE thấp chỉ xét chuỗi có Actual ít nhất bằng trung vị của nhóm có Actual dương. Các tiêu chí bằng nhau dùng series_id tăng dần. Đây là minh họa hồi cứu trên Test. RMSSE từng chuỗi và contribution lấy từ evaluator chính thức, với trọng số từ trước Test.

**Tệp phân tích.** Bảng bổ sung nằm trong reports/tables với tiền tố business_v2_. Có thể mở trực tiếp bằng Excel hoặc Power BI. Workbook chính thức và Forecast_Detail giữ dữ liệu niêm phong."""

BOOT = '''from pathlib import Path
import pandas as pd
from IPython.display import Image, display

current = Path.cwd().resolve()
root = next(p for p in (current, *current.parents) if (p / "reports/manifest_stage5.json").is_file())
tables = root / "reports/tables"
figures = root / "reports/figures"

def show(name, columns, limit):
    frame = pd.read_csv(tables / (name + ".csv"), float_precision="round_trip")
    display(frame[columns].head(limit).round(4))

def figure(name):
    display(Image(filename=str(figures / name), width=1120))
'''


def section_text(i, title, question, notes, decision):
    bullets = "\n".join("- " + note for note in notes)
    return f"## {i}. {title}\n\n**Câu hỏi.** {question}\n\n{bullets}\n\n**Ứng dụng.** {decision}"


def create_notebook(t, b, m, output):
    cells = [nbf.v4.new_markdown_cell(
        "# Phân tích kinh doanh và trực quan hóa · RF-15 + FE-04\n\n"
        + METHOD + "\n\n[Báo cáo và nguồn số liệu](../docs/STAGE5_BUSINESS_REPORT.md) · "
        "[Danh mục hình 300 DPI](../reports/FIGURE_CATALOG.md)."
    ), nbf.v4.new_code_cell(BOOT)]
    for i, (title, question, notes, decision, figures, tables) in enumerate(insights(t, b, m), 1):
        cells.append(nbf.v4.new_markdown_cell(section_text(i, title, question, notes, decision)))
        code = [f"figure({figure_path(name)!r})" for name in figures]
        for name, columns, limit in tables:
            name = PREFIX + name[1:] if name.startswith("@") else name
            code.append(f"show({name!r}, {columns!r}, {limit})")
        if code:
            cells.append(nbf.v4.new_code_cell("\n".join(code)))
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "name": "python3", "language": "python"},
        "language_info": {"name": "python"}, "wrmsse_mode": "read_only_artifacts", "business_version": VERSION,
    })
    with TemporaryDirectory(prefix="wrmsse_business_v2_kernel_") as directory:
        folder = Path(directory) / VERSION
        folder.mkdir()
        (folder / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-B", "-m", "ipykernel_launcher", "-f", "{connection_file}",
                     "--HistoryManager.enabled=False"], "display_name": "Python 3", "language": "python",
            "env": {"IPYTHONDIR": str(Path(directory) / "profile")},
        }), encoding="utf-8")
        manager = KernelManager(kernel_name=VERSION, kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]))
        NotebookClient(notebook, km=manager, timeout=100,
                       resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    notebook.metadata.kernelspec = {"display_name": "Python 3 (ipykernel)", "name": "python3", "language": "python"}
    nbf.validate(notebook)
    nbf.write(notebook, output)
    return describe_notebook(output)


def describe_notebook(path):
    notebook = nbf.read(path, as_version=4)
    nbf.validate(notebook)
    assert notebook.metadata["wrmsse_mode"] == "read_only_artifacts" and notebook.metadata["business_version"] == VERSION
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert [cell.execution_count for cell in code] == list(range(1, len(code)+1))
    outputs = [out for cell in code for out in cell.outputs]
    assert all(out.output_type != "error" for out in outputs)
    image_outputs = sum("image/png" in out.get("data", {}) for out in outputs)
    assert image_outputs == len(SPECS)
    return {"code_cells": len(code), "image_outputs": image_outputs, "sections": 13}


def catalogue(original):
    kept = [
        "Minh họa rõ ba mốc thời gian; giữ cho phần phương pháp.",
        "Sơ đồ quy trình rõ nhưng thưa; giữ làm bằng chứng thiết kế.",
        "Chọn vòng RF đầu; giữ, đọc thêm bảng chính xác khi cột làm tròn bằng nhau.",
        "Ablation dễ so nhờ cùng trục; giữ bằng chứng chọn RF-v1.",
        "Cột ngang hợp 17 cấu hình; giữ cho vòng nâng cao.",
        "So sánh bộ đặc trưng cùng thước đo; giữ cho lựa chọn FE-04.",
        "Tập trung FE-04 và All Features; giữ làm bằng chứng lựa chọn.",
        "Chi phí fit đối chiếu chất lượng; giữ cho phương pháp.",
        "WRMSSE tổng thể dễ đọc; giữ cho notebook đánh giá, v2 dùng thẻ KPI.",
        "Hai cửa sổ và hai context; giữ nguyên bằng chứng.",
        "Đủ 12 cấp; giữ cho notebook đánh giá, v2 bổ sung giá trị và Δ.",
        "Tách trục chỉ số phù hợp; giữ cho notebook đánh giá, v2 có 5 panel gồm WRMSSE.",
    ]
    header = [
        "# Danh mục hình · báo cáo kinh doanh v2", "",
        "Các nhóm hình v2 xuất PNG nền trắng, 300 DPI. Tất cả bảng nguồn dùng dữ liệu Test đã khóa "
        "hoặc lịch sử trước gốc theo phạm vi ghi trong báo cáo.", "",
        "| Hình báo cáo | Loại | Câu hỏi | Bảng nguồn |", "|---|---|---|---|",
    ]
    for slug, label, kind, question, sources in SPECS:
        table_links = []
        for name in sources:
            name = PREFIX + name[1:] if name.startswith("@") else name
            table_links.append(f"[{name}](tables/{name}.csv)")
        header.append(f"| [{label}](figures/{figure_path(slug)}) | {kind} | {question} | {' · '.join(table_links)} |")
    header.extend(["", "## Rà soát 24 hình trước nâng cấp", "", 
                   "| Hình v1 | Câu hỏi/phạm vi gốc | Quyết định và lý do |", "|---|---|---|"])
    for i, row in enumerate(original.to_dict("records"), 1):
        file, topic = row["file"], row["nội_dung"]
        if i <= len(kept):
            header.append(f"| [{file}](figures/{file}) | {topic} | Giữ. {kept[i-1]} |")
        else:
            slugs, note = REPLACEMENTS[i]
            links = " · ".join(f"[{slug}](figures/{figure_path(slug)})" for slug in slugs.split(", "))
            header.append(f"| {file} | {topic} | Thay bằng {links}. {note} Bản gốc trong kho theo SHA-256. |")
    header.extend(["", "[Manifest v2 và ánh xạ bản trước](manifest_business_v2.json) · "
                   "[Biên bản tinh gọn](../docs/PROJECT_CLEANUP_REPORT.md)."])
    return "\n".join(header) + "\n"


def report(t, b, m):
    blocks = ["# Báo cáo phân tích kinh doanh Random Forest · v2", "", METHOD,
              "", "[Notebook phân tích](../notebooks/09_business_insights_and_visualization.ipynb) · "
              "[Danh mục 17 nhóm hình và đối chiếu 24 hình v1](../reports/FIGURE_CATALOG.md) · "
              "[Workbook chính thức](../reports/RandomForest_Forecast_Actual.xlsx) · "
              "[Manifest phiên bản](../reports/manifest_business_v2.json)."]
    for i, (title, question, notes, decision, figures, tables) in enumerate(insights(t, b, m), 1):
        blocks.extend(["", section_text(i, title, question, notes, decision)])
        references = [f"[Hình {name}](../reports/figures/{figure_path(name)})" for name in figures]
        for name, _, _ in tables:
            name = PREFIX + name[1:] if name.startswith("@") else name
            references.append(f"[{name}](../reports/tables/{name}.csv)")
        if references:
            blocks.extend(["", " · ".join(references) + "."])
    return "\n".join(blocks) + "\n"
