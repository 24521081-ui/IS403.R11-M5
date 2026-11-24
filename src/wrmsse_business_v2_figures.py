"""Publication figures for the expanded business analysis."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image

from .wrmsse_business_v2 import ABC_LABELS, PREFIX, VOLUME_LABELS
from .wrmsse_stage5 import FORECAST

COLORS = {"actual": "#263A53", "rf": "#087A72", "sn": "#BD6E23"}
LABEL = {"actual": "Thực tế", "rf": "Random Forest", "sn": "Seasonal Naive"}
UNDER, OVER, GRAY = "#A54A18", "#2468A3", "#536679"
UNITS = [("actual", "actual_sales"), *FORECAST.items()]
WAPE = [("rf", "rf_wape_pct"), ("sn", "sn_wape_pct")]
BIAS = [("rf", "rf_bias_pct"), ("sn", "sn_bias_pct")]
STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white",
         "axes.spines.right": False, "axes.spines.top": False, "axes.titleweight": "bold",
         "axes.labelcolor": "#344558", "text.color": "#263A53", "axes.edgecolor": "#a6b2bf",
         "xtick.color": "#425468", "ytick.color": "#425468", "axes.axisbelow": True}

# @ marks a table generated in version 2.
SPECS = (
    ("dashboard", "KPI và tổng quan", "Thẻ KPI và đường", "Mô hình cải thiện bao nhiêu và còn lệch tổng lượng thế nào?", ("model_comparison", "wrmsse_12_levels", "@daily")),
    ("model_metrics", "Chất lượng hai mô hình", "Cột nhỏ theo chỉ số", "WRMSSE, sai số dòng và Bias thay đổi thế nào?", ("model_comparison",)),
    ("hierarchy", "WRMSSE đủ 12 cấp", "Dumbbell ngang", "Cấp nào được cải thiện nhiều nhất?", ("wrmsse_12_levels",)),
    ("daily", "Dự báo ngày và lũy kế", "Đường và cột lệch ngày", "Khi nào dự báo lệch nhiều và độ lệch có tích lũy không?", ("@daily",)),
    ("weekly", "Kế hoạch theo tuần", "Ba ô cột", "Tuần nào bán nhiều và có Bias lớn?", ("weekly_analysis",)),
    ("weekday", "Nhịp bán theo thứ", "Đường và cột ghép", "Hai mô hình phản ánh chu kỳ tuần ra sao?", ("weekday_analysis",)),
    ("regional", "Quy mô bang và WAPE cửa hàng", "Cột ngang và dumbbell", "Nhu cầu tập trung ở đâu và cửa hàng nào có sai số cao?", ("state_analysis", "store_analysis")),
    ("store_priority", "Ưu tiên theo dõi cửa hàng", "Cột có dấu và scatter bốn vùng", "Ở đâu có độ lệch lớn hoặc WAPE cao cùng quy mô lớn?", ("@store_priority",)),
    ("category", "Quy mô và chất lượng theo ngành", "Ba ô cột ngang", "Ngành nào chiếm nhiều lượng bán hoặc khó dự báo hơn?", ("category_analysis", "department_analysis")),
    ("heatmaps", "WAPE và Bias cửa hàng × bộ phận", "Hai heatmap có nhãn", "Tổ hợp nào có sai số lớn hoặc lệch có hệ thống?", ("store_department",)),
    ("concentration", "Tập trung sai số và contribution", "Pareto tích lũy và cột ngang", "Sai số tuyệt đối và contribution cấp 12 tập trung vào đâu?", ("@pareto", "@pareto_thresholds", "@top_contribution")),
    ("segments", "Quy mô bán và ABC trước Test", "Hai ô cột ghép", "Chất lượng khác nhau thế nào giữa các nhóm biết trước?", ("@volume_segments", "@abc", "@velocity_cutoffs")),
    ("exceptions", "Thiếu và thừa nhiều nhất", "Hai ô cột ngang có dấu", "Chuỗi nào cần rà soát vì lệch tổng lượng lớn?", ("@exceptions",)),
    ("representatives", "Sáu chuỗi sản phẩm đại diện", "Đường 2 × 3", "Mô hình hoạt động ra sao với các kiểu chuỗi khác nhau?", ("@representatives", "@representatives_daily")),
    ("stability", "Phân bố sai số và WAPE chuỗi", "Hai ECDF", "Mức cải thiện có trải đều qua các dòng và chuỗi không?", ("@error_ecdf", "@error_quantiles", "@series", "@series_wins", "@zero_actual")),
    ("snap", "SNAP theo bang và ngày", "Ba ô cột ngang", "Lượng bán bình quân và chất lượng khác thế nào theo SNAP?", ("@snap",)),
    ("importance", "Vai trò đặc trưng", "Hai cột ngang", "Đặc trưng nào được dùng nhiều tại các nút chia?", ("@importance", "@snap_validation")),
)


def vi(value, places=0):
    return f"{value:,.{places}f}".translate(str.maketrans(",.", ".,"))


def figure_path(slug):
    return PREFIX + slug + ".png"


def panel(title, subtitle, width, height, rows=1, cols=1, **kwargs):
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), layout="constrained", **kwargs)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13, fontweight="bold")
    return fig, axes


def key(ax, models=("actual", "rf", "sn"), **kwargs):
    handles = [Line2D([], [], color=COLORS[m], lw=2.5, label=LABEL[m]) for m in models]
    ax.legend(handles=handles, frameon=False, ncols=len(models), fontsize=8, **kwargs)


def axis(ax, direction="x", percent=False, digits=0):
    ax_ = ax.xaxis if direction == "x" else ax.yaxis
    ax_.set_major_locator(MaxNLocator(5))
    ax_.set_major_formatter(PercentFormatter(100, decimals=0) if percent else FuncFormatter(lambda x, _: vi(x, digits)))


def grouped(ax, frame, labels, values, *, horizontal=False, unit="", tagged=True):
    pos = np.arange(len(labels))
    width = .76 / len(values)
    for i, (model, column) in enumerate(values):
        marker = pos + (i - (len(values) - 1) / 2) * width
        if horizontal:
            ax.barh(marker, frame[column], width * .92, color=COLORS[model])
        else:
            ax.bar(marker, frame[column], width * .92, color=COLORS[model])
    if horizontal:
        ax.set(yticks=pos, yticklabels=labels if tagged else [""] * len(labels), xlabel=unit)
        ax.invert_yaxis()
    else:
        ax.set(xticks=pos, xticklabels=labels, ylabel=unit)
    ax.grid(axis="x" if horizontal else "y", alpha=.20)
    axis(ax, "x" if horizontal else "y", "%" in unit)


def trend(ax, frame, suffix=""):
    dates = pd.to_datetime(frame.date)
    for model, column in UNITS:
        ax.plot(dates, frame[column + suffix], color=COLORS[model], lw=1.9,
                ls="--" if model == "sn" else "-")
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.set_xlim(dates.min() - pd.Timedelta(hours=12), dates.max() + pd.Timedelta(hours=12))
    axis(ax, "y")
    ax.grid(axis="y", alpha=.20)


def dumbbell(ax, labels, rf, sn, numbers=False):
    y = np.arange(len(labels))
    ax.hlines(y, rf, sn, color="#aebdca", lw=1.8)
    ax.scatter(rf, y, s=26, color=COLORS["rf"], zorder=3)
    ax.scatter(sn, y, s=26, color=COLORS["sn"], zorder=3)
    if numbers:
        for i, (a, b) in enumerate(zip(rf, sn)):
            for value, model, offset, align in ((a, "rf", -6, "right"), (b, "sn", 6, "left")):
                ax.annotate(vi(value, 3), (value, i), xytext=(offset, 0), textcoords="offset points",
                            ha=align, va="center", fontsize=8, color=COLORS[model])
        ax.margins(x=.19)
    ax.set(yticks=y, yticklabels=labels)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.20)


def dashboard(t, b, m, window):
    rf, sn = [m["models"][model] for model in ("random_forest", "seasonal_naive")]
    fig, (cards, ax) = panel("Dự báo bán lẻ M5 · kết quả Test", window, 12, 6.2, 2, 1, height_ratios=[1, 1.2])
    cards.axis("off")
    level = b["wrmsse_12_levels"]
    won = (level.rf_wrmsse_within_level < level.sn_wrmsse_within_level).sum()
    tiles = [
        ("WRMSSE RF", vi(rf["wrmsse"], 4), "Seasonal Naive " + vi(sn["wrmsse"], 4)),
        ("Giảm WRMSSE", vi(m["improvement_percent"], 2) + "%", f"Thấp hơn ở {won}/{len(level)} cấp"),
        ("WAPE RF", vi(rf["wape_percent"], 2) + "%", "Bias " + vi(rf["forecast_bias_percent"], 2) + "%"),
        ("MAE / RMSE RF", f'{vi(rf["mae"], 3)} / {vi(rf["rmse"], 3)}', "Đơn vị / item–store–day"),
        ("Lượng bán thực tế", vi(rf["actual_sales"]), f'{vi(m["bottom_series"])} chuỗi sản phẩm × cửa hàng'),
        ("Lượng dự báo RF", vi(rf["forecast_sales"]), "Lệch " + vi(rf["sales_difference"]) + " đơn vị"),
    ]
    for i, (label, value, note) in enumerate(tiles):
        x, y = .025 + (i % 3) * .33, .92 - (i // 3) * .5
        cards.text(x, y, label, va="top", fontweight="bold")
        cards.text(x, y - .13, value, va="top", fontsize=17, fontweight="bold", color=COLORS["rf"])
        cards.text(x, y - .32, note, va="top", fontsize=8.5, color=GRAY)
    trend(ax, t["daily"])
    ax.set(ylabel="Đơn vị bán", xlabel="Ngày Test")
    key(ax, loc="upper left")
    return fig


def model_metrics(t, b, m, window):
    fig, axes = panel("So sánh chất lượng hai mô hình", f"{window} · WRMSSE M5 và sai số item–store–day", 12, 4.4, 1, 5)
    frame = b["model_comparison"].set_index("model").loc[["random_forest", "seasonal_naive"]]
    for ax, (col, title, suffix) in zip(axes, [
        ("wrmsse", "WRMSSE", ""), ("mae", "MAE", " đv"), ("rmse", "RMSE", " đv"),
        ("wape_percent", "WAPE", "%"), ("forecast_bias_percent", "Bias", "%"),
    ]):
        a, s = frame[col]
        bars = ax.bar(["RF", "SN"], [a, s], .57, color=[COLORS["rf"], COLORS["sn"]])
        ax.bar_label(bars, labels=[vi(x, 2 if suffix == "%" else 3) + suffix for x in (a, s)], padding=3, fontsize=8)
        ax.margins(y=.30)
        ax.axhline(0, color=GRAY, lw=.8)
        delta = f"Giảm {vi(100*(s-a)/s,1)}%" if col != "forecast_bias_percent" else f"RF − SN: +{vi(a-s,2)} đpt"
        ax.set_title(title + "\n" + delta, fontsize=10)
        axis(ax, "y", suffix == "%", 2 if suffix != "%" else 0)
        ax.grid(axis="y", alpha=.18)
    return fig


def hierarchy(t, b, m, window):
    fig, ax = panel("Random Forest giảm WRMSSE ở 12 cấp", f"{window} · Điểm chuẩn hóa trong cấp, Δ = Seasonal Naive − RF", 11, 7)
    levels = b["wrmsse_12_levels"].sort_values("level")
    gap = levels.sn_wrmsse_within_level - levels.rf_wrmsse_within_level
    labels = [f"{r.level:02d}  {r.hierarchy}   ·   Δ {vi(x,3)}" for r, x in zip(levels.itertuples(), gap)]
    dumbbell(ax, labels, levels.rf_wrmsse_within_level, levels.sn_wrmsse_within_level, numbers=True)
    best = levels.iloc[gap.to_numpy().argmax()]
    ax.set(xlabel="WRMSSE chuẩn hóa trong cấp", title=f"Giảm nhiều nhất ở cấp {best.level} · {best.hierarchy}")
    key(ax, ("rf", "sn"), loc="upper right")
    return fig


def daily(t, b, m, window):
    fig, axes = panel("Dự báo hằng ngày và độ lệch tích lũy", f"{window} · Tổng hệ thống", 11.5, 8, 3, 1,
                      height_ratios=[1.55, .8, 1.15], sharex=True)
    df = t["daily"]
    dates = pd.to_datetime(df.date)
    trend(axes[0], df)
    key(axes[0], loc="upper left")
    axes[0].set_ylabel("Đơn vị bán / ngày")
    peak, error = df.actual_sales.idxmax(), df.rf_error.abs().idxmax()
    notes = {error: "Lệch RF lớn nhất", peak: "Lượng bán cao nhất" if peak != error else "Đỉnh bán và lệch RF lớn nhất"}
    for idx, note in notes.items():
        axes[0].scatter(dates[idx], df.actual_sales[idx], color=COLORS["actual"], zorder=4)
        axes[0].annotate(note, (dates[idx], df.actual_sales[idx]), xytext=(-8, 8), ha="right",
                         fontsize=8, textcoords="offset points", bbox={"fc": "white", "ec": "none", "alpha": .85})
    axes[0].margins(y=.20)
    axes[1].bar(dates, df.rf_error, width=.82, color=np.where(df.rf_error < 0, UNDER, OVER))
    axes[1].axhline(0, color=GRAY, lw=.8)
    axes[1].set_ylabel("RF − thực tế")
    axis(axes[1], "y")
    trend(axes[2], df, "_cumulative")
    axes[2].set(xlabel="Ngày Test", ylabel="Đơn vị tích lũy")
    return fig


def weekly(t, b, m, window):
    fig, axes = panel("Nhu cầu và sai lệch theo từng tuần", f"{window} · Bốn tuần, mỗi tuần 7 ngày", 12, 4.7, 1, 3)
    df = b["weekly_analysis"]
    for ax, title, unit, values in zip(axes, ["Lượng bán", "WAPE", "Forecast Bias"],
                                      ["Đơn vị bán", "WAPE (%)", "Bias (%)"], [UNITS, WAPE, BIAS]):
        grouped(ax, df, [f"Tuần {week}" for week in df.week], values, unit=unit)
        ax.set_title(title)
        ax.axhline(0, color=GRAY, lw=.7)
    key(fig, loc="outside lower center")
    return fig


def weekday(t, b, m, window):
    fig, (a, c) = panel("Chu kỳ bán theo thứ trong tuần", f"{window} · Thứ suy từ ngày dương lịch", 11.5, 4.8, 1, 2)
    frame = b["weekday_analysis"].sort_values("weekday_no")
    labels = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    for model, col in zip(COLORS, ("mean_actual", "mean_rf", "mean_sn")):
        a.plot(labels, frame[col], color=COLORS[model], marker="o", ms=4)
    a.set(title="Lượng bán bình quân", ylabel="Đơn vị / ngày")
    axis(a, "y")
    a.grid(axis="y", alpha=.20)
    key(a, loc="upper left")
    grouped(c, frame, labels, WAPE, unit="WAPE (%)")
    c.set_title("Sai số item–store–day")
    return fig


def regional(t, b, m, window):
    fig, (a, c) = panel("Quy mô theo bang và WAPE theo cửa hàng", f"{window} · WAPE tính từ các dòng item–store–day", 12.5, 5.5, 1, 2)
    frame = b["state_analysis"].sort_values("actual_sales", ascending=False)
    tags = [f"{r.state_id} · {vi(r.actual_share_pct,1)}%" for r in frame.itertuples()]
    grouped(a, frame, tags, UNITS, horizontal=True, unit="Đơn vị bán")
    a.set_title("Quy mô và tỷ trọng thực tế")
    key(fig, loc="outside lower center")
    stores = b["store_analysis"].sort_values("rf_wape_pct", ascending=False)
    dumbbell(c, stores.store_id, stores.rf_wape_pct, stores.sn_wape_pct)
    c.set(xlabel="WAPE (%)", title="Cửa hàng · xếp theo WAPE RF")
    axis(c, percent=True)
    return fig


def store_priority(t, b, m, window):
    fig, (a, c) = panel("Độ lệch và ưu tiên theo dõi cửa hàng", f"{window} · Đường chia theo trung vị của 10 cửa hàng", 12, 5.7, 1, 2)
    frame = t["store_priority"].sort_values("rf_bias_pct")
    group = a.barh(frame.store_id, frame.rf_bias_pct, color=np.where(frame.rf_bias_pct < 0, UNDER, OVER))
    a.bar_label(group, labels=[vi(x,1)+"%" for x in frame.rf_bias_pct], fontsize=8, padding=4)
    a.axvline(0, color=GRAY, lw=.8)
    a.invert_yaxis()
    a.margins(x=.20)
    a.set(xlabel="Forecast Bias RF (%)", title="Sai lệch ròng")
    axis(a, percent=True)
    c.axvline(frame.median_actual.iloc[0], color="#879bad", ls="--", lw=1)
    c.axhline(frame.median_wape.iloc[0], color="#879bad", ls="--", lw=1)
    c.scatter(frame.actual_sales, frame.rf_wape_pct, color=np.where(frame.quadrant.eq("Cao / cao"), UNDER, COLORS["rf"]), s=42)
    offsets = {"CA_1": (7, 10), "CA_2": (7, -12), "TX_3": (9, 10), "TX_2": (9, -14),
               "WI_1": (-7, 9), "WI_3": (-7, -13)}
    for r in frame.itertuples():
        dx, dy = offsets.get(r.store_id, (5, 4))
        c.annotate(r.store_id, (r.actual_sales, r.rf_wape_pct), xytext=(dx, dy),
                   textcoords="offset points", ha="left" if dx > 0 else "right", fontsize=8,
                   arrowprops={"arrowstyle": "-", "color": "#9aabb8", "lw": .65, "shrinkA": 1, "shrinkB": 2})
    c.margins(x=.15, y=.14)
    axis(c)
    axis(c, "y", True)
    c.set(xlabel="Lượng bán thực tế (đơn vị)", ylabel="WAPE RF (%)", title="Quy mô cao và WAPE cao: điểm cam")
    return fig


def category(t, b, m, window):
    fig, axes = panel("Ngành hàng · quy mô, WAPE và Bias", f"{window} · Xếp theo tỷ trọng lượng bán thực tế", 13, 4.6, 1, 3,
                      width_ratios=[1.33, 1, 1])
    frame = b["category_analysis"].sort_values("actual_sales", ascending=False)
    tags = [f"{r.cat_id} · {vi(r.actual_share_pct,1)}%" for r in frame.itertuples()]
    for i, (ax, columns, label) in enumerate(zip(axes, [UNITS, WAPE, BIAS], ["Đơn vị bán", "WAPE (%)", "Bias (%)"])):
        grouped(ax, frame, tags, columns, horizontal=True, unit=label, tagged=i == 0)
    key(fig, loc="outside lower center")
    axes[2].axvline(0, color=GRAY, lw=.8)
    return fig


def heatmaps(t, b, m, window):
    fig, axes = panel("Cửa hàng × bộ phận · WAPE và Bias", f"{window} · 70 tổ hợp, mỗi ô là % ở cấp item–store–day", 14, 6.3, 1, 2)
    stores = b["store_analysis"].sort_values("rf_wape_pct", ascending=False).store_id
    depts = b["department_analysis"].sort_values("actual_sales", ascending=False).dept_id
    bias_map = LinearSegmentedColormap.from_list("forecast_bias", [UNDER, "#f7f8fa", OVER])
    for ax, col, cmap, title in zip(axes, ("rf_wape_pct", "rf_bias_pct"), ("YlGnBu", bias_map), ("WAPE RF", "Forecast Bias RF")):
        mat = b["store_department"].pivot(index="store_id", columns="dept_id", values=col).reindex(index=stores, columns=depts)
        bound = np.abs(mat.to_numpy()).max()
        norm = TwoSlopeNorm(0, vmin=-bound, vmax=bound) if col == "rf_bias_pct" else None
        image = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        for (i, j), val in np.ndenumerate(mat.to_numpy()):
            red, green, blue, _ = image.cmap(image.norm(val))
            color = "white" if .2126*red + .7152*green + .0722*blue < .51 else "#16293c"
            ax.text(j, i, vi(val), ha="center", va="center", color=color, fontsize=8)
        ax.set(xticks=range(len(depts)), xticklabels=mat.columns, yticks=range(len(stores)), yticklabels=mat.index, title=title)
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        fig.colorbar(image, ax=ax, label="%", shrink=.90)
    return fig


def concentration(t, b, m, window):
    fig, (a, c) = panel("Sai số tập trung ở đâu?", f"{window} · Sai số tuyệt đối và contribution WRMSSE là hai đại lượng riêng", 13, 6.6, 1, 2)
    df = t["pareto"]
    a.plot(np.r_[0, df.series_pct], np.r_[0, df.cumulative_error_pct], color=COLORS["rf"], lw=2.1)
    a.plot([0, 100], [0, 100], ls="--", color="#9eb0bf", lw=1)
    for r in t["pareto_thresholds"].itertuples():
        a.scatter(r.series_pct, r.cumulative_error_pct, color=UNDER, s=29)
        a.annotate(f"{r.target_pct}% · {vi(r.rank)} chuỗi", (r.series_pct, r.cumulative_error_pct),
                   xytext=(6, -12), textcoords="offset points", fontsize=8)
    a.set(xlim=(0, 100), ylim=(0, 104), xlabel="% chuỗi, xếp theo sai số RF giảm dần",
          ylabel="% sai số tuyệt đối tích lũy", title="Pareto sản phẩm × cửa hàng")
    a.grid(alpha=.20)
    top = t["top_contribution"]
    c.barh(top.item_id + " · " + top.store_id, top.rf_contribution, color=COLORS["rf"])
    c.set(title=f"Top {len(top)} chiếm {vi(top.share_of_level12_pct.sum(),2)}% contribution cấp 12",
          xlabel="Trọng số doanh thu lịch sử × RMSSE")
    c.invert_yaxis()
    axis(c, digits=5)
    return fig


def segments(t, b, m, window):
    fig, axes = panel("Phân nhóm theo thông tin có trước Test", f"{window} · Nhóm theo d_1886–d_1913, WAPE tính trên Test", 12, 5, 1, 2)
    for ax, item, column, order, title in (
        (axes[0], "volume_segments", "volume", VOLUME_LABELS, "Quy mô lượng bán lịch sử"),
        (axes[1], "abc", "abc", ABC_LABELS, "ABC theo trọng số doanh thu lịch sử"),
    ):
        frame = t[item].set_index(column).loc[order]
        tags = [name.split(" · ")[-1] + f"\n{vi(count)} chuỗi" for name, count in zip(frame.index, frame.n_series)]
        grouped(ax, frame, tags, WAPE, unit="WAPE (%)")
        ax.set_title(title)
    key(axes[0], ("rf", "sn"), loc="upper right")
    axes[0].margins(y=.26)
    return fig


def exceptions(t, b, m, window):
    fig, axes = panel("Các chuỗi lệch tổng lượng lớn nhất", f"{window} · 15 chuỗi mỗi phía, xếp theo độ lệch đơn vị", 12, 7.7, 1, 2)
    for ax, side, color in zip(axes, ("Thiếu", "Thừa"), (UNDER, OVER)):
        df = t["exceptions"][t["exceptions"].direction == side]
        marks = ax.barh(df.item_id + " · " + df.store_id, df.rf_error, color=color)
        ax.bar_label(marks, labels=[vi(x) for x in df.rf_error], padding=4, fontsize=7.5)
        ax.set(title=f"Dự báo {side.lower()}", xlabel="RF − thực tế (đơn vị)")
        ax.invert_yaxis()
        ax.axvline(0, color=GRAY, lw=.8)
        ax.margins(x=.18)
        axis(ax)
    return fig


def representatives(t, b, m, window):
    fig, axes = panel("Sáu chuỗi đại diện của lưới Test", window, 13.4, 7, 2, 3, sharex=True)
    titles = ["Lượng bán cao nhất", "Nhiều ngày không bán", "WAPE thấp", "Thiếu lớn nhất", "Thừa lớn nhất", "Contribution cao nhất"]
    for ax, r, title in zip(axes.flat, t["representatives"].itertuples(), titles):
        data = t["representatives_daily"]
        trend(ax, data[data.series_id == r.series_id].sort_values("day"))
        ax.set_ylim(0, None)
        ax.set_title(f"{title} · {r.item_id} / {r.store_id}\nRF WAPE {vi(r.rf_wape_pct,1)}%  ·  Bias {vi(r.rf_bias_pct,1)}%", fontsize=9)
        ax.set_ylabel("Đơn vị")
    key(fig, loc="outside lower center")
    return fig


def stability(t, b, m, window):
    fig, (a, c) = panel("Độ phân tán sai số và chất lượng theo chuỗi",
                      f"{window} · ECDF dùng mọi quan sát hợp lệ, trục symlog giữ toàn bộ đuôi", 12, 5.2, 1, 2)
    frame = t["error_ecdf"]
    valid = t["series"].actual_sales > 0
    for model in FORECAST:
        a.plot(frame.error_threshold, frame[f"{model}_cdf_pct"], color=COLORS[model], lw=2)
        values = np.sort(t["series"].loc[valid, f"{model}_wape_pct"].to_numpy())
        c.step(values, 100 * np.arange(1, len(values)+1) / len(values), where="post", color=COLORS[model], lw=1.8)
    a.set_xscale("symlog", linthresh=1)
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: vi(x)))
    a.axvline(0, color=GRAY, lw=.8)
    a.set(title="Sai số có dấu theo dòng", xlabel="Forecast − Actual (đơn vị)", ylabel="% dòng ≤ mức trên trục X")
    c.set_xscale("symlog", linthresh=100)
    c.xaxis.set_major_formatter(FuncFormatter(lambda x, _: vi(x) + "%"))
    c.set(title=f"WAPE chuỗi · {vi(valid.sum())} hợp lệ, {vi((~valid).sum())} N/A",
          xlabel="WAPE theo chuỗi (%)", ylabel="% chuỗi ≤ mức trên trục X")
    key(a, ("rf", "sn"), loc="lower right")
    for ax in (a, c):
        axis(ax, "y", True)
        ax.grid(alpha=.18)
    return fig


def snap(t, b, m, window):
    fig, axes = panel("SNAP theo bang · lượng bán bình quân và sai số", f"{window} · Số ngày kèm từng nhóm, so sánh mô tả", 13, 5.4, 1, 3)
    df = t["snap"].query("scope == 'Toàn bộ'")
    tags = [f'{r.state_id} · {"Bật" if r.snap_active else "Tắt"} ({r.n_days} ngày)' for r in df.itertuples()]
    grouped(axes[0], df, tags, [("actual", "actual_per_day")], horizontal=True, unit="Đơn vị bán / ngày")
    for ax, cols, label in zip(axes[1:], (WAPE, BIAS), ("WAPE (%)", "Bias (%)")):
        grouped(ax, df, tags, cols, horizontal=True, unit=label, tagged=False)
        ax.axvline(0, color=GRAY, lw=.7)
    key(fig, ("rf", "sn"), loc="outside lower center")
    return fig


def importance(t, b, m, window):
    fig, axes = panel("Mức sử dụng đặc trưng trong rừng", "Impurity-based importance · RF-15 + FE-04 đã khóa", 11.5, 7.3, 1, 2,
                      width_ratios=[1.4, 1])
    data = t["importance"].sort_values("importance_pct", ascending=False)
    total = data.groupby("feature_group").importance_pct.sum().sort_values(ascending=False)
    for ax, labels, values, title in ((axes[0], data.feature, data.importance_pct, "15 đặc trưng logic"),
                                      (axes[1], total.index, total, "Gộp theo nhóm đặc trưng")):
        group = ax.barh(labels, values, color=COLORS["rf"])
        ax.bar_label(group, labels=[vi(x,2)+"%" for x in values], fontsize=8, padding=4)
        ax.invert_yaxis()
        ax.margins(x=.24)
        ax.set(title=title, xlabel="% tổng importance")
    return fig


def draw(directory, tables, base, metrics):
    window = "–".join(pd.Timestamp(date).strftime("%d/%m/%Y") for date in metrics["test_dates"])
    makers = [dashboard, model_metrics, hierarchy, daily, weekly, weekday, regional, store_priority, category,
              heatmaps, concentration, segments, exceptions, representatives, stability, snap, importance]
    with plt.rc_context(STYLE), warnings.catch_warnings():
        warnings.filterwarnings("error", message="Glyph .* missing from font")
        for spec, make in zip(SPECS, makers, strict=True):
            fig = make(tables, base, metrics, window)
            path = directory / figure_path(spec[0])
            fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
            plt.close(fig)
            with Image.open(path) as picture:
                assert min(picture.info["dpi"]) > 299.9 and picture.width > 2600, path
                picture.verify()
    return [figure_path(spec[0]) for spec in SPECS]
