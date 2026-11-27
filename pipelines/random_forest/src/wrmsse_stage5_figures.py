"""Create the Stage 5 figure set from the verified reports and tables."""

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
from PIL import Image

from .wrmsse_stage4 import read
from .wrmsse_stage5 import FORECAST, LABELS, REPORTS, ROOT, STAGE4, TABLES, csv, rel

FIGURES = REPORTS / "figures"
COLORS = {"actual": "#334155", "rf": "#17675E", "sn": "#C27428"}
ACCENT = "#597D9A"
DPI = 230
WINDOW = "25/04/2016–22/05/2016"


class Gallery:
    def __init__(self):
        FIGURES.mkdir(parents=True, exist_ok=True)
        self.rows = []
        plt.rcParams.update({
            "font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
            "axes.spines.right": False, "axes.labelcolor": "#253445",
            "axes.titleweight": "bold", "figure.facecolor": "white",
        })

    def save(self, stem, title, sources, grain, section, figure=None, copy=None):
        path = FIGURES / f"{stem}.png"
        if copy is not None:
            shutil.copyfile(STAGE4 / copy, path)
        else:
            figure.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=.16)
            plt.close(figure)
        with Image.open(path) as image:
            assert image.width > 1200 and image.height > 600
            assert min(image.info["dpi"]) >= 200
            image.verify()
        for source in sources:
            assert (ROOT / source).is_file()
        self.rows.append((path.name, title, " | ".join(sources), grain, section, DPI))


def table(name):
    return f"reports/tables/{name}.csv"


def chart(width=9.3, height=4.8):
    return plt.subplots(figsize=(width, height), layout="constrained")


def amounts(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))


def experiment_bars(data, winner, title, horizontal=False):
    fig, ax = chart(10.2, max(5.0, len(data) * .34 + 1) if horizontal else 4.8)
    y = data.validation_wrmsse.to_numpy()
    label = data.experiment.str.replace("ABL_rf_v", "RF-v", regex=False).tolist()
    color = [COLORS["rf"] if candidate == winner else ACCENT for candidate in data.experiment]
    if horizontal:
        bars = ax.barh(label, y, color=color)
        ax.invert_yaxis()
        ax.set_xlabel("WRMSSE Validation")
        ax.margins(x=.19)
    else:
        bars = ax.bar(label, y, color=color)
        ax.set_ylabel("WRMSSE Validation")
        ax.margins(y=.15)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_title(title)
    return fig


def sales_bars(data, column, title):
    fig, ax = chart(10.0, max(4.1, len(data) * .47 + 1.9))
    y = np.arange(len(data))
    for offset, key, col in zip((-.24, 0, .24), ("actual", "rf", "sn"), ("actual_sales", *FORECAST.values())):
        ax.barh(y + offset, data[col], .21, color=COLORS[key], label=LABELS[key])
    ax.set(yticks=y, yticklabels=data[column], xlabel="Đơn vị bán", title=title)
    ax.invert_yaxis()
    ax.legend(ncols=3, loc="lower right")
    amounts(ax)
    return fig


def wape_bars(ax, data, labels, title):
    x = np.arange(len(data))
    for offset, key in zip((-.18, .18), FORECAST):
        ax.bar(x + offset, data[f"{key}_wape_pct"], .33, color=COLORS[key], label=LABELS[key])
    ax.set(xticks=x, xticklabels=labels, ylabel="WAPE (%)", title=title)
    ax.yaxis.set_major_formatter(PercentFormatter(100))


def experiments(gallery, tables):
    split = read(ROOT / "configs/experiment.json")
    schedule = pd.DataFrame([
        (name, *split[key]) for name, key in (("Train", "train"), ("Validation", "validation"), ("Test", "test"))
    ], columns=["window", "first_day", "last_day"])
    schedule["days"] = schedule.last_day - schedule.first_day + 1
    schedule.to_csv(TABLES / "split_timeline.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10.6, 4.7), layout="constrained")
    for ax, subset, heading in zip(axes, (schedule, schedule.iloc[1:]), (
        "Toàn bộ lịch sử và hai cửa sổ đánh giá", "Chi tiết 56 ngày Validation và Test",
    )):
        for row in subset.itertuples(index=False):
            color = {"Train": ACCENT, "Validation": COLORS["sn"], "Test": COLORS["rf"]}[row.window]
            ax.barh(row.window, row.days, left=row.first_day - 1, color=color, height=.52)
        ax.invert_yaxis()
        ax.set(title=heading, xlabel="Ngày kể từ trước d_1")
        if len(subset) == 2:
            ax.set_xlim(1885, 1941)
            ax.set_xticks([1885, 1913, 1941])
    gallery.save("01_train_validation_test", "Phân chia Train–Validation–Test", [table("split_timeline")],
                 "Ngày M5", "Thiết kế nghiên cứu", fig)

    receipt = read(STAGE4 / "stage3_optimization_lock.json")
    flow = pd.DataFrame({
        "thứ_tự": range(1, 7),
        "bước": ["Tuning ban đầu", "Ablation đặc trưng", "Tuning nâng cao",
                "Tối ưu đặc trưng", "Fit cuối trên mẫu", "Test sau khóa dự báo"],
        "kết_quả": [receipt["initial_tuning"], receipt["base_features"], receipt["advanced_tuning"],
                    receipt["selected_features"], f'{read(STAGE4 / "final_model_config.json")["train_rows"]:,} dòng'.replace(",", "."),
                    "WRMSSE M5 · 12 cấp"],
    })
    flow.to_csv(TABLES / "experiment_workflow.csv", index=False)
    fig, ax = chart(12.2, 3.15)
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.set_title("Quy trình thực nghiệm Random Forest", pad=15)
    for i, row in enumerate(flow.itertuples(index=False, name=None)):
        x = 1 + 2 * i
        ax.text(x, 1.45, f"{row[1]}\n{row[2]}", ha="center", va="center", fontsize=9,
                color="#1E3548", bbox={"facecolor": "#ECF5F4", "edgecolor": "#779990", "boxstyle": "round,pad=.5"})
        if i < 5:
            ax.annotate("", xy=(x + 1.21, 1.45), xytext=(x + .88, 1.45),
                        arrowprops={"arrowstyle": "->", "color": ACCENT})
    gallery.save("02_quy_trinh_rf", "Trình tự và kết quả các vòng thực nghiệm", [table("experiment_workflow")],
                 "Vòng thực nghiệm", "Thiết kế nghiên cứu", fig)
    results = tables["experiment_results"]
    for index, group, name, winner, title, horizontal in (
        (3, "initial", "rf_01_05", "RF-04", "Tuning ban đầu · RF-01–RF-05", False),
        (4, "ablation", "rf_v1_v4", "ABL_rf_v1", "Ablation · RF-v1–RF-v4", False),
        (5, "advanced", "rf_06_22", "RF-15", "Tuning nâng cao · RF-06–RF-22", True),
        (6, "features", "fe_01_10", "FE-04", "So sánh FE-01–FE-10", False),
        (7, "features", "fe04_va_all_features", "FE-04", "FE-04 và các cấu hình All Features", False),
    ):
        subset = results.loc[results.group == group].sort_values("experiment")
        if index == 7:
            subset = subset.loc[subset.experiment.isin(["FE-04", "FE-09", "FE-10"])]
        fig = experiment_bars(subset, winner, title, horizontal)
        gallery.save(f"{index:02}_{name}", title, [table("experiment_results")],
                     "Ứng viên Validation", "Quá trình thực nghiệm", fig)
    fig, ax = chart(9.7, 5.2)
    for group, label, color in (("initial", "RF-01–05", "#597D9A"), ("ablation", "Ablation", "#9A6593"),
                                 ("advanced", "RF-06–22", "#C27428"), ("features", "FE-01–10", "#17675E")):
        subset = results.loc[results.group == group]
        ax.scatter(subset.fit_seconds, subset.validation_wrmsse, color=color, alpha=.8, label=label)
    for name, offset in (("RF-04", (-45, 10)), ("RF-15", (6, 12)), ("FE-04", (5, -14)),
                         ("FE-09", (5, 8)), ("FE-10", (5, 8))):
        row = results.loc[results.experiment == name].iloc[0]
        ax.annotate(name, (row.fit_seconds, row.validation_wrmsse), textcoords="offset points", xytext=offset)
    ax.set(title="Đánh đổi WRMSSE Validation và thời gian fit", xlabel="Thời gian fit (giây)",
           ylabel="WRMSSE Validation")
    ax.legend(title="Vòng thực nghiệm")
    gallery.save("08_wrmsse_va_chi_phi_fit", "WRMSSE và thời gian fit ghi nhận, có lượt dùng lại", [table("experiment_results")],
                 "Ứng viên Validation", "Quá trình thực nghiệm", fig)


def evaluations(gallery):
    for number, name, title, origin, source, grain in (
        (9, "wrmsse_test", "WRMSSE Test", "test_wrmsse.png", "model_comparison", "Hệ phân cấp M5"),
        (10, "validation_va_test", "WRMSSE Validation và Test", "validation_vs_test_wrmsse.png", "validation_test", "Cửa sổ đánh giá"),
        (11, "wrmsse_12_cap", "WRMSSE theo 12 cấp", "test_wrmsse_by_level.png", "wrmsse_12_levels", "Cấp M5"),
        (12, "chi_so_sai_so_test", "MAE, RMSE, WAPE và Bias", "test_business_metrics.png", "model_comparison", "item–store–day"),
        (13, "feature_importance", "Importance của 15 đặc trưng logic", "final_feature_importance.png", "feature_importance", "Đặc trưng logic"),
        (14, "doanh_so_hang_ngay", "Tổng lượng bán trong 28 ngày Test", "test_daily_sales.png", "daily_sales", "Ngày toàn hệ thống"),
    ):
        gallery.save(f"{number:02}_{name}", title, [table(source), rel(STAGE4 / origin)], grain,
                     "Toàn hệ thống" if number == 14 else "Đánh giá mô hình", copy=origin)


def business_charts(gallery, tables):
    for index, group, key, label, title in (
        (15, "state_analysis", "state_id", "doanh_so_theo_bang", "Lượng bán và dự báo theo bang"),
        (16, "store_analysis", "store_id", "doanh_so_theo_cua_hang", "Lượng bán và dự báo theo cửa hàng"),
    ):
        data = tables[group].sort_values("actual_sales", ascending=False)
        gallery.save(f"{index:02}_{label}", title, [table(group)], key, "Bang và cửa hàng",
                     sales_bars(data, key, title + "  ·  " + WINDOW))
    stores = tables["store_analysis"].sort_values("rf_bias_pct")
    fig, ax = chart(9.5, 6)
    for key, delta in zip(FORECAST, (-.18, .18)):
        ax.barh(np.arange(len(stores)) + delta, stores[f"{key}_bias_pct"], .31, color=COLORS[key], label=LABELS[key])
    ax.set(yticks=range(len(stores)), yticklabels=stores.store_id, xlabel="Forecast Bias (%)",
           title="Sai lệch tổng lượng bán theo cửa hàng  ·  " + WINDOW)
    ax.axvline(0, color="#5E6A74", lw=.8)
    ax.invert_yaxis()
    ax.legend()
    gallery.save("17_bias_cua_hang", "Forecast Bias theo cửa hàng", [table("store_analysis")],
                 "item–store–day theo cửa hàng", "Bang và cửa hàng", fig)

    heat = tables["store_department"].pivot(index="store_id", columns="dept_id", values="rf_wape_pct")
    fig, ax = chart(10.2, 6)
    hm = ax.imshow(heat, cmap="YlGnBu", aspect="auto")
    for (i, j), value in np.ndenumerate(heat.to_numpy()):
        color = "white" if value > (heat.min().min() + heat.max().max()) / 2 else "#102E41"
        ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color)
    ax.set(xticks=range(len(heat.columns)), xticklabels=heat.columns,
           yticks=range(len(heat.index)), yticklabels=heat.index,
           xlabel="Bộ phận", ylabel="Cửa hàng", title="WAPE Random Forest theo cửa hàng × bộ phận")
    fig.colorbar(hm, ax=ax, label="WAPE (%)")
    gallery.save("18_heatmap_wape", "Heatmap WAPE Random Forest", [table("store_department")],
                 "item–store–day theo cửa hàng × bộ phận", "Bang, cửa hàng và bộ phận", fig)

    categories = tables["category_analysis"].sort_values("actual_sales", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for offset, key, col in zip((-.24, 0, .24), ("actual", "rf", "sn"), ("actual_sales", *FORECAST.values())):
        axes[0].bar(np.arange(len(categories)) + offset, categories[col], .22, label=LABELS[key], color=COLORS[key])
    axes[0].set(xticks=range(len(categories)), xticklabels=categories.cat_id, ylabel="Đơn vị bán", title="Lượng bán")
    axes[0].ticklabel_format(style="plain", axis="y")
    axes[0].legend()
    wape_bars(axes[1], categories, categories.cat_id, "WAPE theo ngành hàng")
    axes[1].legend()
    fig.suptitle("Kết quả theo ngành hàng  ·  " + WINDOW)
    gallery.save("19_nganh_hang", "Lượng bán và WAPE theo ngành hàng", [table("category_analysis")],
                 "item–store–day theo ngành hàng", "Ngành hàng và bộ phận", fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained", sharey=True)
    weekly = tables["weekly_analysis"]
    weekdays = tables["weekday_analysis"]
    wape_bars(axes[0], weekly, [f"Tuần {week}" for week in weekly.week], "Theo tuần Test")
    wape_bars(axes[1], weekdays, weekdays.weekday, "Theo thứ trong tuần")
    axes[0].legend(ncols=2)
    fig.suptitle("WAPE theo nhịp tuần  ·  " + WINDOW)
    gallery.save("20_tuan_va_thu", "So sánh WAPE theo tuần và theo thứ", [table("weekly_analysis"), table("weekday_analysis")],
                 "item–store–day theo tuần hoặc thứ", "Tuần, thứ và SNAP", fig)

    for index, short, trend in ((21, "under", "thiếu"), (22, "over", "thừa")):
        group = "top_" + short + "forecast"
        data = tables[group]
        fig, ax = chart(10.1, 8.5)
        label = data.item_id + " · " + data.store_id
        ax.barh(label, data.rf_error, color=COLORS["rf"])
        ax.axvline(0, color="#596875", lw=.8)
        ax.invert_yaxis()
        ax.set(title=f"Top 20 chuỗi Random Forest dự báo {trend}  ·  {WINDOW}",
               xlabel="Tổng dự báo trừ thực tế (đơn vị bán)")
        amounts(ax)
        gallery.save(f"{index:02}_top_du_bao_{short}", f"Top 20 chuỗi dự báo {trend}", [table(group)],
                     "Sản phẩm × cửa hàng, tổng 28 ngày", "Sản phẩm", fig)

    lines = tables["representative_daily"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), layout="constrained", sharex=True)
    for ax, rep in zip(axes.flat, tables["representative_series"].itertuples(index=False)):
        data = lines.loc[lines.series_id == rep.series_id].sort_values("date")
        date = pd.to_datetime(data.date)
        for key, col in zip(("actual", "rf", "sn"), ("actual_sales", *FORECAST.values())):
            ax.plot(date, data[col], color=COLORS[key], label=LABELS[key], lw=1.75)
        ax.set(title=f"{rep.selection}\n{rep.item_id} · {rep.store_id}", ylabel="Đơn vị bán", ylim=(0, None))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Dự báo của bốn chuỗi đại diện  ·  " + WINDOW)
    gallery.save("23_chuoi_dai_dien", "Actual và dự báo của các chuỗi đại diện",
                 [table("representative_series"), table("representative_daily")],
                 "Sản phẩm × cửa hàng × ngày", "Sản phẩm", fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.1), layout="constrained", sharey=True)
    snap = tables["snap_analysis"]
    state_snap = tables["snap_by_state"]
    flag = {False: "Tắt", True: "Bật"}
    wape_bars(axes[0], snap, ["SNAP " + flag[active].lower() for active in snap.snap_active], "Toàn hệ thống")
    wape_bars(axes[1], state_snap, [f"{row.state_id} · {flag[row.snap_active]}" for row in state_snap.itertuples()], "Theo bang")
    axes[0].legend(ncols=2)
    fig.suptitle("So sánh mô tả WAPE theo trạng thái SNAP  ·  " + WINDOW)
    gallery.save("24_snap", "WAPE theo trạng thái SNAP, toàn hệ thống và theo bang",
                 [table("snap_analysis"), table("snap_by_state")],
                 "item–store–day theo trạng thái SNAP", "Tuần, thứ và SNAP", fig)


def create_figures():
    tables = {path.stem: csv(path) for path in TABLES.glob("*.csv")}
    gallery = Gallery()
    experiments(gallery, tables)
    evaluations(gallery)
    business_charts(gallery, tables)
    catalog = pd.DataFrame(gallery.rows, columns=["file", "nội_dung", "nguồn_dữ_liệu", "cấp_phân_tích", "vị_trí_đề_xuất", "dpi"])
    assert len(catalog) == catalog.file.nunique() == 24
    catalog.to_csv(REPORTS / "figure_catalog.csv", index=False, encoding="utf-8-sig")
    lines = "\n".join(
        f"| [{file}](figures/{file}) | {title} | {grain} | {section} | "
        + ", ".join(f"[{source.split('/')[-1]}](../{source})" for source in sources.split(" | ")) + " |"
        for file, title, sources, grain, section, _ in gallery.rows
    )
    (REPORTS / "FIGURE_CATALOG.md").write_text(
        "# Danh mục biểu đồ chính thức\n\n"
        "Mỗi hình là PNG 230 DPI. Các lượt RF, ablation và FE sử dụng cửa sổ Validation. "
        "Nhóm đánh giá và kinh doanh dùng Test d_1914–d_1941, ngoại trừ biểu đồ so sánh hai cửa sổ.\n\n"
        "| File | Nội dung | Cấp phân tích | Vị trí đề xuất | Nguồn |\n"
        "|---|---|---|---|---|\n" + lines + "\n",
        encoding="utf-8",
    )
    return catalog
