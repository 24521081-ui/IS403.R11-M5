"""Business V2 figures for origin-labelled demand, using the published Test grid."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, PercentFormatter

from .wrmsse_business_v2 import table
from .wrmsse_business_v2_figures import (
    COLORS, STYLE, dashboard, exceptions, figure_path, key, panel, representatives, trend, vi,
)
from .wrmsse_demand_groups import GROUP, KEY, join_labels
from .wrmsse_stage4 import read
from .wrmsse_stage5 import STAGE4

WINDOW = "Test · 25/04–22/05/2016"
NAMES = ["Thường xuyên", "Gián đoạn"]
PALETTE = ["#4565A4", "#995D85"]
TITLES = {
    "composition": "Cơ cấu chuỗi, lượng bán và doanh thu",
    "metrics": "Sáu chỉ số Random Forest theo nhóm nhu cầu",
    "region": "WAPE theo cửa hàng và ngành hàng",
    "examples": "Bốn chuỗi theo phân vị sai số",
    "adi": "Sai số dự báo theo khoảng ADI",
}
TIME_TITLES = {
    "sales": "Lượng bán và dự báo từng ngày",
    "error": "Sai lệch ròng từng ngày",
    "daily_quality": "WAPE và Bias từng ngày",
    "weekly": "Chất lượng dự báo trong bốn tuần Test",
    "weekday": "Chất lượng dự báo theo thứ trong tuần",
    "cumulative": "Lượng bán và dự báo tích lũy",
}
GROUPS = ["frequent", "intermittent"]
METRICS = [
    ("rf_mae", "MAE", "đơn vị/chuỗi–ngày"), ("rf_rmse", "RMSE", "đơn vị/chuỗi–ngày"),
    ("rf_wape_pct", "WAPE", "%"), ("rf_bias_pct", "Bias", "%"),
    ("rf_mean_rmsse", "RMSSE trung bình", "cấp sản phẩm–cửa hàng"),
    ("rf_conditional_weighted_rmsse", "Weighted RMSSE cấp 12", "trọng số chuẩn hóa trong nhóm"),
]


def composition(t, detail):
    data = t["summary"].query("split == 'Test'").set_index(GROUP).loc[["frequent", "intermittent"]]
    fig, axes = panel(TITLES["composition"], WINDOW + " · doanh thu lịch sử d_1886–d_1913", 12, 4.7, 1, 2)
    matrices = [data[["series_share_pct", "actual_share_pct", "revenue_share_pct"]].to_numpy(),
                np.array([100 - data.zero_day_pct, data.zero_day_pct])]
    for ax, matrix, labels, legend, colors in zip(
        axes, matrices, [["Số chuỗi", "Lượng bán Test", "Doanh thu lịch sử"], NAMES],
        [NAMES, ["Có bán", "Không bán"]], [PALETTE, ["#297B78", "#718393"]]
    ):
        left = np.zeros(len(labels))
        for values, color, label in zip(matrix, colors, legend):
            marks = ax.barh(labels, values, left=left, height=.56, color=color, label=label)
            ax.bar_label(marks, labels=[vi(v, 1) + "%" for v in values], label_type="center", color="white", fontsize=10)
            left += values
        ax.set_xlim(0, 100)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(100))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False, ncols=2)
    axes[0].set_title("Tỷ trọng trong toàn bộ lưới")
    axes[1].set_title("Tỷ lệ ngày có và không có lượng bán")
    return fig


def metrics(t, detail):
    data = t["metrics"].query("split == 'Test'").set_index(GROUP).loc[GROUPS]
    fig, axes = panel(TITLES["metrics"], WINDOW + " · nhãn tại d_1913", 12.6, 7.8, 2, 3)
    for ax, (metric, title, unit) in zip(axes.flat, METRICS):
        vals = data[metric]
        bars = ax.bar([g.title() for g in GROUPS], vals, .52, color=PALETTE)
        ax.bar_label(bars, labels=[vi(v, 3) for v in vals], padding=4)
        ax.axhline(0, color="#596574", lw=.8)
        ax.margins(y=.23)
        ax.set_title(f"{title}\n{unit}", fontsize=11)
        ax.grid(axis="y", alpha=.15)
    return fig


def date_axis(ax, dates):
    dates = pd.DatetimeIndex(dates)
    ax.set_xticks(dates[[0, 4, 8, 12, 16, 20, 24, 27]])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax.grid(axis="y", alpha=.18)


def volume_figure(t, slug):
    daily = t["daily"]
    stats = t["metrics"].query("split == 'Test'").set_index(GROUP)
    fig, axes = panel(TIME_TITLES[slug], WINDOW + " · trục Y riêng cho từng nhóm", 11.5, 7.4, 2, 1, sharex=True)
    for ax, group in zip(axes, GROUPS):
        data = daily[daily[GROUP].eq(group)].sort_values("day")
        dates = pd.to_datetime(data.date)
        r = stats.loc[group]
        label = group.title()
        if slug == "error":
            values = data.rf_error
            ax.bar(dates, values, color=np.where(values >= 0, "#2468A3", "#A54A18"), width=.72)
            ax.axhline(0, color="#45576b", lw=1)
            extremes = []
            for side, idx in (("Thừa", values.idxmax()), ("Thiếu", values.idxmin())):
                v = values.loc[idx]
                if (side == "Thừa" and v > 0) or (side == "Thiếu" and v < 0):
                    extremes.append(f"{side} nhiều nhất {dates.loc[idx]:%d/%m} ({vi(v, 1)})")
                else:
                    extremes.append(f"Không có ngày dự báo {side.lower()}")
            ax.set_title(label + "   ·   " + "   ·   ".join(extremes), fontsize=11)
            ax.set_ylabel("Forecast − Actual (đơn vị)")
        else:
            suffix = "_cumulative" if slug == "cumulative" else ""
            actual, forecast = data["actual_sales" + suffix], data["random_forest_forecast" + suffix]
            ax.plot(dates, actual, color=COLORS["actual"], lw=2, label="Actual")
            ax.plot(dates, forecast, color=COLORS["rf"], lw=2, ls="--", label="Random Forest Forecast")
            ax.fill_between(dates, actual, forecast, color=COLORS["rf"], alpha=.14)
            if slug == "sales":
                note = (f"{vi(r.observations/28)} chuỗi   ·   Actual {vi(r.actual_sales)}   ·   "
                        f"Forecast {vi(r.random_forest_forecast, 2)}   ·   WAPE {vi(r.rf_wape_pct,2)}%   ·   Bias {vi(r.rf_bias_pct,2)}%")
            else:
                note = f"Chênh lệch cuối kỳ {vi(r.rf_error,2)} đơn vị ({vi(r.rf_bias_pct,2)}%)"
            ax.set_title(label + "   ·   " + note, fontsize=10)
            ax.set_ylabel("Đơn vị bán" + (" tích lũy" if suffix else "/ngày"))
        date_axis(ax, dates)
    if slug != "error":
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", frameon=False, ncols=2)
    return fig


def quality_figure(t, slug):
    daily = slug == "daily_quality"
    data = t["daily" if daily else slug]
    fig, axes = panel(TIME_TITLES[slug], WINDOW + " · tính trên từng quan sát chuỗi–ngày", 11.8,
                      6.7, 2, 1 if daily else 2, sharex=True)
    measures = METRICS[2:4] if daily else [METRICS[i] for i in (2, 0, 1, 3)]
    for ax, (metric, title, unit) in zip(axes.flat, measures):
        for i, group in enumerate(GROUPS):
            sub = data.loc[data[GROUP].eq(group)]
            if daily:
                dates = pd.to_datetime(sub.date)
                ax.plot(dates, sub[metric], color=PALETTE[i], marker=("o", "s")[i], ms=3, label=group.title())
                date_axis(ax, dates)
            else:
                x = np.arange(len(sub))
                ax.bar(x + (i-.5)*.38, sub[metric], width=.36, color=PALETTE[i], label=group.title())
                labels = ["Tuần " + str(w) for w in sub.week] if slug == "weekly" else sub.weekday
                ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel(f"{title} ({unit})")
        ax.grid(axis="y", alpha=.15)
        if metric == "rf_bias_pct":
            ax.axhline(0, lw=.9, color="#45576b")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="outside lower center", frameon=False, ncols=2)
    return fig


def time_figure(t, slug):
    return volume_figure(t, slug) if slug in ("sales", "error", "cumulative") else quality_figure(t, slug)


def region(t, detail):
    matrices = [t[name].pivot(index=dim, columns=GROUP, values="rf_wape_pct").sort_values("intermittent", ascending=False)
                for name, dim in (("store", "store_id"), ("category", "cat_id"))]
    fig, axes = panel(TITLES["region"], WINDOW + " · WAPE Random Forest (%) · xếp theo nhóm gián đoạn", 10.6, 6, 1, 2)
    norm = Normalize(0, max(m.to_numpy().max() for m in matrices))
    for ax, values, title in zip(axes, matrices, ["Cửa hàng", "Ngành hàng"]):
        img = ax.imshow(values, cmap="Blues", norm=norm, aspect="auto")
        ax.set(xticks=[0, 1], xticklabels=NAMES, yticks=range(len(values)), yticklabels=values.index, title=title)
        for (i, j), value in np.ndenumerate(values.to_numpy()):
            ax.text(j, i, vi(value, 1) + "%", ha="center", va="center", fontsize=11,
                    color="white" if norm(value) > .61 else "#17283C")
    fig.colorbar(img, ax=axes, label="WAPE (%)", shrink=.85)
    return fig


def examples(t, detail):
    fig, axes = panel(TITLES["examples"], WINDOW + " · P10 và P90 theo WAPE RF trong mỗi nhóm có bán", 13, 8, 2, 2)
    for ax, r in zip(axes.flat, t["representatives"].itertuples()):
        days = detail.loc[detail.series_id.eq(r.series_id)].sort_values("day")
        trend(ax, days)
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.set(ylim=(0, None), ylabel="Đơn vị bán")
        ax.set_title(f"{r.item_id} / {r.store_id}  ·  {r.demand_group.title()}  ·  ADI {vi(r.ADI,2)}\n"
                     f"P{int(r.quantile*100)}    RF: WAPE {vi(r.rf_wape_pct,1)}%, Bias {vi(r.rf_bias_pct,1)}%\n"
                     f"Seasonal Naive: WAPE {vi(r.sn_wape_pct,1)}%, Bias {vi(r.sn_bias_pct,1)}%", fontsize=10)
    key(fig, loc="outside lower center")
    return fig


def adi(t, detail):
    data = t["adi"].query("n_series > 0")
    fig, axes = panel(TITLES["adi"], WINDOW + " · khoảng đóng trái, mở phải · n là số chuỗi", 12, 5.2, 1, 2)
    labels = [a.replace("1.32", "1,32").replace(", ", " ; ") + f"\nn={vi(n)}" for a, n in zip(data.ADI_bin, data.n_series)]
    for ax, metric, title in zip(axes, ["wape_pct", "mae"], ["WAPE (%)", "MAE (đơn vị/ngày/chuỗi)"]):
        for model, offset in (("rf", -.19), ("sn", .19)):
            vals = data[model + "_" + metric]
            bars = ax.bar(np.arange(len(data)) + offset, vals, .35, color=COLORS[model])
            ax.bar_label(bars, labels=[vi(v, 0 if metric == "wape_pct" else 2) for v in vals], padding=3, fontsize=8)
        ax.set_xticks(range(len(data)), labels, fontsize=8.5)
        ax.set_title(title)
        ax.margins(y=.15)
        ax.grid(axis="y", alpha=.18)
    key(fig, ("rf", "sn"), loc="outside lower center")
    return fig


def updated_v2(t, detail):
    labels = detail[KEY + ["ADI", GROUP]].drop_duplicates()
    published = {name: table("business_v2_" + name) for name in ("daily", "exceptions", "representatives", "representatives_daily")}
    for name in ("exceptions", "representatives"):
        published[name] = join_labels(published[name], labels)
    meta = read(STAGE4 / "test_metrics.json")
    shares = t["summary"].query("split == 'Test'").series_share_pct
    window = WINDOW + f" · {vi(shares.iloc[0],1)}% chuỗi thường xuyên và {vi(shares.iloc[1],1)}% chuỗi gián đoạn"
    yield "dashboard", dashboard(published, {"wrmsse_12_levels": table("wrmsse_12_levels")}, meta, window)
    fig = exceptions(published, {}, meta, WINDOW)
    fig.set_size_inches(15, 8)
    for ax, side in zip(fig.axes, ["Thiếu", "Thừa"]):
        group = published["exceptions"].loc[published["exceptions"].direction.eq(side)]
        ax.set_yticks(range(len(group)), group.item_id + " · " + group.store_id + " · " + group[GROUP].str.title(), fontsize=9)
    yield "exceptions", fig
    fig = representatives(published, {}, meta, WINDOW)
    fig.set_size_inches(14.2, 8)
    for ax, row in zip(fig.axes, published["representatives"].itertuples()):
        ax.set_title(ax.get_title() + f"\n{row.demand_group.title()} · ADI {vi(row.ADI,2)}", fontsize=10)
    yield "representatives", fig


def draw(directory, detail, tables):
    def save(fig, name):
        fig.savefig(directory / name, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
    with plt.rc_context(STYLE):
        for slug, maker in zip(TITLES, (composition, metrics, region, examples, adi), strict=True):
            save(maker(tables, detail), "demand_group_" + slug + ".png")
        for slug in TIME_TITLES:
            save(time_figure(tables, slug), "demand_group_" + slug + ".png")
        for slug, fig in updated_v2(tables, detail):
            save(fig, figure_path(slug))
