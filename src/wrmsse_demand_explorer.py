"""Inspect one sealed 28-day forecast in notebook sessions."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, clear_output, display

from .wrmsse_business_v2_figures import COLORS, STYLE, panel, vi
from .wrmsse_demand_figures import WINDOW, date_axis
from .wrmsse_demand_groups import ENRICHED, GROUP, INPUT, KEY, join_labels
from .wrmsse_stage5 import FORECAST, TABLES, csv


def series_index():
    columns = [*KEY, "rf_mae", "rf_rmse", "rf_wape_pct", "rf_bias_pct", "rf_rmsse",
               "actual_sales", FORECAST["rf"], "zero_actual_days"]
    return join_labels(csv(TABLES / "business_v2_series.csv", usecols=columns),
                       csv(INPUT / "demand_groups_origin_1913.csv")).set_index([GROUP, *KEY]).sort_index()


def series_png(index, group, item, store):
    try:
        r = index.loc[(group, item, store)]
    except KeyError as exc:
        raise ValueError("Tổ hợp nhóm, Item ID và Store ID không có tại d_1913.") from exc
    data = pd.read_parquet(ENRICHED, columns=["date", "actual_sales", FORECAST["rf"]],
                           filters=[(GROUP, "=", group), ("item_id", "=", item), ("store_id", "=", store)])
    data = data.sort_values("date")
    dates = pd.to_datetime(data.date)
    if not pd.DatetimeIndex(dates).equals(pd.date_range("2016-04-25", periods=28)):
        raise ValueError("Chuỗi phải có đủ 28 ngày Test")
    def number(value, digits=2, suffix=""):
        return vi(value, digits) + suffix if pd.notna(value) else "KXĐ"

    with plt.rc_context(STYLE):
        fig, ax = panel(f"{item} / {store}   ·   {group.title()}   ·   ADI {number(r.ADI)}", WINDOW, 10.5, 4.8)
        for col, key, label, line in [("actual_sales", "actual", "Actual", "-"),
                                      (FORECAST["rf"], "rf", "Random Forest Forecast", "--")]:
            ax.plot(dates, data[col], color=COLORS[key], lw=1.8, ls=line, marker="o", ms=3, label=label)
        ax.set_title(f"MAE {number(r.rf_mae)}   ·   RMSE {number(r.rf_rmse)}   ·   WAPE {number(r.rf_wape_pct, 2, '%')}"
                     f"   ·   Bias {number(r.rf_bias_pct, 2, '%')}   ·   RMSSE {number(r.rf_rmsse, 3)}\n"
                     f"Actual {vi(r.actual_sales)}   ·   Forecast {vi(r.random_forest_forecast, 2)}   ·   "
                     f"{vi(r.zero_actual_days)}/28 ngày có lượng bán bằng 0", fontsize=10)
        ax.set(ylabel="Đơn vị bán/ngày", ylim=(0, None))
        date_axis(ax, dates)
        ax.legend(loc="best", frameon=False)
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight", pad_inches=.18)
        plt.close(fig)
    peak = data.iloc[data.actual_sales.argmax()]
    caption = (f"Ngày bán cao nhất đầu tiên {pd.Timestamp(peak.date):%d/%m} ghi nhận {vi(peak.actual_sales)} đơn vị, "
               f"RF dự báo {vi(peak.random_forest_forecast, 2)}. ")
    if r.actual_sales == 0:
        caption += "Cả kỳ không phát sinh lượng bán. WAPE và Bias không xác định."
    else:
        delta = r.random_forest_forecast - r.actual_sales
        caption += f"Tổng dự báo trừ lượng bán là {vi(delta, 2)} đơn vị khi đối chiếu kế hoạch 28 ngày."
    return buffer.getvalue(), caption


def inspect_series(index, group, item, store, export=False):
    png, caption = series_png(index, group, item, store)
    display(Image(data=png, width=1000))
    print(caption)
    if export:
        path = Path.cwd() / f"RF_{group}_{item}_{store}.png"
        path.write_bytes(png)
        print(f"PNG · {path}")


def choose_series(index, group, item, store, prompt=False, export=False):
    """Use linked widgets where available, or kernel input prompts on request."""
    if prompt:
        group = input("Demand group (frequent / intermittent) · ").strip().lower()
        if group not in index.index.levels[0]:
            raise ValueError("Chọn frequent hoặc intermittent.")
        items = index.loc[group].index.get_level_values("item_id").unique()
        print(f"{len(items)} Item ID trong nhóm. Ví dụ: {', '.join(items[:8])}")
        item = input("Item ID · ").strip()
        if item not in items:
            raise ValueError("Item ID không thuộc nhóm đã chọn.")
        stores = index.loc[(group, item)].index
        store = input(f"Store ID ({', '.join(stores)}) · ").strip()
    try:
        import ipywidgets as widgets
    except ImportError:
        inspect_series(index, group, item, store, export)
        return

    groups = widgets.Dropdown(options=list(index.index.levels[0]), value=group, description="Nhóm")
    def items():
        return index.loc[groups.value].index.get_level_values("item_id").unique().tolist()
    products = widgets.Dropdown(options=items(), value=item, description="Item ID")
    def stores():
        return index.loc[(groups.value, products.value)].index.tolist()
    locations = widgets.Dropdown(options=stores(), value=store, description="Store ID")
    output = widgets.Output()
    button = widgets.Button(description="Xuất PNG")

    def render(save=False):
        with output:
            clear_output(wait=True)
            inspect_series(index, groups.value, products.value, locations.value, save)

    def update(change):
        # Replace dependent options before drawing a newly selected group/item.
        for control in (groups, products, locations):
            control.unobserve(update, names="value")
        try:
            if change["owner"] is groups:
                products.options = items()
            if change["owner"] is not locations:
                locations.options = stores()
            render()
        finally:
            for control in (groups, products, locations):
                control.observe(update, names="value")
    for control in (groups, products, locations):
        control.observe(update, names="value")
    button.on_click(lambda _: render(True))
    display(widgets.VBox([groups, products, locations, button, output]))
    render(export)
