"""So sánh bốn cấu hình LightGBM–Tweedie trên cùng kỳ validation CA_1.

Chạy từ thư mục gốc bằng python -m scripts.ml.compare_tweedie_configs.
"""

import logging

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.forecasting import recursive_forecast

matplotlib.use("Agg")
from matplotlib import pyplot as plt

LOGGER = logging.getLogger(__name__)
CONFIGURATIONS = [
    ("Cấu hình gốc", 1.1, 63, 300),
    ("Tăng Tweedie power", 1.5, 63, 300),
    ("Giảm số lá", 1.1, 31, 300),
    ("Giảm số cây", 1.1, 63, 200),
]


def plot_scores(scores, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8))
    labels = [
        f"p={str(row.power).replace('.', ',')}\n{row.num_leaves} lá\n{row.n_estimators} cây"
        for row in scores.itertuples()
    ]
    colors = np.where(scores["selected"], "#156863", "#687a8b")
    for ax, column, title, decimals in zip(
        axes,
        ["within_one_pct", "rmsse", "wrmsse"],
        ["Sai số ≤ 1 đơn vị (%) · Cao hơn tốt hơn", "RMSSE · Thấp hơn tốt hơn", "WRMSSE · Thấp hơn tốt hơn"],
        [2, 5, 5],
    ):
        values = scores[column].to_numpy()
        bars = ax.bar(labels, values, width=0.62, color=colors, zorder=3)
        number_labels = [f"{value:.{decimals}f}".replace(".", ",") for value in values]
        ax.bar_label(bars, labels=number_labels, padding=4, fontsize=9)
        ax.set_ylim(0, values.max() * 1.17)
        ax.set_title(title, fontsize=10, pad=12)
        ax.grid(axis="y", color="#e6eaed")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=9, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        for tick, color in zip(ax.get_xticklabels(), colors):
            tick.set_color(color)
    fig.suptitle("So sánh cấu hình LightGBM–Tweedie trên validation CA_1", fontweight="bold")
    fig.text(0.5, 0.035, "3.049 sản phẩm · 28 ngày · Cùng bật giá và biến động giá · Cột xanh lục là cấu hình được chọn",
             ha="center", color="#455466")
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    from run_baseline_single_file import (
        ALL_FEATURES,
        DATA_DIR,
        HORIZON,
        MODEL_PARAMS,
        ORIGIN_DAY,
        OUTPUT_DIR,
        STORE_ID,
        build_features,
        evaluate,
        load_data,
        scoped_wrmsse,
        training_revenue,
    )

    calendar, prices, wide = load_data(DATA_DIR, STORE_ID, ORIGIN_DAY)
    history_days = [f"d_{day}" for day in range(1, ORIGIN_DAY + 1)]
    target_days = [f"d_{day}" for day in range(ORIGIN_DAY + 1, ORIGIN_DAY + HORIZON + 1)]
    products = wide[["item_id", "store_id", "state_id", "cat_id", "dept_id"]]
    items = products["item_id"].tolist()
    history = wide[history_days].to_numpy(dtype=np.float32)
    revenue = training_revenue(wide, calendar, prices)
    X, y, categories, origin_rows = build_features(wide, calendar, prices, ORIGIN_DAY)
    del prices, wide
    origin_rows = origin_rows.loc[items]
    last_prices = origin_rows["sell_price"].to_numpy()
    last_price_changes = origin_rows["price_change"].to_numpy()
    predictions = []

    for name, power, leaves, trees in CONFIGURATIONS:
        LOGGER.info("Huấn luyện %s", name)
        params = {**MODEL_PARAMS, "tweedie_variance_power": power,
                  "num_leaves": leaves, "n_estimators": trees}
        model = LGBMRegressor(**params)
        model.fit(X, y, categorical_feature=["item_id"])
        forecast = recursive_forecast(
            model, history, items, STORE_ID, products["state_id"].iloc[0],
            calendar, ORIGIN_DAY, ALL_FEATURES, categories, horizon=HORIZON,
            last_prices=last_prices, last_price_changes=last_price_changes,
            model_name=name,
        )
        predictions.append(
            forecast.pivot(index="item_id", columns="target_day", values="y_pred")
            .loc[items, target_days].to_numpy(),
        )
        del model

    actual = pd.read_csv(
        DATA_DIR / "sales_train_evaluation.csv", usecols=["item_id", "store_id", *target_days],
    )
    actual = actual.loc[actual["store_id"].eq(STORE_ID)].set_index("item_id")
    observed = actual.loc[items, target_days].to_numpy()
    wrmsse = scoped_wrmsse(products, revenue, history, observed, predictions)
    rows = []
    for (name, power, leaves, trees), pred, weighted in zip(CONFIGURATIONS, predictions, wrmsse):
        scores = evaluate(observed, history, pred, name)
        rows.append({
            "configuration": name,
            "power": power,
            "num_leaves": leaves,
            "n_estimators": trees,
            "within_one_pct": round(100 * (np.abs(observed - pred) <= 1).mean(), 6),
            "rmsse": scores["RMSSE"],
            "wrmsse": weighted,
        })
    table = pd.DataFrame(rows)
    selected = table.sort_values(["wrmsse", "rmsse", "within_one_pct"],
                                 ascending=[True, True, False]).index[0]
    table["selected"] = table.index == selected
    figure_dir = OUTPUT_DIR / "figures" / f"final_{STORE_ID}_report"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_scores(table, figure_dir / "tweedie_config_selection.png")
    table.to_csv(OUTPUT_DIR / "tweedie_config_selection.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
