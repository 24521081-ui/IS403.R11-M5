"""Dự báo M5 cho CA_1 bằng LightGBM–Tweedie và Seasonal Naive.

Tất cả các bước từ Đọc dữ liệu (kèm sell_prices) -> Tiền xử lý & Giá -> Tạo đặc trưng (Lags, Rolling, Price, Price Change)
-> Huấn luyện LightGBM -> Dự báo 28 ngày -> Chấm điểm chung và theo nhóm sản phẩm.

Chạy:
    python run_baseline_single_file.py
"""

import gc
import logging
import sys
from pathlib import Path
from time import perf_counter

_PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_PIPELINE_DIR.parent / "lightgbm_twopart"))
sys.path.insert(0, str(_PIPELINE_DIR))

import matplotlib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from scripts.score_c7_wrmsse import LEVELS, rmsse_scale
from src.demand_groups import categorize_intermittency, compute_zero_ratio

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# ==============================================================================
# CẤU HÌNH BASELINE CHÍNH THỨC CỦA NHÓM (CÓ BẬT GIÁ BÁN & BIẾN ĐỘNG GIÁ)
# ==============================================================================
STORE_ID = "CA_1"
ORIGIN_DAY = 1885  # Ngày kết thúc tập Train
HORIZON = 28  # Số ngày cần dự báo (d_1886 -> d_1913)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"
CHART_METRICS = ["MAE", "RMSE", "RMSSE", "WRMSSE"]
MODEL_NAMES = ("LightGBM–Tweedie (Giá + Price Change)", "Seasonal Naive (Mốc chuẩn)")
MODEL_LABELS = ("LightGBM–Tweedie", "Seasonal Naive")
MODEL_COLORS = ("#1f77b4", "#ff7f0e")

MODEL_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "num_leaves": 63,
    "min_child_samples": 50,
    "random_state": 42,
    "n_jobs": 6,
    "device_type": "cpu",
    "force_col_wise": True,
    "deterministic": True,
    "verbosity": -1,
}

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
HISTORY_FEATURES = [f"lag_{lag}" for lag in LAGS] + [
    f"rolling_mean_{w}" for w in WINDOWS
]
BASE_FEATURES = [
    "item_id",
    "day_of_week",
    "month",
    "event_flag",
    "snap",
] + HISTORY_FEATURES
ALL_FEATURES = BASE_FEATURES + ["sell_price", "price_change"]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S"
)
LOGGER = logging.getLogger("m5_single_file")


# ==============================================================================
# BƯỚC 1: ĐỌC DỮ LIỆU (LỊCH, GIÁ BÁN, DOANH SỐ)
# ==============================================================================
def load_data(data_dir, store_id, origin_day):
    LOGGER.info("1/6. Đọc dữ liệu cho cửa hàng %s đến ngày d_%d", store_id, origin_day)
    # 1.1 Lịch
    calendar = pd.read_csv(data_dir / "calendar.csv")
    dates = pd.to_datetime(calendar["date"])
    calendar["day"] = calendar["d"].str.removeprefix("d_").astype(int)
    calendar["day_of_week"] = dates.dt.dayofweek.astype("int8")
    calendar["month"] = dates.dt.month.astype("int8")
    calendar["event_flag"] = (
        calendar[["event_name_1", "event_name_2"]].notna().any(axis=1).astype("int8")
    )
    for state in ["CA", "TX", "WI"]:
        calendar[f"snap_{state}"] = calendar[f"snap_{state}"].astype("int8")

    # 1.2 Giá bán của cửa hàng
    parts_price = []
    for chunk in pd.read_csv(
        data_dir / "sell_prices.csv",
        chunksize=100000,
        dtype={"sell_price": "float32", "wm_yr_wk": "int32"},
    ):
        p = chunk.loc[chunk["store_id"] == store_id]
        if not p.empty:
            parts_price.append(p)
    prices = pd.concat(parts_price, ignore_index=True)

    # 1.3 Doanh số bán hàng cửa hàng
    days = [f"d_{d}" for d in range(1, origin_day + 1)]
    dtypes = {
        "id": "string",
        "item_id": "string",
        "store_id": "string",
        "state_id": "string",
        "cat_id": "string",
        "dept_id": "string",
        **{d: "float32" for d in days},
    }
    cols = ["id", "item_id", "store_id", "state_id", "cat_id", "dept_id"] + days

    parts_sales = []
    for chunk in pd.read_csv(
        data_dir / "sales_train_evaluation.csv",
        usecols=cols,
        dtype=dtypes,
        chunksize=1000,
    ):
        selected = chunk.loc[chunk.store_id == store_id]
        if not selected.empty:
            parts_sales.append(selected)

    wide = (
        pd.concat(parts_sales, ignore_index=True)
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    return calendar, prices, wide


# ==============================================================================
# BƯỚC 2 & 3: TIỀN XỬ LÝ VÀ TẠO ĐẶC TRƯNG (KÈM SELL_PRICE & PRICE_CHANGE)
# ==============================================================================
def build_features(wide, calendar, prices, origin_day):
    LOGGER.info("2/6. Chuyển bảng rộng sang bảng dài, ghép lịch và giá bán")
    days = [f"d_{d}" for d in range(1, origin_day + 1)]
    meta = ["item_id", "store_id", "state_id"]

    melted = wide[meta + days].melt(id_vars=meta, var_name="d", value_name="sales")
    melted["day"] = melted["d"].str.removeprefix("d_").astype("int16")

    # Ghép lịch
    cal_subset = calendar[
        [
            "d",
            "wm_yr_wk",
            "day_of_week",
            "month",
            "event_flag",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ]
    ]
    frame = melted.merge(cal_subset, on="d", how="left")
    state = wide["state_id"].iloc[0]
    frame["snap"] = frame[f"snap_{state}"].astype("float32")

    # Ghép giá bán
    frame = frame.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    frame = frame.sort_values(["store_id", "item_id", "day"]).reset_index(
        drop=True
    )

    # ffill giá quá khứ theo chuỗi sản phẩm (không bfill)
    frame["sell_price"] = frame.groupby(
        ["store_id", "item_id"], observed=True
    )["sell_price"].ffill()

    # Tính biến động giá so với 7 ngày trước (1 tuần trước)
    lag_price = frame.groupby(["store_id", "item_id"], observed=True)[
        "sell_price"
    ].shift(7)
    frame["price_change"] = np.where(
        (lag_price > 0) & frame["sell_price"].notna(),
        (frame["sell_price"] - lag_price) / lag_price,
        0.0,
    ).astype("float32")

    LOGGER.info("3/6. Tạo đặc trưng trễ (Lags) và trung bình trượt (Rolling)")
    groups = frame.groupby(["store_id", "item_id"], observed=True)["sales"]

    for lag in LAGS:
        frame[f"lag_{lag}"] = groups.shift(lag).astype("float32")

    for window in WINDOWS:
        frame[f"rolling_mean_{window}"] = groups.transform(
            lambda s: s.shift(1).rolling(window, min_periods=window).mean()
        ).astype("float32")

    # Lưu lại ảnh chụp giá tại ngày origin để dùng cho dự báo đệ quy
    origin_rows = frame.loc[frame["day"] == origin_day].set_index("item_id")

    # Loại bỏ 28 ngày đầu chưa đủ lịch sử cho lag_28
    frame = frame.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)
    frame["item_id"] = frame["item_id"].astype("category")

    X = frame[ALL_FEATURES]
    y = frame["sales"].to_numpy(dtype=np.float32)
    categories = list(X["item_id"].cat.categories)

    return X, y, categories, origin_rows


# ==============================================================================
# BƯỚC 4: HUẤN LUYỆN LIGHTGBM TWEEDIE
# ==============================================================================
def train_lightgbm(X, y):
    LOGGER.info(
        "4/6. Huấn luyện LightGBM–Tweedie (có sell_price và price_change)"
    )
    model = LGBMRegressor(**MODEL_PARAMS)
    start = perf_counter()
    model.fit(X, y, categorical_feature=["item_id"])
    train_time = perf_counter() - start
    LOGGER.info("Huấn luyện hoàn thành trong %.2f giây", train_time)
    return model, train_time


# ==============================================================================
# BƯỚC 5: DỰ BÁO ĐỆ QUY 28 NGÀY
# ==============================================================================
def recursive_predict(
    model,
    history_matrix,
    items,
    calendar,
    state,
    categories,
    origin_rows,
    origin_day,
    horizon=28,
):
    LOGGER.info("5/6. Dự báo đệ quy 28 ngày cho toàn bộ sản phẩm")
    temporary_lgb = history_matrix[:, -28:].copy()
    temporary_naive = history_matrix[:, -28:].copy()
    cal_indexed = calendar.set_index("d")

    # Lấy giá và biến động giá tại origin (last_known policy)
    last_prices = origin_rows.loc[items, "sell_price"].to_numpy(dtype=np.float32)
    last_price_changes = origin_rows.loc[items, "price_change"].to_numpy(
        dtype=np.float32
    )

    lgb_preds = []
    naive_preds = []

    for step in range(1, horizon + 1):
        target_d = f"d_{origin_day + step}"
        cal_row = cal_indexed.loc[target_d]

        # Vector đặc trưng ngày tiếp theo
        X_step = pd.DataFrame(
            {"item_id": pd.Categorical(items, categories=categories)}
        )
        X_step["day_of_week"] = np.int8(cal_row["day_of_week"])
        X_step["month"] = np.int8(cal_row["month"])
        X_step["event_flag"] = np.int8(cal_row["event_flag"])
        X_step["snap"] = np.float32(cal_row[f"snap_{state}"])

        # Lags và Rolling
        for lag in LAGS:
            X_step[f"lag_{lag}"] = temporary_lgb[:, -lag].astype("float32")
        for window in WINDOWS:
            X_step[f"rolling_mean_{window}"] = (
                temporary_lgb[:, -window:]
                .mean(axis=1, dtype=np.float64)
                .astype("float32")
            )

        # Đặc trưng giá
        X_step["sell_price"] = last_prices
        # Tuần đầu dùng biến động tại origin; sau tuần đầu, do giá giữ nguyên nên biến động = 0
        X_step["price_change"] = (
            last_price_changes if step <= 7 else np.zeros_like(last_price_changes)
        )

        # Dự báo LightGBM
        pred_lgb = np.asarray(
            model.predict(X_step[ALL_FEATURES]), dtype=np.float64
        )
        pred_lgb = np.maximum(pred_lgb, 0).astype("float32")
        lgb_preds.append(pred_lgb)

        # Dự báo Seasonal Naive (mùa vụ tuần trước)
        pred_naive = temporary_naive[:, -7].copy()
        naive_preds.append(pred_naive)

        # Cập nhật đệ quy cho bước tiếp theo
        temporary_lgb = np.column_stack([temporary_lgb[:, 1:], pred_lgb])
        temporary_naive = np.column_stack([temporary_naive[:, 1:], pred_naive])

    lgb_matrix = np.column_stack(lgb_preds)
    naive_matrix = np.column_stack(naive_preds)
    return lgb_matrix, naive_matrix


# ==============================================================================
# BƯỚC 6: ĐÁNH GIÁ VÀ XUẤT KẾT QUẢ
# ==============================================================================
def training_revenue(wide, calendar, prices):
    days = [f"d_{day}" for day in range(ORIGIN_DAY - 27, ORIGIN_DAY + 1)]
    weeks = calendar.set_index("d").loc[days, "wm_yr_wk"]
    keys = ["store_id", "item_id"]
    revenue = np.zeros(len(wide), dtype=np.float64)
    for day, week in weeks.items():
        week_prices = prices.loc[prices["wm_yr_wk"] == week, keys + ["sell_price"]]
        price = wide[keys].merge(week_prices, how="left", on=keys, validate="one_to_one")
        price = price["sell_price"].to_numpy(dtype=np.float64)
        units = wide[day].to_numpy(dtype=np.float64)
        if not np.isfinite(price[units > 0]).all():
            raise ValueError(f"Thiếu giá để tính doanh thu tại {day}.")
        revenue += units * np.nan_to_num(price)
    return revenue


def scoped_wrmsse(metadata, revenue, history, actual, predictions):
    """Áp dụng 12 cấp M5 trên phạm vi đang chấm và chuẩn hóa trọng số trong phạm vi."""
    if revenue.sum() == 0:
        return np.full(len(predictions), np.nan)

    varying = set(metadata.columns[metadata.nunique() > 1])
    # Các cấp có cùng cách gộp dùng chung phép tính, nhưng vẫn có trọng số 1/12 mỗi cấp.
    scores_by_keys = {}
    level_scores = []
    for level in LEVELS:
        keys = tuple(sorted(set(level) & varying))
        if keys not in scores_by_keys:
            if keys:
                index = pd.MultiIndex.from_frame(metadata[list(keys)])
                codes, series = pd.factorize(index)
                n_series = len(series)
            else:
                codes = np.zeros(len(metadata), dtype=int)
                n_series = 1

            def aggregate(values):
                totals = np.zeros((n_series, values.shape[1]), dtype=np.float64)
                np.add.at(totals, codes, values)
                return totals

            weights = np.bincount(codes, weights=revenue, minlength=n_series)
            weighted = weights > 0
            scale, valid = rmsse_scale(aggregate(history))
            if not valid[weighted].all():
                raise ValueError(f"Không có thang RMSSE cho chuỗi có doanh thu ở cấp {level}.")
            observed = aggregate(actual)[weighted]
            level_weights = weights[weighted] / weights.sum()
            scores_by_keys[keys] = [
                np.dot(level_weights, np.sqrt(
                    np.square(observed - aggregate(pred)[weighted]).mean(axis=1)
                    / scale[weighted]
                ))
                for pred in predictions
            ]
        level_scores.append(scores_by_keys[keys])
    return np.mean(level_scores, axis=0)


def evaluate(actual_matrix, train_history, pred_matrix, model_name):
    actual = actual_matrix.astype(float)
    pred = pred_matrix.astype(float)

    error = actual - pred
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.square(error).mean()))

    mean_bias = float(error.mean())
    std_residual = float(error.std())
    within_1_pct = float((np.abs(error) <= 1.0).mean() * 100)

    # RMSSE theo M5
    mse_per_item = np.square(actual - pred).mean(axis=1)

    scales, valid = rmsse_scale(train_history)
    item_rmsse = np.sqrt(mse_per_item[valid] / scales[valid])
    mean_rmsse = float(item_rmsse.mean()) if valid.any() else np.nan

    return {
        "Mô hình": model_name,
        "RMSSE": round(mean_rmsse, 5),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Độ chệch (Bias)": round(mean_bias, 4),
        "Độ lệch chuẩn số dư (Std)": round(std_residual, 4),
        "Tỷ lệ lệch <= 1 cái (%)": round(within_1_pct, 2),
    }


def evaluate_scope(products, revenue, history, actual, predictions):
    weighted = scoped_wrmsse(products, revenue, history, actual, predictions)
    rows = []
    for name, prediction, wrmsse in zip(MODEL_NAMES, predictions, weighted):
        row = evaluate(actual, history, prediction, name)
        row["WRMSSE"] = round(float(wrmsse), 5)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_comparison(scores, path, title, daily_totals=None):
    height = 5 if daily_totals is None else 8
    fig, axes = plt.subplots(1 if daily_totals is None else 2, 1, figsize=(9.5, height))
    ax = axes if daily_totals is None else axes[0]
    x = np.arange(len(CHART_METRICS))
    for index, (label, color) in enumerate(zip(MODEL_LABELS, MODEL_COLORS)):
        values = scores.iloc[index][CHART_METRICS].to_numpy(dtype=float)
        bars = ax.bar(
            x + (index - 0.5) * 0.34, np.nan_to_num(values), 0.34,
            label=label, color=color,
        )
        labels = [f"{value:.3f}" if np.isfinite(value) else "N/A" for value in values]
        ax.bar_label(bars, labels=labels, padding=4)
    ax.set_title(title)
    ax.set_xticks(x, CHART_METRICS)
    ax.set_ylabel("Giá trị sai số")
    ax.margins(y=0.22)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    if daily_totals is not None:
        daily = axes[1]
        days = np.arange(1, len(daily_totals[0]) + 1)
        for values, label, color in zip(
            daily_totals, ("Thực tế", *MODEL_LABELS), ("#353535", *MODEL_COLORS)
        ):
            daily.plot(days, values, label=label, color=color, linewidth=1.8)
        daily.set(
            title="Tổng lượng bán theo ngày của nhóm",
            xlabel=f"Ngày dự báo (1 = d_{ORIGIN_DAY + 1})",
            ylabel="Số lượng bán",
        )
        daily.set_xticks([1, 7, 14, 21, 28])
        daily.grid(alpha=0.3)
        daily.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def export_group_results(products, revenue, history, actual, predictions, day_cols, figure_dir):
    output = OUTPUT_DIR / "predictions"
    output.mkdir(parents=True, exist_ok=True)
    horizons = len(day_cols)
    forecasts = products.loc[products.index.repeat(horizons)].reset_index(drop=True)
    forecasts["target_day"] = np.tile(day_cols, len(products))
    forecasts["y_true"] = actual.ravel()
    forecasts["y_pred_lightgbm_tweedie"] = predictions[0].ravel()
    forecasts["y_pred_seasonal_naive"] = predictions[1].ravel()
    forecasts.to_csv(output / f"baseline_{STORE_ID}_validation.csv", index=False, encoding="utf-8-sig")

    group_scores = []
    for column, prefix in (("intermittency_group", "intermittency"), ("cat_id", "category")):
        for label in sorted(products[column].unique()):
            selected = products[column].eq(label).to_numpy(dtype=bool)
            group_actual = actual[selected]
            group_predictions = tuple(pred[selected] for pred in predictions)
            size = int(selected.sum())
            scores = evaluate_scope(
                products.loc[selected], revenue[selected], history[selected],
                group_actual, group_predictions,
            )
            scores.insert(0, "group_type", column)
            scores.insert(1, "group", label)
            scores.insert(2, "n_items", size)
            group_scores.append(scores)
            slug = label.split(".", maxsplit=1)[0] if prefix == "intermittency" else label.lower()
            plot_comparison(
                scores, figure_dir / f"{prefix}_{slug}.png", f"{STORE_ID} · {label} · {size:,} sản phẩm",
                tuple(matrix.sum(axis=0, dtype=np.float64) for matrix in (group_actual, *group_predictions)),
            )
    pd.concat(group_scores, ignore_index=True).to_csv(
        OUTPUT_DIR / "baseline_group_metrics.csv", index=False, encoding="utf-8-sig",
    )


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    print("=" * 80)
    print(
        "SO SÁNH LIGHTGBM–TWEEDIE VÀ SEASONAL NAIVE"
    )
    print(f"Cửa hàng: {STORE_ID} | Horizon: {HORIZON} ngày (d_1886 -> d_1913)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    calendar, prices, wide = load_data(DATA_DIR, STORE_ID, ORIGIN_DAY)
    items = wide["item_id"].tolist()
    state = wide["state_id"].iloc[0]

    # Ma trận lịch sử huấn luyện (3049, 1885)
    train_history = wide[[f"d_{d}" for d in range(1, ORIGIN_DAY + 1)]].to_numpy(
        dtype=np.float32
    )
    products = wide[["item_id", "store_id", "state_id", "cat_id", "dept_id"]].copy()
    products["zero_ratio_train"] = compute_zero_ratio(train_history)
    products["intermittency_group"] = categorize_intermittency(products["zero_ratio_train"].to_numpy())
    revenue = training_revenue(wide, calendar, prices)

    # 2 & 3. Xây dựng đặc trưng kèm sell_price và price_change
    X, y, categories, origin_rows = build_features(
        wide, calendar, prices, ORIGIN_DAY
    )
    del prices, wide
    gc.collect()

    # 4. Huấn luyện
    model, train_time = train_lightgbm(X, y)
    del X, y
    gc.collect()

    # 5. Dự báo đệ quy 28 ngày
    lgb_pred, naive_pred = recursive_predict(
        model,
        train_history,
        items,
        calendar,
        state,
        categories,
        origin_rows,
        ORIGIN_DAY,
        HORIZON,
    )

    # 6. Đọc thực tế 28 ngày holdout
    LOGGER.info("6/6. Đọc dữ liệu thực tế 28 ngày holdout để chấm điểm")
    day_cols = [f"d_{d}" for d in range(ORIGIN_DAY + 1, ORIGIN_DAY + HORIZON + 1)]
    val_cols = ["item_id", "store_id"] + day_cols
    actual_df = pd.read_csv(
        DATA_DIR / "sales_train_evaluation.csv", usecols=val_cols
    )
    actual_df = (
        actual_df[actual_df["store_id"] == STORE_ID]
        .set_index("item_id")
        .loc[items]
    )
    actual_matrix = actual_df[day_cols].to_numpy(dtype=np.float32)

    predictions = (lgb_pred, naive_pred)
    results_df = evaluate_scope(products, revenue, train_history, actual_matrix, predictions)
    results_df["Thời gian Train (s)"] = [round(train_time, 1), 0.0]

    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ CHUNG CUỘC")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # Lưu kết quả
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT_DIR / "figures" / f"final_{STORE_ID}_report"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        results_df, figure_dir / "model_comparison_all_metrics.png",
        f"So sánh MAE, RMSE, RMSSE và WRMSSE tại {STORE_ID}",
    )
    export_group_results(
        products, revenue, train_history, actual_matrix, predictions, day_cols, figure_dir,
    )
    out_file = OUTPUT_DIR / "baseline_single_file_results.csv"
    results_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả tại: {out_file}\n")


if __name__ == "__main__":
    main()
