"""M5 Forecasting - LightGBM Tweedie Baseline (Single-File Complete Pipeline)

Tất cả các bước từ Đọc dữ liệu (kèm sell_prices) -> Tiền xử lý & Giá -> Tạo đặc trưng (Lags, Rolling, Price, Price Change)
-> Huấn luyện LightGBM -> Dự báo đệ quy 28 ngày -> Đánh giá chỉ số (RMSSE, MAE, RMSE, Số dư)
được gom trọn gói trong 1 file duy nhất.

Chạy:
    python run_baseline_single_file.py
"""

import gc
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ==============================================================================
# CẤU HÌNH BASELINE CHÍNH THỨC CỦA NHÓM (CÓ BẬT GIÁ BÁN & BIẾN ĐỘNG GIÁ)
# ==============================================================================
STORE_ID = "CA_1"
ORIGIN_DAY = 1885  # Ngày kết thúc tập Train
HORIZON = 28  # Số ngày cần dự báo (d_1886 -> d_1913)
DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

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
        **{d: "float32" for d in days},
    }
    cols = ["id", "item_id", "store_id", "state_id"] + days

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

    scales = []
    for row in train_history:
        pos = np.flatnonzero(row > 0)
        if len(pos) and len(row) - pos[0] >= 2:
            denom = np.square(np.diff(row[pos[0] :])).mean()
            scales.append(denom if denom > 0 else np.nan)
        else:
            scales.append(np.nan)
    scales = np.array(scales)

    valid = np.isfinite(scales)
    item_rmsse = np.sqrt(mse_per_item[valid] / scales[valid])
    mean_rmsse = float(item_rmsse.mean())

    return {
        "Mô hình": model_name,
        "RMSSE": round(mean_rmsse, 5),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Độ chệch (Bias)": round(mean_bias, 4),
        "Độ lệch chuẩn số dư (Std)": round(std_residual, 4),
        "Tỷ lệ lệch <= 1 cái (%)": round(within_1_pct, 2),
    }


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    print("=" * 80)
    print(
        "CHẠY HUẤN LUYỆN VÀ ĐÁNH GIÁ LIGHTGBM TWEEDIE (TRỌN GÓI 1 FILE - CÓ GIÁ BÁN)"
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

    # 2 & 3. Xây dựng đặc trưng kèm sell_price và price_change
    X, y, categories, origin_rows = build_features(
        wide, calendar, prices, ORIGIN_DAY
    )
    del prices
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

    # Đánh giá chỉ số
    res_lgb = evaluate(
        actual_matrix,
        train_history,
        lgb_pred,
        "LightGBM–Tweedie (Giá + Price Change)",
    )
    res_lgb["Thời gian Train (s)"] = round(train_time, 1)

    res_naive = evaluate(
        actual_matrix, train_history, naive_pred, "Seasonal Naive (Mốc chuẩn)"
    )
    res_naive["Thời gian Train (s)"] = 0.0

    results_df = pd.DataFrame([res_lgb, res_naive])

    print("\n" + "=" * 80)
    print("BẢNG KẾT QUẢ ĐÁNH GIÁ CHUNG CUỘC")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)

    # Lưu kết quả
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / "baseline_single_file_results.csv"
    results_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\nĐã lưu kết quả tại: {out_file}\n")


if __name__ == "__main__":
    main()
