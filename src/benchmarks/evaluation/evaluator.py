"""
Điều phối thực nghiệm mô hình, chuẩn hóa bảng dự báo, tổng hợp Bottom-Up và xuất bảng sai số.
"""

import time
import numpy as np
import pandas as pd

from src.benchmarks.config import (
    HORIZON,
    ALPHA_CROSTON,
    ALPHA_ES,
    GAMMA_ES,
    PREDICTIONS_DIR,
    METRICS_DIR,
)
from src.benchmarks.data_loader import prepare_split_matrices
from src.benchmarks.models.croston import forecast_croston, forecast_croston_sba
from src.benchmarks.models.exponential_smoothing import (
    aggregate_bottom_up,
    forecast_exponential_smoothing_bu,
)
from src.benchmarks.models.seasonal_naive import forecast_seasonal_naive
from src.benchmarks.evaluation.metrics import (
    compute_series_metrics,
    compute_zero_ratio,
    categorize_intermittency,
)

M5_AGGREGATE_LEVELS = (
    (),
    ("state_id",),
    ("store_id",),
    ("cat_id",),
    ("dept_id",),
    ("state_id", "cat_id"),
    ("state_id", "dept_id"),
    ("store_id", "cat_id"),
    ("store_id", "dept_id"),
    ("item_id",),
    ("state_id", "item_id"),
)


def get_benchmark_models():
    """Trả về từ điển 4 mô hình thống kê đối chứng."""
    return {
        "Seasonal_Naive": lambda mat: forecast_seasonal_naive(
            mat, horizon=HORIZON, season_len=7
        ),
        "Croston": lambda mat: forecast_croston(
            mat, horizon=HORIZON, alpha=ALPHA_CROSTON
        ),
        "Croston_SBA": lambda mat: forecast_croston_sba(
            mat, horizon=HORIZON, alpha=ALPHA_CROSTON, with_seasonality=False
        ),
        "ES_bu": lambda mat: forecast_exponential_smoothing_bu(
            mat, horizon=HORIZON, alpha=ALPHA_ES, gamma=GAMMA_ES
        ),
    }


def evaluate_split(df: pd.DataFrame, split: str = "validation"):
    """
    Chạy thực nghiệm cho một giai đoạn ('validation' hoặc 'test').
    Trả về bảng dự báo chuẩn hóa 8 cột và bảng sai số chi tiết từng chuỗi.
    """
    print("\n" + "=" * 70)
    print(f"THỰC NGHIỆM DỰ BÁO GIAI ĐOẠN: {split.upper()} (Cửa sổ {HORIZON} ngày)")
    print("=" * 70)

    X_train, y_target, origin_day, target_day_names = prepare_split_matrices(df, split)

    zero_ratios = compute_zero_ratio(X_train, from_first_sale=True)
    intermittency_groups = categorize_intermittency(zero_ratios)

    models = get_benchmark_models()
    predictions_dict = {}
    metrics_records = []

    for model_name, model_fn in models.items():
        t0 = time.time()
        preds = model_fn(X_train)
        elapsed = time.time() - t0
        predictions_dict[model_name] = preds

        metrics_df = compute_series_metrics(X_train, y_target, preds)
        metrics_df["model"] = model_name
        metrics_df["split"] = split
        metrics_df["cat_id"] = df["cat_id"].values
        metrics_df["store_id"] = df["store_id"].values
        metrics_df["dept_id"] = df["dept_id"].values
        metrics_df["intermittency_group"] = intermittency_groups
        metrics_records.append(metrics_df)

        print(
            f"[{model_name:15s}] Thời gian: {elapsed:.2f}s | "
            f"MAE TB: {metrics_df['mae'].mean():.4f} | "
            f"RMSE TB: {metrics_df['rmse'].mean():.4f} | "
            f"RMSSE TB: {metrics_df['rmsse'].mean():.4f}"
        )

    all_metrics_df = pd.concat(metrics_records, ignore_index=True)

    # Định dạng bảng dự báo theo chuẩn giao tiếp 8 cột của nhóm
    print("Đang định dạng bảng dự báo theo chuẩn giao tiếp chung...")
    t0 = time.time()
    n_series = len(df)
    prediction_rows = []

    for model_name, preds in predictions_dict.items():
        model_sub_df = pd.DataFrame({
            "model": model_name,
            "split": split,
            "origin_day": origin_day,
            "item_id": np.repeat(df["item_id"].values, HORIZON),
            "store_id": np.repeat(df["store_id"].values, HORIZON),
            "target_day": np.tile(target_day_names, n_series),
            "horizon": np.tile(np.arange(1, HORIZON + 1), n_series),
            "y_pred": preds.flatten().round(4),
        })
        prediction_rows.append(model_sub_df)

    pred_table = pd.concat(prediction_rows, ignore_index=True)
    print(f"-> Hoàn tất bảng dự báo ({len(pred_table):,} dòng) trong {time.time() - t0:.2f}s")

    return pred_table, all_metrics_df


def export_es_bottom_up(predictions: pd.DataFrame, df: pd.DataFrame, split: str, store_suffix: str):
    """Cộng dồn dự báo ES_bu từ cấp 12 lên 11 cấp phân cấp cao hơn của M5."""
    metadata = df[["item_id", "store_id", "state_id", "cat_id", "dept_id"]]
    bottom = predictions.loc[predictions["model"].eq("ES_bu")].merge(
        metadata, on=["item_id", "store_id"], validate="many_to_one"
    )
    tables = []
    for level, columns in enumerate(M5_AGGREGATE_LEVELS, start=1):
        aggregated = aggregate_bottom_up(bottom, columns)
        aggregated["level"] = level
        tables.append(aggregated)

    subfolder = "all_stores" if store_suffix == "all_10_stores" else store_suffix.lower()
    target_dir = PREDICTIONS_DIR / subfolder / split
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / "es_bu_aggregated.csv"
    output_columns = [
        "model", "split", "origin_day", "level", "state_id", "store_id",
        "cat_id", "dept_id", "item_id", "target_day", "horizon", "y_pred",
    ]
    pd.concat(tables, ignore_index=True)[output_columns].to_csv(
        output, index=False, float_format="%.4f"
    )
    print(f"-> Đã lưu: {output.name}")


def summarize_and_export_metrics(val_metrics: pd.DataFrame, test_metrics: pd.DataFrame, is_all_stores: bool, store_suffix: str):
    """Tổng hợp và xuất 4 bảng sai số vào results/benchmarks/metrics/all_stores hoặc ca_1."""
    subfolder = "all_stores" if is_all_stores else store_suffix.lower()
    target_dir = METRICS_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = pd.concat([val_metrics, test_metrics], ignore_index=True)

    print("\n" + "=" * 70)
    print(f"TỔNG HỢP VÀ XUẤT CÁC BẢNG KẾT QUẢ ĐÁNH GIÁ ({subfolder})")
    print("=" * 70)

    groupings = [
        ("overall", ["split", "model"], "TỔNG THỂ"),
        ("by_category", ["split", "cat_id", "model"], "THEO NGÀNH HÀNG"),
        ("by_intermittency", ["split", "intermittency_group", "model"], "THEO MỨC ĐỘ GIÁN ĐOẠN"),
        ("by_store", ["split", "store_id", "model"], "THEO CỬA HÀNG"),
    ]

    for key, group_cols, label in groupings:
        summary = (
            all_metrics.groupby(group_cols, as_index=False)
            .agg(
                mae_mean=("mae", "mean"),
                rmse_mean=("rmse", "mean"),
                rmsse_mean=("rmsse", "mean"),
                series_count=("mae", "count"),
            )
            .round(4)
        )
        out_path = target_dir / f"summary_{key}.csv"
        summary.to_csv(out_path, index=False)

        if key == "overall":
            print(f"\n--- BẢNG 1: {label} ---")
            print(summary.to_string(index=False))
        elif key in ("by_category", "by_intermittency"):
            print(f"\n--- {label} (Validation) ---")
            print(summary[summary["split"] == "validation"].to_string(index=False))

    print(f"\n[THÀNH CÔNG] Đã xuất toàn bộ bảng sai số vào {target_dir}.")

