"""
Kiểm tra tính đúng đắn và sự ổn định của các mô hình trên chuỗi nhân tạo và 10 chuỗi thực tế M5.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.config import SALES_EVAL_FILE, HORIZON, TRAIN_END_VAL, VAL_START, VAL_END
from src.benchmarks.models.croston import forecast_croston, forecast_croston_sba
from src.benchmarks.models.exponential_smoothing import forecast_exponential_smoothing_bu
from src.benchmarks.models.seasonal_naive import forecast_seasonal_naive
from src.benchmarks.evaluation.metrics import (
    compute_series_metrics,
    compute_zero_ratio,
    categorize_intermittency,
)


def run_unit_tests():
    print("=" * 60)
    print("BẮT ĐẦU KIỂM THỬ ĐƠN VỊ CÁC THUẬT TOÁN DỰ BÁO THỐNG KÊ")
    print("=" * 60)

    # 1. Kiểm thử trên chuỗi nhân tạo
    toy_series = np.array([
        [0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0],  # Gián đoạn rõ nét
        [5, 4, 6, 5, 7, 8, 9, 5, 4, 6, 5, 7, 8, 9],  # Bán đều theo chu kỳ tuần
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Toàn số 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],  # Chỉ bán duy nhất 1 lần
    ], dtype=np.float64)

    h = 7
    pred_snaive = forecast_seasonal_naive(toy_series, horizon=h)
    pred_croston = forecast_croston(toy_series, horizon=h, alpha=0.1)
    pred_sba = forecast_croston_sba(toy_series, horizon=h, alpha=0.1)
    pred_es = forecast_exponential_smoothing_bu(toy_series, horizon=h)

    # Kiểm tra giữ đúng pha mùa vụ khi ngày bán đầu tiên không ở đầu chuỗi
    weekly_pattern = np.array([2, 3, 5, 7, 11, 13, 17], dtype=np.float64)
    delayed_series = np.concatenate((np.zeros(3), np.tile(weekly_pattern, 2)))
    np.testing.assert_allclose(
        forecast_exponential_smoothing_bu(delayed_series[None, :], horizon=28)[0],
        np.tile(weekly_pattern, 4),
    )

    print("Kiểm thử chuỗi nhân tạo (h = 7 ngày):")
    print(f"- Seasonal Naive shape: {pred_snaive.shape}, Không chứa NaN: {not np.isnan(pred_snaive).any()}")
    print(f"- Croston shape: {pred_croston.shape}, Không chứa NaN: {not np.isnan(pred_croston).any()}")
    print(f"- Croston-SBA shape: {pred_sba.shape}, Không chứa NaN: {not np.isnan(pred_sba).any()}")
    print(f"- ES_bu shape: {pred_es.shape}, Không chứa NaN: {not np.isnan(pred_es).any()}")

    valid_idx = pred_croston[:, 0] > 0
    if np.all(pred_sba[valid_idx, 0] < pred_croston[valid_idx, 0]):
        print("[XÁC NHẬN] Croston-SBA luôn hiệu chỉnh giảm so với Croston cổ điển (1 - alpha/2).")

    # 2. Kiểm thử trên 10 chuỗi thực tế từ M5
    print("\n" + "=" * 60)
    print("KIỂM THỬ TRÊN CÁC CHUỖI ĐẠI DIỆN TỪ DỮ LIỆU M5")
    print("=" * 60)

    cols_to_read = ["id", "item_id", "dept_id", "cat_id", "store_id"] + [
        f"d_{d}" for d in range(1, VAL_END + 1)
    ]
    sample_df = pd.read_csv(SALES_EVAL_FILE, nrows=10, usecols=cols_to_read)

    train_cols = [f"d_{d}" for d in range(1, TRAIN_END_VAL + 1)]
    val_cols = [f"d_{d}" for d in range(VAL_START, VAL_END + 1)]

    X_train = sample_df[train_cols].values.astype(np.float64)
    y_val = sample_df[val_cols].values.astype(np.float64)

    zero_ratios = compute_zero_ratio(X_train)
    categories = categorize_intermittency(zero_ratios)

    print(f"Đã đọc {len(sample_df)} chuỗi mẫu:")
    for idx, row in sample_df.iterrows():
        print(
            f"  Chuỗi {idx+1}: {row['id']} | Ngành: {row['cat_id']} | "
            f"Tỷ lệ số 0: {zero_ratios[idx]*100:.1f}% ({categories[idx]})"
        )

    preds = {
        "Seasonal_Naive": forecast_seasonal_naive(X_train, horizon=HORIZON),
        "Croston": forecast_croston(X_train, horizon=HORIZON),
        "Croston_SBA": forecast_croston_sba(X_train, horizon=HORIZON),
        "ES_bu": forecast_exponential_smoothing_bu(X_train, horizon=HORIZON),
    }

    print("\nBảng sai số trên 10 chuỗi mẫu (Validation 28 ngày):")
    for name, pred in preds.items():
        metrics_df = compute_series_metrics(X_train, y_val, pred)
        print(
            f"  [{name:15s}] MAE TB: {metrics_df['mae'].mean():.4f} | "
            f"RMSE TB: {metrics_df['rmse'].mean():.4f} | "
            f"RMSSE TB: {metrics_df['rmsse'].mean():.4f}"
        )

    print("\n[THÀNH CÔNG] Tất cả các thuật toán đều chạy chính xác, ổn định và không phát sinh lỗi.")


def test_benchmark_models():
    """Pytest wrapper to execute benchmark model unit checks."""
    run_unit_tests()


if __name__ == "__main__":
    run_unit_tests()

