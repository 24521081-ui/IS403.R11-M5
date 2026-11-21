"""
Điểm khởi chạy chính để thực nghiệm mô hình thống kê M5 (Validation & Test).
Ví dụ:
    python scripts/run_benchmark.py --store CA_1
    python scripts/run_benchmark.py --all_stores
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

# Thêm thư mục gốc dự án vào sys.path để import từ src
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.config import PREDICTIONS_DIR, STORES
from src.benchmarks.data_loader import load_sales_dataset
from src.benchmarks.evaluation.evaluator import (
    evaluate_split,
    export_es_bottom_up,
    summarize_and_export_metrics,
)


def main():
    parser = argparse.ArgumentParser(description="Thực nghiệm các mô hình thống kê M5")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--store", choices=STORES, default=None, help="Chỉ định cửa hàng (mặc định: CA_1)")
    scope.add_argument("--all_stores", action="store_true", help="Chạy trên toàn bộ 10 cửa hàng")
    args = parser.parse_args()

    if args.all_stores:
        selected_stores = None
        store_suffix = "all_10_stores"
    elif args.store:
        selected_stores = [args.store]
        store_suffix = args.store
    else:
        selected_stores = ["CA_1"]
        store_suffix = "CA_1"

    print(f"Khởi chạy thực nghiệm cho cấu hình: {store_suffix}")

    df = load_sales_dataset(selected_stores)

    # 1. Chạy giai đoạn Validation (d_1886 đến d_1913)
    val_preds, val_metrics = evaluate_split(df, split="validation")

    # 2. Chạy giai đoạn Test (d_1914 đến d_1941)
    test_preds, test_metrics = evaluate_split(df, split="test")

    # 3. Lưu bảng dự báo chuẩn hóa vào thư mục con phân tầng
    subfolder = "all_stores" if args.all_stores else store_suffix.lower()
    val_dir = PREDICTIONS_DIR / subfolder / "validation"
    test_dir = PREDICTIONS_DIR / subfolder / "test"
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    val_pred_file = val_dir / "forecasts.csv"
    test_pred_file = test_dir / "forecasts.csv"

    print(f"\nĐang lưu tệp dự báo vào {PREDICTIONS_DIR / subfolder}...")
    val_preds.to_csv(val_pred_file, index=False)
    test_preds.to_csv(test_pred_file, index=False)
    print(f"-> Đã lưu: {val_pred_file.relative_to(REPO_ROOT)}")
    print(f"-> Đã lưu: {test_pred_file.relative_to(REPO_ROOT)}")

    export_es_bottom_up(val_preds, df, "validation", store_suffix)
    export_es_bottom_up(test_preds, df, "test", store_suffix)

    # 4. Tổng hợp và xuất 4 bảng sai số
    summarize_and_export_metrics(val_metrics, test_metrics, args.all_stores, store_suffix)


if __name__ == "__main__":
    main()

