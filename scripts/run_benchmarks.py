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
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR))

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

    # 3. Lưu bảng dự báo chuẩn hóa
    val_pred_file = PREDICTIONS_DIR / f"predictions_validation_{store_suffix}.csv"
    test_pred_file = PREDICTIONS_DIR / f"predictions_test_{store_suffix}.csv"

    print(f"\nĐang lưu tệp dự báo vào {PREDICTIONS_DIR}...")
    val_preds.to_csv(val_pred_file, index=False)
    test_preds.to_csv(test_pred_file, index=False)
    print(f"-> Đã lưu: {val_pred_file.name}")
    print(f"-> Đã lưu: {test_pred_file.name}")

    export_es_bottom_up(val_preds, df, "validation", store_suffix)
    export_es_bottom_up(test_preds, df, "test", store_suffix)

    if args.all_stores:
        val_preds.to_csv(PREDICTIONS_DIR / "predictions_validation.csv", index=False)
        test_preds.to_csv(PREDICTIONS_DIR / "predictions_test.csv", index=False)
        print("-> Đã lưu bản quy chuẩn predictions_validation.csv và predictions_test.csv")

    # 4. Tổng hợp và xuất 4 bảng sai số
    summarize_and_export_metrics(val_metrics, test_metrics, args.all_stores, store_suffix)


if __name__ == "__main__":
    main()

