"""
Điểm khởi chạy tập trung cho Hợp phần Mô hình Đối chứng Thống kê (Statistical Benchmarks).
Cung cấp giao diện dòng lệnh linh hoạt để chạy kiểm thử hoặc thực nghiệm dự báo.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.config import STORES


def main():
    parser = argparse.ArgumentParser(
        description="M5 FORECASTING — STATISTICAL BENCHMARKS PIPELINE"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Chạy kiểm thử đơn vị các mô hình thống kê")
    group.add_argument("--store", choices=STORES, help="Chạy thực nghiệm trên 1 cửa hàng chỉ định (VD: CA_1)")
    group.add_argument("--all_stores", action="store_true", help="Chạy thực nghiệm trên toàn bộ 10 cửa hàng")

    args = parser.parse_args()

    if args.test:
        from scripts.benchmarks.test_benchmark_models import run_unit_tests
        run_unit_tests()
    elif args.store or args.all_stores:
        from scripts.benchmarks.run_benchmarks import main as run_benchmarks_main
        run_benchmarks_main()


if __name__ == "__main__":
    main()
