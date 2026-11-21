"""
Biên dịch tự động toàn bộ 4 tài liệu Word báo cáo nghiên cứu và giải thích mô hình vào docs/.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.benchmarks.reports.docx_report import build_docx_report
from src.benchmarks.reports.docx_guide import build_explanation_docx
from src.benchmarks.reports.docx_separate_metrics import build_mae_doc, build_rmse_doc


def main():
    build_docx_report()
    build_explanation_docx()
    build_mae_doc()
    build_rmse_doc()
    print("-> Đã xuất xong toàn bộ 4 tài liệu Word trong thư mục docs/.")


if __name__ == "__main__":
    main()
