"""Script tự động chạy so sánh các cấu hình và phân tích số dư (residuals)."""
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie")
PYTHON_EXE = Path(r"C:\Users\Admin\Desktop\PTDLKD\.venv\Scripts\python.exe")

CONFIGS = [
    ("Baseline (Power=1.2, No Price)", "configs/config_exp0_baseline.yaml"),
    ("Exp 2 (Deep + Reg, Power=1.15)", "configs/config_exp2_deep_reg.yaml"),
    ("Exp 3 (High Zero, Power=1.5)", "configs/config_exp3_high_zero.yaml"),
]

def run_experiment(name, config_path):
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU CHẠY: {name} ({config_path})")
    print(f"{'='*60}")
    cmd = [
        str(PYTHON_EXE),
        "run.py",
        "--mode", "validation",
        "--store", "CA_1",
        "--scope", "full",
        "--config", config_path
    ]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    if res.returncode != 0:
        print(f"LỖI khi chạy {name}:")
        print(res.stderr)
        return False
    print(f"Hoàn thành {name} thành công.")
    return True

if __name__ == "__main__":
    for name, cfg in CONFIGS:
        ok = run_experiment(name, cfg)
        if not ok:
            print("Dừng lại do có lỗi.")
            sys.exit(1)
    print("\nTất cả thử nghiệm đã hoàn thành!")
