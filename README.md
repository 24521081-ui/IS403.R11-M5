# IS403.R11-M5 — Dự báo lượng bán Walmart M5

Dự án nghiên cứu và đối chiếu các mô hình dự báo lượng bán theo sản phẩm × cửa hàng trên bộ dữ liệu **Walmart M5** (30.490 chuỗi, 10 cửa hàng, 3 bang), đánh giá theo **WRMSSE đủ 12 cấp**.

## Cấu trúc thư mục

```text
IS403.R11-M5/
├── data/raw/                          # Dữ liệu M5 gốc dùng chung cho cả 4 pipeline
├── pipelines/
│   ├── benchmarks/                    # Mô hình thống kê đối chứng (Seasonal Naive, Croston, Croston-SBA, ES_bu)
│   ├── lightgbm_tweedie/              # LightGBM với hàm mục tiêu Tweedie (baseline ML)
│   ├── lightgbm_twopart/              # LightGBM Two-Part (C0–C7 & C7_EVENTS)
│   └── random_forest/                 # Random Forest Regression (Stage 1–5, notebooks 00–09, Excel)
├── requirements.txt                   # Thư viện Python cho toàn dự án
└── README.md
```

## Hướng dẫn chạy nhanh

### 1. Benchmarks (Seasonal Naive, Croston, ES_bu)
```powershell
.\.venv\Scripts\python.exe pipelines/benchmarks/run_benchmarks.py --help
```

### 2. LightGBM Tweedie
```powershell
.\.venv\Scripts\python.exe pipelines/lightgbm_tweedie/run.py --help
```

### 3. LightGBM Two-Part (`C7_EVENTS`)
- Tài liệu: [C7_EVENTS.md](pipelines/lightgbm_twopart/docs/C7_EVENTS.md)
- Chạy kiểm thử:
```powershell
.\.venv\Scripts\python.exe -m pytest pipelines/lightgbm_twopart/tests
```

### 4. Random Forest (`RF-15 + FE-04`)
- Tài liệu chi tiết: [pipelines/random_forest/README.md](pipelines/random_forest/README.md)
- Chạy unit test (79 tests):
```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s pipelines/random_forest/tests -t pipelines/random_forest -p "test_wrmsse*.py"
```
- Chạy lại từ đầu theo từng Stage (chạy từ thư mục `pipelines/random_forest`):
```powershell
cd pipelines/random_forest
..\..\.venv\Scripts\python.exe -B scripts/run_preparation_checks.py
..\..\.venv\Scripts\python.exe -B scripts/run_stage2.py
..\..\.venv\Scripts\python.exe -B scripts/run_stage3.py
..\..\.venv\Scripts\python.exe -B scripts/run_stage3_all_features.py
..\..\.venv\Scripts\python.exe -B scripts/run_stage4.py
..\..\.venv\Scripts\python.exe -B scripts/run_stage5.py
```
