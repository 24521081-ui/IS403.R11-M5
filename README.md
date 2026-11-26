# Dự báo doanh số bán lẻ Walmart M5

Đồ án so sánh LightGBM–Tweedie với Seasonal Naive, Exponential Smoothing Bottom-Up, Croston và Croston–SBA trên dữ liệu Walmart M5 gồm 30.490 chuỗi sản phẩm × cửa hàng.

## Cài đặt

Dùng Python 3.11. Đặt `calendar.csv`, `sell_prices.csv` và `sales_train_evaluation.csv` trong `data/raw/`. Chạy các lệnh từ thư mục gốc của repo.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy đồ án

**LightGBM–Tweedie cho CA_1**

Để chọn cấu hình trước khi đối chiếu với Seasonal Naive, chạy `python -m scripts.ml.compare_tweedie_configs`. Bốn lượt cùng dùng `sell_price` và `price_change`. Mỗi lượt thử chỉ thay đổi một trong ba tham số `tweedie_variance_power`, `num_leaves` và `n_estimators` so với cấu hình gốc. Script cập nhật [biểu đồ chọn cấu hình](results/figures/final_CA_1_report/tweedie_config_selection.png) và xuất `results/tweedie_config_selection.csv`.

```powershell
python run_baseline_single_file.py
```

Lệnh train với dữ liệu đến `d_1885`, dự báo `d_1886`–`d_1913` và so sánh LightGBM–Tweedie với Seasonal Naive.

- [Biểu đồ chung](results/figures/final_CA_1_report/model_comparison_all_metrics.png) hiển thị MAE, RMSE, RMSSE và WRMSSE. Bảng điểm được xuất vào `results/baseline_single_file_results.csv`.
- Bảng điểm theo nhóm được xuất vào `results/baseline_group_metrics.csv`. Các biểu đồ `intermittency_*.png` và `category_*.png` trong [thư mục biểu đồ CA_1](results/figures/final_CA_1_report/) trình bày bốn mức gián đoạn và ba ngành hàng M5, kèm tổng lượng bán thực tế và dự báo theo ngày.
- Tệp `results/predictions/baseline_CA_1_validation.csv` lưu doanh số thực tế, dự báo của hai mô hình và nhãn cho từng sản phẩm–ngày. Nhãn gián đoạn tính từ lịch sử train, còn ngành hàng lấy từ `cat_id`.

WRMSSE dùng doanh thu 28 ngày cuối train làm trọng số ở mỗi cấp. Điểm trong biểu đồ được tính trên phạm vi CA_1 hoặc nhóm đang hiển thị. [Hướng dẫn chấm chính thức của M5](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf)

**LightGBM–Tweedie trên 10 cửa hàng**

```powershell
python scripts/ml/run_pipeline.py --mode validation --all-stores
python scripts/ml/run_pipeline.py --mode test --all-stores --confirm-final-test
```

Pipeline đọc `configs/config.yaml` và xuất dự báo theo từng sản phẩm, mô hình đã train và bảng sai số. Kỳ test train đến `d_1913` và dự báo `d_1914`–`d_1941`.

**Mô hình thống kê đối chứng**

```powershell
python run_benchmarks.py --test
python run_benchmarks.py --store CA_1
python run_benchmarks.py --all_stores
```

Bảng tổng hợp của benchmark nằm trong `results/benchmarks/metrics/`. Tham số nằm trong `src/benchmarks/config.py`.

**Chấm WRMSSE trên 12 cấp**

```powershell
python scripts/score_model_wrmsse.py --model LightGBM_Tweedie --forecast 'results/test/test_full_*/lightgbm_tweedie.csv'
```

Lệnh nhận dự báo của đủ 10 cửa hàng với các cột `item_id`, `store_id`, `target_day` và `y_pred`. Mỗi khóa sản phẩm–cửa hàng–ngày xuất hiện đúng một lần. Kết quả mặc định nằm trong `results/wrmsse/runs/<model>/`.

## Báo cáo và kết quả

- Báo cáo và bản thảo được lưu cục bộ trong `docs/`.
- [Biểu đồ thực nghiệm CA_1](results/figures/final_CA_1_report/) và [bảng tổng hợp mô hình đối chứng](results/benchmarks/metrics/all_stores/).

Mã nguồn mô hình nằm trong `src/`. Dự báo chi tiết, checkpoint và kết quả của từng lần chạy được Git bỏ qua. Thư mục lưu trữ cục bộ bên cạnh repo giữ các lượt chạy trước đây theo đường dẫn gốc tính từ thư mục dự án.
