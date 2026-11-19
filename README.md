# Đánh giá thực nghiệm các mô hình dự báo lượng bán sản phẩm trong bán lẻ (Walmart M5)

Đề tài: **Đánh giá thực nghiệm các mô hình dự báo lượng bán sản phẩm trong bán lẻ: Hỗ trợ lập kế hoạch nhu cầu và bổ sung hàng hóa.**  
Bộ dữ liệu chuẩn: **Walmart M5 Forecasting Accuracy Dataset** (30.490 chuỗi thời gian phân cấp, 10 siêu thị, 3 bang).  
Phạm vi nghiên cứu: Đối sánh toàn diện giữa **Hệ thống Học máy tăng cường độ dốc (LightGBM với hàm mục tiêu Tweedie)** và **Hệ thống Mô hình thống kê chuẩn mực & Nhu cầu gián đoạn (Statistical Benchmarks & Intermittent Demand)**.

---

## 1. Kiến trúc hệ thống và Cấu trúc thư mục dự án

```
IS403.R11-M5/
├── configs/                             # Cấu hình tham số mô hình
│   ├── config.yaml                      # Cấu hình pipeline LightGBM–Tweedie
│   └── benchmarks.yaml                  # Cấu hình các mô hình thống kê đối chứng
│
├── data/                                # Thư mục dữ liệu (được .gitignore bảo vệ)
│   └── raw/                             # Đặt các file dữ liệu M5 gốc (.csv) tại đây
│
├── docs/                                # Báo cáo nghiên cứu khoa học & tài liệu kỹ thuật
│   ├── research-report.docx             # Báo cáo nghiên cứu thực nghiệm hoàn chỉnh (Word)
│   ├── research-report.md               # Báo cáo nghiên cứu thực nghiệm (Markdown)
│   ├── model-guide.docx                 # Cẩm nang giải thích bản chất toán học & mã nguồn (Word)
│   ├── model-guide.md                   # Cẩm nang giải thích bản chất toán học & mã nguồn (Markdown)
│   ├── mae-guide.docx                   # Chuyên đề đối sánh & diễn giải chỉ số MAE (Word)
│   ├── rmse-guide.docx                  # Chuyên đề đối sánh & diễn giải chỉ số RMSE (Word)
│   ├── HUONG_DAN_DEMO_DU_BAO.md         # Quy trình demo suy luận mô hình ML
│   ├── reference_comparison.md          # Báo cáo đối chiếu mã nguồn học thuật
│   └── verification.md                  # Nhật ký kiểm chứng độc lập
│
├── results/                             # Kết quả thực nghiệm và trực quan hóa
│   ├── benchmarks/
│   │   ├── metrics/                     # Bảng tổng hợp sai số MAE, RMSE, RMSSE đối chứng
│   │   └── predictions/                 # Kết quả dự báo h=28 ngày của các mô hình thống kê
│   ├── figures/                         # Biểu đồ phân tích độ chính xác & phân phối sai số
│   └── models/                          # Artifacts mô hình Machine Learning đã huấn luyện
│
├── scripts/                             # Các điểm thực thi chi tiết
│   ├── run_benchmarks.py                # Thực thi thực nghiệm các mô hình thống kê đối chứng
│   ├── test_benchmark_models.py         # Kiểm thử đơn vị các thuật toán thống kê
│   ├── build_reports.py                 # Biên dịch tự động toàn bộ tài liệu báo cáo Word (.docx)
│   ├── demo_saved_forecast.py           # Demo phát lại suy luận dự báo từ checkpoint
│   ├── plot_saved_rmsse_report.py       # Xuất biểu đồ so sánh RMSSE tổng hợp
│   └── summarize_validation.py          # Tổng hợp chỉ số validation
│
├── src/                                 # Mã nguồn lõi được đóng gói theo module
│   │
│   ├── benchmarks/                      # HỢP PHẦN CÁC MÔ HÌNH THỐNG KÊ ĐỐI CHỨNG
│   │   ├── config.py                    # Cấu hình tự động nhận diện dữ liệu và siêu tham số
│   │   ├── data_loader.py               # Nạp dữ liệu tối ưu bộ nhớ cho mô hình thống kê
│   │   ├── models/                      # Thuật toán: Seasonal Naïve, ES_bu, Croston, Croston-SBA
│   │   ├── evaluation/                  # Thước đo chuẩn M5: MAE, RMSE, RMSSE & điều phối đánh giá
│   │   └── reports/                     # Tự động hóa biên dịch báo cáo Word (.docx)
│   │
│   ├── data_loader.py                   # HỢP PHẦN MACHINE LEARNING (LIGHTGBM - TWEEDIE)
│   ├── feature_engineering.py           # Kỹ nghệ đặc trưng: Lags, Rolling windows, Calendar
│   ├── preprocessing.py                 # Tiền xử lý ma trận bán hàng và nhãn phân loại
│   ├── training.py                      # Huấn luyện LightGBM tối ưu hàm mất mát Tweedie
│   ├── forecasting.py                   # Dự báo đệ quy đa bước (Recursive Multi-step Forecasting)
│   └── reporting.py                     # Đánh giá chỉ số RMSSE và xuất báo cáo mô hình ML
│
├── run.py                               # Điểm khởi chạy pipeline Machine Learning (LightGBM)
├── run_benchmarks.py                    # Điểm khởi chạy pipeline Statistical Benchmarks
├── requirements.txt                     # Danh mục thư viện phụ thuộc của toàn bộ dự án
├── .gitignore                           # Quy tắc loại trừ tệp nhị phân và dữ liệu lớn
└── README.md                            # Tài liệu tổng quan dự án
```

---

## 2. Các phương pháp dự báo được triển khai

### A. Nhóm mô hình Học máy (Machine Learning)
- **LightGBM với hàm mục tiêu Tweedie:** Mô hình hóa phân phối Poisson hỗn hợp Gamma phù hợp với dữ liệu có tỷ lệ số 0 cao.
- **Kỹ nghệ đặc trưng:** Độ trễ (lags 28, 29,...), giá trị trung bình cuộn (rolling mean 7, 28,...), độ biến động giá, sự kiện lịch (ngày lễ, SNAP food stamps).
- **Chiến lược suy luận:** Dự báo đệ quy (Recursive Forecasting) từng bước qua 28 ngày.

### B. Nhóm mô hình Thống kê Đối chứng (Statistical Benchmarks)
- **Seasonal Naïve (SN):** Chu kỳ tuần 7 ngày bán lẻ ($m = 7$), đóng vai trò là thước đo cơ sở chuẩn mực tính thang đo RMSSE của cuộc thi M5.
- **Exponential Smoothing kết hợp Bottom-Up ($ES_{bu}$):** San bằng mũ Holt-Winters mức và mùa vụ ($lpha = 0.2, \gamma = 0.15$), dự báo tại cấp SKU và tổng hợp lên toàn bộ các cấp phân cấp.
- **Croston cổ điển (Croston, 1972):** Tách biệt quy mô nhu cầu và khoảng cách giữa các lần bán hàng, tối ưu cho nhu cầu gián đoạn.
- **Croston-SBA (Syntetos & Boylan, 2005):** Hiệu chỉnh độ chệch dương bằng hệ số $(1 - lpha/2)$, tăng tốc tính toán hàng chục ngàn chuỗi thời gian bằng **Numba JIT (`@jit(nopython=True)`)**.

---

## 3. Cài đặt môi trường

Sử dụng **Python 3.11 (64-bit)**. Khuyến nghị thiết lập môi trường ảo:

```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Nâng cấp pip và cài đặt thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Hướng dẫn vận hành

### 4.1. Hợp phần Mô hình Thống kê Đối chứng (Statistical Benchmarks)

Điểm khởi chạy chính: `run_benchmarks.py` (hoặc qua `scripts/run_benchmarks.py`):

```powershell
# 1. Kiểm thử đơn vị tính đúng đắn của thuật toán (trên chuỗi nhân tạo và mẫu M5)
python run_benchmarks.py --test

# 2. Chạy thực nghiệm trên 1 cửa hàng thử nghiệm (ví dụ CA_1)
python run_benchmarks.py --store CA_1

# 3. Chạy thực nghiệm toàn diện trên toàn bộ 10 siêu thị M5
python run_benchmarks.py --all_stores

# 4. Biên dịch tự động toàn bộ 4 tài liệu Word báo cáo nghiên cứu và cẩm nang mô hình
python run_benchmarks.py --build_reports
```

### 4.2. Hợp phần Mô hình Học máy (LightGBM–Tweedie)

Điểm khởi chạy chính: `run.py`:

```powershell
# Chạy demo kiểm chứng trên dữ liệu giả lập
python run.py --demo

# Chạy quy trình huấn luyện và dự báo đầy đủ theo cấu hình configs/config.yaml
python run.py
```

### 4.3. Demo và đối chiếu kết quả

```powershell
# Phát lại suy luận dự báo từ checkpoint mô hình đã lưu
python scripts/demo_saved_forecast.py

# Xuất biểu đồ so sánh trực quan chỉ số RMSSE
python scripts/plot_saved_rmsse_report.py
```

---

## 5. Kết quả thực nghiệm và Báo cáo

Toàn bộ các tài liệu nghiên cứu học thuật và báo cáo đối sánh được xuất tự động tại thư mục [`docs/`](docs/):
- **Báo cáo nghiên cứu hoàn chỉnh:** [`docs/research-report.docx`](docs/research-report.docx) / [`docs/research-report.md`](docs/research-report.md).
- **Cẩm nang bản chất thuật toán & mã nguồn:** [`docs/model-guide.docx`](docs/model-guide.docx) / [`docs/model-guide.md`](docs/model-guide.md).
- **Chuyên đề phân tích chỉ số:** [`docs/mae-guide.docx`](docs/mae-guide.docx) và [`docs/rmse-guide.docx`](docs/rmse-guide.docx).

Các bảng tổng hợp sai số định lượng (Overall, Store, Category, Demand Profile) được lưu trữ tại [`results/benchmarks/metrics/`](results/benchmarks/metrics/).
