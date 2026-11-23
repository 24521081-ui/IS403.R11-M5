# Chấm WRMSSE cho nhóm M5

Ae ơi, repo này có bộ chấm chung để mình so sánh 6 mô hình trên cùng dữ liệu và cùng cách tính điểm.

Trước giờ mình dùng RMSSE trung bình không trọng số để so các cấu hình ở cấp sản phẩm × cửa hàng, với mỗi cặp được tính ngang nhau. Giờ mình dùng **WRMSSE trên 12 cấp** làm điểm chính để bám sát cách chấm của bài báo M5. Kết quả RMSSE cũ vẫn dùng để đối chiếu quá trình thử cấu hình nha.

**Ae bắt đầu ở đây:**

1. Giữ lại file dự báo validation của các cấu hình đã thử để chấm lại và chốt cấu hình cho mô hình của mình.
2. Chuẩn bị file dự báo test của cấu hình đã chốt và ba file dữ liệu M5.
3. Mở [hướng dẫn bàn giao WRMSSE](docs/WRMSSE_BAN_GIAO.md) để xem định dạng file, lệnh chấm test và các file cần gửi tui.

Có đủ dự báo đã lưu thì mình chỉ tính lại điểm, không cần train lại. Ai thiếu dự báo của cấu hình nào thì báo tui tên cấu hình và kỳ đang thiếu nha.
## Báo cáo LightGBM Two-Part

- [Xếp hạng WRMSSE validation C0–C7](reports/xep_hang_wrmsse_validation.csv) · [Kết quả validation/test](reports/ket_qua_wrmsse_validation_test.csv).
- [Mô tả và liên kết sáu biểu đồ](reports/mo_ta_bieu_do.md) · [12 cấp Test C7](reports/wrmsse_12_cap/test_C7.csv).

Bản cuối được đề xuất là C7 với λ = 0,75, căn cứ WRMSSE validation. Lệnh chung `scripts/score_model_wrmsse.py` hiện chấm test; validation C0–C7 dùng script riêng `scripts/score_wrmsse_configurations.py` với dự báo Two-Part đã lưu tại máy. Chưa có lệnh chung chấm validation từ CSV tùy ý của thành viên.

---

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
│   ├── reports/                         # Báo cáo thực nghiệm (Word .docx)
│   │   ├── research-report.docx         # Báo cáo nghiên cứu thực nghiệm hoàn chỉnh
│   │   ├── model-guide.docx             # Tài liệu giải thích bản chất toán học & mã nguồn
│   │   ├── mae-guide.docx               # Chuyên đề đối sánh & ý nghĩa kinh doanh của MAE
│   │   └── rmse-guide.docx              # Chuyên đề đối sánh & thiết lập Safety Stock với RMSE
│   └── technical/                       # Tài liệu kỹ thuật & quy trình
│       ├── research-report.md           # Báo cáo nghiên cứu dạng Markdown
│       ├── model-guide.md               # Tài liệu kỹ thuật mô hình dạng Markdown
│       ├── HUONG_DAN_DEMO_DU_BAO.md     # Quy trình demo suy luận mô hình ML
│       └── reference_comparison.md      # Đối chiếu mã nguồn học thuật
│
├── results/                             # Kết quả thực nghiệm và trực quan hóa
│   ├── benchmarks/
│   │   ├── metrics/                     # Bảng chỉ số sai số MAE, RMSE, RMSSE
│   │   │   ├── all_stores/              # Kết quả chính thức toàn bộ 10 siêu thị (30.490 chuỗi)
│   │   │   └── ca_1/                    # Kết quả chạy thử nghiệm mẫu riêng siêu thị CA_1
│   │   └── predictions/                 # Kết quả dự báo 28 ngày
│   │       ├── all_stores/              # Tập dự báo toàn bộ 10 siêu thị (validation & test)
│   │       └── ca_1/                    # Tập dự báo riêng siêu thị CA_1 (validation & test)
│   ├── figures/                         # Biểu đồ phân tích độ chính xác & sai số
│   └── models/                          # Trọng số mô hình Machine Learning đã huấn luyện
│
├── scripts/                             # Các điểm thực thi chi tiết
│   ├── benchmarks/                      # Scripts cho các mô hình thống kê đối chứng
│   │   ├── run_benchmarks.py            # Chạy thực nghiệm (Validation & Test)
│   │   ├── test_benchmark_models.py     # Kiểm thử đơn vị thuật toán
│   │   └── plot_time_series_comparison.py # Vẽ biểu đồ chuỗi thời gian đối chứng
│   └── ml/                              # Scripts cho mô hình Machine Learning (LightGBM)
│       ├── demo_saved_forecast.py       # Demo phát lại suy luận dự báo LightGBM
│       ├── plot_saved_rmsse_report.py   # Xuất biểu đồ so sánh RMSSE
│       └── summarize_validation.py      # Tổng hợp chỉ số validation ML
│
├── src/                                 # Toàn bộ mã nguồn lõi
│   ├── benchmarks/                      # Hợp phần mô hình thống kê đối chứng
│   │   ├── config.py                    # Cấu hình tự động nhận diện dữ liệu và siêu tham số
│   │   ├── data_loader.py               # Nạp ma trận dữ liệu tối ưu bộ nhớ
│   │   ├── models/                      # Thuật toán: Seasonal Naïve, ES_bu, Croston, Croston-SBA
│   │   └── evaluation/                  # Thước đo chuẩn M5: MAE, RMSE, RMSSE & điều phối đánh giá
│   │
│   └── ml/                              # Hợp phần mô hình học máy (LightGBM - Tweedie)
│       ├── data_loader.py               # Nạp dữ liệu bảng sales, calendar, prices
│       ├── feature_engineering.py       # Kỹ nghệ đặc trưng: Lags, Rolling windows, Calendar
│       ├── preprocessing.py             # Tiền xử lý ma trận bán hàng và nhãn phân loại
│       ├── training.py                  # Huấn luyện LightGBM tối ưu hàm mất mát Tweedie
│       ├── forecasting.py               # Dự báo đệ quy đa bước (Recursive Multi-step)
│       └── reporting.py                 # Đánh giá chỉ số RMSSE và xuất báo cáo mô hình ML
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
- **Exponential Smoothing kết hợp Bottom-Up ($ES_{bu}$):** San bằng mũ Holt-Winters mức và mùa vụ ($\alpha = 0.2, \gamma = 0.15$), dự báo tại cấp SKU và tổng hợp lên toàn bộ các cấp phân cấp.
- **Croston cổ điển (Croston, 1972):** Tách biệt quy mô nhu cầu và khoảng cách giữa các lần bán hàng, tối ưu cho nhu cầu gián đoạn.
- **Croston-SBA (Syntetos & Boylan, 2005):** Hiệu chỉnh độ chệch dương bằng hệ số $(1 - \alpha/2)$, tăng tốc tính toán hàng chục ngàn chuỗi thời gian bằng **Numba JIT (`@njit(fastmath=True)`)**.

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

Điểm khởi chạy chính: `run_benchmarks.py` (hoặc qua `scripts/benchmarks/run_benchmarks.py`):

```powershell
# 1. Kiểm thử đơn vị tính đúng đắn của thuật toán (trên chuỗi nhân tạo và mẫu M5)
python run_benchmarks.py --test

# 2. Chạy thực nghiệm trên 1 cửa hàng thử nghiệm (ví dụ CA_1)
python run_benchmarks.py --store CA_1

# 3. Chạy thực nghiệm toàn diện trên toàn bộ 10 siêu thị M5
python run_benchmarks.py --all_stores
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
python scripts/ml/demo_saved_forecast.py

# Xuất biểu đồ so sánh trực quan chỉ số RMSSE
python scripts/ml/plot_saved_rmsse_report.py
```

---

## 5. Kết quả thực nghiệm và Báo cáo

Toàn bộ các tài liệu nghiên cứu học thuật và báo cáo đối sánh được xuất tự động tại thư mục [`docs/`](docs/):
- **Báo cáo nghiên cứu hoàn chỉnh:** [`docs/reports/research-report.docx`](docs/reports/research-report.docx) / [`docs/technical/research-report.md`](docs/technical/research-report.md).
- **Tài liệu kỹ thuật mô hình:** [`docs/reports/model-guide.docx`](docs/reports/model-guide.docx) / [`docs/technical/model-guide.md`](docs/technical/model-guide.md).
- **Chuyên đề phân tích chỉ số:** [`docs/reports/mae-guide.docx`](docs/reports/mae-guide.docx) và [`docs/reports/rmse-guide.docx`](docs/reports/rmse-guide.docx).

Các bảng tổng hợp sai số định lượng (Overall, Store, Category, Demand Profile) được lưu trữ tại [`results/benchmarks/metrics/`](results/benchmarks/metrics/).
