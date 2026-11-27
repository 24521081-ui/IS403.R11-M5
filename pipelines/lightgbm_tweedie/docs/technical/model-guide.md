# TÀI LIỆU KỸ THUẬT VÀ QUY TRÌNH MÃ NGUỒN CÁC MÔ HÌNH THỐNG KÊ ĐỐI CHỨNG
## Đề tài: Dự báo Bán lẻ Phân cấp Walmart M5 (Statistical Benchmarks)

---

## 1. TỔNG QUAN KIẾN TRÚC VÀ QUY TRÌNH THỰC NGHIỆM

Hợp phần mô hình thống kê đối chứng (Statistical Benchmarks) được xây dựng nhằm thiết lập các mốc so sánh định lượng vững chắc cho toàn bộ đề tài nghiên cứu. Quy trình thực nghiệm bao gồm 5 giai đoạn có cấu trúc:

1. **Chuẩn bị ma trận dữ liệu (`src/benchmarks/data_loader.py`):**
   - Đọc dữ liệu bán hàng lịch sử từ `sales_train_evaluation.csv` cho 30.490 chuỗi thời gian ở cấp sản phẩm–cửa hàng (Level 12).
   - Tách ma trận huấn luyện và kiểm thử theo hai mốc thời gian đã thống nhất:
     + *Validation:* Huấn luyện trên $d\_1$ đến $d\_1885$, dự báo 28 ngày ($d\_1886$ đến $d\_1913$).
     + *Test:* Huấn luyện trên $d\_1$ đến $d\_1913$, dự báo 28 ngày ngoài mẫu ($d\_1914$ đến $d\_1941$).
   - Tối ưu hóa cấu trúc dữ liệu dưới dạng ma trận mảng liền khối (`np.ascontiguousarray`) để hạn chế chiếm dụng bộ nhớ và hỗ trợ tính toán song song.

2. **Kiểm thử đơn vị thuật toán (`scripts/benchmarks/test_benchmark_models.py`):**
   - Kiểm tra tính đúng đắn toán học trên 4 chuỗi nhân tạo (chuỗi gián đoạn, chuỗi mùa vụ tuần, chuỗi toàn số 0, chuỗi chỉ có 1 giao dịch bán) và 10 chuỗi thực tế đại diện từ dữ liệu M5.
   - Bảo đảm kết quả không chứa giá trị `NaN`, không mang giá trị âm, và giữ đúng pha chu kỳ tuần 7 ngày.

3. **Thực nghiệm dự báo đa mô hình (`src/benchmarks/models/`):**
   - Thực thi 4 mô hình đối chứng thống kê độc lập:
     + `seasonal_naive.py`: Mô hình lặp lại chu kỳ tuần gần nhất làm mốc cơ sở tính toán thang đo RMSSE.
     + `exponential_smoothing.py` ($ES_{bu}$): San bằng mũ kết hợp thành phần mức và mùa vụ cộng tuần, tổng hợp từ dưới lên (Bottom-Up).
     + `croston.py`: Mô hình Croston cổ điển (1972) và mô hình hiệu chỉnh Croston–SBA (Syntetos & Boylan, 2005) chuyên trị nhu cầu gián đoạn.
   - Tối ưu hóa toàn bộ các vòng lặp đệ quy bằng **Numba JIT (`@njit(fastmath=True)`)**, cho phép xử lý toàn bộ 30.490 chuỗi trong chưa đầy 4 giây trên CPU thông thường.

4. **Đánh giá sai số đa tầng (`src/benchmarks/evaluation/`):**
   - Đo lường sai số dự báo bằng 3 thước đo chuẩn mực: **MAE** (sai số tuyệt đối trung bình), **RMSE** (căn bậc hai sai số bình phương trung bình), và **RMSSE** (sai số chuẩn hóa cuộc thi M5).
   - Phân tích chi tiết hiệu năng theo 3 ngành hàng chính (FOODS, HOBBIES, HOUSEHOLD) và theo 4 nhóm mức độ gián đoạn (tỷ lệ ngày bán bằng 0: $<30\%$, $30-50\%$, $50-70\%$, $\ge 70\%$).

5. **Đóng gói kết quả bàn giao (`results/benchmarks/`):**
   - Bảng dự báo 28 ngày chuẩn hóa 8 cột (`model, split, origin_day, item_id, store_id, target_day, horizon, y_pred`).
   - Bảng tổng hợp dự báo phân cấp Bottom-Up 12 cấp M5 (`es_bu_aggregated.csv`).
   - 4 bảng tổng hợp sai số định lượng dạng CSV (Overall, Category, Intermittency, Store).

---

## 2. CẤU TRÚC THƯ MỤC VÀ VAI TRÒ TỪNG MODULE

```
IS403.R11-M5/
├── configs/
│   └── benchmarks.yaml              # Cấu hình siêu tham số (alpha, gamma, mốc ngày)
├── scripts/
│   └── benchmarks/
│       ├── test_benchmark_models.py # Kịch bản kiểm thử đơn vị thuật toán
│       ├── run_benchmarks.py        # Điểm khởi chạy thực nghiệm (1 hoặc 10 siêu thị)
│       ├── plot_time_series_comparison.py # Vẽ biểu đồ chuỗi thời gian đối chứng
│       └── build_reports.py         # Biên dịch báo cáo tổng hợp
├── src/
│   └── benchmarks/
│       ├── config.py                # Cấu hình đường dẫn và hằng số hệ thống
│       ├── data_loader.py           # Nạp dữ liệu và tách ma trận theo dòng thời gian
│       ├── models/
│       │   ├── seasonal_naive.py    # Thuật toán Seasonal Naïve (chu kỳ m = 7)
│       │   ├── exponential_smoothing.py # Thuật toán ES_bu và cơ chế Bottom-Up
│       │   └── croston.py           # Thuật toán Croston và Croston–SBA
│       ├── evaluation/
│       │   ├── metrics.py           # Tính toán các chỉ số MAE, RMSE, RMSSE
│       │   └── evaluator.py         # Điều phối thực nghiệm và xuất bảng kết quả
│       └── reports/
│           ├── docx_report.py       # Xuất báo cáo nghiên cứu thực nghiệm
│           ├── docx_guide.py        # Xuất tài liệu kỹ thuật mô hình
│           └── docx_separate_metrics.py # Xuất chuyên đề MAE và RMSE
└── results/
    └── benchmarks/
        ├── predictions/             # Tệp dự báo 28 ngày phân tầng theo siêu thị
        └── metrics/                 # 4 bảng tổng hợp chỉ số sai số định lượng
```

---

## 3. CƠ SỞ LÝ THUYẾT VÀ BẢN CHẤT TOÁN HỌC CÁC MÔ HÌNH

### 3.1. Đặc trưng dữ liệu bán lẻ Walmart M5
- **Tỷ lệ số 0 cao (68,37%):** Đa số các mặt hàng thuộc nhóm HOBBIES và HOUSEHOLD có nhu cầu ngắt quãng, xuất hiện nhiều ngày liên tiếp không có giao dịch bán.
- **Hiệu ứng mùa vụ tuần:** Khối lượng mua sắm vào các ngày cuối tuần (thứ Bảy và Chủ Nhật) thường cao hơn trung bình ngày thường khoảng 31%, đòi hỏi mô hình phải phản ánh đúng chu kỳ 7 ngày.

### 3.2. Bản chất các mô hình đối chứng
1. **Seasonal Naïve (Mốc cơ sở chuẩn mực):**
   - Giả định lượng bán tương lai lặp lại chính xác giá trị của chu kỳ 7 ngày gần nhất:
     $$\hat{y}_{T+h} = y_{T+h - 7 \cdot \lceil h/7 \rceil}$$
   - Đóng vai trò là mẫu số đo lường mức độ biến động ngẫu nhiên trong thang đo RMSSE.

2. **Exponential Smoothing kết hợp Bottom-Up ($ES_{bu}$):**
   - Phân rã chuỗi thời gian thành thành phần mức cơ sở ($L_t$) và thành phần mùa vụ cộng ($S_t$):
     $$L_t = \alpha (y_t - S_{t-7}) + (1 - \alpha) L_{t-1}$$
     $$S_t = \gamma (y_t - L_t) + (1 - \gamma) S_{t-7}$$
     $$\hat{y}_{T+h} = \max\left(0, L_T + S_{T + ((h - 1) \pmod 7) + 1}\right)$$
   - Sau khi dự báo tại cấp SKU (Level 12), tổng hợp lên các cấp cao hơn theo công thức:
     $$\hat{y}_{K, T+h} = \sum_{i \in \text{Nhóm } K} \hat{y}_{i, T+h}$$
   - Đạt sai số MAE thấp nhất (1,0608 trên tập Validation), đặc biệt tối ưu ở ngành Thực phẩm (FOODS).

3. **Croston cổ điển (Croston, 1972):**
   - Tách rời quy mô lượng bán khi phát sinh ($z_t$) và khoảng cách giữa các lần bán ($p_t$):
     $$\hat{y}_{T+h}^{\text{Croston}} = \frac{z_T}{p_T}$$
   - Thích hợp cho nhu cầu gián đoạn nhưng mắc độ chệch dương cố hữu do bất đẳng thức Jensen: $E[1/p] > 1/E[p]$.

4. **Hiệu chỉnh Syntetos–Boylan (Croston–SBA, 2005):**
   - Bổ sung hệ số điều chỉnh $(1 - \alpha/2)$ nhằm triệt tiêu phần chệch dương:
     $$\hat{y}_{T+h}^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{z_T}{p_T}$$
   - Với $\alpha = 0,1$, dự báo giảm 5% so với Croston cổ điển, giúp doanh nghiệp tránh hiện tượng tích trữ tồn kho ảo và đạt chỉ số RMSSE tốt nhất trong nhóm thống kê (0,7589).

---

## 4. BẢNG TỔNG HỢP SAI SỐ VÀ NGƯỠNG ĐỐI CHỨNG (BENCHMARK)

Bảng kết quả thực nghiệm trên toàn bộ 30.490 chuỗi thời gian:

| Mô hình | Giai đoạn | MAE TB | RMSE TB | RMSSE TB | Ghi chú vai trò |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Seasonal Naïve** | Validation | 1,2324 | 1,7400 | 0,9756 | Mốc cơ sở chuẩn mực của cuộc thi M5 |
| **Croston (1972)** | Validation | 1,1026 | 1,3986 | 0.7616 | Mô hình gốc cho nhu cầu gián đoạn |
| **Croston–SBA (2005)** | **Validation** | **1,0862** | **1,3909** | **0,7589** | ⭐ **RMSSE tốt nhất nhóm thống kê** |
| **ES_bu** | **Validation** | **1,0608** | **1,3983** | **0,7775** | ⭐ **MAE tốt nhất nhóm thống kê** |
| **Seasonal Naïve** | Test | 1,2440 | 1,7519 | 0,9972 | Kiểm định ngoài mẫu độc lập |
| **Croston–SBA (2005)** | **Test** | **1,0759** | **1,3926** | **0,7687** | ⭐ **RMSSE tốt nhất trên tập Test** |
| **ES_bu** | **Test** | **1,0659** | **1,4121** | **0,7972** | ⭐ **MAE tốt nhất trên tập Test** |

> **Ngưỡng Benchmark kiểm chứng cho các mô hình Học máy:**
> - Ngưỡng đối chứng MAE: **1,0608** (xác lập bởi $ES_{bu}$).
> - Ngưỡng đối chứng RMSSE: **0,7589** (xác lập bởi Croston–SBA).
