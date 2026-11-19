# TÀI LIỆU GIẢI THÍCH BẢN CHẤT TOÁN HỌC & QUY TRÌNH MÃ NGUỒN CÁC MÔ HÌNH THỐNG KÊ ĐỐI CHỨNG
## Đề tài: Dự báo Bán lẻ Phân cấp Walmart M5 (Statistical Benchmarks)

---

# PHẦN 1: QUY TRÌNH THỰC HIỆN TỪ A ĐẾN Z (GIẢI THÍCH BẰNG NGÔN NGỮ ĐỜI THƯỜNG)

Hãy hình dung toàn bộ quy trình xử lý của hợp phần mô hình thống kê giống như **một dây chuyền nấu ăn trong nhà hàng**, gồm đúng **5 bước đơn giản**:

### Bước 1: Chuẩn bị nguyên liệu (`src/data/loader.py`)
- **Làm gì?** Mở file dữ liệu khổng lồ của Walmart (`sales_train_evaluation.csv`), lọc đúng cột cần thiết của 30.490 sản phẩm trong gần 2.000 ngày.
- **Tách làm 2 đề thi:**
  - *Đề thi thử (Validation):* Lấy lịch sử từ ngày 1 đến 1885 để ôn, rồi đoán 28 ngày tiếp theo (1886–1913).
  - *Đề thi thật (Test):* Lấy lịch sử từ ngày 1 đến 1913 để ôn, rồi đoán 28 ngày cuối cùng (1914–1941).
- **Công dụng:** Giúp máy tính không bị ngốn RAM vô ích và đảm bảo mô hình **không gian lận** (không nhìn trước tương lai).

### Bước 2: Thử nghiệm trước trên vài món đồ (`scripts/test_models.py`)
- **Làm gì?** Trước khi chạy cho 30.000 sản phẩm, ta chạy thử trên 4 chuỗi giả lập và 10 sản phẩm thật.
- **Công dụng:** Kiểm tra xem công thức toán học có bị lỗi `NaN`, bị số âm, hay bị lệch ngày cuối tuần không. Nếu 10 món chạy đúng và hợp lý thì mới dám bấm nút chạy đại trà.

### Bước 3: Cho 4 "đầu bếp" cùng dự báo (`src/models/`)
- **Làm gì?** Đưa lịch sử bán hàng vào 4 mô hình dự báo cùng lúc:
  1. `seasonal_naive.py`: Bê nguyên tuần trước sang tuần này (làm mốc sàn).
  2. `croston.py` (Croston & Croston–SBA): Chuyên trị hàng bán chậm nhiều số 0.
  3. `exponential_smoothing.py` (`ES_bu`): Nhìn lịch sử gần + nhịp cuối tuần, rồi cộng dồn từ dưới lên.
- **Công dụng:** Tìm ra dự báo 28 ngày tới cho từng sản phẩm. Nhờ gắn chip tăng tốc **Numba JIT**, 30.490 sản phẩm chạy xong chỉ trong **3–4 giây**.

### Bước 4: Chấm điểm xem ai đoán chuẩn nhất (`src/evaluation/`)
- **Làm gì?** So sánh 28 ngày dự báo với số bán thật trong lịch sử bằng 3 thước đo: **MAE**, **RMSE** và **RMSSE** (trong đó RMSSE < 1,0 nghĩa là giỏi hơn đoán mò).
- **Công dụng:** Biết được mô hình nào thắng ở nhóm nào (ví dụ: `ES_bu` giỏi ở Đồ ăn, `Croston–SBA` giỏi ở Đồ chơi/Hàng bán chậm).

### Bước 5: Đóng gói kết quả bàn giao cho nhóm (`outputs/` & `docs/`)
- **Làm gì?**
  - Lưu file dự báo chuẩn 8 cột vào `outputs/predictions/` để các bạn làm Machine Learning (LightGBM, Random Forest) lấy về so sánh.
  - Lưu 4 bảng tổng hợp sai số vào `outputs/metrics/` để vẽ biểu đồ và làm slide.
  - Tự động xuất báo cáo Word vào `docs/`.
- **Công dụng:** Tạo ra **thước đo chuẩn (Benchmark)**. Nhờ đó cả nhóm biết mô hình Học máy phức tạp có thực sự đáng công làm hay không.

---

# PHẦN 2: CẤU TRÚC THƯ MỤC & GIẢI THÍCH TỪNG FILE CODE

Cấu trúc dự án được chia gọn theo chuẩn phần mềm Python (tên thư mục hoàn toàn bằng tiếng Anh, không dùng gạch dưới `_`):

```
Đồ án/
├── README.md                  # Hướng dẫn tổng quan & cách chạy lệnh
├── docs/                      # Toàn bộ tài liệu báo cáo (.md & .docx)
├── outputs/
│   ├── predictions/           # File dự báo 28 ngày chuẩn 8 cột (.csv)
│   └── metrics/               # 4 bảng tổng hợp sai số (.csv)
├── scripts/                   # Nơi chứa các lệnh chạy trực tiếp
│   ├── test_models.py         # Chạy kiểm thử nhanh trên vài chuỗi mẫu
│   ├── run_benchmark.py       # Chạy thực nghiệm chính (1 hoặc 10 siêu thị)
│   └── build_docs.py          # Chạy xuất ra file Word (.docx)
└── src/                       # Thư viện mã nguồn lõi
    ├── config.py              # Cấu hình tập trung (đường dẫn, tham số, mốc ngày)
    ├── data/
    │   └── loader.py          # Đọc CSV và cắt tập Train/Validation/Test
    ├── models/
    │   ├── croston.py         # Thuật toán Croston & Croston-SBA
    │   ├── exponential_smoothing.py # Thuật toán ES_bu & hàm cộng dồn Bottom-Up
    │   └── seasonal_naive.py  # Thuật toán Seasonal Naive (chu kỳ 7 ngày)
    ├── evaluation/
    │   ├── metrics.py         # Tính MAE, RMSE, RMSSE & phân 4 nhóm gián đoạn
    │   └── evaluator.py       # Chạy 4 mô hình, format bảng 8 cột, gom bảng sai số
    └── reports/
        ├── docx_report.py     # Tạo file docs/research-report.docx
        └── docx_guide.py      # Tạo file docs/model-guide.docx
```

### Chi tiết vai trò từng file code:
1. **`src/config.py`**: Chứa toàn bộ đường dẫn thư mục, mốc chia ngày (1885, 1913, 1941), hệ số $\alpha = 0,1$ cho Croston, $\alpha = 0,2, \gamma = 0,15$ cho ES. Muốn chỉnh thông số chỉ cần sửa đúng 1 file này.
2. **`src/data/loader.py`**: Đọc file gốc `sales_train_evaluation.csv` một cách tiết kiệm RAM, lọc theo cửa hàng (nếu cần) và cắt ra ma trận số để đưa vào mô hình.
3. **`src/models/croston.py`**: Chứa công thức Croston (1972) và Croston–SBA (2005). Dùng `@njit` của Numba để biên dịch sang tốc độ ngôn ngữ C. Có thêm bản `seasonal` nếu muốn gắn hệ số cuối tuần.
4. **`src/models/exponential_smoothing.py`**: Chứa thuật toán Holt-Winters mùa vụ cộng tính (chu kỳ 7 ngày) và hàm `aggregate_bottom_up()` để cộng dồn dự báo từ cấp từng món đồ lên 11 cấp cao hơn (cửa hàng, bang, toàn quốc).
5. **`src/models/seasonal_naive.py`**: Lấy lại đúng 7 ngày gần nhất để lặp lại cho 4 tuần tiếp theo.
6. **`src/evaluation/metrics.py`**: Tính chuẩn xác 3 chỉ số `MAE`, `RMSE`, `RMSSE` theo đúng định nghĩa cuộc thi Walmart M5, kèm hàm đo tỷ lệ số 0 để chia sản phẩm thành 4 nhóm gián đoạn.
7. **`src/evaluation/evaluator.py`**: Đóng vai trò "quản đốc phân xưởng" — gọi cả 4 mô hình chạy tuần tự, bấm giờ, gom bảng dự báo đúng 8 cột chuẩn của nhóm và xuất 4 bảng sai số vào `outputs/metrics/`.
8. **`scripts/test_models.py`**, **`scripts/run_benchmark.py`**, **`scripts/build_docs.py`**: 3 nút bấm để chạy kiểm thử, chạy thực nghiệm toàn bộ, và xuất file Word.

---

# PHẦN 3: BẢN CHẤT 4 MÔ HÌNH & Ý NGHĨA CÁC CHỈ SỐ

### 1. Hai cái bẫy của dữ liệu bán lẻ Walmart:
- **68,37% số ngày bằng 0:** Nhiều món đồ chơi, đồ gia dụng cả tuần không ai mua, đùng cái có người mua 2-3 cái.
- **Hiệu ứng cuối tuần:** Thứ Bảy, Chủ Nhật lượng mua cao hơn ngày thường tới **31%**.

### 2. Bản chất từng mô hình:
- **Seasonal Naïve:** "Tuần trước bán sao, tuần này bê nguyên sang." (Đơn giản nhất, dễ sai khi tuần trước ngẫu nhiên bằng 0 hoặc có người mua sỉ).
- **ES_bu:** "Nhìn cả lịch sử nhưng ưu tiên ngày gần đây + học nhịp cuối tuần. Dự báo từng món ở đáy rồi cộng dồn lên tổng hệ thống." $\to$ **Thắng tuyệt đối ở ngành Thực phẩm (FOODS).**
- **Croston (1972):** Tách chuỗi nhiều số 0 thành 2 câu hỏi: *Mỗi lần mua mấy cái ($z$)?* và *Bao nhiêu ngày mới có 1 người mua ($p$)?* $\to$ Dự báo trung bình = $z / p$.
- **Croston–SBA (2005):** Khắc phục lỗi toán học "lạc quan quá mức" của Croston bằng cách nhân thêm $(1 - \frac{\alpha}{2}) = 0,95$ $\to$ **Giảm tồn kho ảo, đạt RMSSE tốt nhất toàn diện (0,7589).**

---

# PHẦN 4: KỊCH BẢN 2 PHÚT BÁO CÁO TIẾN ĐỘ

> *"Chào các bạn/thầy cô, mình xin báo cáo tiến độ phần việc mô hình thống kê đối chứng (ES_bu, Croston–SBA):*
> 1. **Tiến độ:** Đã hoàn thành **100%** (từ tìm hiểu, cài đặt code, kiểm thử chuỗi mẫu đến chạy toàn bộ 30.490 chuỗi của 10 siêu thị trên cả tập Validation và Test).
> 2. **Tổ chức code:** Đã cấu trúc lại gọn gàng theo chuẩn module Python (`src/`, `scripts/`, `outputs/`, `docs/`) và tối ưu hóa Numba JIT nên chạy hết 30.000 chuỗi chỉ mất chưa tới **4 giây**.
> 3. **Kết quả bàn giao:** Đã xuất file dự báo chuẩn 8 cột (`predictions_validation.csv`, `predictions_test.csv`) để nhóm đối chiếu với Machine Learning.
> 4. **Ngưỡng Benchmark:** **Croston–SBA** đạt RMSSE **0,7589** (vượt trội ở nhóm hàng bán chậm nhiều số 0), còn **ES_bu** đạt MAE **1,0608** (mạnh nhất ở ngành Thực phẩm). Các mô hình ML của nhóm cần vượt qua 2 mốc này để chứng minh hiệu quả."*

