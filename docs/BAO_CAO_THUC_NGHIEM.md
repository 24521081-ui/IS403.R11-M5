# Báo cáo thực nghiệm: LightGBM–Tweedie trên Walmart M5 (có `sell_price` và `price_change`)

> Mọi số liệu trong báo cáo lấy từ các lần chạy Validation thật trên cửa hàng **CA_1**. Số liệu được lưu trong `results/metrics/` và `results/comparison_report.csv`.

---

## 1. Phạm vi

| Mục | Nội dung |
|---|---|
| Bài toán | Dự báo doanh số theo ngày, 28 ngày tới, cho từng sản phẩm (cấp 12 của M5) |
| Dữ liệu | `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv` |
| Cửa hàng chạy | **CA_1** (3.049 sản phẩm). **Chưa chạy 10 cửa hàng.** |
| Train | d_1 → d_1885 (5.661.993 dòng sau khi bỏ 28 ngày đầu) |
| Validation | d_1886 → d_1913 (85.372 điểm dự báo mỗi mô hình) |
| Môi trường | CPU, 6 luồng, Python 3.14, LightGBM 4.7 |
| Mô hình mới | LightGBM, hàm mục tiêu Tweedie, có `sell_price` và `price_change` |
| Mô hình đối chứng | Seasonal Naive |
| Chỉ số | RMSSE trung bình không trọng số, MAE, RMSE, số dư |

**Giới hạn cần nêu trong báo cáo:**
- RMSSE ở đây là **trung bình không trọng số trên 3.049 chuỗi cấp 12**, **không phải WRMSSE** chính thức của M5 (12 cấp, có trọng số doanh thu).
- Mới có kết quả của 1 cửa hàng, 1 lần chạy mỗi cấu hình. Chênh lệch giữa các cấu hình rất nhỏ (xem mục 4).
- Chính sách giá là `last_known`: giữ giá tại ngày origin cho cả 28 ngày. Trong dữ liệu M5, giá tương lai thực ra có sẵn (file giá kéo dài tới tuần 11621), nhưng nghiên cứu này không dùng.
- Không dùng dữ liệu holdout để huấn luyện. Nhãn 28 ngày chỉ được đọc sau khi hai mô hình đã dự báo xong.

---

## 2. Phương pháp cũ (đối chứng): Seasonal Naive

**Nguyên tắc:** doanh số hôm nay giống doanh số cùng thứ của tuần trước.

$$\hat{y}_{t+h} = y_{t+h-7}\quad (h \le 7), \qquad \hat{y}_{t+h} = \hat{y}_{t+h-7}\quad (h > 7)$$

Từ ngày thứ 8 trở đi, mô hình dùng lại chính dự báo của nó. Cài đặt trong code là `SeasonalNaive.predict` trả về `lag_7`.

**Thông số:** chu kỳ 7 ngày, không có tham số học, thời gian huấn luyện 0 giây.

---

## 3. Phương pháp mới: LightGBM–Tweedie

### 3.1 Lý thuyết

**Vì sao Tweedie?** Doanh số bán lẻ có rất nhiều ngày bằng 0, còn ngày có bán thì là số dương. Phân phối Tweedie (1 < p < 2) là hỗn hợp Poisson–Gamma: số lần mua theo Poisson, lượng mỗi lần theo Gamma. Nó cho phép giá trị đúng bằng 0 và dự báo luôn không âm.

Phương sai của Tweedie:

$$\mathrm{Var}(Y) = \phi\,\mu^{p}$$

Hàm mất mát (Tweedie deviance) mà LightGBM tối thiểu hóa, với dự báo \(\hat{y}\) và giá trị thật \(y\):

$$L(y,\hat y)=2\left(\frac{y^{2-p}}{(1-p)(2-p)}-\frac{y\,\hat y^{1-p}}{1-p}+\frac{\hat y^{2-p}}{2-p}\right)$$

Khi p gần 1 thì gần Poisson, khi p gần 2 thì gần Gamma. Chỉ cần hiểu: p càng cao thì mô hình càng chấp nhận phương sai lớn ở các chuỗi bán thưa.

**LightGBM:** nhiều cây quyết định được thêm lần lượt. Mỗi cây mới sửa phần sai của tổng các cây trước (gradient boosting). LightGBM mở rộng cây theo kiểu leaf-wise và gom giá trị đặc trưng vào các bin (histogram), nên chạy nhanh trên hàng triệu dòng.

**Một mô hình chung:** chỉ huấn luyện **1 mô hình cho cả cửa hàng**, với `item_id` là biến phân loại. Không huấn luyện riêng 3.049 mô hình.

### 3.2 Đặc trưng và phân loại nội sinh / ngoại sinh

| Nhóm | Đặc trưng | Ghi chú |
|---|---|---|
| **Nội sinh** (sinh từ doanh số quá khứ) | `lag_1`, `lag_7`, `lag_14`, `lag_28`, `rolling_mean_7`, `rolling_mean_28` | Khi dự báo 28 ngày, các biến này được cập nhật từ **chính dự báo của mô hình** (dự báo đệ quy) |
| **Ngoại sinh** (biết trước) | `day_of_week`, `month`, `event_flag`, `snap` | Lấy từ `calendar.csv`, snap theo đúng bang |
| **Ngoại sinh** (giá) | `sell_price`, `price_change` | Lấy từ `sell_prices.csv`, xem 3.3 |
| Định danh | `item_id` | Biến phân loại |

### 3.3 Đặc trưng giá

- `sell_price`: giá tuần của sản phẩm, ghép theo `(store_id, item_id, wm_yr_wk)`. Chỗ thiếu giá được điền tiếp từ giá quá khứ (`ffill`), **không điền ngược** để tránh dùng thông tin tương lai. Sau `ffill`, tỷ lệ thiếu giá của CA_1 là **19,66%**.
- `price_change` (đặc trưng mới):

$$\text{price\_change}_t=\frac{\text{sell\_price}_t-\text{sell\_price}_{t-7}}{\text{sell\_price}_{t-7}}$$

  Bằng 0 khi không có giá 7 ngày trước hoặc giá đó không dương. Âm nghĩa là đang giảm giá so với tuần trước.

**Cách xử lý khi dự báo (quyết định thiết kế, cần ghi rõ):** `price_change` của origin được dùng cho 7 ngày dự báo đầu. Từ ngày thứ 8 trở đi giá trị này đặt bằng 0, vì giá được giữ cố định nên không có biến động mới. Đây là một giả định đơn giản, chưa được kiểm chứng riêng.

### 3.4 Thông số cài đặt (`configs/config.yaml`)

| Tham số | Giá trị |
|---|---|
| objective | tweedie |
| tweedie_variance_power | 1.1 |
| learning_rate | 0.05 |
| n_estimators | 300 |
| num_leaves | 63 |
| min_child_samples | 50 |
| random_state / n_jobs | 42 / 6 |
| use_prices / price_policy | true / last_known |

### 3.5 Quy trình 6 bước

1. **Đọc dữ liệu** (`data_loader.py`): đọc theo khối, chỉ lấy một cửa hàng, chỉ đến origin.
2. **Chuẩn bị** (`preprocessing.py`): chuyển bảng rộng thành dạng dài, ghép lịch, SNAP, giá; tính `price_change`.
3. **Tạo đặc trưng** (`feature_engineering.py`): lag và rolling mean (đã `shift` để không nhìn ngày đích); bỏ 28 ngày đầu vì chưa đủ lag.
4. **Huấn luyện** (`training.py`): fit 1 mô hình LightGBM trên toàn bộ Train.
5. **Dự báo đệ quy** (`forecasting.py`): mỗi bước dự báo cho cả 3.049 sản phẩm, cắt giá trị âm về 0, rồi đưa dự báo vào lịch sử để tính lag cho bước sau.
6. **Đánh giá** (`reporting.py`): đọc doanh số thật của 28 ngày, tính chỉ số.

### 3.6 Công thức đánh giá

**Mẫu số RMSSE của sản phẩm i** (từ lần bán dương đầu tiên đến origin, gồm n điểm):

$$c_i=\frac{1}{n-1}\sum_{t=2}^{n}(y_{i,t}-y_{i,t-1})^2$$

**Sai số 28 ngày của sản phẩm i:**

$$\mathrm{MSE}_i=\frac{1}{28}\sum_{h=1}^{28}(y_{i,h}-\hat y_{i,h})^2,\qquad \mathrm{RMSSE}_i=\sqrt{\frac{\mathrm{MSE}_i}{c_i}}$$

**Tổng hợp:** lấy trung bình cộng RMSSE của 3.049 sản phẩm (không trọng số). RMSSE < 1 nghĩa là tốt hơn dự báo "bằng hôm qua" trên lịch sử Train.

**MAE** = trung bình của \(|y-\hat y|\) trên 85.372 điểm. **RMSE** = căn của trung bình \((y-\hat y)^2\). **Số dư** \(e=y-\hat y\): Bias là trung bình của e, Std là độ lệch chuẩn của e.

---

## 4. Kết quả

### 4.1 Mô hình mới so với Seasonal Naive (CA_1, cùng lần chạy)

| Chỉ số | Seasonal Naive | LightGBM–Tweedie (giá + price_change) |
|---|---:|---:|
| RMSSE | 0,93603 | **0,72975** |
| MAE | 1,3303 | **1,1768** |
| RMSE | 2,8065 | **2,2463** |
| Bias (trung bình số dư) | **+0,0243** | -0,2208 |
| Std số dư | 2,8064 | **2,2354** |
| Tỷ lệ \|sai số\| ≤ 1 | **72,6%** | 66,7% |
| Thời gian huấn luyện | 0 s | 77,8 s |

RMSSE giảm khoảng **22%** so với Seasonal Naive. LightGBM tốt hơn về RMSSE, MAE, RMSE và Std. Nhưng Seasonal Naive **ít lệch hơn** (Bias gần 0) và có **nhiều dự báo sai trong ±1 hơn**. Mô hình mới có xu hướng dự báo cao hơn thực tế (Bias âm). Hai điểm này cần nêu trung thực khi bảo vệ.

### 4.2 So sánh các cấu hình LightGBM (CA_1)

| Cấu hình | p | Số lá | Số cây | Giá | RMSSE | MAE | RMSE | Bias |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|
| Baseline | 1,2 | 31 | 200 | không | 0,72822 | 1,1630 | 2,2414 | -0,1863 |
| Chỉ giá thô | 1,1 | 63 | 300 | có | 0,72999 | 1,1799 | 2,2577 | -0,2238 |
| **Giá + price_change** | 1,1 | 63 | 300 | có | 0,72975 | 1,1768 | 2,2463 | -0,2208 |
| Sâu + regularization | 1,15 | 127 | 500 | có | 0,73128 | 1,1846 | 2,2649 | -0,2356 |
| p cao | 1,5 | 63 | 300 | không | 0,72603 | 1,1682 | 2,2705 | -0,1872 |

### 4.3 Diễn giải (và những điều chưa kết luận được)

- Chênh lệch RMSSE giữa 5 cấu hình chỉ trong khoảng **0,726 → 0,731**, từ 1 cửa hàng và 1 lần chạy. **Chưa đủ để nói cấu hình nào thật sự tốt hơn.**
- Việc thêm `price_change` so với chỉ giá thô cho RMSSE 0,72999 → 0,72975. Mức giảm này **rất nhỏ, chưa chứng minh được là do `price_change`** (chưa lặp lại nhiều lần, chưa đo độ ngẫu nhiên).
- Các cấu hình ở bảng 4.2 đổi **nhiều tham số cùng lúc** (p, số lá, số cây, giá). Vì vậy không tách được ảnh hưởng của riêng từng yếu tố. Riêng "chỉ giá thô" và "giá + price_change" là cặp duy nhất chỉ khác nhau đúng một đặc trưng.
- Bản có giá **không tốt hơn** Baseline tắt giá về RMSSE, MAE và RMSE trên CA_1. Nhóm vẫn chọn bật giá theo quyết định của nhóm trưởng. Nếu cần lý do khoa học, nên ghi: giá là biến ngoại sinh biết trước theo thiết kế M5, và kết quả CA_1 cho thấy giá chưa cải thiện rõ khi dùng chính sách `last_known` và giá thô.
- Nguyên nhân giá chưa giúp nhiều chưa được kiểm chứng. Giả thuyết hợp lý: giữ giá cố định 28 ngày làm mất tín hiệu khuyến mãi; 19,66% giá thiếu; chưa có đặc trưng giá tương đối.

---

## 5. Hướng dẫn demo

Chạy trong PowerShell, **từ thư mục dự án** (`run.py` nằm trong `m5-lightgbm-tweedie`, không nằm ở thư mục cha):

```powershell
cd C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie
```

**1. Smoke test (~15 giây, 30 sản phẩm):**

```powershell
..\.venv\Scripts\python.exe run.py --mode smoke --store CA_1 --config configs/config.yaml
```

**2. Validation đầy đủ cho CA_1 (khoảng 2 phút, 3.049 sản phẩm):**

```powershell
..\.venv\Scripts\python.exe run.py --mode validation --store CA_1 --scope full --config configs/config.yaml
```

Kết quả nằm ở `results/metrics/validation_full_CA_1_<thời điểm>/summary.csv`, biểu đồ ở `results/figures/`.

**3. Chạy cả 10 cửa hàng (chưa chạy, dự kiến lâu hơn nhiều):**

```powershell
..\.venv\Scripts\python.exe run.py --mode validation --all-stores --scope full --config configs/config.yaml
```

**4. Bảng so sánh các lần chạy và số dư:**

```powershell
..\.venv\Scripts\python.exe scripts/analyze_residuals.py
```

**5. Chạy lại cả bộ cấu hình thử nghiệm:**

```powershell
..\.venv\Scripts\python.exe scripts/run_all_experiments.py
```

**Lưu ý khi demo:**
- `scripts/demo_saved_forecast.py` **từ chối** các mô hình đã lưu có bật giá (có kiểm tra `use_prices`), nên không dùng được với mô hình hiện tại. Mô hình mới có thể xem qua file CSV và biểu đồ trong `results/`.
- Chế độ `--mode test` cần cờ `--confirm-final-test`. Chỉ chạy khi đã chốt cấu hình, không chạy để chọn cấu hình.

---

## 6. Hạn chế và hướng phát triển

1. Tính **WRMSSE** chính thức (12 cấp, trọng số theo doanh thu).
2. Chạy cả **10 cửa hàng** và chạy lặp nhiều seed để đo độ ngẫu nhiên.
3. Thí nghiệm **mỗi lần đổi một yếu tố** (chỉ đổi p, chỉ bật giá, chỉ thêm price_change).
4. Thử **giá tương lai thực** từ `sell_prices.csv` thay vì `last_known`.
5. Thêm đặc trưng giá tương đối (tỷ lệ so với giá cao nhất hoặc giá trung bình gần đây).
6. Chưa có unit test chạy được trong thư mục hiện tại (`pytest` báo 0 test). Các đặc trưng mới (`price_change`) chưa có test chống rò rỉ riêng.
