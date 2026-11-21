# Báo cáo kiểm toán mã LightGBM–Tweedie trên Walmart M5

Ngày kiểm tra: **06/11/2026**, múi giờ Asia/Saigon. Phạm vi: toàn bộ `src/`, `run.py`, YAML, requirements, pytest, README và các kết quả Validation CA_1 có sẵn. Môi trường: Windows, Python 3.14.8, LightGBM 4.7.0, CPU 6 luồng.

## A. Kết luận tổng quan

**PASS WITH WARNINGS — có thể tiếp tục Validation tuần tự cho 10 cửa hàng.** Không tìm thấy rò rỉ nhãn tương lai trong luồng đang dùng. Ba chỉ số Full CA_1 bạn cung cấp đã được kiểm chứng độc lập từ CSV và doanh số thật. Chưa có bằng chứng cần huấn luyện lại lần Full đó vì các sửa đổi lần này.

Đã sửa các lỗ hổng kiểm tra đầu vào khi tái sử dụng module riêng, làm rõ log và tên chỉ số tổng hợp, giảm dữ liệu giữ trong RAM khi đánh giá. Những sửa đổi này không đổi công thức, bộ đặc trưng, cấu hình YAML hoặc mốc chia của thí nghiệm hiện tại.

Không huấn luyện lại Full CA_1, không chạy toàn bộ 10 cửa hàng, không chạy thực nghiệm Test cuối trên dữ liệu thật. Chỉ đọc các kết quả Validation cũ, phát lại 5 sản phẩm từ mô hình đã lưu, chạy hai Smoke CA_1 nhỏ và các unit test giả lập. Các test cũ có mô phỏng mốc Test trong thư mục tạm; đó không phải mở đánh giá Test cuối M5.

Các giới hạn cần giữ: chưa tính WRMSSE chính thức; chưa đo RAM đỉnh Full hoặc kiểm chứng ổn định RAM qua 10 cửa hàng; chưa kiểm chứng trực tiếp bằng Python 3.11.

## B. Bảng kiểm tra

| Thành phần | Kết quả | Ghi chú |
|---|---|---|
| Rò rỉ dữ liệu tương lai | PASS | Đổi ngẫu nhiên 28 nhãn Validation trên CSV giả lập không đổi dự báo; theo dõi luồng đọc holdout sau cả hai mô hình |
| Xây dựng đặc trưng | PASS | Lag 1/7/14/28, Rolling 7/28 đúng ngày; không trộn chuỗi; Train/Predict cùng công thức |
| LightGBM–Tweedie | PASS | Mô hình thật, objective Tweedie, power 1.2, CPU 6 luồng; tham số mô hình cũ khớp metadata |
| Dự báo đệ quy | PASS | Cập nhật dự báo trước; không gọi fit mỗi ngày; phát lại 5 sản phẩm khớp bản lưu |
| MAE/RMSE | PASS | Tính độc lập trên 85.372 dự báo/mô hình, khớp summary |
| RMSSE | PASS | Đúng công thức đang công bố và mẫu số từ Train; tổng hợp trung bình không trọng số, không phải WRMSSE |
| Dự báo 28 ngày | PASS | Cả hai mô hình có 3.049 sản phẩm × 28 ngày liên tục |
| Xuất kết quả CSV | PASS | Đúng tám cột; không trùng khóa, thiếu ngày, NaN, vô cực hoặc âm |
| Bộ nhớ RAM | CHƯA XÁC MINH | Hai Smoke đạt; chưa đo Full 10 cửa hàng. Đã giảm bản sao thừa và chỉ đọc 28 cột holdout |
| Bảo toàn kết quả cũ | PASS | SHA256 của **89 tệp cũ** không đổi |

## C. Những lỗi phát hiện được

Không xác nhận lỗi nghiêm trọng làm sai các chỉ số Full CA_1 đã lưu. Các lỗi có thể tái hiện khi đưa đầu vào sai vào các module:

| Mã | Mức độ | Vị trí cũ | Lỗi và tác động |
|---|---|---|---|
| A01 | Trung bình | `feature_engineering.add_history_features` | Chưa từ chối ngày thiếu/trùng; shift theo hàng có thể bị hiểu sai thành lag theo ngày |
| A02 | Trung bình | `train_lightgbm.train_store` và `add_history_features` | Nhãn NaN/vô cực chưa bị kiểm tra trước fit; tái hiện được việc mô hình vẫn được tạo. Nhãn âm bị LightGBM báo lỗi sâu thay vì thông báo kiểm tra dữ liệu rõ ràng |
| A03 | Trung bình | `evaluation.evaluate_forecast` | Có thể bỏ qua mã dự báo thừa, lấy doanh số của cửa hàng khác có cùng item, dùng origin không đồng nhất hoặc lịch sử scale sai chiều dài khi gọi riêng |
| A04 | Trung bình | `evaluation.rmsse_scale` | Lịch sử chứa NaN có thể bị coi như scale không xác định và bị loại khỏi trung bình, che giấu dữ liệu hỏng |
| A05 | Trung bình | `feature_engineering.next_day_features` | Mã ngoài danh mục mô hình âm thầm thành giá trị phân loại thiếu thay vì bị từ chối |
| A06 | Nhỏ | `recursive_forecasting.recursive_forecast` | Hai vòng log giống nhau, không thể thấy vòng nào thuộc LightGBM/Seasonal Naive |
| A07 | Nhỏ | Tên cột summary | `RMSSE` chưa nêu cách tổng hợp ngay trong tên cột, dễ bị nhầm là chỉ số gộp hoặc WRMSSE |
| A08 | Nhỏ | `utils.export_forecasts` | Khóa cũ có origin; khi ghép nhiều origin có thể lọt cặp item/ngày đích trùng theo khóa chuẩn nhóm |

Các đường đọc M5 bình thường đã kiểm tra doanh số đầu vào, dùng bảng rộng đủ ngày và danh sách mã cố định. Vì vậy A01–A05 không đồng nghĩa lần Full CA_1 đã mắc những lỗi đó. Đối chiếu độc lập hiện vật không cho thấy ảnh hưởng đến kết quả cũ.

Về hiệu năng: đánh giá trước đây đọc lại toàn bộ 1.913 ngày mặc dù chỉ cần 28 nhãn holdout; xây đặc trưng có `.copy()` ngay sau một phép sắp xếp đã tạo bảng mới. Đây là chi phí thừa, không phải thay đổi mô hình.

## D. Những lỗi đã sửa và tính đúng đắn từng phần

### 1. Đọc dữ liệu, nối lịch/giá và chống rò rỉ

`load_store` dùng `usecols` để chỉ đưa d_1–d_1885 vào bảng Train Validation. Dù CSV gốc có các cột sau đó, chúng không được đưa vào bảng huấn luyện. Test theo thiết kế tương tự cắt tại d_1913. Lấy mẫu chỉ dùng mã và seed, không dùng nhãn Validation/Test để chọn sản phẩm.

`prepare_history` chỉ melt các cột đến origin. Nối lịch theo `d` với `validate='many_to_one'`; nối giá theo `store_id,item_id,wm_yr_wk` cũng kiểm tra quan hệ nhiều–một. Không nhân bản doanh số do nối nhiều–nhiều. Giữ nguyên các số 0, kể cả đầu chuỗi. SNAP lấy đúng CA/TX/WI.

Thứ trong tuần, tháng, ngày lễ theo lịch và kế hoạch SNAP được coi là biến ngoại sinh biết trước. Kết luận không rò rỉ phụ thuộc giả định lịch này có sẵn tại thời điểm dự báo. Mặc định giá tắt trong cả YAML và metadata Full CA_1. Khi bật, giá Train chỉ ffill trong từng chuỗi; giá dự báo giữ nguyên mức đã biết tại origin, không truy cập giá của tuần tương lai, không bfill và không gán thiếu thành 0 USD.

Test mới `test_mutating_validation_csv_cannot_change_forecast` thực hiện đúng phép kiểm tra yêu cầu: huấn luyện một mô hình thật trên ba sản phẩm giả lập, dự báo 28 ngày, thay ngẫu nhiên toàn bộ nhãn d_1886–d_1913 trong bản CSV thử, đọc lại Train và dự báo bằng chính mô hình đã lưu. **Dự báo trước/sau bằng nhau tuyệt đối**, với cả giá bật/tắt. Không sửa CSV M5 thật.

Test `test_pipeline_reads_holdout_only_after_both_forecasts` theo dõi luồng thực tế: đọc Train → một lần fit → hoàn tất LightGBM → hoàn tất Seasonal Naive → đọc holdout. Chưa có luồng đánh giá nhãn thật trước dự báo.

### 2. Đặc trưng và huấn luyện

Với ngày t: lag_k = y[t−k]; Rolling 7 lấy t−7…t−1; Rolling 28 lấy t−28…t−1. GroupBy theo cửa hàng–sản phẩm. Ví dụ chuỗi 1,2,… ở ngày 29: lag_1=28, lag_7=22, lag_14=15, lag_28=1, rolling_mean_7=25, rolling_mean_28=14,5. Các test đối chiếu từng công thức giữa Train và Predict, hai sản phẩm, hai cửa hàng, SNAP ba bang và lịch.

Đã bổ sung kiểm tra lịch sử liên tục/duy nhất, mã/ngày/nhãn không thiếu, lượng bán hữu hạn và không âm. `train_store` còn yêu cầu đúng một cửa hàng, mọi chuỗi bắt đầu d_1 và kết thúc cùng mốc. Mã không có trong danh mục mô hình bị từ chối khi dự báo.

Mô hình Full cũ là `LGBMRegressor`, Booster objective=`tweedie`, power=1.2, 200 cây theo cấu hình, learning_rate=0.05, num_leaves=31, seed=42, CPU và 6 luồng. Tham số `get_params()` khớp cấu hình lưu cùng lần chạy; danh sách đặc trưng và thứ tự khớp Booster, danh mục item khớp `pandas_categorical`. Không có cột `sales` trong X. Cửa hàng chỉ có một `model.joblib` và một bản xuất văn bản của chính mô hình đó, không có mô hình riêng từng sản phẩm.

Pipeline thêm `importance_type='gain'` cho cách báo cáo độ quan trọng; tùy chọn này không đổi việc fit. Metadata mới ghi cả tham số thực tế, số dòng bỏ do cửa sổ đầu và mốc fit. YAML, bộ đặc trưng và siêu tham số học không thay đổi.

Số dòng:

```text
3049 × 1885        = 5.747.365 hàng lịch sử
3049 × 28          =    85.372 hàng đầu chưa đủ lag_28/Rolling 28
3049 × (1885 − 28) = 5.661.993 hàng fit
```

Như vậy số dòng bạn thấy **hợp lý**. Mốc lịch sử vẫn là d_1–d_1885, nhưng hàng mục tiêu dùng trong fit bắt đầu d_29. Không phải loại các ngày bán bằng 0 hoặc rút ngắn lịch sử tùy ý.

### 3. Đệ quy, lưu/nạp và log

Hàm dự báo không nhận doanh số holdout. Dùng 28 ngày gần nhất để tạo đặc trưng, gọi cùng mô hình, nối dự báo vào cuối lịch sử tạm, dịch cửa sổ sang ngày kế. Đủ 28 bước, không fit lại, giữ thứ tự item và không trộn chuỗi. Dự báo âm cắt về 0; NaN/vô cực bị từ chối. Không làm tròn số lượng nguyên; có chuyển float32 theo chủ đích tiết kiệm RAM. Test mới xác nhận phần thập phân vẫn được giữ và truyền vào các bước sau.

Hai vòng 1/28…28/28 là **hai mô hình**, không phải huấn luyện hoặc dự báo LightGBM lặp nhầm. Log mới:

```text
Dự báo lightgbm_tweedie | CA_1: bước 1/28
Dự báo seasonal_naive | CA_1: bước 1/28
```

Các unit test so sánh dự báo mô hình vừa fit và mô hình nạp lại bằng nhau tuyệt đối. Với mô hình Full cũ, phát lại năm mã đầu theo danh mục, 140 dự báo, khớp chính xác các giá trị cũ khi đọc CSV về float32. Không cần huấn luyện lại Full để thực hiện phép kiểm tra này.

### 4. Đánh giá và tên chỉ số

MAE = trung bình sai số tuyệt đối của toàn bộ item × ngày. RMSE = căn trung bình bình phương sai số trên cùng tập điểm, không phải trung bình RMSE từng sản phẩm.

Với từng chuỗi i, đặt q_i là vị trí lần bán dương đầu trong **Train đầy đủ**. Mẫu số `s_i = mean(diff(y_train[q_i:])²)`. Tử số là MSE 28 ngày holdout. `RMSSE_i = sqrt(MSE_i / s_i)`. Bỏ tiền tố 0 chỉ khi tính scale, không bỏ khỏi dữ liệu học. Quy ước này phù hợp mô tả trong [bài báo M5 Accuracy của ban tổ chức](https://doi.org/10.1016/j.ijforecast.2021.11.013).

Chuỗi toàn 0, hằng sau lần bán đầu hoặc không đủ hai quan sát: scale và RMSSE để NaN; không thay bằng epsilon. Trung bình chỉ dùng chuỗi xác định và ghi rõ số hợp lệ/bị loại. Khi tất cả không xác định, trung bình để NaN, MAE/RMSE vẫn tính toàn bộ. Dữ liệu hỏng NaN/vô cực/âm nay bị từ chối thay vì gộp chung với trường hợp scale không xác định.

**`RMSSE = 0.728222` cũ là trung bình số học không trọng số của RMSSE từng sản phẩm–cửa hàng**, trên 3.049 chuỗi hợp lệ của CA_1. Không phải căn của MSE gộp chia scale gộp; cũng không phải WRMSSE chính thức. Chưa triển khai 12 cấp phân cấp và trọng số doanh thu M5.

Đổi tên cột summary mới thành **`RMSSE_mean_unweighted`**; công thức không đổi. Cột chi tiết từng item vẫn là `RMSSE`. README giải thích cách ánh xạ cột cũ khi ghép bảng. Không chỉnh tên/cột hay giá trị trong các tệp cũ.

Đánh giá hiện kiểm tra model/split/store/origin duy nhất, đúng tập item, lịch sử có đúng origin ngày và đúng cửa hàng thật. `validate_forecasts` dùng chung cho đánh giá/xuất CSV và kiểm tra cả khóa sản phẩm–ngày đích theo định dạng nhóm.

### 5. RAM và hiệu năng

`load_store` mới hỗ trợ `start_day` và `item_ids`. Khi đánh giá, chỉ đọc d_1886–d_1913 và các item đã dự báo thay vì giữ thêm bản sao lịch sử đầy đủ. Riêng CA_1 Full, ma trận nhãn float32 giảm từ khoảng 22,25 MiB xuống 0,326 MiB; đây là tính toán kích thước phần tử, chưa tính overhead DataFrame. Test xác nhận nhãn từ đường đọc mới bằng nhãn từ đường đọc cũ.

Bỏ bản `.copy()` thừa sau sort; vẫn giữ bảng riêng để không sửa lịch sử gọi vào. Sales, lag, rolling, SNAP và cửa sổ đệ quy dùng float32; mã lặp dùng category; lịch dùng int8/int32 và số ngày int16 trong phạm vi M5. Phần tính chỉ số dùng float64 để giảm sai số số học.

`run.py` gọi từng cửa hàng tuần tự. Các bảng lớn nằm trong phạm vi hàm `run_store`; Train giải phóng frame/X khi trả về, pipeline xóa wide/history/prices khi không cần và gọi gc; matplotlib đóng các figure. Không tích lũy danh sách dữ liệu dài của 10 cửa hàng. Tuy nhiên bộ cấp phát Python/NumPy/LightGBM có thể giữ bộ nhớ đệm, nên không thể suy từ gc rằng RAM hệ điều hành luôn về mức ban đầu.

Đã đo hai Smoke thật cùng CA_1 liên tiếp trong **cùng tiến trình**, bằng bộ đếm Windows:

| Mốc | RAM cư trú hiện tại (MiB) | RAM cư trú đỉnh (MiB) | Bộ nhớ riêng đã cam kết (MiB) |
|---|---:|---:|---:|
| Trước hai Smoke, sau import | 174,23 | 174,23 | 977,00 |
| Sau Smoke 1 và gc | 193,24 | 258,11 | 1.024,34 |
| Sau Smoke 2 và gc | 196,81 | 269,78 | 1.028,41 |

Hai Smoke hoàn tất, không giữ toàn bộ bảng dài sau khi hàm kết thúc; RAM cư trú sau lần hai tăng khoảng 3,57 MiB so với sau lần một. Bằng chứng này **không đủ kết luận không có rò rỉ bộ nhớ trên Full 10 cửa hàng**. Bộ nhớ riêng đã cam kết không đồng nghĩa toàn bộ đang nằm trong RAM vật lý. Chưa đo RAM đỉnh Full; giữ trạng thái CHƯA XÁC MINH cho yêu cầu đó. Không giảm số sản phẩm/lịch sử hoặc đổi số luồng để làm đẹp phép đo.

## E. Kết quả unit test và bằng chứng

- Bộ test ban đầu: **23 đạt, 0 lỗi**.
- Đợt tái hiện đầu vào sai trước sửa: **12 lỗi kỳ vọng được tái hiện, 1 ca đã bị chặn** trong 13 ca thử ban đầu. Đây là test cố tình mong chương trình từ chối dữ liệu hỏng, không phải 12 lỗi trên dữ liệu CA_1 thật.
- Bộ test sau sửa và bổ sung đầy đủ: **43 đạt, 0 lỗi, 0 bỏ qua**, thời gian lần chạy cuối **11,59 giây**.
- Đã kiểm tra phép sửa nhãn tương lai, thứ tự đọc holdout, công thức và tách nhóm, mã/ngày trùng, horizon đầy đủ, phân số dự báo, nhãn không hợp lệ, MAE/RMSE/RMSSE tính tay, scale không xác định và lưu/nạp.

Tái chạy từ thư mục project:

```powershell
..\.venv\Scripts\python.exe -m pytest -q --junitxml=audit_artifacts/pytest_after.xml
..\.venv\Scripts\python.exe scripts/audit_saved_results.py
..\.venv\Scripts\python.exe scripts/audit_model_replay.py
..\.venv\Scripts\python.exe scripts/audit_smoke_memory.py
```

| Bằng chứng | Nội dung |
|---|---|
| [pytest_after.xml](audit_artifacts/pytest_after.xml) | 43 ca kiểm tra và thời gian từng ca |
| [saved_results_check.json](audit_artifacts/saved_results_check.json) | Chỉ số tính độc lập, chênh lệch, số dòng, kiểm tra tham số |
| [model_replay.json](audit_artifacts/model_replay.json) | Phát lại mô hình Full trên 5 sản phẩm |
| [smoke_memory.json](audit_artifacts/smoke_memory.json) | Bộ nhớ trước/sau hai Smoke cùng tiến trình |
| [smoke_run.log](audit_artifacts/smoke_run.log) | Log Smoke với tên mô hình rõ ràng và chỉ số sau sửa |
| [smoke_prediction_comparison.json](audit_artifacts/smoke_prediction_comparison.json) | Cả hai Smoke sau sửa khớp chính xác 840 dự báo LightGBM của Smoke cũ |
| [old_results_sha256.json](audit_artifacts/old_results_sha256.json) | Dấu vân tay các tệp cũ trước chỉnh sửa |
| [old_results_integrity.json](audit_artifacts/old_results_integrity.json) | 89 tệp cũ, 0 tệp thay đổi |

Các Smoke kiểm toán lưu riêng trong `audit_artifacts/smoke_runs/`. Các file nguồn dataset và thư mục kết quả cũ không bị sửa hay xóa. Ngày/giờ UTC trong tên lần chạy là tên định danh, không phải thay đổi múi giờ báo cáo.

## F. Đánh giá kết quả Full CA_1 hiện tại

Lần được kiểm toán: `validation_full_CA_1_20261106T145400435654Z`.

- Đủ 3.049 item, 28 horizon mỗi item, **85.372 dự báo cho mỗi mô hình**.
- Origin d_1885; target đúng d_1886–d_1913; horizon trong từng item đúng 1…28.
- Không thiếu ngày, trùng khóa chuẩn, NaN, vô cực hay dự báo âm.
- CSV đúng `model,split,origin_day,item_id,store_id,target_day,horizon,y_pred`.
- Cả hai mô hình dùng đúng cùng tập item và nhãn thật.
- Seasonal Naive bằng chính xác bảy lượng bán cuối Train lặp lại bốn lần. Không lấy lượng bán thật mới ở tuần thứ hai/ba/bốn.

Script độc lập đọc CSV bằng float64, tự tính công thức, **không gọi `src.evaluation`**:

| Mô hình | Chỉ số | summary cũ | Tính lại từ CSV | Chênh lệch tuyệt đối |
|---|---|---:|---:|---:|
| LightGBM | MAE | 1,163019702145 | 1,163019701955 | 1,90 × 10⁻¹⁰ |
| LightGBM | RMSE | 2,241372552761 | 2,241372550166 | 2,60 × 10⁻⁹ |
| LightGBM | RMSSE trung bình | 0,728222371205 | 0,728222371225 | 1,98 × 10⁻¹¹ |
| Seasonal Naive | MAE | 1,330260506958 | 1,330260506958 | 0 |
| Seasonal Naive | RMSE | 2,806477821445 | 2,806477821445 | 0 |
| Seasonal Naive | RMSSE trung bình | 0,936025106614 | 0,936025106614 | ≈ 1,11 × 10⁻¹⁶ |

Chênh lệch LightGBM do CSV ghi số float32 thành chuỗi thập phân ngắn rồi được đọc lại bằng float64. Đã kiểm tra thêm: đưa dự báo CSV trở lại float32 như lần chạy gốc rồi tính bằng float64, **cả ba chỉ số LightGBM khớp chính xác summary**. Không phải lỗi căn chỉnh hay thay công thức. Cả 3.049 chuỗi đều có scale xác định.

Thời gian fit 27,968670900 giây đúng giá trị metadata đã lưu, chỉ là thời gian quanh `model.fit`, không gồm đọc CSV/tạo đặc trưng/dự báo. Kiểm toán không đo lại thời gian Full và không suy ra thời gian từ tệp mô hình.

**MAE=1,163020; RMSE=2,241373; RMSSE=0,728222 được xác minh đến sáu chữ số thập phân. Không cần chạy lại Full CA_1 vì các lỗi đã sửa lần này.** Kết quả dùng được cho so sánh ở cấp sản phẩm–cửa hàng theo thiết kế hiện tại; không đại diện WRMSSE toàn M5. Nếu nhóm cần WRMSSE hoặc thay bộ đặc trưng/công thức thì đó là thay đổi thiết kế riêng, phải lập báo cáo mới.

## G. Kết luận cuối cùng

**PASS WITH WARNINGS.** Có thể tiếp tục Validation cho 10 cửa hàng theo chế độ tuần tự đã có. Giữ nguyên lịch sử/sản phẩm Full và CPU 6 luồng; quan sát RAM thực tế khi chạy. Nếu thiếu bộ nhớ, dừng và xử lý tài nguyên thay vì tự cắt dữ liệu. Khi gộp summary, ánh xạ `RMSSE` cũ sang `RMSSE_mean_unweighted` mới và báo rõ số chuỗi hợp lệ.

Chưa mở Test cuối. Chỉ sử dụng Validation để quyết định cấu hình; Test cuối vẫn dành cho thời điểm đã khóa thiết kế.
