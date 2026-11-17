# LightGBM–Tweedie dự báo đệ quy trên Walmart M5

Đề tài: **Đánh giá thực nghiệm các mô hình dự báo lượng bán sản phẩm trong bán lẻ: Hỗ trợ lập kế hoạch nhu cầu và bổ sung hàng hóa.**

Đây là **bản tự triển khai phục vụ học tập và thực nghiệm**, kế thừa có chọn lọc ý tưởng mô hình theo cửa hàng và dự báo đệ quy của nhánh `recursive_store`. Không phải bản sao chính xác, không tái tạo kết quả đội thắng. Tham số ban đầu chưa được tối ưu. Xem [đối chiếu mã nguồn](docs/reference_comparison.md).

## 1. Cài Python và môi trường

Dùng Python **3.11 bản 64 bit**, VS Code và Terminal PowerShell trên Windows. Tải bộ cài từ [trang Python](https://www.python.org/downloads/windows/), bật tùy chọn thêm Python vào PATH. Không cần CUDA hoặc GPU NVIDIA.

Mở Terminal tại thư mục project:

```powershell
cd C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Các ví dụ bên dưới dùng `python` sau khi kích hoạt môi trường:

```powershell
.\.venv\Scripts\Activate.ps1
python --version
```

Nếu PowerShell chặn kích hoạt, thay `python` bằng `.\.venv\Scripts\python.exe` trong mọi lệnh; không cần thay chính sách hệ thống. Trong VS Code, chọn **Python: Select Interpreter** rồi chọn `.venv\Scripts\python.exe`.

Trong workspace bàn giao đã có môi trường dùng để kiểm chứng tại `PTDLKD\.venv` (nằm ngoài project). Có thể dùng trực tiếp `..\.venv\Scripts\python.exe` khi đứng trong project. Phiên bản thực tế và kết quả kiểm thử được ghi trong [biên bản kiểm chứng](docs/verification.md). Mã dùng cú pháp tương thích Python 3.11; chưa mặc nhiên coi việc kiểm tra trên phiên bản khác là kiểm tra Python 3.11.

## 2. Đặt dữ liệu thật

Tự tải bộ **M5 Forecasting Accuracy** từ Kaggle, giải nén và đặt đúng tên:

```text
C:\Users\Admin\Desktop\PTDLKD\m5-lightgbm-tweedie\data\raw\
    sales_train_evaluation.csv
    calendar.csv
    sell_prices.csv
```

Không dùng `sales_train_validation.csv` vì không đủ doanh số đến d_1941 cho Test cuối. Không đặt file ZIP vào thay CSV. Chương trình báo trạng thái cả ba file; `sell_prices.csv` chỉ bắt buộc khi `use_prices: true`. Phiên bản đầu mặc định tắt giá.

Dữ liệu đầu vào được lưu duy nhất trong `data/raw/`, gồm ba tệp ở trên. Thư mục `dataset/` và dữ liệu giả lập đã được dọn bỏ.

## 3. Kiểm tra nhanh trước

Sau khi có dữ liệu thật, **lệnh đầu tiên** là:

```powershell
python run.py --mode smoke --store CA_1
```

Smoke lấy tối đa 30 mã sản phẩm bằng lấy mẫu ngẫu nhiên có seed, không dùng doanh số Validation/Test để chọn. Kết quả Smoke không đại diện toàn bộ M5.

## 4. Validation và thực nghiệm đầy đủ

| Giai đoạn | Lịch sử huấn luyện | Dự báo và đánh giá |
|---|---|---|
| Validation | d_1–d_1885 | d_1886–d_1913 |
| Test cuối | d_1–d_1913, huấn luyện lại | d_1914–d_1941 |

Chạy đủ 3049 sản phẩm của CA_1:

```powershell
python run.py --mode validation --store CA_1 --scope full
```

Chạy lần lượt 10 cửa hàng:

```powershell
python run.py --mode validation --all-stores --scope full
```

Mỗi cửa hàng có một mô hình chung cho các sản phẩm. Không chạy song song các cửa hàng. Full kiểm tra đủ 3049 sản phẩm/cửa hàng trên dữ liệu thật và không tự cắt lịch sử. `--all-stores` yêu cầu dữ liệu thật có đủ 10 cửa hàng. Có thể thử luồng Validation trên mẫu nhỏ bằng `--scope smoke`.

Dùng Validation để so sánh cấu hình. Mỗi lần chạy lưu ở thư mục có thời điểm UTC riêng để tránh ghi đè. Mô hình dùng số cây cố định, không dùng holdout làm `eval_set` hay dừng sớm; mọi đặc trưng Train chỉ được tính đến origin.

## 5. Đọc dự báo và tìm kết quả

Thư mục một lần chạy có tên như `validation_smoke_CA_1_<thời_điểm_UTC>`. Dữ liệu thật được lưu ở:

```text
results/
  validation/<tên_lần_chạy>/lightgbm_tweedie.csv
  validation/<tên_lần_chạy>/seasonal_naive.csv
  test/<tên_lần_chạy>/...                 # chỉ khi gọi chế độ test
  models/<tên_lần_chạy>/model.joblib
  models/<tên_lần_chạy>/model.txt
  models/<tên_lần_chạy>/metadata.json
  metrics/<tên_lần_chạy>/summary.csv
  metrics/<tên_lần_chạy>/*_per_item.csv
  figures/<tên_lần_chạy>/*.png
```


CSV dự báo có đúng tám cột:

| Cột | Ý nghĩa |
|---|---|
| model | `lightgbm_tweedie` hoặc `seasonal_naive` |
| split | `validation` hoặc `test` |
| origin_day | Ngày cuối được quan sát, như `d_1885` |
| item_id | Mã sản phẩm |
| store_id | Mã cửa hàng |
| target_day | Ngày được dự báo, như `d_1886` |
| horizon | Số bước từ origin, từ 1 đến 28 |
| y_pred | Dự báo không âm, không làm tròn |

Mỗi sản phẩm có đúng 28 dòng trên mỗi mô hình. File này dùng để ghép với kết quả thành viên khác; không phải định dạng nộp Kaggle. Doanh số thật holdout chỉ được đọc sau khi cả LightGBM và baseline hoàn thành dự báo.

`model.joblib` lưu mô hình, danh sách đặc trưng đúng thứ tự và các mã phân loại. Chỉ mở tệp joblib do bạn hoặc nguồn tin cậy tạo. `model.txt` là mô hình LightGBM dạng văn bản. `metadata.json` ghi cấu hình, mã sản phẩm đã chọn, số dòng Train, thời gian `fit`, phiên bản Python/thư viện và cờ demo.

## 6. Hiểu các chỉ số

`summary.csv` mới có các cột `model, split, store_id, n_items, MAE, RMSE, RMSSE_mean_unweighted, training_time_seconds` cùng số chuỗi RMSSE hợp lệ/không xác định.

Sau kiểm toán, cột tổng hợp đổi tên từ `RMSSE` thành **`RMSSE_mean_unweighted`** để nêu rõ đây là trung bình không trọng số của RMSSE từng sản phẩm–cửa hàng. Công thức và giá trị không đổi. Các summary cũ được giữ nguyên; khi ghép kết quả cũ/mới, ánh xạ cột `RMSSE` cũ sang tên mới. Trong bảng chi tiết từng sản phẩm, cột vẫn tên `RMSSE` vì đó là chỉ số của một chuỗi, chưa tổng hợp. Không phải WRMSSE chính thức của M5.

- **MAE** = trung bình `abs(y_thật - y_dự_báo)` trên tất cả sản phẩm và ngày. Đơn vị là số sản phẩm bán.
- **RMSE** = căn của trung bình bình phương sai số. Sai số lớn bị phạt mạnh hơn MAE.
- **RMSSE** cho từng chuỗi = `sqrt(MSE_holdout / mean(diff(Train_sau_lần_bán_dương_đầu_tiên)^2))`. Báo cáo trung bình không trọng số của RMSSE các chuỗi xác định.
- Chuỗi toàn 0, chuỗi hằng sau lần bán đầu hoặc chỉ còn một điểm có RMSSE không xác định. Lưu trống (NaN) trong CSV chi tiết; bỏ khỏi trung bình và báo `rmsse_undefined_items`. Nếu không có chuỗi hợp lệ, RMSSE tổng cũng để trống. MAE/RMSE vẫn bao gồm mọi chuỗi.

Việc bỏ tiền tố 0 chỉ áp dụng vào **mẫu số RMSSE** theo cách tính M5; dữ liệu Train không bỏ các ngày này. Không thay mẫu số 0 bằng epsilon. So sánh RMSSE phải đọc kèm số chuỗi hợp lệ.

**Chưa tính WRMSSE chính thức**: chưa tổng hợp đủ 12 cấp và trọng số doanh thu M5. Không gọi RMSSE trung bình của một cửa hàng hoặc Smoke là WRMSSE. Chỉ số càng nhỏ càng tốt khi đánh giá trên cùng dữ liệu và cùng quy ước.

**Seasonal Naive** lặp lại lượng bán cùng thứ tuần trước: ngày T+1 lấy T−6; ngày T+8 lấy dự báo T+1. Baseline không sử dụng doanh số thực tế phát sinh trong 28 ngày holdout.

## 7. Xem hình

Mở các PNG độ phân giải 220 dpi trong thư mục figures của lần chạy:

1. `actual_vs_prediction.png`: doanh số thật và hai dự báo của sản phẩm đầu tiên theo mã; không chọn sản phẩm theo sai số holdout.
2. `error_by_horizon.png`: MAE gộp trên sản phẩm ở từng bước 1–28.
3. `feature_importance.png`: độ quan trọng theo tổng gain của LightGBM; không phải quan hệ nhân quả.
4. `model_comparison.png`: MAE của LightGBM và Seasonal Naive.

## 8. Chạy Test cuối sau khi khóa cấu hình

Lưu bản cấu hình đã chọn và ghi lại lý do lựa chọn từ Validation. Chỉ sau đó chạy:

```powershell
python run.py --mode test --store CA_1 --scope full --confirm-final-test
```

Hoặc thêm `--all-stores` thay `--store CA_1` để chạy tuần tự đủ cửa hàng. CLI từ chối Test nếu thiếu cờ xác nhận. Mọi chế độ Smoke/Validation đều không tự chạy Test. Cờ xác nhận thể hiện quyết định của người chạy; phần mềm không thể tự xác định bạn đã khóa thiết kế nghiên cứu. Không quay lại tối ưu tham số dựa trên kết quả Test.

## 9. Thay cấu hình và dùng giá an toàn

Sửa `configs/config.yaml`, hoặc sao chép thành cấu hình khác rồi chạy:

```powershell
python run.py --mode validation --store CA_1 --scope full --config configs/config.yaml
```

Đường dẫn tương đối trong YAML và `--config` được tính từ project. Các tham số khởi điểm: `objective=tweedie`, `tweedie_variance_power=1.2`, `learning_rate=0.05`, `n_estimators=200`, `num_leaves=31`, `random_state=42`, `n_jobs=6`. Đây không phải tham số gốc đội thắng và không bảo đảm tối ưu. Giữ các mốc chia và horizon cố định để nhất quán thiết kế nhóm.

Đặc trưng gồm mã sản phẩm dạng phân loại, thứ trong tuần (thứ Hai=0), tháng, cờ sự kiện (có event_name_1 hoặc event_name_2), SNAP đúng bang, lag 1/7/14/28 và trung bình trượt 7/28 ngày. Bỏ 28 hàng đầu mỗi chuỗi khỏi phần `fit` do chưa đủ cửa sổ; giữ nguyên toàn bộ lịch sử và số 0 trong dữ liệu. Không sử dụng mã ngày hay doanh số hiện tại làm đầu vào.

Khi bật `use_prices: true`, code nối `sell_prices.csv` theo cửa hàng/sản phẩm/tuần. Trong Train chỉ điền tiến (ffill) theo từng chuỗi. Giá chưa từng biết vẫn là NaN, được LightGBM xử lý; không điền bằng 0 hoặc bằng giá tương lai. Tại origin, lấy giá cuối cùng đã biết và **giữ cố định trong 28 ngày** (`price_policy: last_known`). Chương trình không hỗ trợ tự dùng bảng giá tương lai; nếu có kế hoạch giá biết trước, cần thiết kế giao diện riêng và ghi rõ giả định trước khi mở rộng.

## 10. Cách hoạt động và tái sử dụng

`run.py` đọc cấu hình và điều phối trực tiếp 6 giai đoạn. `src/` chỉ có 6 file nghiệp vụ, mỗi file tương ứng một giai đoạn, có comment và log đánh số:

| Giai đoạn | File chính | Đầu ra |
|---|---|---|
| 1. Đọc dữ liệu | `data_loader.py` | Doanh số theo cửa hàng, lịch và giá nếu bật |
| 2. Chuẩn bị dữ liệu | `preprocessing.py` | Bảng dài Train và ma trận lịch sử |
| 3. Tạo đặc trưng | `feature_engineering.py` | X, y, lag, rolling và mã phân loại |
| 4. Huấn luyện | `training.py` | Model và thông tin huấn luyện |
| 5. Dự báo | `forecasting.py` | Dự báo đệ quy 28 ngày của hai mô hình |
| 6. Đánh giá và xuất kết quả | `reporting.py` | Metrics, biểu đồ và metadata |

Chạy đủ 6 giai đoạn cho một cửa hàng:

```powershell
python run.py --mode validation --store CA_1 --scope full
```

Thêm `--all-stores` thay cho `--store CA_1` để chạy lần lượt 10 cửa hàng. Dùng `--mode smoke --store CA_1` để kiểm tra toàn bộ luồng trên 30 sản phẩm. Các hàm tính chỉ số, vẽ biểu đồ và xuất báo cáo đều nằm trong `reporting.py`.

Train xây đặc trưng bằng GroupBy theo sản phẩm–cửa hàng. Rolling dùng `shift(1).rolling(window).mean()`, nên ngày t không nhìn doanh số ngày t. Dự báo giữ ma trận 28 ngày gần nhất, tạo cùng các đặc trưng rồi gọi `predict()`, thêm kết quả vào lịch sử tạm thời. Mô hình không được huấn luyện lại mỗi ngày. Mỗi bước xử lý toàn bộ sản phẩm theo thứ tự mã cố định.

`add_history_features()` và `next_day_features()` trong `src/feature_engineering.py` có thể tái dùng. `recursive_forecast()` nhận mô hình bất kỳ có `predict(X)`. Mô hình khác cần hỗ trợ cùng cách mã hóa phân loại hoặc tự thêm bước mã hóa phù hợp; Random Forest không trực tiếp hiểu mã chuỗi dạng category như LightGBM.

| Tệp | Vai trò |
|---|---|
| `run.py` | Đọc CLI, chặn Test chưa xác nhận, xử lý cửa hàng tuần tự |
| `src/data_loader.py` | Kiểm tra và đọc CSV theo khối |
| `src/preprocessing.py` | Bảng dài Train, nối lịch/giá, giữ số 0 |
| `src/feature_engineering.py` | Lag, Rolling, đặc trưng cho ngày kế tiếp |
| `src/training.py` | Nhận đặc trưng đã tạo, fit và lưu mô hình |
| `src/forecasting.py` | Đệ quy 28 ngày và Seasonal Naive |
| `src/reporting.py` | Đánh giá hai mô hình, xuất metrics, hình và metadata |
| `tests/` | Kiểm tra công thức, leakage, định dạng và CLI |

## 11. Thuật ngữ dành cho người mới

| Thuật ngữ | Giải thích |
|---|---|
| Train | Huấn luyện: cho mô hình học quy luật từ lịch sử |
| Validation | Xác thực: đánh giá cấu hình trên đoạn tương lai chưa học |
| Test | Kiểm thử cuối: đánh giá độc lập sau khi khóa thiết kế |
| Feature Engineering | Xây dựng đặc trưng: biến dữ liệu thành thông tin đầu vào |
| Lag | Độ trễ: lag_7 của ngày t là lượng bán ngày t−7 |
| Rolling Mean | Trung bình trượt: ví dụ trung bình 7 ngày trước ngày t |
| Recursive Forecasting | Dự báo đệ quy: dùng dự báo hôm nay làm lịch sử cho ngày mai |
| Model | Mô hình: hàm đã học từ dữ liệu để tạo dự báo |
| Prediction | Giá trị dự báo, có thể là số thập phân |
| Metric | Chỉ số đánh giá độ sai lệch dự báo |
| Origin | Mốc cuối cùng có quan sát khi bắt đầu dự báo |
| Horizon | Khoảng cách từ mốc origin đến ngày cần dự báo |
| Data Leakage | Rò rỉ: dùng thông tin chưa có tại thời điểm dự báo |
| Tweedie | Họ phân phối hỗ trợ mô hình hóa mục tiêu không âm có nhiều số 0 |

## 12. Lỗi thường gặp

- **Không tìm thấy CSV**: kiểm tra tên, phần mở rộng và vị trí `data/raw/`. Terminal in đường dẫn cụ thể.
- **`python` mở Microsoft Store hoặc không chạy**: dùng đường dẫn đầy đủ đến Python hoặc `.venv\Scripts\python.exe`. Có thể dùng `py -3.11` nếu đã cài Python 3.11.
- **`No module named ...`**: cài requirements bằng đúng Python dùng chạy chương trình; tránh cài vào môi trường khác.
- **Không tải được thư viện**: kiểm tra kết nối PyPI và proxy; dùng bản Python 64 bit được thư viện hỗ trợ.
- **Thiếu DLL LightGBM trên Windows**: cài Microsoft Visual C++ Redistributable x64, mở lại Terminal.
- **Thiếu RAM**: đóng ứng dụng khác, giảm `chunk_size` khi đọc CSV và giảm `n_jobs` nếu cần. Các thay đổi này không loại dữ liệu Full. Nếu vẫn thiếu RAM, dùng máy có RAM lớn hơn; dùng Smoke để kiểm tra kỹ thuật và ghi rõ giới hạn. Không tự rút ngắn lịch sử rồi báo cáo như Full.
- **Full báo không đủ 3049 sản phẩm**: kiểm tra đúng file gốc M5, không dùng dữ liệu đã lọc; demo là ngoại lệ có nhãn riêng.
- **Mã trùng, doanh số thiếu/âm, lịch thiếu ngày, giá trùng khóa**: sửa nguồn dữ liệu; chương trình dừng thay vì tự xóa hay gán 0.
- **RMSSE trống**: xem cột scale và số chuỗi không xác định; đây là chính sách chỉ số có chủ đích.
- **Huấn luyện chậm**: Full một cửa hàng có khoảng 5,7 triệu hàng trước khi bỏ cửa sổ đầu. Thời gian phụ thuộc máy, cấu hình và dữ liệu. Lần Full CA_1 bạn chạy có thời gian fit khoảng 27,97 giây; kiểm toán đã kiểm tra hiện vật, chưa đo lại RAM đỉnh Full hoặc chạy 10 cửa hàng.

## 13. Tài liệu

Nguồn mã đối chiếu và phạm vi khác biệt: [docs/reference_comparison.md](docs/reference_comparison.md). Biên bản kiểm thử thực tế: [docs/verification.md](docs/verification.md). Hai nghiên cứu do nhóm cung cấp là định hướng học thuật; project không tuyên bố tái lập toàn bộ thực nghiệm của các bài báo.

## 14. Kiểm toán trước khi mở rộng 10 cửa hàng

Xem [CODE_AUDIT_REPORT.md](CODE_AUDIT_REPORT.md) và các bằng chứng trong `audit_artifacts/`. Kiểm toán không đổi mốc chia, bộ đặc trưng, cấu hình YAML hoặc công thức chỉ số; không ghi đè kết quả cũ. Log nay hiển thị tên LightGBM/Seasonal Naive cho hai vòng dự báo. Đánh giá chỉ đọc 28 cột holdout thay vì đọc lại toàn bộ lịch sử.

Các module kiểm tra chặt hơn: từ chối chuỗi thiếu/trùng ngày, nhãn không hữu hạn/âm, mã phân loại chưa biết và đầu vào đánh giá lệch mã/cửa hàng/origin. Những lỗi này được tái hiện bằng dữ liệu giả lập; không có bằng chứng chúng làm sai kết quả Full CA_1 đã lưu.

Các script kiểm toán chạy từ thư mục project:

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe scripts/audit_saved_results.py
..\.venv\Scripts\python.exe scripts/audit_model_replay.py
..\.venv\Scripts\python.exe scripts/audit_smoke_memory.py
```

Script đối chiếu model phát lại 5 sản phẩm từ lần Full CA_1 cụ thể đã kiểm toán; script đo RAM chỉ chạy hai Smoke CA_1 liên tiếp, lưu riêng trong `audit_artifacts/smoke_runs/`. Không script nào trong ba script trên huấn luyện Full hoặc gọi Test cuối. Pytest dùng dữ liệu giả lập trong thư mục tạm, gồm kiểm tra mốc Test bằng dữ liệu giả; không đọc 28 nhãn Test thật hay mở thực nghiệm Test cuối.
