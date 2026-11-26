# LightGBM Two-Part C7_EVENTS

Cấu hình được chọn trên Validation d_1886–d_1913 là **C7_EVENTS**, λ = 0,75. Test d_1914–d_1941 chỉ dùng đánh giá sau lựa chọn. Các file nhỏ được công bố từ lượt đã kiểm chứng `results/analysis/c7_events_20261126/`; đợt đưa lên Git không Train, hiệu chuẩn hoặc dự báo lại.

| Cấu hình | Kỳ | WRMSSE chung 12 cấp | RMSSE trọng số cấp sản phẩm × cửa hàng | Sai lệch tổng lượng |
|---|---|---:|---:|---:|
| C7 gốc | Validation d_1886–d_1913 | 0,5825613324 | 0,8278839952 | −0,5206% |
| C7_EVENTS | Validation d_1886–d_1913 | 0,5785505731 | 0,8276652417 | −0,3511% |
| C7_EVENTS | Test d_1914–d_1941 | 0,6284865670 | 0,8258527913 | −4,3041% |

Điểm cấp 12 là điểm trọng số trong cấp đó, chưa chia 12; WRMSSE chung là trung bình 12 điểm cấp. Xem giá trị chưa làm tròn trong [scores.csv](../reports/c7_events/scores.csv) và [test_levels.csv](../reports/c7_events/test_levels.csv). Validation không có ngày gắn nhãn sự kiện; chênh lệch phản ánh tác động học và hiệu chuẩn, chưa đánh giá trực tiếp ngày lễ.

## Bộ 7 hình chính

1. [Sơ đồ vận hành Two-Part](../reports/bieu_do/so_do_two_part.png): hai nhánh, Platt, phối hợp xác suất và lượng bán.
2. [Lịch sử C0 → C7 → C7_EVENTS](../reports/bieu_do/lich_su_cau_hinh.png): C6 là mốc Platt ngắn; C7_EVENTS được làm nổi bật là bản chốt.
3. [Đối chứng Validation](../reports/bieu_do/so_sanh_validation.png): WRMSSE C7/C7_EVENTS và chênh lệch −0,0040107594.
4. [Tổng thực tế và dự báo Test](../reports/bieu_do/thuc_te_va_du_bao_test.png): 30.490 cặp, đúng 28 ngày.
5. [Sai lệch Test tại 10 cửa hàng](../reports/bieu_do/sai_lech_test_theo_cua_hang.png): âm là thiếu, dương là dư.
6. [Điểm theo ngành hàng](../reports/bieu_do/rmsse_theo_nganh_hang.png): FOODS, HOBBIES, HOUSEHOLD.
7. [Điểm theo tần suất bán](../reports/bieu_do/rmsse_theo_tan_suat_ban.png): thường xuyên và gián đoạn.

Hai hình nhóm chỉ phân tích C7_EVENTS trên Test: RMSSE có trọng số doanh thu chuẩn hóa trong từng nhóm tại cấp sản phẩm × cửa hàng. Nhãn ADI cố định từ lịch sử đến d_1885; trọng số doanh thu Test từ d_1886–d_1913. Không ghép thành xếp hạng sáu mô hình khi chưa có đủ kết quả cùng kỳ và cùng quy tắc.

Hình C0–C7/C6–C7 cũ được giữ trong `reports/archive/wrmsse_C0_C7_20261124/bieu_do/`, không xóa công trình trước. Các bảng cũ ở `reports/wrmsse_12_cap/`, `ket_qua_cuoi.csv` và `sai_lech_test_theo_cua_hang.csv` vẫn là **C7 gốc**, không phải C7_EVENTS.

## Chạy từ clone sạch

Python 3.11 trở lên, 64 bit; cài `requirements.txt`. Giữ các dependency benchmark của thành viên khác. Không thay `configs/config.yaml` đang dùng cho Tweedie. C7_EVENTS dùng riêng [configs/c7_events.json](../configs/c7_events.json).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_wrmsse.py tests/test_c7_events.py tests/test_c7_release.py -q
.\.venv\Scripts\python.exe scripts/plot_c7_events.py --published-report
```

Lệnh vẽ chỉ đọc bảng nhỏ đã commit: không cần raw data, model, log hoặc CSV dự báo 853.720 dòng. Nó dựng lại đúng bảy PNG; không chấm lại hoặc sửa bảng số liệu. Hash các PNG đã duyệt và bảng nguồn nằm trong [published_manifest.json](../reports/c7_events/published_manifest.json); text CSV/JSON được chuẩn hóa xuống dòng LF khi kiểm tra hash để chạy được cả Windows/Linux. Tái xuất PNG trên môi trường khác có thể khác hash do font/phiên bản Matplotlib.

Mã Two-Part nằm ở sáu module `src/data_loader.py`, `feature_engineering.py`, `preprocessing.py`, `training.py`, `forecasting.py`, `reporting.py`; không thay `src/ml/`, `src/benchmarks/` hoặc entrypoint Tweedie. Đường dẫn `src.training` được giữ để nạp checkpoint Two-Part đã lưu. Các script C4/C5/C6/Stage B được bổ sung vì cung cấp helper đặc trưng giá và Platt mà C7_EVENTS đã sử dụng; không chạy các entrypoint lịch sử bằng cấu hình YAML Tweedie.

## Nhận file dự báo để nhóm chấm chung

File không nằm trong commit. Chủ repo `24521081-ui` giữ bản thực nghiệm trên máy:

```text
D:\Phân tích dữ liệu kinh doanh\Code\IS403.R11-M5\results\analysis\c7_events_20261126\forecast_test_selected.csv
```

Nhận riêng file này từ người giữ artifact. Chép vào `results/analysis/c7_events_20261126/` hoặc dùng trực tiếp đường dẫn nhận được. Cột: `item_id,store_id,target_day,y_pred`; 853.720 dòng, 30.490 cặp sản phẩm × cửa hàng, mỗi cặp đúng 28 ngày d_1914–d_1941, không trùng khóa. SHA-256:

```text
96409130fe0b573f0268795bc4c6b245f43fc4059b49415559ad135598912a5c
```

Kiểm tra bằng `Get-FileHash -Algorithm SHA256`, đối chiếu [test_verified.json](../reports/c7_events/test_verified.json). Tệp mang tên C7 cũ không thay thế được file này. Không có URL tải công khai được xác minh; không suy ra file có trên GitHub.

Chấm lại bằng bộ chấm chung hiện có, không Train hoặc dự báo lại:

```powershell
.\.venv\Scripts\python.exe scripts/score_model_wrmsse.py --model C7_EVENTS --forecast "results/analysis/c7_events_20261126/forecast_test_selected.csv" --data-dir data/raw --output-dir .local_evidence/wrmsse_C7_EVENTS_recheck
```

Đặt `sales_train_evaluation.csv` đầy đủ d_1–d_1941, `calendar.csv`, `sell_prices.csv` trong `data/raw/`. Chọn thư mục output mới; bộ chấm từ chối ghi đè kết quả cũ. Dữ liệu M5, model, raw predictions, log và cache được Git bỏ qua.

## Tái lập hoặc dùng checkpoint — chỉ khi chủ động yêu cầu chạy

Đây là các lệnh thực nghiệm nặng, không phải bước cần chạy để xem báo cáo. Đợt công bố không thực hiện các lệnh này.

1. Khôi phục artifact C7 gốc: `results/analysis/stage_C7_validation_d1885_lambda075/folds/<store>/forecast_C7.csv` cho đủ 10 cửa hàng để đối chứng Validation; không dùng Test làm baseline chọn cấu hình. Cần yêu cầu chủ repo cung cấp bộ này riêng nếu muốn chấm lại quyết định chọn.
2. Khi có raw data, `run_c7_events.py` đọc cấu hình đóng băng riêng, không cần manifest/model C5 để tạo cấu hình. Mô hình phụ học đến d_1857; Platt học từ OOF d_1858–d_1885; mô hình chính học đến d_1885. Từ điển sự kiện chỉ lấy lịch đến d_1857.

```powershell
# Có thể thêm --stores CA_1 để chạy một cửa hàng; bỏ tùy chọn để chạy cả 10.
.\.venv\Scripts\python.exe scripts/run_c7_events.py --stage validation
.\.venv\Scripts\python.exe scripts/run_c7_events.py --stage select
# Chỉ sau khi select chọn C7_EVENTS:
.\.venv\Scripts\python.exe scripts/run_c7_events.py --stage test
.\.venv\Scripts\python.exe scripts/run_c7_events.py --stage verify-test
.\.venv\Scripts\python.exe scripts/run_c7_events.py --stage groups
```

Test: mô hình phụ học đến d_1885, Platt từ OOF d_1886–d_1913, mô hình chính đến d_1913. Mỗi bước đệ quy cập nhật lag/rolling bằng chính dự báo C7_EVENTS. Hai nhánh cùng 48 đặc trưng; xác suất `p_blend = 0.25*p_raw + 0.75*p_cal`, lượng bán `y_pred = p_blend*mu_positive`. Chính sách giá là `previous_completed_week`, cố định tại origin.

Để tái dùng checkpoint, yêu cầu riêng toàn bộ `models/<store>/d_<origin>/` và `validation/`, `test/` của lượt `c7_events_20261126`, giữ cả bundle và file classifier/regressor, `training.json`, Platt, calibration và `completed.json`. Script kiểm tra và tái dùng cache đã hoàn tất. Chỉ nạp joblib từ nguồn tin cậy. Không thể khôi phục checkpoint chỉ từ các bảng tổng hợp trong Git.

Các bộ điểm C0–C7 dùng vẽ lịch sử đã được công bố trong `reports/lich_su_cau_hinh_validation.csv`; không cần tái chạy lịch sử C0–C7 để đọc hình. Hồ sơ chi tiết, log và dự báo gốc vẫn được giữ tại máy thực nghiệm để truy xuất, không bị xóa khi công bố.
