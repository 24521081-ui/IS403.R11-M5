# Giai đoạn 3 mở rộng với toàn bộ đặc trưng

Đề xuất giữ **FE-04** cho Giai đoạn 4. FE-09 đạt **0,6321382731** và FE-10 đạt **0,6360743504**. Mốc FE-04 là **0,6239308540**. Kết luận giới hạn ở các ứng viên đã đánh giá trên cửa sổ này.

## So sánh trên Validation

Seasonal Naive đạt WRMSSE **0,9227813345**. Chênh lệch là WRMSSE của ứng viên trừ WRMSSE FE-04. Số âm là cải thiện. Các tỷ lệ lấy từ điểm gốc chưa làm tròn. Thời gian và RAM do các lượt ghi nhận, nên chịu ảnh hưởng của tải máy.

| Lượt | Bộ đặc trưng | Logic / encoded | WRMSSE | Δ WRMSSE | Δ % | Fit (s) | Predict (s) | RAM đỉnh (MiB) | n_jobs | Đủ điều kiện |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FE-01 | rf_v1 | 14 / 3,081 | 0,66477108 | 0,04084023 | 6,55% | 252,65 | 2,79 | 725,25 | 8 | Có |
| FE-02 | rf_v1 + recency | 15 / 3,082 | 0,66942518 | 0,04549433 | 7,29% | 223,27 | 2,67 | 725,18 | 8 | Có |
| FE-03 | rf_v1 + prices | 21 / 3,088 | 0,65477696 | 0,03084611 | 4,94% | 260,34 | 2,77 | 782,07 | 8 | Có |
| FE-04 | rf_v1 + snap | 15 / 3,082 | 0,62393085 | 0,00000000 | 0,00% | 225,05 | 2,77 | 736,48 | 8 | Có |
| FE-05 | rf_v1 + events | 18 / 3,103 | 0,66476314 | 0,04083229 | 6,54% | 221,13 | 2,68 | 738,43 | 8 | Thăm dò |
| FE-06 | rf_v1 + behavior | 17 / 3,084 | 0,68256328 | 0,05863243 | 9,40% | 254,41 | 2,74 | 768,38 | 8 | Có |
| FE-07 | rf_v1 + behavior + snap | 18 / 3,085 | 0,63821465 | 0,01428380 | 2,29% | 268,46 | 2,93 | 790,20 | 8 | Có |
| FE-08 | rf_v1 + behavior + prices | 24 / 3,091 | 0,66909255 | 0,04516169 | 7,24% | 313,64 | 2,80 | 845,27 | 8 | Có |
| FE-09 | rf_v4 + prices + snap + behavior | 29 / 3,096 | 0,63213827 | 0,00820742 | 1,32% | 446,49 | 2,97 | 919,76 | 8 | Có |
| FE-10 | rf_v4 + prices + snap + behavior + events | 33 / 3,118 | 0,63607435 | 0,01214350 | 1,95% | 389,46 | 2,91 | 923,64 | 8 | Có |

FE-01–FE-08 được lấy từ kết quả và trạng thái selection v2 đã lưu. Hai lượt mới đều dùng RF-15, cùng Train 600.000 dòng, Validation 853.720 dòng và khung dự báo 28 ngày. FE-09 hợp nhất RF-v4, Historical Prices, SNAP và Behavior. FE-10 thêm Events. Đếm theo schema thực tế được 29 và 33 đặc trưng logic, 3.096 và 3.118 cột encoded, với mỗi tên và định nghĩa feature xuất hiện một lần. Recency đã có trong RF-v4.

## Chi phí tính toán và vai trò của Events

- **FE-09** so với FE-04: fit chênh 221,44 giây (98,39%), predict chênh 0,21 giây, RAM đỉnh chênh 183,28 MiB (24,89%).
- **FE-10** so với FE-04: fit chênh 164,40 giây (73,05%), predict chênh 0,15 giây, RAM đỉnh chênh 187,16 MiB (25,41%).

Thêm Events làm WRMSSE **tăng 0,0039360773** so với FE-09, tương ứng 0,62%. Cả 22 cột của Events khớp khi dựng bằng quy tắc ngày lễ và loại sự kiện học từ lịch đến d_1885. **FE-10 đủ điều kiện được cân nhắc trên cửa sổ này.** Trạng thái thăm dò trong request/result của worker được ghi trước khi kiểm tra riêng hoàn tất. Bảng trên áp dụng kết luận hậu kiểm cho FE-10, giữ nguyên trạng thái lịch sử của FE-05 trong selection v2. Có **0 ngày** bật cờ Events trong Validation. Do đó chênh lệch FE-09–FE-10 phản ánh ảnh hưởng của việc thêm feature vào huấn luyện và các điểm chia của rừng trong cửa sổ này. Nó chưa cho thấy chất lượng vào ngày có sự kiện. [Hậu kiểm Events](events_origin_audit.json)

## WRMSSE theo 12 cấp phân cấp

| Cấp | Số chuỗi | FE-04 | FE-09 | FE-10 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0,40077153 | 0,40092306 | 0,41050059 |
| 2 | 3 | 0,46312403 | 0,48225677 | 0,48903341 |
| 3 | 10 | 0,52667505 | 0,55127802 | 0,55441737 |
| 4 | 3 | 0,46456841 | 0,46422962 | 0,47092000 |
| 5 | 7 | 0,55106781 | 0,55931790 | 0,56430145 |
| 6 | 9 | 0,54151831 | 0,55663638 | 0,56139898 |
| 7 | 21 | 0,62254824 | 0,64106403 | 0,64514128 |
| 8 | 30 | 0,61860884 | 0,63574372 | 0,63874025 |
| 9 | 70 | 0,68521033 | 0,70436759 | 0,70732252 |
| 10 | 3,049 | 0,87977706 | 0,87117670 | 0,87198493 |
| 11 | 9,147 | 0,87347831 | 0,86628104 | 0,86676896 |
| 12 | 30,490 | 0,85982232 | 0,85238445 | 0,85236246 |

Các giá trị trong bảng là WRMSSE chuẩn hóa trong từng cấp. WRMSSE chung là trung bình của 12 cấp, trên 42.840 chuỗi.

## Kiểm tra trước và sau fit

69 kiểm thử đã đạt trước khi mở gate. Checksum của dữ liệu, ma trận, evaluator và selection v2 đều khớp. Lưới khóa và nhãn được đối chiếu với mẫu gốc. Feature doanh số dừng ở t−28 và không vượt d_1885 trên Validation. Historical Prices dùng tuần đã hoàn tất tại cutoff với median từ Train. Toàn bộ cột SNAP khớp quy tắc d_1885 đã khóa. Các ma trận huấn luyện và dự báo giữ CSR float32. Hai lượt chạy tuần tự với n_jobs=8 và giới hạn RAM của dự án.

Test vẫn khóa, không truy cập hoặc đánh giá. Giai đoạn 4 và final retrain chưa chạy. Selection v2, các báo cáo chính thức, kết quả cũ, raw, legacy và dự án nguồn giữ nguyên.

## Tệp kết quả

- [Bảng FE-01–FE-10 CSV](comparison_fe01_fe10.csv) · [Bảng Parquet](comparison_fe01_fe10.parquet) · [Bảng 12 cấp](comparison_by_level.csv).
- [WRMSSE theo lượt](wrmsse_fe01_fe10.png) · [WRMSSE theo số feature](feature_count_vs_wrmsse.png) · [WRMSSE theo thời gian fit](fit_time_vs_wrmsse.png) · [WRMSSE theo 12 cấp](wrmsse_by_level.png) · [Chi phí](compute_cost.png).
- FE-09 [manifest](FE-09_20261124T102016236931Z/manifest.json) · [đặc trưng](FE-09_20261124T102016236931Z/feature_metadata.json) · [cấu hình](FE-09_20261124T102016236931Z/request.json) · [điểm](FE-09_20261124T102016236931Z/result.json) · [điểm 12 cấp](FE-09_20261124T102016236931Z/wrmsse_by_level.csv) · [log CPU/RAM](FE-09_20261124T102016236931Z/resources.jsonl).
- FE-10 [manifest](FE-10_20261124T102016828681Z/manifest.json) · [đặc trưng](FE-10_20261124T102016828681Z/feature_metadata.json) · [cấu hình](FE-10_20261124T102016828681Z/request.json) · [điểm](FE-10_20261124T102016828681Z/result.json) · [điểm 12 cấp](FE-10_20261124T102016828681Z/wrmsse_by_level.csv) · [log CPU/RAM](FE-10_20261124T102016828681Z/resources.jsonl).
- [Kiểm tra trước fit](preflight.json) · [Kết quả giám sát](execution_summary.json) · [Manifest báo cáo](manifest.json).

Kho lịch sử giữ model, dự báo Validation, điểm từng chuỗi và bản báo cáo đã niêm phong của cả hai lượt theo checksum gốc. [Biên bản và chỉ mục](../../../docs/PROJECT_CLEANUP_REPORT.md).
