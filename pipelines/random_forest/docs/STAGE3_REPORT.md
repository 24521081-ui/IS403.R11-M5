# Giai đoạn 3 tối ưu Random Forest theo WRMSSE

**Chọn RF-15 + FE-04 với RF-v1 + SNAP, WRMSSE Validation 0,6239308540.**
Seasonal Naive đạt 0,9227813345 trên cùng Validation. Mô hình được chọn giảm WRMSSE tuyệt đối 0,2988504806, tương đương 32,39%.

Giai đoạn tối ưu đã kết thúc với **36 lượt thực nghiệm**, gồm **32 lượt fit thành công** và **4 lượt dùng lại**. Selection v2 khóa RF-15 + FE-04 theo WRMSSE thấp nhất trong 33 bản ghi hợp lệ tại thời điểm hậu kiểm SNAP. Hai lượt mở rộng và hậu kiểm Events xác nhận FE-04 vẫn có WRMSSE thấp nhất.

## Dữ liệu và quy trình

Train là d_1–d_1885. Mẫu gồm 600.000 dòng lấy theo store × dept × quý với seed 42. Validation d_1886–d_1913 gồm đủ 853.720 dòng, 30.490 chuỗi và horizon 28 ngày. Điểm chọn mô hình là WRMSSE M5 tính trên 42.840 chuỗi ở đủ 12 cấp.

Mọi lượt sử dụng mẫu và ma trận đặc trưng đã tạo ở giai đoạn 2. Đặc trưng doanh số tại ngày t lấy dữ liệu tối đa đến t−28 và d_1885. Đặc trưng giá lấy tuần đã kết thúc tại cutoff, với median học từ Train. FE-04 và FE-07 đủ điều kiện ở selection v2 vì toàn bộ cờ SNAP Validation khớp quy tắc chỉ dùng lịch trước gốc. FE-07 dùng Behavior + SNAP, FE-08 dùng Behavior + Prices. [Hậu kiểm SNAP](STAGE3_SNAP_REVIEW.md).

Các lượt dùng lại tham chiếu artifact cùng batch khi siêu tham số mô hình, bộ đặc trưng, mẫu Train và seed trùng nhau. Khi thay đổi số luồng, record dùng lại ghi cả số luồng của fit nguồn và số luồng được yêu cầu. RF-03 được fit mới trong giai đoạn 3. Kết quả smoke của giai đoạn 2 và mọi score legacy nằm ngoài tập ứng viên.

## RF-01 đến RF-05

Bước 1 chọn RF-04 trên rf_v1 (WRMSSE 0,67572915). Bộ đặc trưng là RF-v1 cho cả năm lượt.

| Lượt | Bộ đặc trưng | Logic / cột mã hóa | WRMSSE | Fit (s) | Predict (s) | RAM đỉnh (MiB) | n_jobs | Kết quả |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RF-01 | rf_v1 | 14 / 3081 | 0,79870418 | 14,38 | 1,12 | 512,39 | 4 | Fit mới |
| RF-02 | rf_v1 | 14 / 3081 | 0,74738979 | 36,12 | 1,23 | 514,27 | 4 | Fit mới |
| RF-03 | rf_v1 | 14 / 3081 | 0,74595361 | 57,48 | 2,25 | 521,14 | 4 | Fit mới |
| RF-04 | rf_v1 | 14 / 3081 | 0,67572915 | 196,30 | 2,77 | 544,10 | 4 | Fit mới |
| RF-05 | rf_v1 | 14 / 3081 | 0,67611550 | 268,08 | 3,95 | 559,14 | 4 | Fit mới |

## So sánh RF-v1 đến RF-v4

Dùng siêu tham số của RF-04 trên cùng Train sample. Bộ thắng là **rf_v1**.

| Lượt | Bộ đặc trưng | Logic / cột mã hóa | WRMSSE | Fit (s) | Predict (s) | RAM đỉnh (MiB) | n_jobs | Kết quả | Δ WRMSSE so với RF-v1 |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| ABL_rf_v1 | rf_v1 | 14 / 3081 | 0,67572915 | 196,30 | 2,77 | 544,10 | 4 | Dùng lại RF-04 | 0,00000000 |
| ABL_rf_v2 | rf_v2 | 16 / 3083 | 0,68081577 | 225,83 | 2,68 | 557,36 | 4 | Fit mới | 0,00508662 |
| ABL_rf_v3 | rf_v3 | 17 / 3084 | 0,68421630 | 232,89 | 2,71 | 559,59 | 4 | Fit mới | 0,00848714 |
| ABL_rf_v4 | rf_v4 | 18 / 3085 | 0,68505223 | 245,48 | 2,71 | 567,39 | 4 | Fit mới | 0,00932308 |

## RF-06 đến RF-22

Các bộ siêu tham số giữ theo thiết kế lịch sử và dùng bộ thắng ablation. Cấu hình chuyển sang FE là RF-15 trên rf_v1 (WRMSSE 0,66477108).

| Lượt | Bộ đặc trưng | Logic / cột mã hóa | WRMSSE | Fit (s) | Predict (s) | RAM đỉnh (MiB) | n_jobs | Kết quả |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RF-06 | rf_v1 | 14 / 3081 | 0,74344410 | 98,01 | 3,17 | 527,50 | 4 | Fit mới |
| RF-07 | rf_v1 | 14 / 3081 | 0,70640476 | 199,26 | 3,80 | 563,09 | 4 | Fit mới |
| RF-08 | rf_v1 | 14 / 3081 | 0,69831073 | 293,64 | 4,54 | 600,58 | 4 | Fit mới |
| RF-09 | rf_v1 | 14 / 3081 | 0,75018403 | 93,99 | 3,16 | 525,96 | 4 | Fit mới |
| RF-10 | rf_v1 | 14 / 3081 | 0,75815930 | 87,35 | 3,12 | 525,48 | 4 | Fit mới |
| RF-11 | rf_v1 | 14 / 3081 | 0,84898815 | 22,85 | 2,88 | 512,46 | 4 | Fit mới |
| RF-12 | rf_v1 | 14 / 3081 | 0,78700427 | 64,63 | 3,18 | 525,23 | 4 | Fit mới |
| RF-13 | rf_v1 | 14 / 3081 | 0,69897908 | 157,99 | 3,17 | 526,48 | 4 | Fit mới |
| RF-14 | rf_v1 | 14 / 3081 | 0,68335842 | 259,72 | 4,07 | 576,92 | 4 | Fit mới |
| RF-15 | rf_v1 | 14 / 3081 | 0,66477108 | 252,65 | 2,79 | 725,25 | 8 | Fit mới |
| RF-16 | rf_v1 | 14 / 3081 | 0,71065806 | 112,41 | 2,40 | 654,45 | 8 | Fit mới |
| RF-17 | rf_v1 | 14 / 3081 | 0,67259607 | 169,08 | 2,45 | 655,28 | 8 | Fit mới |
| RF-18 | rf_v1 | 14 / 3081 | 0,68553357 | 202,62 | 2,84 | 735,30 | 8 | Fit mới |
| RF-19 | rf_v1 | 14 / 3081 | 0,71065806 | 112,41 | 2,40 | 654,45 | 8 | Dùng lại RF-16 |
| RF-20 | rf_v1 | 14 / 3081 | 0,67922078 | 281,08 | 2,98 | 831,57 | 8 | Fit mới |
| RF-21 | rf_v1 | 14 / 3081 | 0,66477108 | 252,65 | 2,79 | 725,25 | 8 | Dùng lại RF-15 |
| RF-22 | rf_v1 | 14 / 3081 | 0,68910358 | 151,03 | 2,66 | 713,55 | 8 | Fit mới |

## FE-01 đến FE-08

Các thử nghiệm giữ siêu tham số thắng vòng Random Forest và thêm nhóm đặc trưng lên rf_v1. FE hợp lệ tốt nhất là **FE-04 trên RF-v1 + SNAP (WRMSSE 0,62393085)**.

| Lượt | Bộ đặc trưng | Logic / cột mã hóa | WRMSSE | Fit (s) | Predict (s) | RAM đỉnh (MiB) | n_jobs | Kết quả |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FE-01 | rf_v1 | 14 / 3081 | 0,66477108 | 252,65 | 2,79 | 725,25 | 8 | Dùng lại RF-15 |
| FE-02 | rf_v1 + recency | 15 / 3082 | 0,66942518 | 223,27 | 2,67 | 725,18 | 8 | Fit mới |
| FE-03 | rf_v1 + prices | 21 / 3088 | 0,65477696 | 260,34 | 2,77 | 782,07 | 8 | Fit mới |
| FE-04 | rf_v1 + snap | 15 / 3082 | 0,62393085 | 225,05 | 2,77 | 736,48 | 8 | Fit mới, hợp lệ v2 |
| FE-05 | rf_v1 + events | 18 / 3103 | 0,6647631394 | 221,13 | 2,68 | 738,43 | 8 | Fit mới, hợp lệ theo hậu kiểm Events |
| FE-06 | rf_v1 + behavior | 17 / 3084 | 0,68256328 | 254,41 | 2,74 | 768,38 | 8 | Fit mới |
| FE-07 | rf_v1 + behavior + snap | 18 / 3085 | 0,63821465 | 268,46 | 2,93 | 790,20 | 8 | Fit mới, hợp lệ v2 |
| FE-08 | rf_v1 + behavior + prices | 24 / 3091 | 0,66909255 | 313,64 | 2,80 | 845,27 | 8 | Fit mới |
| FE-09 | rf_v4 + prices + snap + behavior | 29 / 3096 | 0,6321382731 | 446,49 | 2,97 | 919,76 | 8 | Fit mới, hợp lệ |
| FE-10 | rf_v4 + prices + snap + behavior + events | 33 / 3118 | 0,6360743504 | 389,46 | 2,91 | 923,64 | 8 | Fit mới, hợp lệ theo hậu kiểm Events |

FE-05 và FE-10 dùng chung khối 22 cột Events đã khóa trong manifest đặc trưng, với cùng dữ liệu, forecast origin d_1885 và cửa sổ Validation. Hậu kiểm tái tạo các cột từ quy tắc ngày lễ và loại sự kiện xác định từ lịch đến d_1885. Toàn bộ giá trị khớp trên 600.000 dòng Train và 853.720 dòng Validation. Validation có 0 ngày bật cờ Events. Kết luận hợp lệ áp dụng cho cửa sổ này. Bản ghi selection v2 lưu trạng thái thăm dò của FE-05 tại thời điểm công bố. [Hậu kiểm Events](../results/wrmsse_v1/stage3_all_features/events_origin_audit.json).

## Cấu hình khóa theo Validation

Mô hình được chọn dùng **15 đặc trưng logic**, **3.082 cột sau mã hóa**, fit **225,05 giây**, predict **2,77 giây** và đỉnh RAM worker **736,48 MiB**.

| Siêu tham số | Giá trị |
|---|---:|
| `n_estimators` | `150` |
| `max_depth` | `20` |
| `max_features` | `0.5` |
| `max_samples` | `0.5` |
| `min_samples_leaf` | `5` |
| `bootstrap` | `True` |
| `random_state` | `42` |
| `n_jobs` | `8` |

Điểm trong từng cấp được chuẩn hóa theo trọng số của cấp. Mỗi cấp đóng góp 1/12 điểm trong cấp vào WRMSSE toàn hệ phân cấp.

| Cấp | Số chuỗi | WRMSSE trong cấp | Đóng góp vào tổng |
|---:|---:|---:|---:|
| 1 | 1 | 0,40077153 | 0,03339763 |
| 2 | 3 | 0,46312403 | 0,03859367 |
| 3 | 10 | 0,52667505 | 0,04388959 |
| 4 | 3 | 0,46456841 | 0,03871403 |
| 5 | 7 | 0,55106781 | 0,04592232 |
| 6 | 9 | 0,54151831 | 0,04512653 |
| 7 | 21 | 0,62254824 | 0,05187902 |
| 8 | 30 | 0,61860884 | 0,05155074 |
| 9 | 70 | 0,68521033 | 0,05710086 |
| 10 | 3,049 | 0,87977706 | 0,07331475 |
| 11 | 9,147 | 0,87347831 | 0,07278986 |
| 12 | 30,490 | 0,85982232 | 0,07165186 |

## Bảng, biểu đồ và bằng chứng

- [Bảng tổng hợp CSV](../results/wrmsse_v1/stage3_20261124T082126Z/all_results.csv) và [Parquet](../results/wrmsse_v1/stage3_20261124T082126Z/all_results.parquet).
- [Cấu hình v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_model.json), [manifest v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/manifest.json), [lựa chọn v1](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_model_v1.json) và [danh sách ứng viên v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/candidate_eligibility.csv), [manifest](../results/wrmsse_v1/stage3_20261124T082126Z/manifest_completed.json) và [đối chiếu hậu kỳ](../results/wrmsse_v1/stage3_20261124T082126Z/postflight_verification.json).
- [WRMSSE theo cấp](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_wrmsse_by_level.csv) và [so sánh baseline](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/baseline_comparison.csv).
- [RF-01–RF-05](../results/wrmsse_v1/stage3_20261124T082126Z/initial_wrmsse.png)
- [RF-v1–RF-v4](../results/wrmsse_v1/stage3_20261124T082126Z/ablation_wrmsse.png)
- [RF-06–RF-22](../results/wrmsse_v1/stage3_20261124T082126Z/advanced_wrmsse.png)
- [Biểu đồ FE-01–FE-10](../results/wrmsse_v1/stage4_20261124T104618287922Z/stage3_fe01_fe10.png), [bảng tổng hợp](../results/wrmsse_v1/stage4_20261124T104618287922Z/stage3_fe01_fe10.csv) và [biên bản khóa tối ưu](../results/wrmsse_v1/stage4_20261124T104618287922Z/stage3_optimization_lock.json).
- [Báo cáo mở rộng FE-09 và FE-10](../results/wrmsse_v1/stage3_all_features/REPORT.md).
- [WRMSSE và thời gian fit](../results/wrmsse_v1/stage3_20261124T082126Z/wrmsse_vs_fit_time.png)
- [Mô hình chọn và Seasonal Naive](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/baseline_comparison.png)
- [Theo max_depth](../results/wrmsse_v1/stage3_20261124T082126Z/wrmsse_vs_max_depth.png), [theo max_features](../results/wrmsse_v1/stage3_20261124T082126Z/wrmsse_vs_max_features.png), [theo min_samples_leaf](../results/wrmsse_v1/stage3_20261124T082126Z/wrmsse_vs_min_samples_leaf.png). Các đường chỉ so sánh các lượt có những tham số mô hình khác cố định.
- Request, điểm theo cấp và số liệu tài nguyên tổng hợp nằm trong thư mục từng lượt. Model và dự báo FE-04 cùng dữ liệu cần cho hậu kiểm SNAP nằm trong dự án. Các model, dự báo và log trung gian khác có đường dẫn và checksum trong chỉ mục lưu trữ lịch sử. Các bảng hiển thị thời gian và RAM của lượt fit nguồn khi dùng lại. [Biên bản tinh gọn](PROJECT_CLEANUP_REPORT.md).

## Kiểm tra và giới hạn

Tất cả lượt theo kế hoạch đều có kết quả thành công hoặc dùng lại hợp lệ.

69 kiểm thử đạt trước khi mở tuning. Hậu kiểm đối chiếu 34 bản ghi, 30 artifact fit, checksum của dữ liệu và kết quả, tổng đóng góp theo cấp, và điểm trong CSV, Parquet, JSON. Mẫu Train được tái tạo chính xác và lưới Validation được kiểm tra đầy đủ. Bộ kiểm tra trước và sau batch dùng cùng contract dữ liệu.

Đây là kết quả lựa chọn trên một cửa sổ Validation. Kết luận về SNAP và Events áp dụng cho cửa sổ đã xác minh và quy tắc lịch khóa trước gốc. Batch chuyển từ 4 lên 8 luồng sau RF-14. RF-15 attempt_001 dừng theo yêu cầu và được tiếp tục bằng attempt_002 với 8 luồng. 18 checkpoint giữ nguyên. Thời gian giữa các lượt 4 và 8 luồng phản ánh cả điều kiện thực thi.

Trong Giai đoạn 3, Test d_1914–d_1941 ở trạng thái khóa và quá trình tối ưu chỉ sử dụng Validation. Tuning đã đóng sau batch mở rộng với quyết định RF-15 + FE-04.
