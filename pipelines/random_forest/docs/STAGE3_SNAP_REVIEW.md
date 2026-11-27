# Hậu kiểm SNAP và lựa chọn Random Forest giai đoạn 3

**FE-04 đủ điều kiện và được chọn với WRMSSE Validation 0,6239308540.** FE-07 cũng đủ điều kiện. Cả hai dùng lịch SNAP có thể tái tạo bằng dữ liệu lịch đã biết đến gốc dự báo d_1885. Events của FE-05 chưa được xác minh riêng nên vẫn chỉ dùng để tham khảo.

## Ý nghĩa và tính biết trước của SNAP

Mục Dataset trong M5 Competitors Guide định nghĩa `snap_CA`, `snap_TX`, `snap_WI` là cờ 0/1 về khả năng mua bằng SNAP theo bang và ngày. Chúng thuộc `calendar.csv`. Số đơn vị bán thực tế được cung cấp ở bảng sales riêng. USDA cũng mô tả lịch cấp quyền lợi theo ngày cố định trong tháng cho từng nhóm hộ. Lịch cấp quyền lợi ngoài thực tế có thể phụ thuộc nhóm hộ và thay đổi theo chính sách. Cuộc kiểm tra này dùng đúng cờ M5. [M5 Competitors Guide, mục Dataset](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf) · [USDA ERS SNAP Distribution Schedule Database](https://www.ers.usda.gov/data-products/snap-policy-data-sets)

Tại gốc d_1885 (27/03/2016), bảng tra chỉ dùng `calendar.csv` từ d_1 đến d_1885. Với từng bang và ngày trong tháng, tất cả dòng lịch trước gốc cùng mang một giá trị. Áp dụng bảng tra đó vào ngày đích d_1886–d_1913 (28/03–24/04/2016) tái tạo **84/84** cờ. Phép dựng bảng tra chỉ dùng ngày và cờ lịch trước gốc. Việc đối chiếu với lịch Validation sau khi áp dụng quy tắc cho thấy đúng cột đã cung cấp cho mô hình.

| Bang | Ngày trong tháng có cờ SNAP bằng 1 trong dữ liệu M5 |
|---|---|
| CA | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 |
| TX | 1, 3, 5, 6, 7, 9, 11, 12, 13, 15 |
| WI | 2, 3, 5, 6, 8, 9, 11, 12, 14, 15 |

CSV M5 không có ngày công bố hoặc lịch sử phiên bản. Bằng chứng ở đây xác lập tính tái tạo trước gốc trong cửa sổ đã kiểm tra. Khi áp dụng ở giai đoạn 4, dùng quy tắc đã khóa hoặc lịch SNAP đã biết tại gốc dự báo và ghi rõ nguồn. Nếu có thay đổi lịch thực tế, cần kiểm tra tính sẵn có tại gốc.

## Cột đặc trưng và căn chỉnh dữ liệu

`snap_features` tra `series_id → state_id` rồi lấy `calendar[target_day, snap_<state>]`. FE-04 nối cột `snap_state` vào RF-v1. FE-07 nối Behavior và cùng cột SNAP. Ngày được xác định từ `2011-01-29 + (d−1)` ngày. Các khóa theo series × ngày và tiền tố bang của cửa hàng đều khớp CA/TX/WI. [Mã snap_features](../src/wrmsse_extensions.py#L16) · [Mã tạo và lưu ma trận](../src/wrmsse_stage2_data.py#L162)

- Đã đối chiếu **600.000 dòng Train** và **853.720 dòng Validation** với cả bảng lịch và quy tắc trước gốc. Số sai khác là 0.
- Ba cột Behavior của FE-07 được tính lại trên toàn bộ dòng, tổng cộng **4.361.160 giá trị**. Cửa sổ doanh số kết thúc ở t−28. Ngày mới nhất là d_1857 với mẫu Train và d_1885 với Validation. Cột `snap_state` dùng ngày đích vì lịch là đầu vào biết trước.
- Đã đối chiếu SHA-256 của model, dự báo, manifest và dữ liệu. Model lưu sẵn tái tạo dự báo Validation với sai khác tuyệt đối tối đa 2,84 × 10⁻¹⁴. Điểm WRMSSE tính lại trên 42.840 chuỗi thuộc 12 cấp khớp điểm gốc của cả hai ứng viên.

## Kết quả selection v2

| Ứng viên | WRMSSE Validation | Quyết định |
|---|---:|---|
| Seasonal Naive | 0,9227813345 | Baseline từ lịch sử đến d_1885 |
| FE-03 | 0,6547769630 | Lựa chọn v1 được lưu vào lịch sử |
| FE-04 | 0,6239308540 | Chọn chính thức |
| FE-07 | 0,6382146523 | Đủ điều kiện |

Sau hậu kiểm, **33/34** bản ghi đủ điều kiện. Mức WRMSSE thấp nhất là **RF-15 + FE-04**, dùng RF-v1 + `snap_state`, 15 đặc trưng logic, 3.082 cột, 150 cây, `max_depth=20`, `max_features=0.5`, `max_samples=0.5`, `min_samples_leaf=5`, bootstrap, seed 42 và `n_jobs=8`. Mô hình giảm WRMSSE **0,2988504806** so với Seasonal Naive, tương đương **32,39%**.

Bản v2 có manifest riêng và giữ bản chọn v1. Cấu hình được khóa cho giai đoạn 4 với chính sách SNAP nêu trên. Giai đoạn 4 chưa bắt đầu. Không có lần fit hoặc tuning mới và Test d_1914–d_1941 vẫn khóa. Các tệp kết quả gốc, dữ liệu raw, legacy và dự án nguồn được giữ nguyên.

- [Selected model v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_model.json) và [bản v1 lưu lịch sử](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_model_v1.json).
- [Danh sách ứng viên v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/candidate_eligibility.csv), [WRMSSE theo 12 cấp](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_wrmsse_by_level.csv) và [manifest v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/manifest.json).
- [Dữ liệu hậu kiểm](../preparation/stage3_snap_audit.json) và [báo cáo giai đoạn 3 đã cập nhật](STAGE3_REPORT.md).
