# Giai đoạn 2 — Data pipeline, Feature Engineering và smoke test

Hoàn thành ngày 24/11/2026, múi giờ Asia/Bangkok. Pipeline đã chạy thành công từ dữ liệu raw qua xử lý, tạo mẫu/feature, fit RF-03 và WRMSSE Validation. Chỉ thực hiện một lần fit smoke; không tuning hàng loạt, không huấn luyện mô hình cuối và không mở hoặc chấm actual Test.

## Dữ liệu và kiểm thử

Train d_1–d_1885; Validation d_1886–d_1913; Test d_1914–d_1941 được khai báo nhưng chưa materialize nhãn. Horizon 28; 30.490 chuỗi bottom-level, 42.840 chuỗi phân cấp khi đánh giá.

- Train đầy đủ: 57.473.650 quan sát; vùng đủ lag từ d_57 có 55.766.210 quan sát.
- Mẫu Train: **600.000** dòng mới, seed 42, không hoàn lại, phân tầng **store × dept × quý** với 1.470 tầng. Giữ cách phân bổ một dòng/tầng rồi theo năng lực còn lại và largest remainder của thiết kế cũ.
- Validation: **853.720** dòng mới, đủ khóa series × 28 ngày, không trùng với Train.
- Dữ liệu mới dùng NPY/Parquet; ma trận feature float32 CSR và nhãn/khóa tách riêng.
- 43 kiểm thử giai đoạn 1 đã chạy lại và đạt. 58 bài đạt trước smoke; kiểm tra bổ sung cuối cùng đạt **63/63**, không có lỗi hoặc bỏ qua.
- Kiểm tra lại toàn bộ mẫu 600.000 dòng bằng seed 42 cho kết quả khớp tuyệt đối; các manifest mới lưu SHA-256 và contract ID.
- Test bị chặn trước mọi truy cập actual. Training chỉ được mở trong scope có bằng chứng kiểm thử; sau smoke đã đóng lại. Policy tuning vẫn tắt.
- Các kiểm thử bổ sung bao phủ cutoff, cửa sổ ngắn đầu kỳ, recency khi chưa từng bán, giá chỉ dùng tuần hoàn tất, encoding thống nhất, khóa dự báo, loại smoke khỏi selection và thứ tự workflow.

## Các bộ đặc trưng

| Feature set | Logic | Encoded |
|---|---:|---:|
| rf_v1 | 14 | 3081 |
| rf_v2 | 16 | 3083 |
| rf_v3 | 17 | 3084 |
| rf_v4 | 18 | 3085 |
| FE-01 | 17 | 3084 |
| FE-02 | 18 | 3085 |
| FE-03 | 24 | 3091 |
| FE-04 | 18 | 3085 |
| FE-05 | 21 | 3106 |
| FE-06 | 20 | 3087 |

RF-v1 có 5 định danh one-hot, wday/month/year, lag 28/35/42/56, rolling mean 7/28. RF-v2 thêm zero_ratio_28/56; RF-v3 thêm nonzero_count_28; RF-v4 thêm days_since_last_sale.

Các định nghĩa được đối chiếu source và notebook 03/05/08 trong legacy ở chế độ chỉ đọc. Số cột được suy ra từ metadata và vocabulary Train, không hardcode theo artifact cũ. Cùng encoder metadata Train được dùng cho mọi split.

Feature doanh số chỉ đọc tới t−28, không vượt d_1885 khi dự báo Validation. Cửa sổ 56 ngày đầu kỳ dùng đúng số ngày đã quan sát, không thêm ngày giả bằng 0. Behavior dùng std ddof=0 và chênh lệch hai cửa sổ không chồng lấp. Recency tính tới cutoff; nếu chưa bán thì dùng số ngày đã quan sát.

Historical prices dùng các tuần có ngày kết thúc ≤ cutoff; không trừ số học mã tuần. Median điền feature giá chỉ học từ Train và giữ ba cờ thiếu. Ngày nguồn giá lớn nhất: d_1855 ở Train, d_1883 ở Validation. SNAP được ghép theo bang. Events chỉ giữ các ngày lễ dân sự có quy tắc lịch xác định; vocabulary chỉ học trên Train.

Đã tạo thêm hai template chưa chấm điểm: behavior + SNAP (21 logic/3.088 cột) và behavior + prices (27 logic/3.094 cột). Không gắn chúng làm lựa chọn FE-07/08 mặc định. FE-07/08 mới sẽ được lập thích nghi từ WRMSSE của các nhóm đơn mới.

## Kết quả Validation mới

| Mô hình | WRMSSE 12 cấp |
|---|---:|
| Seasonal Naive, chu kỳ 7 | 0,9227813345358948 |
| RF-03 smoke, RF-v1 | 0,7459536083918473 |

Seasonal Naive lặp lại đúng 7 ngày d_1879–d_1885 cho toàn bộ 28 ngày dự báo; không cập nhật bằng actual Validation.

RF-03: 100 cây, max_depth=12, max_features=0.3, max_samples=0.5, min_samples_leaf=20, bootstrap=True, random_state=42, n_jobs=4. Fit trên 600.000 dòng/3.081 cột; predict đủ 853.720 dòng. Fit **68,6258 giây**; predict **2,2932 giây**; đánh giá **0,0855 giây**. RAM high-water của worker: **1.557,42 MiB**; RSS cây tiến trình lấy mẫu mỗi 2 giây đạt **1.442,20 MiB**. Hai cách đo có tần suất/phạm vi khác nhau; bộ giám sát không chạm ngưỡng dừng.

Dự báo được lưu và băm trước khi chấm WRMSSE, có bản Parquet chứa series_id + ngày để kiểm chứng khóa. Điểm của từng cấp dưới đây được chuẩn hóa trọng số trong cấp; tổng WRMSSE là trung bình 12 điểm này.

| Level | Series | Seasonal Naive | RF-03 smoke |
|---|---:|---:|---:|
| 1 | 1 | 0.66767900 | 0.62173360 |
| 2 | 3 | 0.72448361 | 0.65169605 |
| 3 | 10 | 0.82518182 | 0.67604735 |
| 4 | 3 | 0.75425320 | 0.65210316 |
| 5 | 7 | 0.97382376 | 0.72062457 |
| 6 | 9 | 0.83472386 | 0.69747153 |
| 7 | 21 | 0.98439012 | 0.76017209 |
| 8 | 30 | 0.92051311 | 0.73702499 |
| 9 | 70 | 1.02895551 | 0.78723249 |
| 10 | 3049 | 1.11427001 | 0.90223237 |
| 11 | 9147 | 1.11921401 | 0.88414989 |
| 12 | 30490 | 1.12588800 | 0.86095521 |

Smoke có purpose=smoke và selection_eligible=false. Không tạo selected_model từ kết quả này và không tuyên bố RF-03 là cấu hình tối ưu.

## Workflow đã chuẩn bị

Entry point tương lai: src.wrmsse_workflows.run_ordered_tuning. Thứ tự được kiểm tra:

1. RF-01–RF-05 với RF-v1.
2. RF-03 so sánh RF-v1–RF-v4.
3. RF-06 với RF-v3, rồi RF-07–RF-13 đơn yếu tố.
4. RF-14–RF-18 kết hợp các giá trị thắng theo Validation mới.
5. RF-19–RF-22 với RF-v3.
6. FE-01–FE-06 với hyperparameter được lựa chọn lại từ kết quả mới.
7. FE-07/08 từ các nhóm đơn có cải thiện mới.
8. Chọn WRMSSE Validation thấp nhất trong các record tuning hợp lệ.

Có lưu kế hoạch, hyperparameter, thời gian, RAM, prediction/model, bảng điểm theo cấp/từng chuỗi, CSV so sánh, biểu đồ và manifest. Kiểm tra nguồn matrix bằng hash trước khi dùng. Không nhập mô hình/score/prediction hoặc processed/features từ legacy.

configs/stage3.json giữ allow_tuning=false và allow_test_evaluation=false. Scope tuning ở giai đoạn sau yêu cầu policy được cho phép rõ ràng và bằng chứng kiểm thử còn khớp mã. Policy thực thi tách khỏi contract dữ liệu để mở tuning sau này không làm mất hiệu lực dữ liệu mới đã chuẩn bị. Chưa chạy workflow hàng loạt.

## File mới và chỉnh sửa

Mới: configs/stage2.json, configs/stage3.json, src/wrmsse_stage2_data.py, src/wrmsse_extensions.py, src/wrmsse_workflows.py, scripts/run_stage2.py, tests/test_wrmsse_stage2.py, requirements-stage2-lock.txt, báo cáo này và các thư mục artifact có run riêng.

Chỉnh sửa: src/wrmsse_config.py (scope có bằng chứng), src/wrmsse_pipeline.py (loại smoke khỏi selection), scripts/run_preparation_checks.py (lưu contract/hash mã), README.md (liên kết giai đoạn 2).

Đường dẫn run: results/wrmsse_v1/stage2_20261123T202104Z. Processed/features và các template có namespace/run riêng. Xem summary.json, manifest_completed.json, comparison_by_level.csv, feature_counts.csv và resource_summary.json trong run. Biên bản cuối và kiểm thử nằm trong preparation/.

## Tính toàn vẹn và giới hạn

Dự án nguồn và toàn bộ legacy giữ nguyên tên tệp, kích thước và thời gian sửa trước/sau run. Không sửa dữ liệu raw. Snapshot mã/config đúng thời điểm fit được giữ trong source_snapshot; các kiểm tra/phần điều phối được hoàn thiện sau smoke có hash riêng ở biên bản postflight. Không fit lại mô hình sau các sửa đó.

Tất cả tài liệu mới được ghi bằng UTF-8 tường minh. Giữ các biên bản giai đoạn 1 làm lịch sử.

Chưa đo hiệu quả RF-v2–v4, các nhóm FE hoặc tổ hợp qua huấn luyện; mới kiểm tra feature và tạo ma trận. Tuning hàng loạt được chuẩn bị nhưng chưa chạy end-to-end toàn bộ nhánh. SNAP có giả định lịch được công bố trước origin cần kiểm chứng khi triển khai. Test mới từng là Validation của nghiên cứu cũ nên không phải cửa sổ chưa từng được quan sát trong lịch sử. Các giới hạn này cần giữ trong báo cáo nghiên cứu tiếp theo.

Manifest completed hashes all artifacts after the worker and resource monitor have closed their logs. The earlier worker manifest is a checkpoint; its live worker.log/resources.jsonl entries are superseded by manifest_completed.json. No model was rerun.
