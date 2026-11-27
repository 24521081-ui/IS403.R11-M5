# Giai đoạn 4 · Huấn luyện cuối và đánh giá Test

RF-15 + FE-04 đạt **WRMSSE Test 0,7039124143**. Seasonal Naive đạt **0,8697007023**. Random Forest giảm WRMSSE **19,06%** trên cửa sổ **25/04/2016–22/05/2016**.

## 1. Cấu hình cuối cùng

RF-15 + FE-04 được chọn theo WRMSSE Validation thấp nhất trong các cấu hình đã đánh giá. FE-04 đạt 0,6239308540, FE-09 đạt 0,6321382731 và FE-10 đạt 0,6360743504. Seasonal Naive Validation đạt 0,9227813345. Quá trình tối ưu kết thúc với 36 lượt gồm 32 fit và 4 reuse. [Báo cáo Giai đoạn 3](STAGE3_REPORT.md) · [Cấu hình selection v2](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_model_config.json).

| Siêu tham số | Giá trị |
|---|---:|
| n_estimators | 150 |
| max_depth | 20 |
| max_features | 0.5 |
| max_samples | 0.5 |
| min_samples_leaf | 5 |
| bootstrap | True |
| random_state | 42 |
| n_jobs | 8 |

FE-04 gồm 15 đặc trưng logic theo thứ tự item_id, dept_id, cat_id, store_id, state_id, wday, month, year, lag_28, lag_35, lag_42, lag_56, rolling_mean_7, rolling_mean_28, snap_state. One-hot cho năm trường ID dùng vocabulary đã khóa, xác minh đầy đủ trên mẫu huấn luyện, cho 3.082 cột CSR float32.

## 2. Quy trình huấn luyện lại

Random Forest được khởi tạo mới và fit một lần trên **mẫu 609.185 quan sát lấy từ Train + Validation** trong d_1–d_1913. Giai đoạn này lấy mẫu từ lịch sử gồm hàng triệu quan sát. 600.000 khóa Train gốc được giữ đúng thứ tự và tái tạo chính xác. Mẫu thêm 9.185 khóa từ d_1886–d_1913 theo mật độ 600.000 × 28 / 1.829 và làm tròn đến số nguyên gần nhất. Cả hai mẫu phân tầng theo cửa hàng × bộ phận × quý, lấy không hoàn lại với seed 42. Mỗi tầng nhận số lượng tối thiểu một dòng, phân bổ phần còn lại theo sức chứa và dùng phần dư lớn nhất để khớp tổng mẫu. Đã kiểm tra khóa trùng, phạm vi d và nhãn.

Nguồn doanh số Test feature tối đa là t−28, trong khoảng cutoff d_1886–d_1913. SNAP được dựng từ ngày trong tháng và bang với quy tắc kiểm chứng từ lịch đến d_1885. Lưới 30.490 chuỗi × 28 ngày và hai bộ dự báo được khóa SHA-256 trước khi mở actual Test. Seasonal Naive lặp d_1907–d_1913 cho cả cửa sổ.

## 3. Chi phí tính toán

| Chỉ tiêu | Giá trị |
|---|---:|
| Dòng huấn luyện | 609.185 |
| Đặc trưng logic / cột mã hóa | 15 / 3.082 |
| Thời gian fit (giây) | 224,92 |
| Thời gian predict Random Forest (giây) | 2,34 |
| Thời gian predict Seasonal Naive (giây) | 0,0018 |
| RAM đỉnh worker (MiB) | 1.483,72 |
| RAM đỉnh cây tiến trình worker (MiB) | 1.487,85 |
| RAM khả dụng thấp nhất (MiB) | 3.752,45 |
| CPU vật lý / logic / n_jobs | 8 / 16 / 8 |
| CPU đỉnh cây worker (% của một lõi) | 809,1 |

RSS được lấy mẫu mỗi giây. Phiên bản môi trường gồm python 3.12.10, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, joblib 1.6.0, pyarrow 26.0.1, psutil 7.2.2.

## 4. Kết quả đánh giá Test

Test gồm 853.720 bản ghi item–store–day từ d_1914 đến d_1941. WRMSSE M5 là thước đo chính, với scale từ d_1–d_1913 và trọng số doanh thu lịch sử units × sell_price từ d_1886–d_1913. Cột chênh lệch tính Random Forest trừ Seasonal Naive.

| Chỉ số | Random Forest | Seasonal Naive | Chênh lệch |
|---|---:|---:|---:|
| WRMSSE | 0,7039124143 | 0,8697007023 | -0,1657882880 |
| MAE (đơn vị) | 1,0850 | 1,2440 | -0,1590 |
| RMSE (đơn vị) | 2,2327 | 2,6769 | -0,4442 |
| WAPE (%) | 75,1990 | 86,2181 | -11,0191 |
| Forecast Bias (%) | -4,3279 | -7,3592 | 3,0313 |
| Tổng lượng bán thực tế | 1.231.764 | 1.231.764 | 0 |
| Tổng lượng bán dự báo | 1.178.454,73 | 1.141.116,00 | 37.338,73 |
| Dự báo trừ thực tế (đơn vị) | -53.309,27 | -90.648,00 | 37.338,73 |
| Tỷ lệ dự báo thừa (%) | 0,0000 | 0,0000 | 0,0000 |
| Tỷ lệ dự báo thiếu (%) | 4,3279 | 7,3592 | -3,0313 |

Chênh lệch WRMSSE tuyệt đối là 0,1657882880. Mức cải thiện tương đối tính bằng (WRMSSE Seasonal Naive − WRMSSE Random Forest) / WRMSSE Seasonal Naive × 100%, bằng 19,0627%. Bias âm biểu thị dự báo thiếu. Bias dương biểu thị dự báo thừa. Tỷ lệ thừa và thiếu lần lượt lấy phần dương và phần âm theo độ lớn của Bias. Khi tổng actual bằng 0, WAPE, Bias và tỷ lệ thừa hoặc thiếu được ghi null.

## 5. Đánh giá theo hệ thống phân cấp

| Cấp | Tên cấp | Số chuỗi | Random Forest | Seasonal Naive | RF trừ SN |
|---:|---|---:|---:|---:|---:|
| 1 | Toàn hệ thống | 1 | 0,561465 | 0,751430 | -0,189965 |
| 2 | Bang | 3 | 0,580430 | 0,754396 | -0,173965 |
| 3 | Cửa hàng | 10 | 0,652730 | 0,799304 | -0,146573 |
| 4 | Ngành hàng | 3 | 0,601073 | 0,762047 | -0,160974 |
| 5 | Bộ phận | 7 | 0,657674 | 0,817086 | -0,159412 |
| 6 | Bang × ngành hàng | 9 | 0,635997 | 0,782983 | -0,146985 |
| 7 | Bang × bộ phận | 21 | 0,684688 | 0,830436 | -0,145748 |
| 8 | Cửa hàng × ngành hàng | 30 | 0,701347 | 0,839263 | -0,137916 |
| 9 | Cửa hàng × bộ phận | 70 | 0,750878 | 0,891855 | -0,140977 |
| 10 | Sản phẩm | 3.049 | 0,887196 | 1,032248 | -0,145052 |
| 11 | Sản phẩm × bang | 9.147 | 0,872752 | 1,074959 | -0,202207 |
| 12 | Sản phẩm × cửa hàng | 30.490 | 0,860718 | 1,100402 | -0,239684 |

Random Forest có WRMSSE thấp hơn tại 12 cấp, cao hơn tại 0 cấp và bằng nhau tại 0 cấp. Chênh lệch theo hướng có lợi cho Random Forest lớn nhất ở cấp 12 (Sản phẩm × cửa hàng), với Seasonal Naive trừ Random Forest bằng 0,239684. Cấp 8 (Cửa hàng × ngành hàng) có mức chênh lệch theo hướng đó thấp nhất, bằng 0,137916. Mỗi cấp có trọng số 1/12. Tổng đóng góp của 42.840 chuỗi khớp WRMSSE toàn hệ thống ở cả hai mô hình.

## 6. Feature importance

| Đặc trưng logic | Tỷ trọng (%) |
|---|---:|
| rolling_mean_28 | 40,9543 |
| rolling_mean_7 | 24,6255 |
| lag_28 | 9,2371 |
| lag_35 | 7,5212 |
| lag_42 | 4,5409 |
| lag_56 | 4,5245 |
| wday | 1,5986 |
| month | 1,5811 |
| item_id | 1,4741 |
| store_id | 1,0278 |
| year | 1,0179 |
| state_id | 0,6784 |
| dept_id | 0,5278 |
| snap_state | 0,4493 |
| cat_id | 0,2415 |

Importance trích từ mức giảm impurity trong model đã fit và cộng các cột one-hot theo đặc trưng logic gốc. Tổng sau gộp bằng 100% trước khi làm tròn. SNAP chiếm 0,45%, các lag chiếm 25,82% và hai rolling mean chiếm 65,58%. Các tỷ trọng mô tả cách rừng dùng đặc trưng để chia nút. Đặc trưng tương quan có thể chia sẻ importance, và các tỷ trọng này không xác lập quan hệ nhân quả.

## 7. Nhận xét kết quả

Trên tổng lượng bán Test, Random Forest dự báo thiếu với Bias -4,33%, còn Seasonal Naive dự báo thiếu với Bias -7,36%. Bảng 12 cấp cho thấy mức chênh lệch thay đổi theo độ chi tiết của chuỗi. Biểu đồ doanh số theo ngày giúp đối chiếu độ lệch tổng lượng với các ngày trong cả cửa sổ.

So sánh Validation và Test dùng hai context scale và trọng số khác nhau. Điểm Validation dùng mô hình chọn đã fit trên mẫu Train, còn điểm Test dùng mô hình cuối đã fit trên mẫu Train + Validation. MAE, RMSE, WAPE và Bias bổ sung góc nhìn sai số ở cấp item–store–day. Quyết định mô hình được khóa theo WRMSSE Validation.

## 8. Giới hạn nghiên cứu

- Mô hình dùng mẫu Train + Validation từ lịch sử, gồm 609.185 dòng.
- Tối ưu dựa trên một cửa sổ Validation 28 ngày.
- Test d_1914–d_1941 từng được dùng làm Validation trong nghiên cứu cũ. Actual Test chỉ được mở sau khi khóa mô hình và cả hai bộ dự báo trong pipeline WRMSSE mới. Cửa sổ này đã từng được quan sát trong lịch sử dự án.
- Kết quả phản ánh phạm vi dữ liệu, cách lấy mẫu và điều kiện thực nghiệm hiện tại.

## 9. Artifact

- [Mô hình](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_model.joblib) · [cấu hình](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_model_config.json) · [schema mã hóa](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_feature_metadata.json).
- [Mẫu huấn luyện](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_training_sample.parquet) · [tầng lấy mẫu](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_training_strata.parquet) · [manifest mẫu](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_training_sample_manifest.json).
- [Ma trận Test](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_feature_matrix.npz) · [khóa ma trận](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_feature_keys.parquet) · [Random Forest](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_predictions.parquet) · [Seasonal Naive](../results/wrmsse_v1/stage4_20261124T104618287922Z/seasonal_naive_predictions.parquet).
- [Actual và hai dự báo cho Giai đoạn 5](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_actual_and_forecast.parquet) · [metrics JSON](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_metrics.json) · [so sánh CSV](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_model_comparison.csv).
- [WRMSSE theo cấp](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse_by_level.csv) · [RMSSE, scale, trọng số và đóng góp từng chuỗi](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse_by_series.parquet) · [importance CSV](../results/wrmsse_v1/stage4_20261124T104618287922Z/feature_importance.csv).
- [Kiểm tra trước fit](../results/wrmsse_v1/stage4_20261124T104618287922Z/preflight.json) · [giám sát tài nguyên](../results/wrmsse_v1/stage4_20261124T104618287922Z/resource_summary.json) · [khóa dự báo](../results/wrmsse_v1/stage4_20261124T104618287922Z/forecast_lock_manifest.json) · [biên bản mở actual](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_access.json).
- [Manifest hoàn thành](../results/wrmsse_v1/stage4_20261124T104618287922Z/manifest_completed.json) · [hậu kiểm](../results/wrmsse_v1/stage4_20261124T104618287922Z/postflight_verification.json).

- [WRMSSE tổng thể trên Test](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse.png)
- [WRMSSE theo 12 cấp phân cấp](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_wrmsse_by_level.png)
- [Tổng doanh số theo ngày Test](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_daily_sales.png)
- [WRMSSE trên Validation và Test](../results/wrmsse_v1/stage4_20261124T104618287922Z/validation_vs_test_wrmsse.png)
- [MAE, RMSE, WAPE và Forecast Bias](../results/wrmsse_v1/stage4_20261124T104618287922Z/test_business_metrics.png)
- [Feature importance của mô hình cuối](../results/wrmsse_v1/stage4_20261124T104618287922Z/final_feature_importance.png)

## 10. Xác nhận hoàn thành

Giai đoạn 4 hoàn thành với 71 kiểm thử đạt. Mô hình RF-15 + FE-04 và hai bộ dự báo được khóa trước khi đánh giá toàn bộ Test bằng WRMSSE. Training và Test đã đóng. Không có tuning hoặc refit sau Test.
