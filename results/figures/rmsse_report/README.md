# Biểu đồ RMSSE — Validation Full 10 cửa hàng

## Phạm vi và nguồn

Nguồn tổng hợp mới nhất đã được kiểm chứng: `results\aggregated\validation_full_10_stores_20261111T151420617263Z`.
Validation d_1886–d_1913; 30.490 chuỗi sản phẩm–cửa hàng, 853.720 dự báo/mô hình.
Không huấn luyện lại, không nạp mô hình, không chạy Test cuối, không đọc dữ liệu
gốc trong data/ hoặc dataset/. Chỉ đọc CSV kết quả và JSON biên bản kiểm chứng.
Không sử dụng CSV demo hoặc dữ liệu giả lập.

- `metrics_overall.csv`: điểm RMSSE tổng hợp, giữ đầy đủ độ chính xác khi vẽ.
- `metrics_by_store.csv`: điểm RMSSE của hai mô hình ở mỗi cửa hàng, tên lần chạy nguồn.
- `metrics_per_series.csv`: RMSSE, scale Train và tổng bình phương sai số của 28 ngày.
- `summary_crosscheck.csv`: đối chiếu chỉ số tính lại với summary nguồn.
- `integrity_check.json`: biên bản kiểm chứng của bản tổng hợp, số tệp cũ thay đổi bằng 0.

Các tên CSV trên đều nằm trong thư mục nguồn tổng hợp đã nêu. Đã đối chiếu lại
điểm tổng hợp và từng cửa hàng bằng trung bình RMSSE trong metrics_per_series.csv;
đồng thời kiểm tra `sqrt(squared_error_sum / 28 / scale)` cho từng chuỗi hợp lệ.

## 5.6 – Kết quả thực nghiệm

Chèn hai hình sau (có bản PDF cùng tên):

1. **rmsse_comparison_overall.png**: so sánh trên toàn bộ 30.490 chuỗi.
   LightGBM–Tweedie = 0.748373; Seasonal Naive =
   0.975596. Mức giảm sai số = 23.29%, khoảng 23,29%,
   tính bằng `100 * (1 - điểm_LightGBM / điểm_Seasonal_Naive)`.
2. **rmsse_comparison_by_store.png**: mỗi cửa hàng có hai cột, mỗi cột là trung bình
   RMSSE 28 ngày của 3.049 chuỗi. Nhãn cột hiển thị ba chữ số thập phân.

Chỉ số dùng là **RMSSE_mean_unweighted**. Càng thấp càng tốt trên cùng tập đánh giá.
Điểm tổng hợp là trung bình các chuỗi hợp lệ, không phải căn của trung bình tỷ số
MSE/scale. MAE và RMSE vẫn là chỉ số bổ sung trong kết quả cũ, được giữ nguyên.
Đây không phải phần trăm sai số hoặc độ chính xác; giảm 23,29% không có nghĩa
độ chính xác tăng 23,29 điểm phần trăm.

**RMSSE trung bình không trọng số, không phải WRMSSE chính thức.**
Không đánh giá đủ 12 cấp phân cấp và trọng số doanh thu của WRMSSE M5.

## 5.7 – Trực quan hóa kết quả dự báo: chưa đủ dữ liệu horizon

**Không tạo rmsse_by_horizon.png hoặc rmsse_by_horizon.pdf**, vì thiếu nhãn thật
theo từng chuỗi và từng ngày trong CSV kết quả Validation Full đã lưu.
Đã kiểm tra header của 20 CSV dự báo nguồn bên dưới: chúng có y_pred nhưng
không có y_true/actual. Bảng metrics_per_series.csv có scale và tổng sai số
28 ngày; không thể phân rã duy nhất tổng này thành sai số từng horizon.
comparison.csv trong kết quả demo một sản phẩm không đủ cho toàn bộ 30.490 chuỗi.

Công thức cần dùng khi có đủ dữ liệu:

```text
V = các chuỗi có scale Train hữu hạn và > 0
mean_scaled_absolute_error_by_horizon(h)
    = mean_i_in_V(sqrt((y_true[i,h] - y_pred[i,h])^2 / scale[i]))
    = mean_i_in_V(abs(y_true[i,h] - y_pred[i,h]) / sqrt(scale[i]))
```

Đây là **RMSSE một bước theo horizon, trung bình trên các chuỗi**. Nó khác
RMSSE 28 ngày của một chuỗi: `sqrt(mean_h(error[i,h]^2) / scale[i])`.
Trung bình 28 giá trị trên đường horizon không đồng nhất với RMSSE_mean_unweighted
28 ngày. Không chia MAE cho hằng số, không lấy điểm tổng hợp làm đường phẳng.

Dữ liệu còn thiếu là CSV kết quả chứa `store_id, item_id, target_day/horizon,
y_true` cho đủ 30.490 × 28 điểm. Có thể nối CSV đó với y_pred và scale đã lưu;
không cần huấn luyện lại. Báo cáo này không đọc sales gốc để bổ sung nhãn,
đúng giới hạn chỉ đọc CSV kết quả của yêu cầu hiện tại.

Khi đủ nhãn, **rmsse_by_horizon.png** phù hợp mục **5.7**. Hiện tại có thể dùng
biểu đồ dự báo–thực tế đã có cho một sản phẩm để minh họa tại 5.7, kèm phạm vi
một chuỗi; không gọi đó là biểu đồ RMSSE horizon của 10 cửa hàng.

## Danh sách CSV dự báo đã kiểm tra schema

- `results\validation\validation_full_CA_1_20261106T152044652294Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_CA_1_20261106T152044652294Z\seasonal_naive.csv`
- `results\validation\validation_full_CA_2_20261106T152128571471Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_CA_2_20261106T152128571471Z\seasonal_naive.csv`
- `results\validation\validation_full_CA_3_20261106T152215498574Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_CA_3_20261106T152215498574Z\seasonal_naive.csv`
- `results\validation\validation_full_CA_4_20261106T152304815096Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_CA_4_20261106T152304815096Z\seasonal_naive.csv`
- `results\validation\validation_full_TX_1_20261106T152349831355Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_TX_1_20261106T152349831355Z\seasonal_naive.csv`
- `results\validation\validation_full_TX_2_20261106T152435185415Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_TX_2_20261106T152435185415Z\seasonal_naive.csv`
- `results\validation\validation_full_TX_3_20261106T152523230095Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_TX_3_20261106T152523230095Z\seasonal_naive.csv`
- `results\validation\validation_full_WI_1_20261106T152616359231Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_WI_1_20261106T152616359231Z\seasonal_naive.csv`
- `results\validation\validation_full_WI_2_20261106T152702387404Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_WI_2_20261106T152702387404Z\seasonal_naive.csv`
- `results\validation\validation_full_WI_3_20261106T152749597725Z\lightgbm_tweedie.csv`
- `results\validation\validation_full_WI_3_20261106T152749597725Z\seasonal_naive.csv`

## Định dạng và bảo toàn

PNG 300 dpi, PDF dạng vector; tiêu đề, trục và ghi chú tiếng Việt.
Chèn PNG vào Google Docs; PDF phù hợp lưu trữ/in ấn. Điểm thấp hơn là tốt hơn.
`verification.json` ghi dấu SHA256 của nguồn, kiểm tra nguồn không thay đổi,
schema dự báo và lý do chưa tạo biểu đồ horizon. Mọi tệp cũ được giữ nguyên.
Script tái tạo: `scripts/plot_saved_rmsse_report.py`. Khi thư mục rmsse_report
đã tồn tại, lần chạy sau tạo thư mục con theo thời điểm UTC để tránh ghi đè.
