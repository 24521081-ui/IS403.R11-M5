# Phân tích Seasonal Naïve theo ngành hàng và tần suất bán

## Phạm vi

Phân tích này chỉ dùng kết quả dự báo Seasonal Naïve đã lưu, không huấn luyện và không chạy suy luận lại. Mỗi dòng ở cấp phân tích là một chuỗi `item_id × store_id`; đây không phải WRMSSE đầy đủ 12 cấp.

Nguồn đầu vào:

- `data/raw/sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`.
- `reports/demand_groups/demand_groups_origin_1885.csv`, bảng nhãn chung lấy từ repo `IS403.R11-M5-main`.
- `results/benchmarks/metrics/seasonal_naive_wrmsse/{validation,test}/rmsse_va_trong_so_cap_san_pham_cua_hang.csv`.
- `results/benchmarks/metrics/seasonal_naive_wrmsse/{validation,test}/diem_theo_12_cap.csv` và `diem_chung.csv` để kiểm tra đối chiếu.

## Phương pháp

Nhãn ngành hàng là `cat_id`: `FOODS`, `HOBBIES`, `HOUSEHOLD`. Nhãn tần suất là `demand_group` từ ADI với ngưỡng `ADI <= 1.32` cho `frequent`; các chuỗi còn lại là `intermittent`. Bảng nhãn được kiểm tra có đúng 30.490 cặp duy nhất.

Với mỗi nhóm `G`, các chỉ số được tính như sau:

- `n_series`: số chuỗi trong nhóm.
- `revenue_share`: tổng `revenue_weight` của nhóm; tổng sáu nhóm bằng 1.
- `weighted_rmsse = SUM(revenue_weight × RMSSE) / SUM(revenue_weight)`.
- `unweighted_mean_rmsse`: RMSSE trung bình, chỉ dùng tham khảo.
- `actual_units`, `predicted_units`: tổng lượng thực tế và dự báo trong 28 ngày đánh giá.
- `unit_difference = predicted_units - actual_units` và `forecast_bias_pct = unit_difference / actual_units × 100`.

Trọng số doanh thu và RMSSE từng chuỗi được lấy trực tiếp từ bộ chấm Seasonal Naïve đã lưu. Do đó kết quả không thay đổi dự báo hoặc mã chấm cũ.

## Kết quả đối chiếu

| Kỳ | Chuỗi | Tổng trọng số cấp sản phẩm–cửa hàng | WRMSSE 12 cấp đã lưu | Tham chiếu |
|---|---:|---:|---:|---:|
| Validation (origin 1885) | 30.490 | 1,000000000000 | 0,922781334536 | 0,922781334536 |
| Test (origin 1913) | 30.490 | 1,000000000000 | 0,869700702331 | 0,869700702331 |

WRMSSE 12 cấp là trung bình của 12 điểm cấp trong `diem_theo_12_cap.csv`. RMSSE có trọng số ở cấp sản phẩm–cửa hàng được báo riêng trong các CSV phân tích và không được gọi là WRMSSE chung.

## Diễn giải

Seasonal Naïve có ưu điểm là đơn giản, tái lập, không cần huấn luyện và tận dụng mùa vụ tuần 7 ngày. Điểm yếu là mô hình lặp lại mẫu tuần trước nên khó phản ứng với thay đổi xu hướng, giá, khuyến mại và các chuỗi bán gián đoạn. Vì vậy bảng giao giữa ngành hàng và tần suất giúp chỉ ra nhóm mà sai lệch tổng lượng hoặc RMSSE cao, thay vì chỉ nhìn một điểm chung toàn hệ thống.

## Chạy lại

Đứng tại thư mục gốc `IS403.R11-M5` và dùng môi trường đã cài:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_seasonal_naive_groups.py
```

Mặc định script đọc kết quả đã lưu và ghi vào `results/benchmarks/metrics/seasonal_naive_wrmsse/group_analysis/`. Script từ chối chạy nếu thư mục đầu ra đã có tệp để tránh ghi đè. Có thể truyền `--metrics-dir`, `--labels` và `--output-dir` khi dùng đường dẫn khác.