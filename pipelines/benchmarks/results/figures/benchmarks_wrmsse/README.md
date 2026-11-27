# Bộ biểu đồ trực quan WRMSSE cho ES_bu và Croston (M5 Forecasting)

Thư mục này chứa bộ biểu đồ phân tích thực nghiệm cho các mô hình thống kê đối chứng (**ES_bu**, **Croston**, **Croston-SBA**) theo tiêu chuẩn **WRMSSE 12 cấp M5**, được tạo bởi `scripts/benchmarks/plot_wrmsse_benchmarks.py`.

| STT | Tệp biểu đồ | Mục đích & Nội dung chính |
|---:|---|---|
| 1 | `01_thuc_te_vs_du_bao_test_28_ngay.png` | So sánh chuỗi thời gian tổng lượng bán **Thực tế vs Dự báo 28 ngày Test** (`d_1914–d_1941`) của `ES_bu`, `Croston`, `Croston-SBA` kèm WRMSSE và % sai lệch tổng lượng. |
| 2 | `02_wrmsse_validation_vs_test.png` | So sánh **WRMSSE 12 cấp** giữa hai kỳ **Validation** (`d_1886–d_1913`) và **Test** (`d_1914–d_1941`) của 4 mô hình thống kê. |
| 3 | `03_diem_12_cap_es_bu_vs_croston_test.png` | Phân rã điểm **weighted_RMSSE trên đủ 12 cấp phân cấp M5** (kỳ Test), lý giải vì sao `Croston` yếu ở các cấp tổng hợp (1–9) còn `ES_bu` ổn định hơn. |
| 4 | `04_sai_lech_test_theo_cua_hang.png` | Sai lệch tổng lượng Test (%) theo từng cửa hàng trong **10 cửa hàng Walmart** (`CA_1`–`WI_3`). |
| 5 | `05_phan_tich_nganh_hang_va_gian_doan.png` | Phân tích hiệu năng `ES_bu` vs `Croston` / `Croston-SBA` theo **3 ngành hàng** (`FOODS`, `HOBBIES`, `HOUSEHOLD`) và **4 nhóm mức độ bán gián đoạn**. |
| 6 | `06_xep_hang_wrmsse_toan_du_an.png` | Bảng xếp hạng đối chiếu giữa **WRMSSE 12 cấp** (tiêu chí chính) và **RMSSE trung bình cấp 12** (không trọng số) trên toàn dự án. |
