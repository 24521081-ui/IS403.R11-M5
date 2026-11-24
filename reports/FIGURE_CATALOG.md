# Danh mục hình · báo cáo kinh doanh v2

Các nhóm hình v2 xuất PNG nền trắng, 300 DPI. Tất cả bảng nguồn dùng dữ liệu Test đã khóa hoặc lịch sử trước gốc theo phạm vi ghi trong báo cáo.

| Hình báo cáo | Loại | Câu hỏi | Bảng nguồn |
|---|---|---|---|
| [KPI và tổng quan](figures/business_v2_dashboard.png) | Thẻ KPI và đường | Mô hình cải thiện bao nhiêu và còn lệch tổng lượng thế nào? | [model_comparison](tables/model_comparison.csv) · [wrmsse_12_levels](tables/wrmsse_12_levels.csv) · [business_v2_daily](tables/business_v2_daily.csv) |
| [Chất lượng hai mô hình](figures/business_v2_model_metrics.png) | Cột nhỏ theo chỉ số | WRMSSE, sai số dòng và Bias thay đổi thế nào? | [model_comparison](tables/model_comparison.csv) |
| [WRMSSE đủ 12 cấp](figures/business_v2_hierarchy.png) | Dumbbell ngang | Cấp nào được cải thiện nhiều nhất? | [wrmsse_12_levels](tables/wrmsse_12_levels.csv) |
| [Dự báo ngày và lũy kế](figures/business_v2_daily.png) | Đường và cột lệch ngày | Khi nào dự báo lệch nhiều và độ lệch có tích lũy không? | [business_v2_daily](tables/business_v2_daily.csv) |
| [Kế hoạch theo tuần](figures/business_v2_weekly.png) | Ba ô cột | Tuần nào bán nhiều và có Bias lớn? | [weekly_analysis](tables/weekly_analysis.csv) |
| [Nhịp bán theo thứ](figures/business_v2_weekday.png) | Đường và cột ghép | Hai mô hình phản ánh chu kỳ tuần ra sao? | [weekday_analysis](tables/weekday_analysis.csv) |
| [Quy mô bang và WAPE cửa hàng](figures/business_v2_regional.png) | Cột ngang và dumbbell | Nhu cầu tập trung ở đâu và cửa hàng nào có sai số cao? | [state_analysis](tables/state_analysis.csv) · [store_analysis](tables/store_analysis.csv) |
| [Ưu tiên theo dõi cửa hàng](figures/business_v2_store_priority.png) | Cột có dấu và scatter bốn vùng | Ở đâu có độ lệch lớn hoặc WAPE cao cùng quy mô lớn? | [business_v2_store_priority](tables/business_v2_store_priority.csv) |
| [Quy mô và chất lượng theo ngành](figures/business_v2_category.png) | Ba ô cột ngang | Ngành nào chiếm nhiều lượng bán hoặc khó dự báo hơn? | [category_analysis](tables/category_analysis.csv) · [department_analysis](tables/department_analysis.csv) |
| [WAPE và Bias cửa hàng × bộ phận](figures/business_v2_heatmaps.png) | Hai heatmap có nhãn | Tổ hợp nào có sai số lớn hoặc lệch có hệ thống? | [store_department](tables/store_department.csv) |
| [Tập trung sai số và contribution](figures/business_v2_concentration.png) | Pareto tích lũy và cột ngang | Sai số tuyệt đối và contribution cấp 12 tập trung vào đâu? | [business_v2_pareto](tables/business_v2_pareto.csv) · [business_v2_pareto_thresholds](tables/business_v2_pareto_thresholds.csv) · [business_v2_top_contribution](tables/business_v2_top_contribution.csv) |
| [Quy mô bán và ABC trước Test](figures/business_v2_segments.png) | Hai ô cột ghép | Chất lượng khác nhau thế nào giữa các nhóm biết trước? | [business_v2_volume_segments](tables/business_v2_volume_segments.csv) · [business_v2_abc](tables/business_v2_abc.csv) · [business_v2_velocity_cutoffs](tables/business_v2_velocity_cutoffs.csv) |
| [Thiếu và thừa nhiều nhất](figures/business_v2_exceptions.png) | Hai ô cột ngang có dấu | Chuỗi nào cần rà soát vì lệch tổng lượng lớn? | [business_v2_exceptions](tables/business_v2_exceptions.csv) |
| [Sáu chuỗi sản phẩm đại diện](figures/business_v2_representatives.png) | Đường 2 × 3 | Mô hình hoạt động ra sao với các kiểu chuỗi khác nhau? | [business_v2_representatives](tables/business_v2_representatives.csv) · [business_v2_representatives_daily](tables/business_v2_representatives_daily.csv) |
| [Phân bố sai số và WAPE chuỗi](figures/business_v2_stability.png) | Hai ECDF | Mức cải thiện có trải đều qua các dòng và chuỗi không? | [business_v2_error_ecdf](tables/business_v2_error_ecdf.csv) · [business_v2_error_quantiles](tables/business_v2_error_quantiles.csv) · [business_v2_series](tables/business_v2_series.csv) · [business_v2_series_wins](tables/business_v2_series_wins.csv) · [business_v2_zero_actual](tables/business_v2_zero_actual.csv) |
| [SNAP theo bang và ngày](figures/business_v2_snap.png) | Ba ô cột ngang | Lượng bán bình quân và chất lượng khác thế nào theo SNAP? | [business_v2_snap](tables/business_v2_snap.csv) |
| [Vai trò đặc trưng](figures/business_v2_importance.png) | Hai cột ngang | Đặc trưng nào được dùng nhiều tại các nút chia? | [business_v2_importance](tables/business_v2_importance.csv) · [business_v2_snap_validation](tables/business_v2_snap_validation.csv) |

## Rà soát 24 hình trước nâng cấp

| Hình v1 | Câu hỏi/phạm vi gốc | Quyết định và lý do |
|---|---|---|
| [01_train_validation_test.png](figures/01_train_validation_test.png) | Phân chia Train–Validation–Test | Giữ. Minh họa rõ ba mốc thời gian; giữ cho phần phương pháp. |
| [02_quy_trinh_rf.png](figures/02_quy_trinh_rf.png) | Trình tự và kết quả các vòng thực nghiệm | Giữ. Sơ đồ quy trình rõ nhưng thưa; giữ làm bằng chứng thiết kế. |
| [03_rf_01_05.png](figures/03_rf_01_05.png) | Tuning ban đầu · RF-01–RF-05 | Giữ. Chọn vòng RF đầu; giữ, đọc thêm bảng chính xác khi cột làm tròn bằng nhau. |
| [04_rf_v1_v4.png](figures/04_rf_v1_v4.png) | Ablation · RF-v1–RF-v4 | Giữ. Ablation dễ so nhờ cùng trục; giữ bằng chứng chọn RF-v1. |
| [05_rf_06_22.png](figures/05_rf_06_22.png) | Tuning nâng cao · RF-06–RF-22 | Giữ. Cột ngang hợp 17 cấu hình; giữ cho vòng nâng cao. |
| [06_fe_01_10.png](figures/06_fe_01_10.png) | So sánh FE-01–FE-10 | Giữ. So sánh bộ đặc trưng cùng thước đo; giữ cho lựa chọn FE-04. |
| [07_fe04_va_all_features.png](figures/07_fe04_va_all_features.png) | FE-04 và các cấu hình All Features | Giữ. Tập trung FE-04 và All Features; giữ làm bằng chứng lựa chọn. |
| [08_wrmsse_va_chi_phi_fit.png](figures/08_wrmsse_va_chi_phi_fit.png) | WRMSSE và thời gian fit ghi nhận, có lượt dùng lại | Giữ. Chi phí fit đối chiếu chất lượng; giữ cho phương pháp. |
| [09_wrmsse_test.png](figures/09_wrmsse_test.png) | WRMSSE Test | Giữ. WRMSSE tổng thể dễ đọc; giữ cho notebook đánh giá, v2 dùng thẻ KPI. |
| [10_validation_va_test.png](figures/10_validation_va_test.png) | WRMSSE Validation và Test | Giữ. Hai cửa sổ và hai context; giữ nguyên bằng chứng. |
| [11_wrmsse_12_cap.png](figures/11_wrmsse_12_cap.png) | WRMSSE theo 12 cấp | Giữ. Đủ 12 cấp; giữ cho notebook đánh giá, v2 bổ sung giá trị và Δ. |
| [12_chi_so_sai_so_test.png](figures/12_chi_so_sai_so_test.png) | MAE, RMSE, WAPE và Bias | Giữ. Tách trục chỉ số phù hợp; giữ cho notebook đánh giá, v2 có 5 panel gồm WRMSSE. |
| 13_feature_importance.png | Importance của 15 đặc trưng logic | Thay bằng [importance](figures/business_v2_importance.png). Chú thích cũ sát nhãn trục; bản mới chia theo đặc trưng và nhóm. Bản gốc trong kho theo SHA-256. |
| 14_doanh_so_hang_ngay.png | Tổng lượng bán trong 28 ngày Test | Thay bằng [daily](figures/business_v2_daily.png). Đường rõ; bổ sung ngày cực trị, lệch có dấu và lũy kế. Bản gốc trong kho theo SHA-256. |
| 15_doanh_so_theo_bang.png | Lượng bán và dự báo theo bang | Thay bằng [regional](figures/business_v2_regional.png). Quy mô bang dễ đọc; thêm tỷ trọng và WAPE cửa hàng. Bản gốc trong kho theo SHA-256. |
| 16_doanh_so_theo_cua_hang.png | Lượng bán và dự báo theo cửa hàng | Thay bằng [regional](figures/business_v2_regional.png). Cột lượng bán lặp với phân tích bang; thay bằng so sánh WAPE tại 10 cửa hàng, giữ lượng bán ở bảng. Bản gốc trong kho theo SHA-256. |
| 17_bias_cua_hang.png | Forecast Bias theo cửa hàng | Thay bằng [store_priority](figures/business_v2_store_priority.png). Bias có đường 0 nhưng thiếu góc nhìn quy mô; thêm quadrant và số lệch ở bảng. Bản gốc trong kho theo SHA-256. |
| 18_heatmap_wape.png | Heatmap WAPE Random Forest | Thay bằng [heatmaps](figures/business_v2_heatmaps.png). Heatmap có giá trị; thêm panel Bias, đổi thứ tự theo mức độ ưu tiên. Bản gốc trong kho theo SHA-256. |
| 19_nganh_hang.png | Lượng bán và WAPE theo ngành hàng | Thay bằng [category](figures/business_v2_category.png). Giữ các trục riêng; thêm tỷ trọng và Bias. Bản gốc trong kho theo SHA-256. |
| 20_tuan_va_thu.png | So sánh WAPE theo tuần và theo thứ | Thay bằng [weekly](figures/business_v2_weekly.png) · [weekday](figures/business_v2_weekday.png). Chỉ có WAPE; tách nhịp tuần, lượng bán, Bias và bình quân theo thứ. Bản gốc trong kho theo SHA-256. |
| 21_top_du_bao_under.png | Top 20 chuỗi dự báo thiếu | Thay bằng [exceptions](figures/business_v2_exceptions.png). Nhãn dài và thiếu giá trị ở đầu cột; gom hai phía và thêm nhãn số. Bản gốc trong kho theo SHA-256. |
| 22_top_du_bao_over.png | Top 20 chuỗi dự báo thừa | Thay bằng [exceptions](figures/business_v2_exceptions.png). Gộp vào panel lệch dương có màu riêng và trục 0. Bản gốc trong kho theo SHA-256. |
| 23_chuoi_dai_dien.png | Actual và dự báo của các chuỗi đại diện | Thay bằng [representatives](figures/business_v2_representatives.png). Mở rộng từ bốn lên sáu tiêu chí; thêm WAPE và Bias trực tiếp. Bản gốc trong kho theo SHA-256. |
| 24_snap.png | WAPE theo trạng thái SNAP, toàn hệ thống và theo bang | Thay bằng [snap](figures/business_v2_snap.png). So sánh WAPE rõ nhưng thiếu lượng bán bình quân, Bias và số ngày. Bản gốc trong kho theo SHA-256. |

[Manifest v2 và ánh xạ bản trước](manifest_business_v2.json) · [Biên bản tinh gọn](../docs/PROJECT_CLEANUP_REPORT.md).
