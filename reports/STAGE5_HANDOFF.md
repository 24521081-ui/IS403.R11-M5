# Bàn giao Giai đoạn 5

Bộ báo cáo dùng kết quả RF-15 + FE-04 đã khóa. Bảng và workbook được đối chiếu với artifact Stage 4. Mười notebook được thực thi ở chế độ đọc và lưu output. [Hướng dẫn mở dự án](../README.md) · [Phát hiện và ưu tiên kinh doanh](../docs/STAGE5_BUSINESS_REPORT.md).

## Excel và số dòng

[RandomForest_Forecast_Actual.xlsx](RandomForest_Forecast_Actual.xlsx) gồm 32 sheet. Bảng dưới liệt kê 23 sheet gốc và số dòng dữ liệu, chưa tính hai dòng tiêu đề. Chín sheet dành cho phân tích nhóm nhu cầu được mô tả ở cuối tài liệu.

| Sheet | Dòng dữ liệu |
|---|---|
| Summary | 26 |
| Model_Comparison | 2 |
| Daily_Sales | 28 |
| State_Analysis | 3 |
| Store_Analysis | 10 |
| Category_Analysis | 3 |
| Department_Analysis | 7 |
| Weekly_Analysis | 4 |
| Weekday_Analysis | 7 |
| SNAP_Analysis | 2 |
| SNAP_By_State | 6 |
| Store_Department | 70 |
| WRMSSE_12_Levels | 12 |
| Top_Underforecast | 20 |
| Top_Overforecast | 20 |
| Top_WRMSSE_Series | 20 |
| Top_Items_Sales | 20 |
| Top_Series_Sales | 20 |
| Representative_Series | 4 |
| Feature_Importance | 15 |
| Experiment_Results | 36 |
| Forecast_Detail | 853.720 |
| Data_Dictionary | 35 |

## Biểu đồ chính thức

[Danh mục nguồn và vị trí đề xuất](FIGURE_CATALOG.md).

- [01_train_validation_test.png](figures/01_train_validation_test.png)
- [02_quy_trinh_rf.png](figures/02_quy_trinh_rf.png)
- [03_rf_01_05.png](figures/03_rf_01_05.png)
- [04_rf_v1_v4.png](figures/04_rf_v1_v4.png)
- [05_rf_06_22.png](figures/05_rf_06_22.png)
- [06_fe_01_10.png](figures/06_fe_01_10.png)
- [07_fe04_va_all_features.png](figures/07_fe04_va_all_features.png)
- [08_wrmsse_va_chi_phi_fit.png](figures/08_wrmsse_va_chi_phi_fit.png)
- [09_wrmsse_test.png](figures/09_wrmsse_test.png)
- [10_validation_va_test.png](figures/10_validation_va_test.png)
- [11_wrmsse_12_cap.png](figures/11_wrmsse_12_cap.png)
- [12_chi_so_sai_so_test.png](figures/12_chi_so_sai_so_test.png)

## Notebook theo thứ tự nghiên cứu

- [00_project_overview.ipynb](../notebooks/00_project_overview.ipynb) — Tổng quan đề tài
- [01_data_preprocessing.ipynb](../notebooks/01_data_preprocessing.ipynb) — Chia dữ liệu và lấy mẫu
- [02_wrmsse_evaluation_method.ipynb](../notebooks/02_wrmsse_evaluation_method.ipynb) — WRMSSE M5 và hệ thống phân cấp
- [03_feature_engineering.ipynb](../notebooks/03_feature_engineering.ipynb) — Đặc trưng và giới hạn lịch sử
- [04_initial_hyperparameter_tuning.ipynb](../notebooks/04_initial_hyperparameter_tuning.ipynb) — Vòng siêu tham số ban đầu
- [05_feature_ablation.ipynb](../notebooks/05_feature_ablation.ipynb) — Ablation RF-v1 đến RF-v4
- [06_advanced_hyperparameter_tuning.ipynb](../notebooks/06_advanced_hyperparameter_tuning.ipynb) — Vòng siêu tham số nâng cao
- [07_feature_optimization.ipynb](../notebooks/07_feature_optimization.ipynb) — Tối ưu đặc trưng và chốt selection v2
- [08_final_training_and_test.ipynb](../notebooks/08_final_training_and_test.ipynb) — Fit cuối và đánh giá Test
- [09_business_insights_and_visualization.ipynb](../notebooks/09_business_insights_and_visualization.ipynb) — Phân tích kinh doanh trên Test

## Kiểm tra và tính toàn vẹn

| Kiểm tra | Kết quả |
|---|---|
| Unit test evaluator và pipeline | 71 đạt |
| Toàn lưới Forecast_Detail | 853.720 dòng khớp Parquet theo từng trường |
| WRMSSE và trọng số | 12 cấp khớp evaluator, contribution cộng đúng WRMSSE |
| Notebook | 10 tệp chạy Run All và có bảng/hình thật |
| Hình | 24 PNG với nguồn đã định danh và từ 200 DPI |
| Manifest Giai đoạn 1–4 | Bằng chứng và các manifest niêm phong đã đối chiếu |
| Dữ liệu raw, dự án nguồn và legacy | Danh sách tệp, kích thước và mốc sửa đổi khớp snapshot |
| Training, tuning, Test | Gate đóng trong toàn bộ Giai đoạn 5 |

## File mới và file chỉnh sửa

README là file hướng dẫn được cập nhật. Báo cáo, workbook, bảng, hình, notebook, mã điều phối và file khóa thư viện dưới đây được thêm trong Giai đoạn 5.

- [STAGE5_BUSINESS_REPORT.md](../docs/STAGE5_BUSINESS_REPORT.md)
- [00_project_overview.ipynb](../notebooks/00_project_overview.ipynb)
- [01_data_preprocessing.ipynb](../notebooks/01_data_preprocessing.ipynb)
- [02_wrmsse_evaluation_method.ipynb](../notebooks/02_wrmsse_evaluation_method.ipynb)
- [03_feature_engineering.ipynb](../notebooks/03_feature_engineering.ipynb)
- [04_initial_hyperparameter_tuning.ipynb](../notebooks/04_initial_hyperparameter_tuning.ipynb)
- [05_feature_ablation.ipynb](../notebooks/05_feature_ablation.ipynb)
- [06_advanced_hyperparameter_tuning.ipynb](../notebooks/06_advanced_hyperparameter_tuning.ipynb)
- [07_feature_optimization.ipynb](../notebooks/07_feature_optimization.ipynb)
- [08_final_training_and_test.ipynb](../notebooks/08_final_training_and_test.ipynb)
- [09_business_insights_and_visualization.ipynb](../notebooks/09_business_insights_and_visualization.ipynb)
- [README.md](../README.md)
- [figure_catalog.csv](../reports/figure_catalog.csv)
- [FIGURE_CATALOG.md](../reports/FIGURE_CATALOG.md)
- [01_train_validation_test.png](../reports/figures/01_train_validation_test.png)
- [02_quy_trinh_rf.png](../reports/figures/02_quy_trinh_rf.png)
- [03_rf_01_05.png](../reports/figures/03_rf_01_05.png)
- [04_rf_v1_v4.png](../reports/figures/04_rf_v1_v4.png)
- [05_rf_06_22.png](../reports/figures/05_rf_06_22.png)
- [06_fe_01_10.png](../reports/figures/06_fe_01_10.png)
- [07_fe04_va_all_features.png](../reports/figures/07_fe04_va_all_features.png)
- [08_wrmsse_va_chi_phi_fit.png](../reports/figures/08_wrmsse_va_chi_phi_fit.png)
- [09_wrmsse_test.png](../reports/figures/09_wrmsse_test.png)
- [10_validation_va_test.png](../reports/figures/10_validation_va_test.png)
- [11_wrmsse_12_cap.png](../reports/figures/11_wrmsse_12_cap.png)
- [12_chi_so_sai_so_test.png](../reports/figures/12_chi_so_sai_so_test.png)
- [RandomForest_Forecast_Actual.xlsx](../reports/RandomForest_Forecast_Actual.xlsx)
- [stage5_inputs.json](../reports/stage5_inputs.json)
- [stage5_unit_tests.log](../reports/stage5_unit_tests.log)
- [category_analysis.csv](../reports/tables/category_analysis.csv)
- [daily_alignment.csv](../reports/tables/daily_alignment.csv)
- [daily_sales.csv](../reports/tables/daily_sales.csv)
- [department_analysis.csv](../reports/tables/department_analysis.csv)
- [experiment_results.csv](../reports/tables/experiment_results.csv)
- [experiment_workflow.csv](../reports/tables/experiment_workflow.csv)
- [feature_importance.csv](../reports/tables/feature_importance.csv)
- [model_comparison.csv](../reports/tables/model_comparison.csv)
- [representative_daily.csv](../reports/tables/representative_daily.csv)
- [representative_series.csv](../reports/tables/representative_series.csv)
- [snap_analysis.csv](../reports/tables/snap_analysis.csv)
- [snap_by_state.csv](../reports/tables/snap_by_state.csv)
- [split_timeline.csv](../reports/tables/split_timeline.csv)
- [state_analysis.csv](../reports/tables/state_analysis.csv)
- [store_analysis.csv](../reports/tables/store_analysis.csv)
- [store_department.csv](../reports/tables/store_department.csv)
- [system.csv](../reports/tables/system.csv)
- [top_items_sales.csv](../reports/tables/top_items_sales.csv)
- [top_overforecast.csv](../reports/tables/top_overforecast.csv)
- [top_series_sales.csv](../reports/tables/top_series_sales.csv)
- [top_underforecast.csv](../reports/tables/top_underforecast.csv)
- [top_wrmsse_contribution.csv](../reports/tables/top_wrmsse_contribution.csv)
- [validation_test.csv](../reports/tables/validation_test.csv)
- [weekday_analysis.csv](../reports/tables/weekday_analysis.csv)
- [weekly_analysis.csv](../reports/tables/weekly_analysis.csv)
- [wrmsse_12_levels.csv](../reports/tables/wrmsse_12_levels.csv)
- [requirements-lock.txt](../requirements-lock.txt)
- [run_stage5.py](../scripts/run_stage5.py)
- [wrmsse_stage5.py](../src/wrmsse_stage5.py)
- [wrmsse_stage5_docs.py](../src/wrmsse_stage5_docs.py)
- [wrmsse_stage5_excel.py](../src/wrmsse_stage5_excel.py)
- [wrmsse_stage5_figures.py](../src/wrmsse_stage5_figures.py)
- [wrmsse_stage5_notebooks.py](../src/wrmsse_stage5_notebooks.py)
- [stage5_verification.json](../reports/stage5_verification.json)

- [Biên bản bàn giao](STAGE5_HANDOFF.md)
- [Manifest Giai đoạn 5](manifest_stage5.json)

Biên bản Giai đoạn 6 ghi cấu trúc và đường dẫn hiện tại. Kho lịch sử giữ nguyên bản đã niêm phong của biên bản này. [Biên bản tinh gọn](../docs/PROJECT_CLEANUP_REPORT.md).

## Giới hạn cần lưu ý

WAPE cấp item–store–day vẫn cao dù WRMSSE thấp hơn Seasonal Naive. Cửa sổ Test từng là Validation ở phiên bản nghiên cứu cũ. Ứng dụng nhập hàng cần dữ liệu tồn kho, lead time, chi phí và chính sách tồn kho an toàn.

## Trực quan hóa báo cáo v2

Danh mục hiện hành có 17 nhóm hình kinh doanh mới hoặc thay thế. Bản v1 của hình và tài liệu được đối chiếu bằng SHA-256. [Danh mục và bảng nguồn](FIGURE_CATALOG.md) · [Manifest v2](manifest_business_v2.json).

## Phân tích theo đặc điểm nhu cầu

Phần 14 bổ sung 5 hình và sửa 3 hình trong 17 hình kinh doanh V2. Forecast_Detail có 853.720 dòng và 17 cột. Năm sheet đầu tiên trong nhóm Demand_Group_* trình bày cơ cấu, sai số, cửa hàng, ngành hàng và khoảng ADI. Sáu bảng CSV của phần 14 gồm năm bảng tương ứng và bảng chọn bốn chuỗi minh họa. [Danh mục hình](FIGURE_CATALOG.md) · [Báo cáo kinh doanh](../docs/STAGE5_BUSINESS_REPORT.md) · [Manifest và danh sách tệp](manifest_demand_groups.json)

Hai gốc đều tái tạo đúng 30.490 giá trị ADI từ lịch sử. Phần 15 mở rộng hình so sánh thành sáu chỉ số RF và thêm sáu hình theo thời gian. Workbook có ba sheet thống kê ngày, tuần, thứ và một sheet chứa tám hình, tổng cộng 32 sheet. Công cụ cuối notebook tra cứu từng chuỗi và xuất một PNG theo lựa chọn. Kiểm tra lại bằng `.venv/Scripts/python.exe -B scripts/verify_isolation.py --demand-groups`. Công cụ đọc workbook trước từ lịch sử Git. [Số liệu theo ngày](tables/demand_group_daily.csv) · [Theo tuần](tables/demand_group_weekly.csv) · [Theo thứ](tables/demand_group_weekday.csv)

Bản checkout Windows có một số mã nguồn lịch sử dùng CRLF và thiếu thư mục đặc trưng Stage 2 đã được Git bỏ qua. Lần chạy trình kiểm tra nguyên bản trước khi cập nhật dừng ở checksum của preparation/publish_stage4_report.py. Bản lưu đầy đủ vượt qua trình kiểm tra nguyên bản. Với bản hiện hành, dấu vân tay bao trùm từng tệp hiện có trong configs, data, preparation và results. Mô hình, các tệp Stage 4 và khóa dự báo được đối chiếu trực tiếp với SHA-256 gốc. Hai manifest Stage 5 và V2 tiếp tục lưu biên bản của từng phiên bản.
