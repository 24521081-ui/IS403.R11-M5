# Dự báo lượng bán M5 bằng Random Forest

Dự án dự báo lượng bán theo sản phẩm × cửa hàng trong 28 ngày, so sánh Random Forest Regression với Seasonal Naive chu kỳ 7 và đánh giá bằng **WRMSSE đủ 12 cấp M5**. Random Forest lấy trung bình dự báo của các cây hồi quy được fit với mẫu bootstrap và tập con đặc trưng khi chia nút. Mục tiêu là đánh giá chất lượng ở nhiều cấp tổng hợp và tìm nơi cần theo dõi trong kế hoạch bán hàng.

Bộ dữ liệu M5 có lượng bán hằng ngày, lịch và giá tuần của Walmart. Lưới dưới cùng gồm **30.490** chuỗi tại 10 cửa hàng thuộc CA, TX và WI.

| Tập | Ngày M5 | Cách dùng |
|---|---|---|
| Train | d_1–d_1885 | Lấy mẫu cho các vòng thử nghiệm |
| Validation | d_1886–d_1913 | Chọn siêu tham số và đặc trưng |
| Train + Validation | d_1–d_1913 | Lấy mẫu fit mô hình cuối |
| Test | d_1914–d_1941 | Đánh giá 25/04/2016–22/05/2016 từ gốc d_1913 |

Train dùng **600.000** quan sát phân tầng cửa hàng × bộ phận × quý, không hoàn lại, seed 42. Fit cuối giữ mẫu Train và thêm **9.185** quan sát Validation ở mật độ tương đương, tổng **609.185** dòng.

## Đặc trưng và đánh giá

FE-04 gồm 15 đặc trưng logic và 3.082 cột CSR float32. Thứ tự logic là item_id, dept_id, cat_id, store_id, state_id, wday, month, year, lag_28, lag_35, lag_42, lag_56, rolling_mean_7, rolling_mean_28, snap_state. Feature doanh số sử dụng lịch sử đến t−28 và forecast origin. SNAP xác định từ bang và ngày trong tháng. Vocabulary one-hot của năm trường ID giữ nhất quán. [Hậu kiểm SNAP](docs/STAGE3_SNAP_REVIEW.md) · [Notebook đặc trưng](notebooks/03_feature_engineering.ipynb).

WRMSSE cộng contribution qua 42.840 chuỗi. Contribution là RMSSE nhân trọng số doanh thu lịch sử. Mỗi cấp có tổng trọng số 1/12. RMSSE dùng scale sai phân lịch sử từ lần bán khác 0 đầu tiên. Điểm trong cấp bằng 12 lần tổng contribution cấp. [Đặc tả evaluator](docs/WRMSSE_PREPARATION.md) · [Notebook WRMSSE](notebooks/02_wrmsse_evaluation_method.ipynb).

## Thực nghiệm và mô hình khóa

Các vòng lần lượt chọn **RF-04 → RF-v1 → RF-15 → FE-04**. Có 36 lượt gồm 32 fit và 4 reuse. FE-09 và FE-10 kiểm tra All Features trên cùng Validation. [Báo cáo tối ưu](docs/STAGE3_REPORT.md) · [Bảng thực nghiệm](reports/tables/experiment_results.csv).

**RF-15 + FE-04** dùng n_estimators=150, max_depth=20, max_features=0.5, max_samples=0.5, min_samples_leaf=5, bootstrap=True, random_state=42, n_jobs=8.

| Cửa sổ | Random Forest | Seasonal Naive |
|---|---|---|
| Validation d_1886–d_1913 | 0,6239308540 | 0,9227813345 |
| Test d_1914–d_1941 | 0,7039124143 | 0,8697007023 |

Trên Test, Random Forest giảm WRMSSE **19,06%** và đạt WAPE **75,20%** ở cấp item–store–day. [Báo cáo Test](docs/STAGE4_REPORT.md) · [Báo cáo kinh doanh](docs/STAGE5_BUSINESS_REPORT.md).

### Hiệu quả theo tần suất bán

Trên 28 ngày Test, 2.754 chuỗi Frequent đóng góp 36,28% lượng bán và 27.736 chuỗi Intermittent đóng góp 63,72%. WAPE RF / Seasonal Naive lần lượt là 50,99% / 59,67% và 88,98% / 101,33%. Riêng 3.598 chuỗi có ADI ≥ 8, Seasonal Naive đạt WAPE 150,83% so với 182,46% của RF. Phần 14 của báo cáo và notebook 09 giải thích tiêu chí phân nhóm, sai số theo cửa hàng và các ngày bán đột biến. [Báo cáo kinh doanh](docs/STAGE5_BUSINESS_REPORT.md) · [Notebook 09](notebooks/09_business_insights_and_visualization.ipynb) · [Bảng kết quả](reports/tables/demand_group_metrics.csv)

Nhãn được ghép sau dự báo theo gốc d_1885 cho Validation và d_1913 cho Test. Workbook gồm 32 sheet, trong đó Forecast_Detail có 853.720 dòng kèm ADI và demand_group. Có 17 hình Business V2 và 11 hình phân tích nhu cầu. Phần 15 của notebook 09 có bộ chọn chuỗi và xuất PNG. [Excel](reports/RandomForest_Forecast_Actual.xlsx) · [Danh mục hình](reports/FIGURE_CATALOG.md) · [Manifest phân tích nhu cầu](reports/manifest_demand_groups.json)

## Xem kết quả đã công bố

1. Mở [Excel dự báo và thực tế](reports/RandomForest_Forecast_Actual.xlsx). Bắt đầu từ Summary, các sheet theo nhóm và Data_Dictionary.
2. Dùng [danh mục biểu đồ](reports/FIGURE_CATALOG.md) để chọn hình và tìm CSV nguồn.
3. Mở [notebook tổng quan](notebooks/00_project_overview.ipynb) đến [notebook phân tích kinh doanh](notebooks/09_business_insights_and_visualization.ipynb). Chọn kernel của .venv. Run All đọc artifact đã công bố. Notebook mới có output thực thi đã lưu.
4. Xem [biên bản bàn giao](reports/STAGE5_HANDOFF.md) và [manifest](reports/manifest_stage5.json) để tra số dòng, đường dẫn và checksum.

## Thư mục

| Thư mục | Nội dung |
|---|---|
| data/raw/ | CSV gốc được bảo toàn |
| data/processed/ | Doanh số lịch sử, khóa và mẫu |
| data/features/ | Ma trận đặc trưng và vocabulary |
| src/ và scripts/ | Evaluator, đặc trưng, điều phối, báo cáo và kiểm tra |
| configs/ | Split, metric và stage gate |
| notebooks/ | Mười notebook trình bày ở chế độ chỉ đọc |
| results/wrmsse_v1/ | Manifest và artifact Giai đoạn 2–4 |
| reports/figures/ và reports/tables/ | Hình PNG và bảng CSV nguồn |
| reports/ | Excel, danh mục hình và biên bản bàn giao |
| docs/ | Đặc tả và báo cáo |
| tests/ | Unit test evaluator, pipeline, data leakage |
| preparation/ | Biên bản chuẩn bị và audit |
| .venv/ | Môi trường cục bộ, ẩn trong VS Code và loại khỏi bản nộp |

## Môi trường

Môi trường thực nghiệm là Python 3.12.10. Mở workbook bằng Excel hỗ trợ XLSX. Dùng VS Code với Python và Jupyter extensions để mở notebook. Để cài từ thư mục dự án:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip check
```

requirements.txt dùng requirements-lock.txt, gồm đúng các phiên bản thực nghiệm, Excel và notebook đã kiểm tra. Mười notebook dùng các kết quả được giữ tại chỗ. [Biên bản và chỉ mục tinh gọn](docs/PROJECT_CLEANUP_REPORT.md).

## Kiểm tra và tái hiện

Dùng Python của môi trường dự án với lệnh dưới đây để xác minh đầu ra và chạy unit test.

```powershell
.venv/Scripts/python.exe -B scripts/verify_isolation.py --demand-groups
.venv/Scripts/python.exe -B -m unittest discover -s tests -t . -p "test_wrmsse*.py"
```

Các entry point dưới đây dành cho việc tái hiện giai đoạn tương ứng theo stage gate đã thiết lập, với namespace đầu ra mới.

| Giai đoạn | Entry point và điều kiện |
|---|---|
| 1 | [Kiểm thử](scripts/run_preparation_checks.py) và [audit context lịch sử](scripts/audit_wrmsse_contexts.py). Thực hiện theo báo cáo chuẩn bị. |
| 2 | [Điều phối Stage 2](scripts/run_stage2.py) tạo processed/features, baseline và smoke fit qua gate kiểm thử. |
| 3 | [Điều phối Stage 3](scripts/run_stage3.py) và [All Features](scripts/run_stage3_all_features.py) dùng gate tuning, evidence và namespace đã thiết lập. Selection v2 đã khóa. |
| 4 | [Điều phối Stage 4](scripts/run_stage4.py) kiểm tra selection v2, tạo run mới, fit một lần và khóa forecast trước Test. |
| 5 | [Điều phối Stage 5](scripts/run_stage5.py) đọc run Stage 4 đã khóa rồi tạo lớp báo cáo. Dùng tham số --verify để kiểm tra bản hoàn thành. |

## Giới hạn

Mô hình học từ mẫu và được tối ưu trên một cửa sổ Validation. Nghiên cứu này chuyển từ Mean RMSSE sang WRMSSE M5 đủ 12 cấp. Test mới từng là Validation của nghiên cứu cũ. Validation và Test dùng các context scale và trọng số khác nhau. Quyết định tồn kho và nhập hàng cần thêm dữ liệu vận hành.
