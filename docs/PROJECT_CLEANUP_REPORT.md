# Giai đoạn 6 · Tinh gọn dự án Random Forest WRMSSE

Thư mục làm việc đã giảm từ **5,55 GiB** xuống **1,77 GiB**, gồm cả môi trường cục bộ. Đã gỡ **1.491 tệp** và **180 thư mục rỗng**. RF-15 + FE-04, dữ liệu raw, model, dự báo, điểm và manifest niêm phong được bảo toàn. [Kiểm kê SHA-256 và bảng ánh xạ](PROJECT_CLEANUP_INVENTORY.csv).

## Kiểm kê trước và sau

| Phạm vi | Tệp trước | Tệp sau | MiB trước | MiB sau |
|---|---:|---:|---:|---:|
| . | 6 | 4 | 0,01 | 0,01 |
| .venv | 23.177 | 23.177 | 554,19 | 554,19 |
| .vscode | 0 | 1 | 0,00 | 0,00 |
| configs | 4 | 3 | 0,02 | 0,01 |
| data | 45 | 40 | 706,38 | 657,55 |
| docs | 7 | 9 | 0,08 | 0,68 |
| legacy | 1.178 | 0 | 1.341,10 | 0,00 |
| notebooks | 13 | 10 | 7,54 | 3,59 |
| preparation | 56 | 20 | 6,03 | 0,41 |
| reports | 58 | 58 | 62,79 | 62,79 |
| results | 584 | 318 | 3.006,73 | 528,17 |
| scripts | 8 | 8 | 0,08 | 0,08 |
| src | 16 | 17 | 0,17 | 0,17 |
| tests | 7 | 7 | 0,03 | 0,03 |
| **Tổng** | **25.159** | **23.672** | **5.685,14** | **1.807,67** |

Có 3.050 thư mục con trước dọn và 2.871 sau dọn. Ký hiệu . là các tệp ở thư mục gốc. Bảng tính dung lượng logic bằng byte, quy đổi sang MiB hoặc GiB. Toàn bộ .venv giữ nguyên 23.177 tệp và được thống kê gộp. Hai tệp làm việc tạm dùng cho lần dọn được loại khỏi số liệu.

Trước dọn, kiểm kê SHA-256 ngoài .venv phát hiện **94 nhóm tệp trùng byte**. Đối chiếu gồm mã Python, import, cấu hình, manifest, README, Markdown và các ô notebook, cùng đường dẫn ghép theo mã lượt, phiên bản feature và split. Tên tệp chỉ là điểm bắt đầu để tìm dependency. Các tệp lớn nhất có phân loại như sau.

| Đường dẫn trước dọn | MiB | Quyết định |
|---|---:|---|
| results/wrmsse_v1/stage3_20261124T082126Z/advanced/RF-20/attempt_001/model.joblib | 244,21 | ARCHIVE_EXTERNAL |
| data/raw/sell_prices.csv | 226,09 | KEEP |
| results/wrmsse_v1/stage4_20261124T104618287922Z/final_model.joblib | 153,58 | KEEP |
| results/wrmsse_v1/stage3_20261124T082126Z/advanced/RF-18/attempt_001/model.joblib | 152,59 | ARCHIVE_EXTERNAL |
| results/wrmsse_v1/stage3_20261124T082126Z/features/FE-04/attempt_001/model.joblib | 149,99 | KEEP |
| results/wrmsse_v1/stage3_20261124T082126Z/features/FE-07/attempt_001/model.joblib | 145,36 | ARCHIVE_EXTERNAL |
| results/wrmsse_v1/stage3_20261124T082126Z/advanced/RF-15/attempt_002/model.joblib | 141,87 | ARCHIVE_EXTERNAL |
| results/wrmsse_v1/stage3_20261124T082126Z/features/FE-06/attempt_001/model.joblib | 140,77 | ARCHIVE_EXTERNAL |
| results/wrmsse_v1/stage3_20261124T082126Z/features/FE-05/attempt_001/model.joblib | 139,72 | ARCHIVE_EXTERNAL |
| results/wrmsse_v1/stage3_20261124T082126Z/features/FE-02/attempt_001/model.joblib | 138,12 | ARCHIVE_EXTERNAL |

## Dung lượng trên ổ đĩa

- Dung lượng logic thư mục làm việc giảm **3,79 GiB**.
- Từ thời điểm ZIP đã được ghi và xác minh, dung lượng trống ổ D tăng **3,79 GiB** sau khi gỡ các tệp.
- ZIP dùng **0,90 GiB** trên cùng ổ D. Sau khi tính dung lượng ZIP, mức giải phóng ròng ước tính **2,89 GiB**. Số ròng trừ dung lượng archive và có thể khác số đo logic do cách hệ tệp cấp phát khối.

## Đã loại khỏi thư mục làm việc

- **legacy/** gồm toàn bộ 1.178 tệp. 990 tệp trùng byte với đúng đường dẫn ở dự án Mean RMSSE gốc. 188 tệp khác bản gốc hoặc riêng có trong bản clone được bảo toàn trong ZIP. Đối chiếu SHA ngoài legacy và raw chỉ gặp tệp khởi tạo test rỗng, gồm bản chuẩn cần giữ và các snapshot trùng.
- Bốn run Stage 4 dừng đã được chuyển ra kho. Cả bốn dừng trước khi fit và mở Test. RF-15 attempt_001 đã dừng trong vòng tuning khi chuyển cấu hình luồng. Giữ attempt_002 thành công và biên bản chuyển tiếp.
- 31 model fit trung gian, các mảng dự báo Validation, điểm theo từng chuỗi và log tài nguyên ít dùng được đưa ra kho. Request, kết quả, điểm theo cấp, thời gian, RAM tổng hợp và liên kết reuse vẫn ở từng lượt.
- Các snapshot mã Stage 2 và Stage 3 được tinh gọn theo hash. Những bản trùng byte trỏ tới tệp còn giữ. Các bản mã khác nhau có từng đối tượng riêng trong ZIP.
- Bốn ma trận ghép sẵn và manifest trong stage2_20261123T202104Z_combinations được lưu ngoài dự án. FE-07 và FE-08 đọc các block từ feature manifest chính, ghép trong bộ nhớ theo request. Tất cả block nguồn và dữ liệu xử lý chính thức còn nguyên tại chỗ.
- Đã loại notebook chuẩn bị cũ, notebook kiểm kê và bản TXT kiểm kê. Nội dung dẫn nhập nằm trong 00, 01 và 02. Chương trình sinh notebook hiện dùng đúng danh sách 00–09.
- Đã dọn cache Matplotlib, log và receipt cũ, script xuất báo cáo một lần, cấu hình n_jobs chuyển tiếp và lock lịch sử. Cache niêm phong được đối chiếu qua archive hoặc bản trùng byte. Cache tái tạo ngoài manifest được xóa. Các báo cáo cũ được giữ để mô tả đúng diễn biến nghiên cứu.

| Run dừng | Nguyên nhân đã ghi nhận |
|---|---|
| stage4_20261124T104419354631Z | API kiểm tra ổ đĩa cần chuỗi thay vì WindowsPath. |
| stage4_20261124T104425897245Z | Log worker Stage 2 đã có dòng ghi bổ sung sau checkpoint. Biên bản hoàn tất Stage 2 là bản niêm phong dùng để xác minh log. |
| stage4_20261124T104451222266Z | Biên bản nguồn có BOM UTF-8. |
| stage4_20261124T104520821109Z | Phép so sánh cột state nhạy với khác kiểu dữ liệu dù giá trị khớp. |
| RF-15 attempt_001 | Dừng theo kế hoạch chuyển từ 4 sang 8 luồng. Attempt_002 là bản fit chính thức của lượt RF-15. |

<details>
<summary>Đường dẫn của 180 thư mục đã gỡ</summary>

```text
  legacy/data/features/rf_feature_optimization_round2/fe_02/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_03/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_04/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_05/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_06/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_07/artifact_001/
  legacy/data/features/rf_feature_optimization_round2/fe_08/artifact_001/
  legacy/data/processed/m5_shared_v1/sales/test/
  legacy/data/processed/m5_shared_v1/sales/train/
  legacy/data/processed/m5_shared_v1/sales/validation/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-02/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-03/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-04/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-05/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-06/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-07/attempt_001/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-08/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/attempt_003/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/attempt_004/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/attempt_005/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/attempt_003/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/attempt_004/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/attempt_005/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/attempt_003/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/attempt_004/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/attempt_005/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-10/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-10/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-11/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-11/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-12/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-12/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/attempt_003/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/attempt_004/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/attempt_005/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-14/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-15/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-16/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-17/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-18/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-20/attempt_001/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-20/attempt_002/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-22/attempt_001/
  results/wrmsse_v1/stage3_20261124T082126Z/advanced/RF-15/attempt_001/
  data/features/wrmsse_v1/stage2_20261123T202104Z_combinations/template_behavior_prices/
  data/features/wrmsse_v1/stage2_20261123T202104Z_combinations/template_behavior_snap/
  legacy/data/features/rf_feature_optimization_round2/fe_02/
  legacy/data/features/rf_feature_optimization_round2/fe_03/
  legacy/data/features/rf_feature_optimization_round2/fe_04/
  legacy/data/features/rf_feature_optimization_round2/fe_05/
  legacy/data/features/rf_feature_optimization_round2/fe_06/
  legacy/data/features/rf_feature_optimization_round2/fe_07/
  legacy/data/features/rf_feature_optimization_round2/fe_08/
  legacy/data/processed/m5_shared_v1/metrics/
  legacy/data/processed/m5_shared_v1/prices/
  legacy/data/processed/m5_shared_v1/sales/
  legacy/results/archive/pre_feature_optimization_round2_20261122/notebooks/
  legacy/results/archive/pre_feature_optimization_round2_20261122/scripts/
  legacy/results/archive/pre_feature_optimization_round2_20261122/src/
  legacy/results/archive/pre_feature_optimization_round2_20261122/tests/
  legacy/results/archive/pre_round4_20261122/notebooks/
  legacy/results/archive/pre_round4_20261122/scripts/
  legacy/results/archive/pre_round4_20261122/src/
  legacy/results/archive/pre_round4_20261122/tests/
  legacy/results/archive/pre_selection_20261122/notebooks/
  legacy/results/archive/pre_selection_20261122/src/
  legacy/results/archive/pre_selection_20261122/tests/
  legacy/results/rf_experiments/RF-01/attempt_001/
  legacy/results/rf_experiments/RF-02/attempt_001/
  legacy/results/rf_experiments/RF-03/attempt_001/
  legacy/results/rf_experiments/RF-04/attempt_001/
  legacy/results/rf_experiments/RF-05/attempt_001/
  legacy/results/rf_experiments/rf_feature_ablation/rf_v2/
  legacy/results/rf_experiments/rf_feature_ablation/rf_v3/
  legacy/results/rf_experiments/rf_feature_ablation/rf_v4/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-02/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-03/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-04/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-05/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-06/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-07/
  legacy/results/rf_experiments/rf_feature_optimization_round2/FE-08/
  legacy/results/rf_experiments/rf_final_test/execution_sources/
  legacy/results/rf_experiments/rf_final_test/figures/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-07/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-08/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-09/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-10/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-11/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-12/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-13/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-14/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-15/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-16/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-17/
  legacy/results/rf_experiments/rf_v3_tuning_round3/RF-18/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-19/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-20/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-21/
  legacy/results/rf_experiments/rf_v3_tuning_round4/RF-22/
  results/wrmsse_v1/stage2_20261123T202104Z/prepared_code_snapshot/configs/
  results/wrmsse_v1/stage2_20261123T202104Z/prepared_code_snapshot/scripts/
  results/wrmsse_v1/stage2_20261123T202104Z/prepared_code_snapshot/src/
  results/wrmsse_v1/stage2_20261123T202104Z/prepared_code_snapshot/tests/
  results/wrmsse_v1/stage2_20261123T202104Z/source_snapshot/configs/
  results/wrmsse_v1/stage2_20261123T202104Z/source_snapshot/scripts/
  results/wrmsse_v1/stage2_20261123T202104Z/source_snapshot/src/
  results/wrmsse_v1/stage2_20261123T202104Z/source_snapshot/tests/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot/scripts/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot/src/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot/tests/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot_n_jobs8/configs/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot_n_jobs8/scripts/
  results/wrmsse_v1/stage3_20261124T082126Z/code_snapshot_n_jobs8/tests/
  data/features/wrmsse_v1/stage2_20261123T202104Z_combinations/
  legacy/data/features/rf_feature_optimization_round2/
  legacy/data/features/rf_v1/
  legacy/data/features/rf_v2/
  legacy/data/features/rf_v3/
  legacy/data/features/rf_v3_test/
  legacy/data/features/rf_v4/
  legacy/data/processed/m5_shared_v1/
  legacy/experiments/fe09_all_features/attempt_001/
  legacy/results/archive/pre_feature_optimization_round2_20261122/
  legacy/results/archive/pre_round4_20261122/
  legacy/results/archive/pre_selection_20261122/
  legacy/results/report_artifacts_selected/RF-10_1d7ded1253bb/
  legacy/results/report_artifacts_selected/RF-18 + FE-07_e94d1215bca3_rf_feature_optimization_round2/
  legacy/results/report_artifacts_selected/RF-18_986b891aad8c_rf_v3_tuning_round4/
  legacy/results/rf_experiments/RF-01/
  legacy/results/rf_experiments/RF-02/
  legacy/results/rf_experiments/RF-03/
  legacy/results/rf_experiments/RF-04/
  legacy/results/rf_experiments/RF-05/
  legacy/results/rf_experiments/rf_feature_ablation/
  legacy/results/rf_experiments/rf_feature_optimization_round2/
  legacy/results/rf_experiments/rf_final_test/
  legacy/results/rf_experiments/rf_v3_tuning_round2/
  legacy/results/rf_experiments/rf_v3_tuning_round3/
  legacy/results/rf_experiments/rf_v3_tuning_round4/
  results/wrmsse_v1/stage2_20261123T202104Z/mplconfig/
  results/wrmsse_v1/stage2_20261123T202104Z/prepared_code_snapshot/
  results/wrmsse_v1/stage2_20261123T202104Z/source_snapshot/
  results/wrmsse_v1/stage3_20261124T082126Z/mplconfig/
  results/wrmsse_v1/stage3_all_features/plot_cache/
  legacy/data/features/
  legacy/data/processed/
  legacy/docs/notebook_documentation/
  legacy/docs/visualization_review/
  legacy/experiments/fe09_all_features/
  legacy/reports/figures/
  legacy/reports/tables/
  legacy/results/archive/
  legacy/results/baseline_seasonal_naive/
  legacy/results/report_artifacts_selected/
  legacy/results/rf_experiments/
  legacy/results/rf_test_final/
  results/wrmsse_v1/stage4_20261124T104419354631Z/
  results/wrmsse_v1/stage4_20261124T104425897245Z/
  results/wrmsse_v1/stage4_20261124T104451222266Z/
  results/wrmsse_v1/stage4_20261124T104520821109Z/
  legacy/data/
  legacy/docs/
  legacy/experiments/
  legacy/notebooks/
  legacy/reports/
  legacy/results/
  legacy/scripts/
  legacy/src/
  legacy/tests/
  preparation/matplotlib_cache/
  legacy/
```

</details>

## Kho lịch sử và manifest

Kho nằm cạnh dự án tại **D:/project/PTDLKD_RandomForest_WRMSSE_history.zip**. Kích thước ZIP **961.559.242 byte**. SHA-256 của ZIP là **60e43fc5bbe89cbba2711211cebfcc8536057a10bd603f4b1854613dcfc862c1**.

Kho giữ **370 đối tượng duy nhất theo SHA-256**, gồm dữ liệu legacy khác bản nguồn, artifact WRMSSE trung gian, run dừng và phiên bản cũ của mã, cấu hình, tài liệu được chỉnh. Các đường dẫn tham chiếu đối tượng trong ZIP tương ứng **2,47 GiB** trước nén, tính theo từng đường dẫn. Chỉ mục CSV lưu đường dẫn ban đầu, hành động, kích thước, SHA-256, nơi thay thế và manifest liên quan. Dòng có archive_member cho biết tệp trong ZIP. Dòng project: trỏ tới bản giống byte còn trong dự án. Dòng source: trỏ tới bản Mean RMSSE đã đối chiếu tại dự án gốc.

Các manifest Giai đoạn 1–5 và forecast lock giữ nguyên byte. Kiểm tra bản gọn xác minh file tại chỗ hoặc tham chiếu đối tượng theo bảng ánh xạ. Kiểm tra kho đọc từng đối tượng và đối chiếu SHA-256. Nếu cần phục hồi một bản lịch sử, tra archive_member theo đường dẫn cũ trong CSV, trích đối tượng vào vị trí riêng và so SHA-256 trước khi dùng. [Chỉ mục kiểm kê](PROJECT_CLEANUP_INVENTORY.csv).

```powershell
.venv/Scripts/python.exe -B scripts/verify_isolation.py
.venv/Scripts/python.exe -B scripts/verify_isolation.py --archive D:/project/PTDLKD_RandomForest_WRMSSE_history.zip
.venv/Scripts/python.exe -B scripts/run_stage5.py --verify
```

## Giữ lại và chỉnh đường dẫn

- Giữ nguyên 8 CSV raw, processed split, feature block của RF-v1–RF-v4, Prices, SNAP, Events, Recency và Behavior. Giữ từng bảng kết quả RF-01–22, ablation và FE-01–10 cùng provenance của 32 fit và 4 reuse. Giữ model và dự báo FE-04/FE-07 vì hậu kiểm SNAP cần cả hai.
- Selection v2 là đầu mối chính thức. Bản chọn FE-03 được nhận diện rõ bằng selected_model_v1.json. Bản trùng ở gốc run được gỡ. [Selection v2](../results/wrmsse_v1/stage3_20261124T082126Z/selection_v2/selected_model.json).
- Giữ nguyên toàn bộ run Test chính thức, kể cả khóa dự báo, mã băm, model, dữ liệu theo chuỗi và 6 PNG nguồn trong manifest. reports/figures là nơi lấy hình đưa vào báo cáo. Hình thử nghiệm, context hoặc checksum khác nhau vẫn được bảo toàn theo mục đích. [Báo cáo Test](STAGE4_REPORT.md) · [Danh mục hình](../reports/FIGURE_CATALOG.md).
- Các script Stage 2–5 và kiểm tra tính toàn vẹn dùng đường dẫn cục bộ. Siêu tham số RF-14–18 đã nằm đầy đủ trong config, trường provenance mô tả nguồn lịch sử. Hai tài liệu Stage 3 và biên bản Stage 5 trỏ tới bản hoạt động hoặc chỉ mục lưu trữ. Kiểm tra import Stage 4 hiện dùng đường dẫn module của dự án. 
- Cài môi trường qua requirements.txt và một requirements-lock.txt cố định theo môi trường thực nghiệm đã kiểm tra. Các lock cũ có trong ZIP. VS Code ẩn môi trường và cache theo .vscode/settings.json. [README](../README.md).

## Hậu kiểm

| Hạng mục | Kết quả |
|---|---|
| Unit test | 71/71 đạt, gồm evaluator, pipeline và leakage. |
| Import | 30 module active nạp thành công. |
| Notebook | 10/10 thực thi Run All chỉ đọc sau khi bỏ legacy. Bản đã công bố giữ nguyên. |
| Excel và báo cáo | Workbook mở để kiểm tra bằng trình đọc XLSX. 853.720 dòng Forecast_Detail khớp Parquet. 24 PNG và 26 CSV đã đối chiếu. |
| Model và lựa chọn | Nạp được model RF-15 + FE-04, 150 cây và 3.082 cột. Mẫu fit 609.185. Thông số của model, config và selection khớp. |
| Validation và Test | WRMSSE Validation 0.6239308540. WRMSSE Test RF 0.7039124143, Seasonal Naive 0.8697007023. Rà soát đủ 12 cấp. |
| Manifest cục bộ | 38 tệp run Test và 11 tệp forecast lock khớp SHA-256. Bảy manifest đầu vào đã đối chiếu nơi giữ. Stage 5 có 69 tệp nguyên trạng tại chỗ và 7 phiên bản trước chỉnh sửa có chỉ mục kho ngoài. |
| Kho lịch sử | 370 đối tượng đọc và băm lại thành công. Stage 3 v1 được kiểm tra bằng điểm từng lượt. FE-04 vẫn là lựa chọn v2. |
| Raw và dự án nguồn | 8 CSV raw trùng SHA-256 với biên bản clone. 29.454 tệp ở dự án gốc khớp danh sách tên, kích thước và mốc sửa đổi trước/sau. |
| Stage gate | Training, tuning và Test evaluation ở trạng thái khóa. Cleanup dùng dữ liệu và kết quả đã lưu, không fit, tuning hay sinh dự báo. |

Bản gọn đọc báo cáo, notebook và các kết quả chính thức từ bên trong dự án. Kho ZIP chỉ phục vụ khôi phục và kiểm tra đầy đủ các byte đã chuyển. Dự án Mean RMSSE vẫn ở vị trí cũ. Cảnh báo deprecation của pandas trong bộ test đã tồn tại ở bộ phụ thuộc hiện tại. Môi trường và phiên bản thư viện giữ nguyên.

## Cấu trúc cuối cùng

```text
PTDLKD_RandomForest_WRMSSE/
├── .venv/                 môi trường cục bộ, ẩn trong VS Code
├── .vscode/settings.json
├── configs/               experiment.json, stage2.json, stage3.json
├── data/
│   ├── raw/               8 CSV gốc
│   ├── processed/wrmsse_v1/
│   └── features/wrmsse_v1/
├── docs/                  phương pháp, báo cáo và chỉ mục cleanup
├── notebooks/             00–09 đã kiểm tra Run All chỉ đọc
├── preparation/           hậu kiểm SNAP/Events, WRMSSE và evidence chính
├── reports/
│   ├── figures/           24 PNG chính thức
│   ├── tables/            26 CSV
│   └── RandomForest_Forecast_Actual.xlsx
├── results/wrmsse_v1/
│   ├── stage2_20261123T202104Z/
│   ├── stage3_20261124T082126Z/
│   │   └── selection_v2/
│   ├── stage3_all_features/
│   └── stage4_20261124T104618287922Z/
├── scripts/
├── src/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

Nghiên cứu chuyển từ Mean RMSSE sang WRMSSE M5 đầy đủ 12 cấp. Cửa sổ Test d_1914–d_1941 từng được dùng làm Validation trong nghiên cứu cũ. Các kết quả và giới hạn đó được ghi trong báo cáo Test và báo cáo kinh doanh. [Báo cáo kinh doanh](STAGE5_BUSINESS_REPORT.md).
