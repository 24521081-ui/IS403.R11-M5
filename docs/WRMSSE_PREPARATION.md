# Báo cáo chuẩn bị Random Forest / WRMSSE — 24/11/2026

Trạng thái: đã tạo bản sao độc lập và xác minh hàm WRMSSE cùng các context lịch sử. Chưa chạy tuning, huấn luyện, sinh dự báo hoặc đánh giá Test. Cấu hình vẫn ở `stage=preparation`, ba cờ thực nghiệm đều `false`.

## Bản sao và cấu trúc

Đích: `D:/project/PTDLKD_RandomForest_WRMSSE`. Đã sao chép và kiểm chứng SHA-256 **1.185 tệp, 1.894.632.548 byte**. Loại trừ môi trường ảo nguồn, Git, cache Python/Jupyter/matplotlib và bytecode; không loại dữ liệu raw hoặc artifact thực nghiệm cần lưu. Tám CSV raw giữ nguyên byte. `.venv/` trong bản mới được tạo và cài mới hoàn toàn; `requirements-lock.txt` khóa môi trường đã kiểm tra.

```text
configs/experiment.json           cấu hình split, metric, output namespace, stage
src/wrmsse_config.py              contract ID, Fold, stage gates
src/project_paths.py              bảo vệ đường dẫn và ghi file độc quyền
src/wrmsse.py                     hierarchy, scale, revenue weights, evaluator
src/wrmsse_data.py                raw loaders và join giá theo ngày/tuần
src/wrmsse_features.py            đặc trưng RF cơ sở theo cutoff mới
src/wrmsse_pipeline.py            WRMSSE selection, keyed forecasts, gated RF
scripts/                         unit checks, history-only audit, isolation audit
tests/                           WRMSSE, oracle độc lập, leakage và isolation
notebooks/00_wrmsse_preparation.ipynb  chỉ đọc cấu hình và báo cáo
preparation/                     manifest sao chép, log kiểm thử và bằng chứng
data/raw/                        cả 8 CSV gốc, không sửa
results/wrmsse_v1/                rỗng, dành cho kết quả mới sau này
legacy/                          source, notebook, data processed/features,
                                 model, forecast, report và lịch sử cũ
```

Không có junction/symlink hoặc hard link có chủ đích tới nguồn. Các đường dẫn active đều được suy ra từ vị trí module trong bản mới. Hàm ghi từ chối đường dẫn tuyệt đối, đường dẫn thoát root và `legacy/`; file được mở bằng chế độ tạo mới để tránh ghi đè. Các entry point Python và mọi code cell notebook trong hồ sơ cũ đã thêm guard chặn thực thi. Tất cả điểm số, mô hình và dự báo cũ vẫn là dữ liệu lịch sử; không có kết quả thực nghiệm WRMSSE mới được tạo.

## Cấu hình thời gian và chống leakage

| Context | Lịch sử tính scale | Doanh thu tính trọng số | Khoảng dự báo |
|---|---|---|---|
| Validation | d_1–d_1885 | d_1858–d_1885 | d_1886–d_1913 |
| Test | d_1–d_1913 | d_1886–d_1913 | d_1914–d_1941 |

Train ban đầu d_1–d_1885; horizon cố định 28. Với Test, lịch sử được mở rộng tới d_1913 vì Validation đã trở thành quá khứ tại gốc dự báo. Đây là chính sách refit Train + Validation đã khai báo; không lấy Test để tính scale hoặc trọng số.

`load_history` chỉ chọn cột đến đúng origin. Validation và lịch sử Test lấy từ `sales_train_validation.csv`; Validation actual là d_1886–d_1913 trong cùng CSV. `sales_test_validation.csv` chứa Test mới, chỉ được phép mở qua entry point Test khi stage cho phép. `sales_test_evaluation.csv` thuộc cửa sổ cũ d_1942–d_1969, chỉ được giữ để bảo toàn dữ liệu.

Trọng số luôn tính mới từ `units * sell_price`. Không dùng sẵn `weights_validation.csv` hoặc `weights_evaluation.csv` làm input evaluator. Tệp weights_validation đi kèm tương ứng gốc d_1913, nên chỉ được dùng làm fixture đối chiếu cho **context Test mới**; không được dùng cho Validation mới. Tệp weights_evaluation tương ứng gốc cũ d_1941, không dùng cho hai context mới.

Calendar bản dữ liệu này không có cột `d`: loader tính `day=(date-2011-01-29).days+1`, kiểm tra trùng ngày và đối chiếu nếu có nhãn `d`. Giá ghép chính xác theo `(store_id,item_id,wm_yr_wk)` từ ngày mục tiêu trong cửa sổ doanh thu, không forward-fill từ tương lai. Giá thiếu với số lượng bán dương gây lỗi; giá thiếu tại ngày không bán có doanh thu bằng 0. Giá âm, vô cực và khóa giá trùng bị từ chối.

Đặc trưng direct dùng lag 28,35,42,56 và rolling mean 7,28, với cutoff `min(target_day-28, origin)`. Dòng Train cơ sở bắt đầu d_57 vì lag 56. Khóa dự báo chứa cả series và ngày, nên thứ tự dòng thay đổi không làm ghép nhầm. Nhãn Training phải nằm trước hoặc bằng d_1885.

## Định nghĩa WRMSSE

Triển khai theo [M5 Competitors Guide](https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf) và [bài báo của ban tổ chức M5](https://doi.org/10.1016/j.ijforecast.2021.11.013). Tổng hợp actual và forecast từ 30.490 chuỗi item-store lên mọi cấp trước khi tính lỗi. Thứ tự trường khóa không làm đổi một nhóm phân cấp; fixture M5 Level11 lưu state trước item, evaluator lưu item trước state và audit chuyển đổi rõ ràng.

| Cấp | Nhóm | Số chuỗi |
|---|---|---:|
| 1 | Tổng | 1 |
| 2 | Bang | 3 |
| 3 | Cửa hàng | 10 |
| 4 | Ngành hàng | 3 |
| 5 | Bộ phận | 7 |
| 6 | Bang × ngành hàng | 9 |
| 7 | Bang × bộ phận | 21 |
| 8 | Cửa hàng × ngành hàng | 30 |
| 9 | Cửa hàng × bộ phận | 70 |
| 10 | Sản phẩm | 3.049 |
| 11 | Sản phẩm × bang | 9.147 |
| 12 | Sản phẩm × cửa hàng | 30.490 |
| Tổng | | **42.840** |

Với chuỗi tổng hợp i, loại đúng các số 0 đầu lịch sử, giữ mọi số 0 kể từ lần bán đầu tiên. Nếu đoạn còn lại có m quan sát:

`scale_i = sum((y_t-y_(t-1))^2)/(m-1)` trên các sai phân bên trong đoạn đó.

`RMSSE_i = sqrt(mean((actual_i-forecast_i)^2)/scale_i)` trên 28 ngày.

`revenue_i = sum(units * price)` tổng hợp trong 28 ngày cuối lịch sử.

`weight_i = revenue_i / total_bottom_revenue / 12`.

`WRMSSE = sum(weight_i * RMSSE_i)` trên 42.840 chuỗi.

Mỗi cấp có tổng trọng số 1/12; toàn bộ có tổng 1. Báo cáo tách contribution và score trong từng cấp. Không lấy trung bình RMSSE bottom-level thay cho WRMSSE; không lấy căn sau khi cộng weighted MSE.

Quy tắc chuỗi đặc biệt: không cộng epsilon vào scale, không bỏ chuỗi rồi chuẩn hóa lại trọng số. Chuỗi toàn 0, chỉ có một quan sát hoạt động hoặc scale=0 được đánh dấu riêng. Nếu có trọng số dương, evaluator dừng vì tỷ số M5 không xác định; nếu trọng số bằng 0, contribution bằng 0 và RMSSE không xác định vẫn hiển thị NaN. Đây là quy tắc xử lý tường minh cho trường hợp ngoài dữ liệu M5 hợp lệ, không khẳng định M5 định nghĩa một điểm hữu hạn cho scale bằng 0. Dữ liệu thật ở cả hai origin đều có 42.840 scale dương hợp lệ. Forecast âm hữu hạn được giữ nguyên khi chấm, không ngầm clip.

## Kết quả kiểm thử

**43 unit test đạt; 0 lỗi, 0 bỏ qua.** Bao gồm:

- Dự báo hoàn hảo = 0; scale=1 và lỗi=1 cho WRMSSE=1.
- `[0,0,2,0,4]` có scale=10; lỗi ±1 cho `1/sqrt(10)`.
- Hai chuỗi có revenue ratio 1:4 cho kết quả tính tay **0,3**.
- Lỗi trái dấu triệt tiêu khi tổng hợp đúng cấp cho kết quả **0,25**.
- Bất biến theo đơn vị, căn chỉnh khóa, forecast âm, thiếu/trùng ngày, NaN/Inf, scale đặc biệt và thiếu giá.
- Cửa sổ trọng số chính xác; thay giá sau origin không làm đổi kết quả; đặc trưng không dùng doanh số sau cutoff.
- Chặn training/tuning/Test trước mọi truy cập dữ liệu; từ chối score cũ, sai contract và ghi thoát root; không ghi đè file đã có.

`tests/reference_wrmsse.py` là oracle scalar độc lập, dùng vòng lặp Python/math.fsum với định nghĩa nhóm riêng, không gọi roll-up, scale hoặc scorer production. Đối chiếu score, scale, weight và contribution trên 10 bộ dữ liệu ngẫu nhiên nhiều bang/cửa hàng, horizon 28; sai số score dưới 1e-12. Đây là đối chiếu bằng một cách triển khai độc lập về thuật toán, chưa phải chứng nhận từ một gói scorer bên ngoài.

Audit chỉ dùng lịch sử thật, không chấm forecast:

| Context | Chuỗi phân cấp | Scale hợp lệ | Doanh thu cửa sổ |
|---|---:|---:|---:|
| Validation | 42.840 | 42.840 | 3.665.392,06 |
| Test, chỉ lịch sử đến d_1913 | 42.840 | 42.840 | 3.748.180,05 |

Trọng số mọi nút được so với một cách tính pandas groupby riêng; scale Total và ba chuỗi bottom được tính lại bằng scalar. Đối chiếu toàn bộ 42.840 trọng số context d_1913 với fixture M5 đạt: sai số doanh thu lớn nhất 7,50e-8; sai số trọng số trong từng cấp lớn nhất **5,00e-13**. Tham khảo mã độc lập công khai: [pmrgn/m5-wrmsse](https://github.com/pmrgn/m5-wrmsse); không sử dụng precomputed weights/test labels của gói đó.

## File thay đổi và kiểm kê ảnh hưởng

Mã mới: `src/__init__.py`, `src/wrmsse_config.py`, `src/project_paths.py`, `src/wrmsse.py`, `src/wrmsse_data.py`, `src/wrmsse_features.py`, `src/wrmsse_pipeline.py`.

Cấu hình/môi trường: `configs/experiment.json`, `requirements.txt`, `requirements-lock.txt`, `.gitignore`. Kiểm thử: `tests/__init__.py`, `tests/reference_wrmsse.py`, `tests/test_wrmsse.py`, `tests/test_wrmsse_pipeline.py`. Script: `scripts/run_preparation_checks.py`, `scripts/audit_wrmsse_contexts.py`, `scripts/verify_isolation.py`. Tài liệu: README, báo cáo này, `docs/affected_components.csv`, `legacy/ARCHIVE_NOTICE.md` và notebook chuẩn bị mới.

68 tệp source/script/test/notebook cấp chính được kiểm kê từng tệp trong CSV. Tổng 177 entry point Python/notebook kể cả snapshot cũ được chặn. Các thay đổi trong archive ở bước chuẩn bị gồm guard và metadata đường dẫn; log `preparation/archive_changes.json` ghi nhóm thao tác, còn `preparation/modified_archive_files.json` ghi hash trước/sau từng tệp thay đổi. Không thay điểm/mô hình/dự báo lịch sử bằng giá trị mới; một số checksum source/notebook cũ không còn áp dụng sau khi thêm guard.

Nhóm ảnh hưởng chính: data_loader/preprocessing (split, lịch sử và schema); metrics (Mean RMSSE thành 12-level WRMSSE); rf_features và feature optimization (sampling/lag/window/cutoff và dữ liệu rebuild); training/tuning/ablation/selection (selection WRMSSE + contract mới); final training/Test (origin d_1913 và Test gate); reporting/Excel/PowerPoint/visualization (tên metric, nhóm phân cấp, biểu đồ và provenance); notebook và test artifact cũ (cần dữ liệu và output mới).

## Những phần cần chạy lại sau bước chuẩn bị

1. Tạo dữ liệu processed mới, thiết kế lại sampling Train trong d_1–d_1885 và tạo feature matrices mới; không dùng parquet/features cũ.
2. Port các biến thể RF-v1–v4, tối ưu feature, ablation, monitoring và báo cáo chuyên biệt sang contract mới nếu tiếp tục sử dụng. Pipeline cơ sở mới đã có loader, feature builder và entry point RF/selection, nhưng các workflow nâng cao cũ hiện được lưu và chặn, chưa coi là đã chuyển xong.
3. Chạy lại baseline, RF tuning và feature selection theo WRMSSE Validation, với cấu hình và seed được ghi mới. Không lấy RF-18/FE-07 hoặc score cũ làm lựa chọn mặc định.
4. Sau khi khóa lựa chọn bằng Validation, fit model mới với lịch sử Train + Validation đến d_1913 theo sampling đã công bố; lưu dự báo đủ 30.490 × 28 ngày, checksum và manifest trước khi mở actual Test.
5. Chỉ khi bước Test được yêu cầu, chấm d_1914–d_1941 bằng context Test đã xác minh và tạo báo cáo mới trong namespace mới.

## Vấn đề và giới hạn còn tồn tại

- Chưa có model, prediction hoặc điểm thực nghiệm mới; chưa xác minh end-to-end qua một lần fit/predict thực tế vì bước này không được phép chạy. Hai context lịch sử và hàm metric đã được xác minh riêng.
- Các workflow nâng cao cũ và báo cáo chuyên biệt cần port như danh sách trên; notebook mới chỉ phục vụ chuẩn bị. Không tự đổi hàng loạt số ngày trong mã cũ rồi xem là đã migrate.
- Oracle scalar do dự án viết độc lập; chưa chạy scorer bên thứ ba trên dữ liệu forecast thật. Đối chiếu fixture trọng số chính thức đã hoàn tất.
- Test mới d_1914–d_1941 từng là Validation của nghiên cứu cũ. Không sử dụng lại artifacts giúp tránh leakage trong lần chạy mới, nhưng không thể khôi phục tính chưa từng được quan sát của cửa sổ này. Cần công bố giới hạn đó khi báo cáo nghiên cứu.
- Chưa cài Jupyter/ipykernel trong môi trường tối thiểu mới; đọc notebook tĩnh được. Pinned môi trường lịch sử vẫn ở legacy, môi trường mới có lock riêng.

## Bằng chứng cách ly và nguồn bất biến

`preparation/source_manifest.json`: hash trước sao chép của 1.185 tệp được giữ. `source_metadata_before.json`: tập tệp/kích thước/mtime của toàn bộ 29.454 tệp nguồn, kể cả môi trường bị loại khi copy. `isolation_audit.json`: đối chiếu lại SHA-256 nguồn, tính đủ của bản sao, dữ liệu raw bất biến và quét đường dẫn nguồn cả trong archive nén. `source_metadata_after_check.json`: so sánh toàn bộ metadata trước/sau bằng .NET ticks.

Chỉ đọc nguồn để sao chép, kiểm kê và băm chứng minh bất biến; không sửa, di chuyển hoặc xóa tệp nguồn. Mọi thay đổi và báo cáo nằm trong dự án đích. Không chạy training, tuning hoặc Test, không tạo model/prediction/score mới, không tái sử dụng kết quả cũ làm kết quả mới.

## Sửa mã hóa tiếng Việt — 24/11/2026

Bốn tệp mới tạo (README, báo cáo này, ARCHIVE_NOTICE và notebook chuẩn bị) bị mất dấu khi nội dung Unicode đi qua pipeline PowerShell dùng mã hóa ASCII. Các tệp đã được ghi lại từ nội dung tiếng Việt bằng byte UTF-8 tường minh, không đi qua bước chuyển mã đó.

Một đoạn chú giải vốn đã bị mất dấu trong artifact lịch sử `legacy/results/rf_experiments/rf_final_test/analysis_summary.json` cũng được sửa phần chữ. Mọi giá trị số và phần chữ còn lại được giữ nguyên. Biên bản sửa mã hóa lưu riêng trong `preparation/`; các biên bản chuẩn bị trước đó vẫn được giữ làm lịch sử. Thao tác này không thay đổi mã tính toán, dữ liệu, mô hình, dự báo hoặc điểm đánh giá.
