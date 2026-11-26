# Bàn giao bộ chấm WRMSSE cho nhóm M5

Ae ơi, từ trước tới giờ nhóm mình dùng **RMSSE trung bình không trọng số** để so các cấu hình. Cách đó cho mỗi cặp **sản phẩm × cửa hàng** ảnh hưởng ngang nhau khi tính điểm. Mấy kết quả đã chạy vẫn dùng để đối chiếu quá trình thử cấu hình nha.

Để bám sát cách chấm trong bài báo M5, nhóm mình dùng **WRMSSE trên đủ 12 cấp** làm tiêu chí chính. Ae đã lưu đủ dự báo đúng kỳ thì dùng lại để chấm, **không cần train lại**.

## 1. Ae cần làm gì?

| Kỳ của nhóm | Dùng để làm gì? | Mốc dự báo | 28 ngày tính trọng số doanh thu |
|---|---|---|---|
| **Validation** `d_1886–d_1913` | So các cấu hình của từng mô hình rồi chốt cấu hình đại diện | `d_1885` | `d_1858–d_1885` |
| **Test** `d_1914–d_1941` | So sánh 6 mô hình sau khi chốt cấu hình | `d_1913` | `d_1886–d_1913` |

1. **Rà lại dự báo validation của các cấu hình ae đã thử.** Ai thiếu file nào thì báo tui tên mô hình và cấu hình đó để mình xác định cần xuất lại dự báo hay chạy lại bước nào.
2. **Chấm validation và gửi tui tên cấu hình cùng điểm WRMSSE** để chốt cấu hình đại diện. Lệnh chung ở mục 4 hiện chạy **test**. Phần chạy validation cho CSV riêng của từng thành viên cần được bổ sung vào bộ chấm chung trước khi ae tự chạy.
3. **Sau khi chốt cấu hình, chấm file test** bằng lệnh bên dưới và gửi tui các tệp kết quả ở mục 5. Tui sẽ tổng hợp bảng xếp hạng 6 mô hình khi tất cả đã được chấm cùng kỳ và cùng cách tính.

## 2. WRMSSE tính như thế nào?

Mô hình vẫn dự báo **số lượng bán từng ngày cho từng sản phẩm tại từng cửa hàng**. Bộ chấm xử lý như sau.

1. Từ dự báo chi tiết, cộng lượng bán thực tế và dự báo theo từng ngày để tạo các chuỗi ở 12 cấp M5, từ sản phẩm × cửa hàng đến toàn hệ thống.
2. Tính **RMSSE cho từng chuỗi**. Mẫu số chuẩn hóa lấy từ lịch sử đến mốc dự báo của kỳ đang chấm. Khi tính mẫu số, bỏ các số 0 đứng trước lần bán dương đầu tiên của chuỗi.
3. Trong mỗi cấp, tính **tỷ trọng doanh thu** của mỗi chuỗi từ 28 ngày trước kỳ dự báo. Doanh thu là tổng của **số lượng bán thực tế × giá bán tương ứng**. Nhân RMSSE của từng chuỗi với tỷ trọng doanh thu của chuỗi đó, rồi cộng lại để ra điểm của cấp.
4. Lấy **trung bình 12 điểm cấp** để ra WRMSSE chung. **Điểm càng thấp càng tốt.**

Ví dụ, ở cấp cửa hàng, mình cộng lượng bán của tất cả sản phẩm trong từng cửa hàng theo ngày, tính RMSSE cho từng chuỗi cửa hàng, rồi gộp các điểm theo tỷ trọng doanh thu của 10 cửa hàng. Ở cấp toàn hệ thống chỉ có một chuỗi, nên trọng số trong cấp đó bằng 1.

Ae gửi **file dự báo chi tiết** để bộ chấm cộng lên đủ 12 cấp. Con số RMSSE trung bình trong bảng cũ chỉ dùng để đối chiếu.

## 3. Ae cần chuẩn bị file gì?

### Ba file M5 dùng chung

Cả nhóm dùng cùng một bộ dữ liệu. Đặt ba file trong `data/raw/` hoặc truyền thư mục khác qua `--data-dir`.

| File | Bộ chấm dùng để làm gì? |
|---|---|
| `sales_train_evaluation.csv` | Lấy mã sản phẩm, cửa hàng, bang, ngành hàng, nhóm hàng, lượng bán lịch sử và lượng bán thực tế của kỳ chấm. |
| `calendar.csv` | Ghép mã ngày M5 với tuần giá qua `wm_yr_wk`. |
| `sell_prices.csv` | Lấy giá theo sản phẩm, cửa hàng và tuần để tính doanh thu dùng đặt trọng số. |

Ba file này cần chuẩn bị riêng sau khi clone repo. Với ngày có phát sinh bán trong cửa sổ tính doanh thu, bộ chấm cần giá tương ứng.

### File dự báo riêng của mỗi mô hình

Dùng CSV có bốn cột sau.

| Cột | Nội dung |
|---|---|
| `item_id` | Mã sản phẩm |
| `store_id` | Mã cửa hàng |
| `target_day` | Ngày M5 thuộc kỳ đang chấm |
| `y_pred` | Số lượng bán dự báo |

Mỗi kỳ đầy đủ có **30.490 cặp × 28 ngày = 853.720 dòng**. Mỗi bộ `item_id–store_id–target_day` xuất hiện đúng một lần. Giữ nguyên giá trị dự báo, kể cả phần thập phân. Bộ chấm kiểm tra độ phủ và báo lỗi nếu thiếu khóa, trùng khóa hoặc giá trị dự báo không hợp lệ.

Ae lưu dự báo thành một CSV hoặc nhiều CSV theo cửa hàng đều được. Nếu chia nhiều file, truyền các đường dẫn sau `--forecast` để script gộp và kiểm tra chung.

## 4. Chạy bộ chấm test

Mở PowerShell trong thư mục gốc của repo. Nếu đã có `.venv` cùng `numpy` và `pandas` thì chạy thẳng dòng chấm cuối.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "numpy>=1.26,<3" "pandas>=2.2,<3"
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/score_model_wrmsse.py --model MoHinhCuaToi --forecast data/processed/du_bao_test_MoHinhCuaToi.csv
```

Đổi `MoHinhCuaToi` và đường dẫn CSV thành tên mô hình và file thật của ae. Ví dụ trên dùng test `d_1914–d_1941`, origin 1913. Bản bộ chấm trong lượt bàn giao này là `WRMSSE_TEST_1913_V2`.

- `scripts/score_model_wrmsse.py` là script ae chạy.
- `scripts/score_c7_wrmsse.py` chứa hàm tính điểm dùng chung. Hai script cần nằm cùng thư mục.
- Nếu chạy lại và thư mục điểm đã có, dùng `--output-dir` để chọn thư mục mới. Bộ chấm giữ kết quả đã xuất trước đó.
- Ae dùng cùng phiên bản bộ chấm và cùng dữ liệu M5 để tui đối chiếu kết quả giữa các mô hình.

## 5. Chấm xong gửi tui file gì?

Với tên mẫu ở trên, kết quả mặc định nằm trong `.local_evidence/wrmsse_6_models/MoHinhCuaToi/`.

| File | Dùng để làm gì? |
|---|---|
| **`diem_chung.csv`** | Lấy WRMSSE chung để xếp hạng. |
| **`diem_theo_cap.csv`** | Xem điểm và phần đóng góp của từng cấp trong 12 cấp M5. |
| **`rmsse_theo_cap_san_pham.csv`** | Xem sai số từng cặp sản phẩm × cửa hàng và ghép để phân tích theo nhóm sản phẩm. |
| **`manifest.json`** | Kiểm tra kỳ chấm, phiên bản bộ chấm và mã kiểm tra dữ liệu, trọng số. |

Gửi tui **ba bảng CSV và `manifest.json`**. File dự báo chuẩn hóa `forecast_test.csv` giữ ngoài Git. Khi cần đối chiếu, ae gửi link file riêng.

Phần phân tích theo ngành hàng hoặc nhóm hàng M5 có thể dùng `cat_id` và `dept_id` trong dữ liệu bán hàng. Nếu nhóm muốn phân tích riêng theo **mức độ bán gián đoạn**, mình cần một bảng gán nhóm chung theo `item_id,store_id`, được tạo từ dữ liệu trước test.

## 6. Kết quả LightGBM Two-Part đã có

### Validation, dùng để so C0–C7

Đã chấm đủ tám cấu hình từ dự báo lưu sẵn. Mỗi cấu hình có **10 cửa hàng, 30.490 cặp sản phẩm × cửa hàng, 853.720 dòng dự báo và đủ 12 cấp**.

| Hạng | Cấu hình | WRMSSE validation |
|---:|---|---:|
| 1 | **C7 (λ = 0,75)** | **0,582561** |
| 2 | C6 | 0,625033 |
| 3 | C5 | 0,754339 |
| 4 | C2 | 0,764362 |
| 5 | C3 | 0,767454 |
| 6 | C0 | 0,768046 |
| 7 | C4 | 0,768366 |
| 8 | C1 | 0,780582 |

Theo WRMSSE validation, tui đề xuất **C7 (λ = 0,75)** đại diện cho LightGBM Two-Part. C7 thấp hơn C6 khoảng **0,042472 điểm**. Trước đó C6 được ưu tiên theo RMSSE trung bình không trọng số. Tiêu chí WRMSSE được bổ sung sau khi nhóm đã xem kết quả test trước đó. Căn cứ cho đề xuất C7 ở đây là **bảng validation**.

### Test, đối chiếu C6 và C7

| Cấu hình | WRMSSE test | Tổng thực tế | Tổng dự báo | Sai lệch tổng lượng |
|---|---:|---:|---:|---:|
| C6 | 0,726174 | 1.231.764 | 1.147.896,79 | −6,81% |
| **C7** | **0,652481** | 1.231.764 | 1.177.561,77 | **−4,40%** |

Sai lệch tổng lượng âm nghĩa là mô hình dự báo thiếu khi cộng toàn hệ thống. Bảng này đối chiếu **hai cấu hình của Two-Part trên test**. Bảng xếp hạng 6 mô hình sẽ được tổng hợp khi có đủ kết quả từ cả nhóm.

| Cấp | Nhóm chuỗi | Số chuỗi | Điểm cấp C6 | Điểm cấp C7 |
|---:|---|---:|---:|---:|
| 1 | Toàn hệ thống | 1 | 0,655983 | 0,515126 |
| 2 | Bang | 3 | 0,650406 | 0,536679 |
| 3 | Cửa hàng | 10 | 0,666674 | 0,578591 |
| 4 | Ngành hàng | 3 | 0,662992 | 0,539163 |
| 5 | Nhóm hàng | 7 | 0,722082 | 0,619003 |
| 6 | Bang × ngành hàng | 9 | 0,675343 | 0,577526 |
| 7 | Bang × nhóm hàng | 21 | 0,728361 | 0,649967 |
| 8 | Cửa hàng × ngành hàng | 30 | 0,692500 | 0,621901 |
| 9 | Cửa hàng × nhóm hàng | 70 | 0,745898 | 0,695634 |
| 10 | Sản phẩm | 3.049 | 0,850235 | 0,838412 |
| 11 | Bang × sản phẩm | 9.147 | 0,836268 | 0,831429 |
| 12 | Cửa hàng × sản phẩm | 30.490 | 0,827340 | 0,826345 |

Mỗi dòng là điểm có trọng số **trong cấp đó**. Lấy trung bình 12 dòng trước khi làm tròn mới ra WRMSSE chung.

### Mở bảng và hình ở đâu?

- [Bảng xếp hạng WRMSSE validation](../reports/xep_hang_wrmsse_validation.csv)
- [Bảng điểm validation và test](../reports/ket_qua_wrmsse_validation_test.csv)
- [Kết quả C7](../reports/ket_qua_cuoi.csv)
- [Điểm 12 cấp test C6](../reports/wrmsse_12_cap/test_C6.csv) và [điểm 12 cấp test C7](../reports/wrmsse_12_cap/test_C7.csv)
- [Biểu đồ lịch sử WRMSSE validation](../reports/archive/wrmsse_C0_C7_20261124/bieu_do/lich_su_wrmsse_validation.png)
- [Biểu đồ so sánh C6–C7 trên 12 cấp test](../reports/archive/wrmsse_C0_C7_20261124/bieu_do/c6_c7_diem_12_cap_test.png)
- [Cách đọc và nhận xét từng biểu đồ](../reports/mo_ta_bieu_do.md)

## 7. Kiểm tra đã làm và phần còn lại

- Đã chấm **8 cấu hình validation và 2 cấu hình test** bằng dự báo đã lưu. Mỗi lượt đủ **853.720 khóa duy nhất**, **12 cấp** và **42.840 chuỗi sau tổng hợp**.
- Đã kiểm tra tổng trọng số mỗi cấp bằng 1 và WRMSSE khớp trung bình 12 điểm cấp, trong dung sai `10⁻¹²`.
- Lượt kiểm tra toàn repo đạt **30 kiểm thử**. Bộ tệp chia sẻ chạy riêng được **9 kiểm thử của bộ chấm** và tái lập WRMSSE test của C7 là **0,652481319570**.
- Điểm hiện tại đã được đối chiếu trong dự án. Phần so với bộ chấm độc lập hoặc leaderboard M5 vẫn chưa thực hiện.
- Để tái lập lượt chấm lô Two-Part, cần có dự báo gốc và manifest trên máy. `results/metrics/wrmsse_C0_C7_20261124/manifest.json` ghi thông tin đối chiếu của lượt đã chạy.

Lệnh chấm lô và dựng báo cáo Two-Part đã dùng trên máy của tui:

```powershell
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe scripts/score_wrmsse_configurations.py --output-dir results/metrics/wrmsse_C0_C7_20261124
.\.venv\Scripts\python.exe scripts/build_report.py --scores-dir results/metrics/wrmsse_C0_C7_20261124
```

Thư mục kết quả trên đã tồn tại. Khi tái lập, dùng một thư mục mới cho `--output-dir`, rồi truyền cùng đường dẫn đó vào `--scores-dir`.

## 8. Tệp đề xuất chia sẻ qua Git

Đây là **31 tệp đề xuất** để ae chạy bộ chấm và đọc báo cáo. Danh sách tệp thực tế trong commit sẽ được duyệt riêng vì nhánh hiện vẫn kế thừa tệp từ commit trước.

```text
.gitignore
README.md
requirements.txt
docs/WRMSSE_BAN_GIAO.md
scripts/score_model_wrmsse.py
scripts/score_c7_wrmsse.py
scripts/score_wrmsse_configurations.py
scripts/build_report.py
tests/test_wrmsse.py
reports/ket_qua_cuoi.csv
reports/ket_qua_wrmsse_validation_test.csv
reports/mo_ta_bieu_do.md
reports/sai_lech_test_theo_cua_hang.csv
reports/tep_da_luu_tru.csv
reports/xep_hang_wrmsse_validation.csv
reports/archive/wrmsse_C0_C7_20261124/bieu_do/ban_cuoi_test_28_ngay.png
reports/archive/wrmsse_C0_C7_20261124/bieu_do/c6_c7_diem_12_cap_test.png
reports/archive/wrmsse_C0_C7_20261124/bieu_do/c6_c7_sai_lech_test_theo_cua_hang.png
reports/archive/wrmsse_C0_C7_20261124/bieu_do/c6_c7_wrmsse_validation_test.png
reports/archive/wrmsse_C0_C7_20261124/bieu_do/lich_su_sai_lech_validation.png
reports/archive/wrmsse_C0_C7_20261124/bieu_do/lich_su_wrmsse_validation.png
reports/wrmsse_12_cap/validation_C0.csv
reports/wrmsse_12_cap/validation_C1.csv
reports/wrmsse_12_cap/validation_C2.csv
reports/wrmsse_12_cap/validation_C3.csv
reports/wrmsse_12_cap/validation_C4.csv
reports/wrmsse_12_cap/validation_C5.csv
reports/wrmsse_12_cap/validation_C6.csv
reports/wrmsse_12_cap/validation_C7.csv
reports/wrmsse_12_cap/test_C6.csv
reports/wrmsse_12_cap/test_C7.csv
```

Ba CSV M5 gốc, model đã train, dự báo chi tiết, cache và kết quả trung gian giữ trên máy. `.gitignore` giúp Git bỏ qua các tệp này khi chúng chưa được theo dõi. Bộ chấm test chỉ cần hai script chấm, thư viện `numpy` và `pandas`, ba file M5 và CSV dự báo của ae.