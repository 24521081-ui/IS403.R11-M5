# Mô tả biểu đồ theo WRMSSE 12 cấp M5

Bản cuối/cấu hình đại diện đề xuất là **C7**, được xác định sau khi chấm đủ WRMSSE validation C0–C7. WRMSSE được bổ sung sau khi nhóm đã xem Test trước đó. Các mô tả RMSSE cũ được giữ nguyên tại `.local_evidence/archive_wrmsse_20261124/reports/mo_ta_bieu_do.md`; không dùng chúng để diễn giải hình hiện tại.

## 1. Lịch sử WRMSSE validation

[Mở hình](bieu_do/lich_su_wrmsse_validation.png)

### 1. Chú thích dưới hình

Validation d_1886–d_1913, origin 1885, đủ C0–C7. Mỗi điểm là WRMSSE chung trên 12 cấp.

### 2. Cách đọc

Trục ngang là cấu hình, trục dọc WRMSSE; thấp hơn tốt hơn. Chấm hồng đánh dấu cấu hình đại diện, đường xanh nối lịch sử cấu hình.

### 3. Nhận xét

C7 đạt 0,582561332, thấp nhất trong đủ 8 cấu hình. C6 đạt 0,625032992; chênh lệch được tính từ điểm chưa làm tròn.

### 4. Câu nói khi thuyết trình

“Chọn theo validation trên cùng bộ chấm; Test được báo cáo riêng.”

Đối chiếu: [bảng số liệu](xep_hang_wrmsse_validation.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## 2. C6/C7 theo WRMSSE trên hai kỳ

[Mở hình](bieu_do/c6_c7_wrmsse_validation_test.png)

### 1. Chú thích dưới hình

Hai ô tách validation và Test; không gộp hoặc xếp hạng chéo hai kỳ.

### 2. Cách đọc

C6 chấm tròn xanh, C7 hình thoi hồng; trục dọc WRMSSE, thấp hơn tốt hơn. Δ là C7 trừ C6.

### 3. Nhận xét

Validation C6/C7: 0,625033/0,582561. Test C6/C7: 0,726174/0,652481. Kết luận được đọc theo đúng kỳ.

### 4. Câu nói khi thuyết trình

“Ưu tiên WRMSSE đã làm thay đổi cách nhìn từ tiêu chí RMSSE ngang nhau trước đây.”

Đối chiếu: [bảng số liệu](ket_qua_wrmsse_validation_test.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## 3. Phân tích 12 điểm cấp Test

[Mở hình](bieu_do/c6_c7_diem_12_cap_test.png)

### 1. Chú thích dưới hình

Test d_1914–d_1941, origin 1913, cùng dữ liệu và bộ trọng số.

### 2. Cách đọc

Trục dọc là 12 cấp; trục ngang là điểm có trọng số trong cấp, chưa chia 12. C6 xanh tròn, C7 hồng thoi; thấp hơn tốt hơn.

### 3. Nhận xét

WRMSSE chung C6/C7 là 0,726174/0,652481. Đây là trung bình 12 điểm cấp, không phải trung bình RMSSE không trọng số của 30.490 cặp.

### 4. Câu nói khi thuyết trình

“Mỗi cấp tổng hợp chuỗi theo ngày trước, tính RMSSE, gộp theo doanh thu; trung bình 12 điểm mới ra WRMSSE.”

Đối chiếu: [bảng số liệu](wrmsse_12_cap/test_C7.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## 4. Bản cuối trong 28 ngày Test

[Mở hình](bieu_do/ban_cuoi_test_28_ngay.png)

### 1. Chú thích dưới hình

Thực tế và dự báo C7 trong Test d_1914–d_1941, tổng ngày toàn hệ thống.

### 2. Cách đọc

Trục ngang mã ngày M5, trục dọc lượng bán/ngày; thực tế xám, dự báo hồng. Điểm WRMSSE trong ô được tính trên đủ 12 cấp, không chỉ trên đường tổng ngày.

### 3. Nhận xét

C7 dự báo 1177561,77 so với thực tế 1231764 đơn vị; sai lệch tổng -4,40%.

### 4. Câu nói khi thuyết trình

“Bản cuối có điểm WRMSSE tốt hơn nhưng tổng lượng vẫn thiếu trên Test.”

Đối chiếu: [bảng số liệu](ket_qua_cuoi.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## 5. Sai lệch tổng lượng validation

[Mở hình](bieu_do/lich_su_sai_lech_validation.png)

### 1. Chú thích dưới hình

Validation d_1886–d_1913, origin 1885; chỉ tiêu phân tích bổ sung.

### 2. Cách đọc

Trục ngang cấu hình; trục dọc sai lệch %. Vạch 0 là tổng lượng khớp thực tế; đỏ dương là dư, xanh âm là thiếu.

### 3. Nhận xét

C6 thiếu 3,47%, C7 thiếu 0,52%. Không dùng riêng sai lệch tổng để thay thế WRMSSE khi xếp hạng.

### 4. Câu nói khi thuyết trình

“Biểu đồ này giải thích tổng lượng; bảng xếp hạng chính dùng WRMSSE 12 cấp.”

Đối chiếu: [bảng số liệu](xep_hang_wrmsse_validation.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## 6. Sai lệch Test theo cửa hàng

[Mở hình](bieu_do/c6_c7_sai_lech_test_theo_cua_hang.png)

### 1. Chú thích dưới hình

Test d_1914–d_1941; mỗi cửa hàng 3.049 chuỗi, gộp toàn bộ 28 ngày.

### 2. Cách đọc

Trục dọc cửa hàng; trục ngang sai lệch có dấu %. C6 tròn xanh, C7 thoi hồng; vạch 0 là khớp tổng lượng. Độ lớn là trị tuyệt đối, không phải giá trị có dấu.

### 3. Nhận xét

TX_3: C6 −10,86%, C7 −8,47%; CA_3 chuyển từ −1,89% sang +1,19%. Dù đổi dấu tại CA_3, độ lớn sai lệch vẫn giảm.

### 4. Câu nói khi thuyết trình

“Kết quả cửa hàng giúp giải thích tổng lượng, không phải WRMSSE riêng từng cửa hàng.”

Đối chiếu: [bảng số liệu](sai_lech_test_theo_cua_hang.csv); script [build_report.py](../scripts/build_report.py). Điểm gốc tại `results/metrics/wrmsse_C0_C7_20261124/`, được chấm bằng [score_c7_wrmsse.py](../scripts/score_c7_wrmsse.py).

## Quy tắc trọng số và giới hạn

Validation dùng doanh thu thực tế d_1858–d_1885; Test dùng d_1886–d_1913. Doanh thu bằng lượng bán thực tế nhân giá đúng tuần, chuẩn hóa tổng trọng số bằng 1 trong mỗi cấp. Code cộng lịch sử, thực tế và dự báo theo cấp trước khi tính RMSSE; giữ scale sau leading zeros, rồi gộp bằng doanh thu và lấy trung bình 12 điểm cấp. Ở cấp 1 có một chuỗi nên trọng số trong cấp bằng 100%. Không đồng nhất điểm cấp với đóng góp WRMSSE: đóng góp bằng điểm cấp / 12.

RMSSE từng chuỗi và doanh thu/trọng số chi tiết được giữ trong `results/metrics/wrmsse_C0_C7_20261124/<ky>_<cau_hinh>/rmsse_tung_chuoi.csv`. Chưa có đánh giá bởi bộ chấm độc lập; chưa đủ dự báo của 6 thuật toán trong nhóm để xếp hạng liên mô hình.
