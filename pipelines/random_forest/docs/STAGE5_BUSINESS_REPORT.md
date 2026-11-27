# Báo cáo phân tích kinh doanh Random Forest · v2

**Phương pháp.** Error = Forecast − Actual. MAE và RMSE tính ở cấp item–store–day trừ khi ghi rõ. WAPE = 100 × tổng |Error| / tổng Actual. Bias = 100 × tổng Error / tổng Actual. Tổng Actual bằng 0 trả về N/A. Tổng sai số tuyệt đối và độ lớn lệch tổng là hai đại lượng riêng.

**Nhóm biết trước.** Quy mô bán dùng tổng đơn vị d_1886–d_1913. Hai ngưỡng tercile tính theo nội suy tuyến tính trên các chuỗi có bán. Giá trị bằng ngưỡng thuộc nhóm thấp hơn, chuỗi không bán có nhóm riêng. ABC dùng tỷ trọng từ trọng số cấp 12 của evaluator, tương đương tỷ trọng doanh thu units × sell_price ở cùng cửa sổ. Xếp giảm theo trọng số, hòa theo series_id tăng dần. Chuỗi làm vượt 80% hoặc 95% vẫn ở lớp trước khi vượt. Khi đã chạm ngưỡng trong sai số số học 10⁻¹², chuỗi kế tiếp thuộc lớp sau. Doanh thu bằng 0 vào C. Evaluator gán giá 0 khi thiếu giá và lượng bán bằng 0, đồng thời dừng nếu thiếu giá tại dòng có bán.

**Các ví dụ.** Chọn theo thứ tự trình bày và loại chuỗi đã dùng. Ví dụ gián đoạn có ít nhất một ngày bán, ưu tiên số ngày 0 và sau đó là lượng bán. Ví dụ WAPE thấp chỉ xét chuỗi có Actual ít nhất bằng trung vị của nhóm có Actual dương. Các tiêu chí bằng nhau dùng series_id tăng dần. Đây là minh họa hồi cứu trên Test. RMSSE từng chuỗi và contribution lấy từ evaluator chính thức, với trọng số từ trước Test.

**Tệp phân tích.** Các bảng trong reports/tables dùng tiền tố business_v2_ và demand_group_. Workbook trình bày giá trị dự báo đã khóa, kèm hai thuộc tính phân tích ở cuối Forecast_Detail.

[Notebook phân tích](../notebooks/09_business_insights_and_visualization.ipynb) · [Danh mục 22 nhóm hình kinh doanh](../reports/FIGURE_CATALOG.md) · [Workbook](../reports/RandomForest_Forecast_Actual.xlsx) · [Manifest V2](../reports/manifest_business_v2.json) · [Manifest phân tích nhu cầu](../reports/manifest_demand_groups.json).

## 1. Mục tiêu và phạm vi

**Câu hỏi.** Kết quả nào có thể dùng để ưu tiên theo dõi hàng bán lẻ?

- Phạm vi Test 25/04/2016–22/05/2016, 30.490 chuỗi và 853.720 dòng. Forecast có gốc d_1913.
- Đọc Actual, hai forecast và trọng số từ artifact niêm phong. Các nhóm biết trước dùng lịch sử đến forecast origin, còn xếp hạng ngoại lệ mô tả riêng cửa sổ Test.

**Ứng dụng.** Dùng bảng tổng hợp để xác định điểm cần rà soát, sau đó mở bảng chi tiết theo chuỗi.

## 2. Tổng quan

**Câu hỏi.** Mô hình mới đem lại mức cải thiện nào và còn lệch bao nhiêu?

- WRMSSE RF **0,703912** so với **0,869701** của Seasonal Naive, giảm **19,06%**.
- RF dự báo **1.178.455** trên **1.231.764** đơn vị đã bán, lệch **-53.309** đơn vị.

**Ứng dụng.** Dùng cả sai số chi tiết và độ lệch tổng lượng khi đánh giá phương án kế hoạch.

[Hình dashboard](../reports/figures/business_v2_dashboard.png).

## 3. So sánh hiệu quả mô hình

**Câu hỏi.** Mức cải thiện có nhất quán giữa các chỉ số và cấp tổng hợp?

- WAPE giảm từ **86,22%** xuống **75,20%**, giảm **11,02 điểm %**. Bias RF là **-4,33%**.
- RF có WRMSSE thấp hơn ở **12/12 cấp**. Cấp 12 (Sản phẩm × cửa hàng) giảm tuyệt đối nhiều nhất, **0,240** điểm chuẩn hóa.
- Điểm trong cấp bằng contribution cấp nhân 12. Tổng contribution của các cấp cho WRMSSE toàn hệ thống.

**Ứng dụng.** Dùng WRMSSE M5 để đối chiếu thí nghiệm và WAPE/Bias để theo dõi sai số vận hành theo đúng độ phân giải.

[Hình model_metrics](../reports/figures/business_v2_model_metrics.png) · [Hình hierarchy](../reports/figures/business_v2_hierarchy.png) · [model_comparison](../reports/tables/model_comparison.csv).

## 4. Nhu cầu theo thời gian

**Câu hỏi.** Khi nào chênh lệch tích lũy và nhịp tuần thay đổi?

- Đỉnh bán là **15/05/2016** với **59.921** đơn vị. Ngày có sai lệch RF lớn nhất là **15/05/2016**, lệch **-9.558** đơn vị.
- Đến cuối kỳ, sai lệch tích lũy của RF là **-53.309**, của SN là **-90.648** đơn vị. Sai lệch trái dấu có thể bù trừ trong tổng tích lũy.
- Tuần **3** bán nhiều nhất (**326.832**), tuần **3** có Bias RF âm nhất (**-8,16%**), tuần **1** có WAPE RF cao nhất (**76,86%**).
- **Chủ nhật** dẫn đầu bình quân với **53.225** đơn vị/ngày. Tương quan Pearson giữa tổng ngày dự báo và thực tế là **0,919** (RF) và **0,888** (SN).

**Ứng dụng.** Rà soát lịch nhập hàng theo tuần và ngày bán cao, kiểm tra lượng giao trước khi điều chỉnh kế hoạch.

[Hình daily](../reports/figures/business_v2_daily.png) · [Hình weekly](../reports/figures/business_v2_weekly.png) · [Hình weekday](../reports/figures/business_v2_weekday.png).

## 5. Bang và cửa hàng

**Câu hỏi.** Nơi nào cần ưu tiên do vừa có quy mô bán lớn vừa khó dự báo?

- **CA** chiếm **42,31%** lượng bán. RF tại bang này lệch **-22.034** đơn vị.
- **CA_4** có WAPE RF cao nhất (**90,00%**). **CA_2** có độ lệch tổng lớn nhất (**-12.418** đơn vị).
- Nhóm quy mô cao và WAPE cao gồm **CA_1, TX_3**, theo trung vị **115.934** đơn vị và **74,99%**. Giá trị bằng ngưỡng thuộc phía cao.

**Ứng dụng.** Rà soát trước nhóm cao/cao, duy trì theo dõi nhóm cao/thấp, xem nguyên nhân cục bộ ở nhóm thấp/cao và kiểm tra định kỳ nhóm thấp/thấp. Phân nhóm mô tả Test theo lượng bán và WAPE.

[Hình regional](../reports/figures/business_v2_regional.png) · [Hình store_priority](../reports/figures/business_v2_store_priority.png) · [state_analysis](../reports/tables/state_analysis.csv) · [business_v2_store_priority](../reports/tables/business_v2_store_priority.csv).

## 6. Ngành hàng và bộ phận

**Câu hỏi.** Ngành nào chi phối tổng lượng và bộ phận nào có ngoại lệ?

- FOODS chiếm **67,59%** thực tế, với lệch RF **-48.337** đơn vị.
- **HOBBIES_2** có WAPE RF **141,65%** so với **139,02%** của SN.
- Ô cao nhất theo WAPE là **CA_4 × HOBBIES_2** (**197,15%**, 608 đơn vị thực tế). Ô Bias âm nhất là **WI_3 × FOODS_1** (**-22,22%**).

**Ứng dụng.** Xem quy mô thực tế và sai số tuyệt đối cùng tỷ lệ, nhất là các tổ hợp có lượng bán ít.

[Hình category](../reports/figures/business_v2_category.png) · [Hình heatmaps](../reports/figures/business_v2_heatmaps.png) · [department_analysis](../reports/tables/department_analysis.csv).

## 7. Sản phẩm và danh mục ABC

**Câu hỏi.** Sai số tập trung ở đâu và có thể phân nhóm từ lịch sử ra sao?

- **5.704**, **15.714**, **21.063** chuỗi đầu tiên tạo ít nhất 50%, 80% và 90% sai số tuyệt đối RF. Mốc 80% tương ứng **51,54%** số chuỗi.
- Top 15 contribution chiếm **3,82%** contribution cấp 12, theo RMSSE nhân trọng số doanh thu lịch sử.
- Phân vị của chuỗi có bán lịch sử là **11,0** và **31,0** đơn vị. WAPE RF của nhóm chưa bán trong 28 ngày lịch sử là **119,06%**, của nhóm cao là **59,53%**.
- ABC A gồm **10.715** chuỗi và **80,00%** tỷ trọng doanh thu lịch sử. WAPE RF của A là **58,96%**. Ở C, WAPE RF **135,83%** cao hơn SN **125,40%**.

**Ứng dụng.** Ưu tiên chuỗi doanh thu lịch sử cao hoặc contribution lớn. Giữ danh sách Pareto riêng khi mục tiêu là giảm tổng sai số đơn vị.

[Hình concentration](../reports/figures/business_v2_concentration.png) · [Hình segments](../reports/figures/business_v2_segments.png) · [business_v2_pareto_thresholds](../reports/tables/business_v2_pareto_thresholds.csv) · [business_v2_volume_segments](../reports/tables/business_v2_volume_segments.csv) · [business_v2_abc](../reports/tables/business_v2_abc.csv).

## 8. Ngoại lệ cấp chuỗi

**Câu hỏi.** Sai lệch lớn nhất về số lượng nằm ở chuỗi nào?

- RF thiếu nhiều nhất ở **FOODS_3_234 × WI_2** (**-988** đơn vị), và thừa nhiều nhất ở **FOODS_3_362 × CA_3** (**+694** đơn vị).
- Ví dụ bán gián đoạn **HOBBIES_2_124 × WI_2** có **27** ngày không bán trong 28 ngày. Panel riêng cho thấy hình dạng chuỗi trước khi so sánh WAPE.

**Ứng dụng.** Đối chiếu lịch bán, trạng thái mặt hàng và dữ liệu cửa hàng ở những chuỗi cực trị. Đây là tín hiệu kiểm tra dự báo.

[Hình exceptions](../reports/figures/business_v2_exceptions.png) · [Hình representatives](../reports/figures/business_v2_representatives.png) · [business_v2_exceptions](../reports/tables/business_v2_exceptions.csv) · [business_v2_representatives](../reports/tables/business_v2_representatives.csv).

## 9. Sai số và độ ổn định

**Câu hỏi.** Lợi thế RF có xuất hiện nhất quán ở từng chuỗi và từng kiểu ngày bán?

- RF có MAE thấp hơn ở **18.161** chuỗi (**59,56%**); tương đương **15**; SN thấp hơn **12.314**. Tolerance tuyệt đối **1e-09** đơn vị mỗi quan sát.
- **29.678** chuỗi có WAPE xác định và **812** chuỗi có tổng Actual bằng 0 nên để N/A. Phân vị 95 sai số tuyệt đối theo dòng lần lượt **3,66** (RF) và **5,00** đơn vị (SN).
- Trên **464.725** dòng Actual = 0, MAE RF **0,642** so với **0,540** của SN. Trên dòng có bán, MAE là **1,614** và **2,085**.

**Ứng dụng.** Theo dõi riêng chuỗi bán gián đoạn và so sánh với baseline theo nhóm, nhất là ở ngày không bán.

[Hình stability](../reports/figures/business_v2_stability.png) · [business_v2_zero_actual](../reports/tables/business_v2_zero_actual.csv) · [business_v2_series_wins](../reports/tables/business_v2_series_wins.csv).

## 10. SNAP

**Câu hỏi.** Sau khi tính theo số ngày, lượng bán và chất lượng thay đổi ra sao?

- CA có **10** ngày SNAP bật và **18** ngày tắt. Lượng bán bình quân **19.070** và **18.359** đơn vị/ngày.
- Ở CA, WAPE RF lần lượt **75,53%** và **75,16%**; trong FOODS bình quân là **12.520** và **11.909** đơn vị/ngày.

**Ứng dụng.** Dùng lịch SNAP để phân lớp kiểm tra lịch bán. So sánh theo bang có thể chịu ảnh hưởng của thứ trong tuần và cơ cấu SKU.

[Hình snap](../reports/figures/business_v2_snap.png) · [business_v2_snap](../reports/tables/business_v2_snap.csv).

## 11. Giải thích mô hình

**Câu hỏi.** Rừng sử dụng dữ liệu lịch sử và SNAP như thế nào?

- Rolling Mean chiếm **65,58%** importance, SNAP chiếm **0,45%**.
- Trên Validation, FE-01 đạt **0,664771**, FE-04 đạt **0,623931**, chênh **0,040840** WRMSSE trong lần so sánh hai tập đặc trưng.

**Ứng dụng.** Giữ kiểm tra chất lượng lịch và lags. Importance theo impurity nhạy với số điểm chia và tương quan, chỉ mô tả mức sử dụng đặc trưng.

[Hình importance](../reports/figures/business_v2_importance.png).

## 12. Hàm ý ứng dụng

**Câu hỏi.** Bắt đầu theo dõi ở đâu trong quy trình vận hành?

- Dùng danh sách **CA_1, TX_3** để ưu tiên rà soát dự báo và dùng **CA_2** để kiểm tra chênh lệch tổng lượng.
- Đối chiếu nhóm ABC A và top contribution trước khi chọn chuỗi cần theo dõi. Nhóm A tương ứng **80,00%** doanh thu lịch sử trong cửa sổ trọng số.

**Ứng dụng.** Kết hợp tồn kho khả dụng, lead time, chi phí và mức dịch vụ trước khi quyết định lượng nhập hoặc phân bổ.

## 13. Hạn chế và kết luận

**Câu hỏi.** Kết quả dùng được đến đâu?

- Test gồm 28 ngày và WAPE RF ở độ chi tiết item–store–day vẫn là **75,20%**. Cửa sổ này từng là Validation trong nghiên cứu Mean RMSSE cũ.
- Lợi thế WRMSSE **19,06%** là bằng chứng so với Seasonal Naive trong cửa sổ này. Lượng bán, SNAP và các nhóm minh họa chỉ hỗ trợ phân tích mô tả.

**Ứng dụng.** Kiểm tra lại trên các forecast origin khác và ghép dữ liệu vận hành trước khi quy đổi sai lệch dự báo thành tác động tài chính.

## 14. Phân tích hiệu quả dự báo theo đặc điểm nhu cầu

### ADI, gốc dự báo và kiểm tra dữ liệu

ADI đo khoảng cách bình quân giữa các ngày có bán. Với từng cặp sản phẩm – cửa hàng, lấy số ngày từ **lần bán dương đầu tiên đến hết ngày gốc**, kể cả hai đầu, chia cho số ngày có bán trong khoảng đó. Ví dụ, nếu giai đoạn này dài 100 ngày và có 25 ngày bán dương, ADI bằng 4 ngày. Nhãn Frequent (thường xuyên) ứng với ADI < 1,32 và Intermittent (gián đoạn) ứng với ADI ≥ 1,32. Hai mẫu đều không có chuỗi đúng tại ngưỡng, nên dữ liệu tự nó không phân biệt được hai quy ước tại điểm bằng.

| Gốc | Kỳ sử dụng nhãn | Thường xuyên | Gián đoạn | Tổng chuỗi | Trung vị ADI thường xuyên / gián đoạn |
|---|---|---:|---:|---:|---:|
| d_1885 | Validation, d_1886–d_1913 | 2.749 | 27.741 | 30.490 | 1,190 / 3,010 |
| d_1913 | Test, d_1914–d_1941 | 2.754 | 27.736 | 30.490 | 1,190 / 3,005 |

Mỗi CSV gốc có năm cột định danh dạng chuỗi, ADI dạng số thực và nhãn dạng chuỗi. Cả hai phủ đúng danh mục chính thức, bao gồm bang, ngành và bộ phận. Số khóa kép trùng, ô trống và ADI vô hạn đều bằng 0. Bảng tổng hợp có bốn dòng duy nhất theo gốc × nhóm, gồm hai cột số nguyên, bốn cột số thực và một cột nhãn. Số chuỗi, tỷ trọng, trung vị và tỷ trọng doanh thu khớp đối chiếu. Giữa hai gốc, 27 chuỗi chuyển từ Frequent sang Intermittent và 32 chuyển ngược lại. Có 2.722 chuỗi giữ nhãn Frequent và 27.709 chuỗi giữ nhãn Intermittent.

Việc tái tính từ doanh số d_1–d_1885 và d_1–d_1913 tái tạo đúng toàn bộ 60.980 giá trị ADI. Kiểm tra xác minh được giá trị và cửa sổ lịch sử cần thiết để tạo nhãn, dù ba CSV không kèm mã của tác giả. Nhãn Test được xác định độc lập với lượng bán từ d_1914 trở đi.

Hai phép ghép trái dùng khóa item_id và store_id, kiểm tra quan hệ nhiều–một và đối chiếu từng cột gốc theo thứ tự. Validation lấy dự báo FE-04 và Seasonal Naive đã lưu cùng doanh số Validation. Test dùng trực tiếp test_actual_and_forecast.parquet của Stage 4. Bảng chi tiết bổ sung có 853.720 dòng, 28 ngày cho mỗi chuỗi và 100% nhãn. ADI và nhãn nhóm chỉ xuất hiện sau dự báo. [Bảng chi tiết Test](../reports/tables/demand_group_test.parquet)

### Cơ cấu chuỗi và đóng góp kinh doanh

**Câu hỏi.** Một nhóm chiếm bao nhiêu danh mục, lượng bán và doanh thu?

| Chỉ tiêu | Frequent | Intermittent |
|---|---:|---:|
| Chuỗi (% danh mục) | 2.754 (9,03%) | 27.736 (90,97%) |
| Dòng Test | 77.112 | 776.608 |
| ADI trung bình / trung vị (ngày) | 1,183 / 1,190 | 4,562 / 3,005 |
| Lượng bán Test (% tổng lượng) | 446.854 (36,28%) | 784.910 (63,72%) |
| Doanh thu lịch sử 28 ngày (% tổng doanh thu) | 1.043.215,92 (27,83%) | 2.704.964,13 (72,17%) |
| Ngày không bán (% dòng trong nhóm) | 13,25% | 58,52% |
| Đơn vị bán mỗi chuỗi trong 28 ngày | 162,26 | 28,30 |
| Đơn vị trên mỗi ngày có bán của một chuỗi | 6,68 | 2,44 |

Nhóm Frequent chỉ chiếm khoảng một phần mười số chuỗi nhưng đóng góp hơn một phần ba lượng bán Test. Nhóm Intermittent vẫn chiếm phần lớn doanh thu lịch sử dù hơn nửa số ngày Test có lượng bán bằng 0. Khi phân bổ thời gian rà soát danh mục, nên kết hợp ADI với tỷ trọng doanh thu hoặc nhóm ABC. Doanh thu trong bảng thuộc d_1886–d_1913, còn lượng bán thuộc 28 ngày kế tiếp. [Bảng theo hai gốc](../reports/tables/demand_group_summary.csv) · [Hình cơ cấu](../reports/figures/demand_group_composition.png)

### Chất lượng dự báo và đóng góp cấp 12

**Câu hỏi.** Mô hình nào có sai số nhỏ hơn và kết quả có nhất quán giữa các chuỗi?

| Nhóm Test | Mô hình | MAE | RMSE | WAPE | Bias | Tổng dự báo | Tổng sai số tuyệt đối | Tổng sai lệch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frequent | Random Forest | 2,955 | 4,807 | 50,99% | −5,65% | 421.619,28 | 227.850,50 | −25.234,72 |
| Frequent | Seasonal Naive | 3,458 | 5,552 | 59,67% | −5,73% | 421.248 | 266.658 | −25.606 |
| Intermittent | Random Forest | 0,899 | 1,785 | 88,98% | −3,58% | 756.835,46 | 698.424,18 | −28.074,54 |
| Intermittent | Seasonal Naive | 1,024 | 2,195 | 101,33% | −8,29% | 719.868 | 795.346 | −65.042 |

Sai số có dấu bằng dự báo trừ lượng bán. MAE và RMSE tính từ các dòng chuỗi–ngày. WAPE và Bias dùng tổng lượng bán nhóm làm mẫu số cho tổng sai số tuyệt đối hoặc tổng sai lệch có dấu. Mức cải thiện tương đối bằng 100 × (chỉ số SN − chỉ số RF) / chỉ số SN. Khi đánh giá Bias dùng độ lớn của Bias. Mẫu số 0 cho kết quả không xác định.

Random Forest giảm WAPE tương đối **14,55%** ở Frequent và **12,19%** ở Intermittent. Với Intermittent, RF sai ít đơn vị hơn mỗi dòng nhưng sai nhiều hơn so với lượng bán. Cả hai nhóm đều có Bias RF âm. So sánh MAE của từng chuỗi trong 28 ngày, hòa nếu chênh lệch tuyệt đối ≤ 10⁻⁹. RF thắng / hòa / thua lần lượt **80,54% / 0,04% / 19,43%** ở Frequent và **57,48% / 0,05% / 42,47%** ở Intermittent. Trên Validation, WAPE RF / SN lần lượt là 50,91% / 63,98% và 91,81% / 103,29%. [Chỉ số hai gốc](../reports/tables/demand_group_metrics.csv) · [Hình so sánh](../reports/figures/demand_group_metrics.png)

| Nhóm Test | Tổng trọng số cấp 12 | Đóng góp RF | Đóng góp SN | RMSSE có trọng số trong nhóm RF | RMSSE có trọng số trong nhóm SN |
|---|---:|---:|---:|---:|---:|
| Frequent | 0,023194 | 0,019082 | 0,023074 | 0,822725 | 0,994836 |
| Intermittent | 0,060140 | 0,052644 | 0,068626 | 0,875370 | 1,141115 |

Mỗi đóng góp bằng tổng trọng số chính thức nhân RMSSE của từng chuỗi trong nhóm. Hai nhóm cộng thành **0,07172649** cho RF và **0,09170016** cho SN tại cấp 12. Chia đóng góp cho trọng số nhóm thu được RMSSE có trọng số trong nhóm. Tổng trọng số cấp 12 bằng 1/12. WRMSSE chính thức **đủ 12 cấp** là **0,7039124143** và **0,8697007023**. Các chuỗi ở cấp 1–11 có thể chứa cả hai nhóm và được giữ ở cấp tổng hợp ban đầu. Khi cần giải thích tác động lên điểm đủ 12 cấp, dùng cột đóng góp. Khi so chất lượng giữa hai nhóm, đọc cột đã chuẩn hóa theo trọng số nhóm.

### Cửa hàng và ngành hàng cần theo dõi

**Câu hỏi.** Cửa hàng và ngành hàng nào có WAPE cao nhất trong nhóm gián đoạn?

TX_1 và CA_4 đứng đầu theo WAPE RF của nhóm Intermittent, lần lượt **100,26%** và **99,91%**. Seasonal Naive tương ứng là 108,59% và 115,13%. Trong ba ngành, HOBBIES đạt **107,75%**, tiếp theo là HOUSEHOLD 98,52% và FOODS 81,57%. Có thể bắt đầu rà soát chuỗi tại hai cửa hàng và ngành HOBBIES, sau đó xếp hạng theo tổng sai số tuyệt đối để thấy quy mô đơn vị liên quan. Các thứ hạng này là hai lát cắt riêng của 28 ngày Test. [Bảng cửa hàng](../reports/tables/demand_group_store.csv) · [Bảng ngành](../reports/tables/demand_group_category.csv) · [Heatmap](../reports/figures/demand_group_region.png)

### Bốn chuỗi đại diện và các đỉnh bán

**Câu hỏi.** Sai số đến từ độ cao của đỉnh bán hay dự báo dương vào ngày không bán?

Trong mỗi nhóm, chọn chuỗi có WAPE RF gần nhất với phân vị 10% và 90% (nội suy tuyến tính). Hòa thì xếp item_id rồi store_id tăng dần. Quy tắc áp dụng cho toàn bộ 2.712 chuỗi Frequent và 26.966 chuỗi Intermittent có bán trong Test. Loại 42 và 770 chuỗi khỏi riêng phép chọn minh họa do WAPE chuỗi không xác định. Các chuỗi này vẫn có mặt trong chỉ số gộp và tỷ lệ thắng theo MAE. Việc chọn theo WAPE tuyệt đối diễn ra trước khi so sánh với Seasonal Naive.

| Chuỗi / cửa hàng | Nhóm, phân vị | Ngày bán bằng 0 | Đỉnh Actual | RF / SN vào ngày đỉnh đầu tiên |
|---|---|---:|---:|---:|
| HOUSEHOLD_1_334 / CA_2 | Frequent, P10 | 0 | 15 | 8,19 / 19 |
| FOODS_1_046 / WI_2 | Frequent, P90 | 8 | 17 | 4,01 / 10 |
| HOUSEHOLD_1_170 / WI_3 | Intermittent, P10 | 7 | 3 | 0,75 / 1 |
| FOODS_2_389 / TX_3 | Intermittent, P90 | 24 | 1 | 0,26 / 0 |

Ngày 13/05, FOODS_1_046 tại WI_2 bán 17 đơn vị trong khi RF dự báo 4,01. Với FOODS_2_389 tại TX_3, cả kỳ chỉ có bốn ngày bán đúng một đơn vị, còn RF phân bổ tổng cộng **7,33 đơn vị** vào 24 ngày không bán. Khi rà soát chuỗi, nên xem cả thời điểm bán và độ lớn đơn hàng. Bốn ví dụ minh họa cơ chế sai số quan sát được, không ước lượng tần suất bỏ lỡ đỉnh của toàn bộ danh mục. [Bảng chọn mẫu và ngày đỉnh](../reports/tables/demand_group_representatives.csv) · [Hình bốn chuỗi](../reports/figures/demand_group_examples.png)

### ADI và sai số quan sát được

**Câu hỏi.** Lợi thế của RF có còn khi khoảng cách giữa các lần bán dài hơn?

| Khoảng ADI | Chuỗi | WAPE RF / SN | MAE RF / SN |
|---|---:|---:|---:|
| Từ 1 đến dưới 1,32 | 2.754 | 50,99% / 59,67% | 2,955 / 3,458 |
| Từ 1,32 đến dưới 2 | 7.188 | 71,81% / 86,69% | 1,590 / 1,919 |
| Từ 2 đến dưới 4 | 10.518 | 99,66% / 113,86% | 0,859 / 0,981 |
| Từ 4 đến dưới 8 | 6.432 | 137,42% / 137,70% | 0,518 / 0,519 |
| Từ 8 trở lên, hữu hạn | 3.598 | 182,46% / 150,83% | 0,320 / 0,264 |

Các khoảng được ấn định theo ngưỡng phân nhóm và các mốc 2, 4, 8 ngày. Bảng nguồn dành dòng riêng cho ADI vô hạn và thiếu, đều có 0 chuỗi. **Tại ADI ≥ 8, Seasonal Naive có WAPE thấp hơn RF 31,63 điểm phần trăm.** Bias RF chuyển sang +32,32%, trong khi Bias SN là −13,53%. RMSE của RF vẫn thấp hơn (0,552 so với 0,701). Thứ hạng mô hình vì vậy phụ thuộc vào chỉ số sử dụng. Đọc WAPE cùng MAE và lượng bán để tránh đánh giá quá mức các tỷ lệ có mẫu số nhỏ. Các nhóm khác nhau cả về quy mô bán và cơ cấu mặt hàng, nên số liệu thể hiện mối liên hệ quan sát được. [Bảng khoảng ADI](../reports/tables/demand_group_adi.csv) · [Hình sai số theo ADI](../reports/figures/demand_group_adi.png)

### Ứng dụng và giới hạn

Ưu tiên theo dõi nhóm ADI ≥ 8 khi mục tiêu là giảm sai số tuyệt đối, nhất là khi WAPE RF vượt Seasonal Naive như trong cửa sổ này. Ở nghiên cứu sau, có thể so sánh Croston, hiệu chỉnh SBA và TSB trên nhiều gốc thời gian với cùng chân trời 28 ngày. TSB cập nhật xác suất phát sinh bán mỗi kỳ, kể cả các kỳ không bán. Đánh giá riêng độ chính xác của thời điểm phát sinh, độ lớn khi có bán và điểm đủ 12 cấp trước khi chọn phương pháp. [Syntetos và Boylan](https://doi.org/10.1016/j.ijforecast.2004.10.001) · [Teunter, Syntetos và Babai](https://doi.org/10.1016/j.ejor.2011.05.018)

Actual ghi lượng bán quan sát được. Muốn ước lượng nhu cầu chưa được đáp ứng, thiếu hàng, dư tồn hoặc giá trị tài chính, cần thêm dữ liệu khả dụng trên kệ, tồn kho, thời gian cung ứng và chi phí. [Danh mục hình](../reports/FIGURE_CATALOG.md) · [Workbook phân tích](../reports/RandomForest_Forecast_Actual.xlsx)

## 15. Random Forest theo nhóm nhu cầu và thời gian

### Sáu chỉ số trong toàn kỳ

Trên 28 ngày từ 25/04 đến 22/05/2016, Frequent có WAPE **50,99%** và Intermittent **88,98%**. Một dự báo Frequent sai trung bình 2,955 đơn vị trên mỗi chuỗi–ngày, so với 0,899 của Intermittent. RMSE tương ứng là 4,807 và 1,785 đơn vị. Do mức bán của hai nhóm khác nhau, thứ hạng theo số đơn vị sai khác với thứ hạng theo WAPE. Với kế hoạch phân bổ xuống từng mặt hàng, nhóm gián đoạn có sai số lớn hơn trên mỗi đơn vị bán. [Hình sáu chỉ số](../reports/figures/demand_group_metrics.png) · [Bảng chỉ số](../reports/tables/demand_group_metrics.csv)

Gọi A là lượng bán, F là dự báo và e = F − A trên một dòng sản phẩm–cửa hàng–ngày. Trong mỗi nhóm và khoảng thời gian, với N dòng, **MAE = Σ|e|/N**, **RMSE = √(Σe²/N)** (đơn vị bán trên một quan sát), **WAPE = 100 × Σ|e|/ΣA**, **Bias = 100 × Σe/ΣA** (%). Khi ΣA = 0, WAPE và Bias không xác định. Bias dương biểu thị dự báo thừa, âm biểu thị dự báo thiếu. Các tổng bao gồm cả dòng có A = 0. Nhãn tại d_1913 cố định trong suốt kỳ Test và chỉ dùng sau khi dự báo đã khóa.

Với chuỗi i, **RMSSEᵢ = √[(Σₜeᵢₜ²/28)/sᵢ]**, trong đó sᵢ là trung bình bình phương sai phân lượng bán liên tiếp từ lần bán dương đầu tiên đến d_1913. Trung bình cộng các RMSSE xác định là **0,757** cho 2.754 chuỗi Frequent và **0,776** cho 27.736 chuỗi Intermittent. **Weighted RMSSE cấp 12 trong nhóm = Σᵢ(wᵢ × RMSSEᵢ)/Σᵢwᵢ**, với wᵢ là trọng số cấp 12 theo doanh thu lịch sử 28 ngày trước Test. Kết quả là **0,823** và **0,875**. Cả hai chỉ số RMSSE đều không có đơn vị. Frequent có sai số chuẩn hóa thấp hơn trên cả hai cách tính. WRMSSE chính thức toàn hệ thống dùng đủ 12 cấp và bằng **0,7039124143**.

### Nhịp bán và đỉnh ngày 15/05

Tương quan giữa tổng Actual và Forecast hằng ngày là **0,908** ở Frequent và **0,923** ở Intermittent. Độ lệch chuẩn tổng lượng bán ngày chia trung bình 28 ngày lần lượt là **15,0%** và **15,2%**. Hai nhóm có mức biến động tương đối gần nhau sau khi gộp chuỗi. Trục Y riêng trên mỗi panel thể hiện quy mô khác nhau. Cùng ngày đỉnh 15/05, Frequent bán **21.622** đơn vị trong khi RF dự báo **17.899,85**. Intermittent bán **38.299** so với dự báo **32.463,25**. RF bám được nhịp lên xuống của tổng lượng nhưng đánh giá thấp quy mô đỉnh ở cả hai nhóm. [Hình lượng bán](../reports/figures/demand_group_sales.png) · [Bảng theo ngày](../reports/tables/demand_group_daily.csv)

### Sai lệch ngày và các đợt dự báo thiếu

Dự báo thừa nhiều nhất vào **08/05**, với **+1.808,87** đơn vị ở Frequent và **+3.707,28** ở Intermittent. Mức thiếu lớn nhất cùng rơi vào **15/05**, lần lượt **−3.722,15** và **−5.835,75** đơn vị. Frequent thiếu 24/28 ngày, gồm **14 ngày liên tiếp từ 09–22/05**. Intermittent thiếu 21/28 ngày, có hai đợt 9 ngày từ 29/04–07/05 và 09–17/05. Khi rà soát kế hoạch nhóm, các đợt cùng dấu cho thấy chênh lệch kéo dài qua nhiều lần bán. Mỗi cột là ΣF − ΣA của một ngày. Sai số trái dấu giữa các chuỗi có thể bù trừ trong cột này. [Hình sai lệch ngày](../reports/figures/demand_group_error.png) · [Bảng theo ngày](../reports/tables/demand_group_daily.csv)

### Chất lượng theo ngày

WAPE ngày dao động **47,09–57,50%** ở Frequent và **77,83–100,34%** ở Intermittent. Ngày 15/05, Intermittent có WAPE thấp nhất kỳ (**77,83%**) đồng thời có Bias âm sâu nhất (**−15,24%**). Frequent cũng đạt Bias thấp nhất ngày đó (**−17,21%**). Đối với ngày bán mạnh, WAPE cho biết độ lớn sai số tương đối trên các mặt hàng, còn Bias cho thấy kế hoạch tổng đã đánh giá thấp lượng bán bao nhiêu. [Hình WAPE và Bias ngày](../reports/figures/demand_group_daily_quality.png) · [Bảng theo ngày](../reports/tables/demand_group_daily.csv)

### Chất lượng theo tuần

Tuần 3 (09–15/05) có tổng lượng bán cao nhất ở cả hai nhóm. Bias đạt **−9,27%** ở Frequent và **−7,54%** ở Intermittent, âm sâu nhất trong bốn tuần. RMSE của Intermittent cũng cao nhất tuần 3 (**1,928** đơn vị), trong khi Frequent đạt RMSE cao nhất ở tuần 2 (**5,114**). Với kế hoạch tuần, tín hiệu cần theo dõi là lượng dự báo thiếu tập trung vào tuần cao điểm, dù WAPE tuần 3 của Intermittent thấp nhất kỳ (**86,59%**). [Hình bốn tuần](../reports/figures/demand_group_weekly.png) · [Bảng theo tuần](../reports/tables/demand_group_weekly.csv)

### Thứ trong tuần và ngày bán cao

Gộp trực tiếp sai số chuỗi–ngày của tám ngày cuối tuần và hai mươi ngày trong tuần. Với Intermittent, MAE tăng từ **0,856 lên 1,008** đơn vị vào cuối tuần, Bias chuyển từ **−2,92% sang −4,82%**, còn WAPE giảm từ **92,26% xuống 82,75%**. Frequent có cùng chiều thay đổi, MAE tăng từ **2,776 lên 3,401**, WAPE giảm từ **52,05% xuống 48,96%**. Như vậy, cuối tuần cần lưu ý độ lệch kế hoạch tính bằng đơn vị, trong khi tỷ lệ sai số trên lượng bán ở cả hai nhóm thấp hơn.

Chọn riêng bảy ngày có tổng Actual cao nhất của từng nhóm để kiểm tra ảnh hưởng của mức bán. Ở Intermittent, WAPE là **81,48%** so với **92,31%** trong 21 ngày còn lại, nhưng Bias âm sâu hơn (**−6,94%** so với **−2,09%**). Ở Frequent, Bias tương ứng **−8,38%** và **−4,44%**. Hiện tượng đánh giá thấp tổng lượng ngày bán mạnh xuất hiện ở cả hai nhóm trong cửa sổ Test này. Tiêu chí bảy ngày chỉ dùng để mô tả sau dự báo. [Hình theo thứ](../reports/figures/demand_group_weekday.png) · [Bảng theo thứ](../reports/tables/demand_group_weekday.csv) · [Bảng theo ngày](../reports/tables/demand_group_daily.csv)

### Chênh lệch kế hoạch tích lũy

Đến 22/05, Frequent tích lũy **446.854** đơn vị bán và **421.619,28** đơn vị dự báo, chênh **−25.234,72** (**−5,65%**). Intermittent tích lũy **784.910** và **756.835,46**, chênh **−28.074,54** (**−3,58%**). Intermittent đóng góp lượng chênh lớn hơn, còn Frequent có tỷ lệ thiếu dự báo lớn hơn. Cộng hai nhóm được **1.231.764** đơn vị bán và **1.178.454,74** đơn vị dự báo. Đây là chênh lệch giữa kế hoạch và lượng bán ghi nhận. Đánh giá thiếu hàng hoặc dư tồn thực tế cần dữ liệu tồn kho và khả năng phục vụ. [Hình tích lũy](../reports/figures/demand_group_cumulative.png) · [Bảng lũy kế theo ngày](../reports/tables/demand_group_daily.csv)

### Tra cứu một chuỗi

Phần cuối notebook 09 cho phép chọn Demand group, Item ID và Store ID, xem các chỉ số của chuỗi và xuất PNG cho lựa chọn đang xem. Hình bốn chuỗi ở phần 14 dùng chung quy tắc P10/P90 trong mỗi nhóm. Ở hai ví dụ P90, RF dự báo 4,01 trong ngày bán 17 đơn vị của chuỗi Frequent và phân bổ 7,33 đơn vị vào 24 ngày không bán của chuỗi Intermittent. Tra cứu từng chuỗi giúp phân biệt sai độ cao của đỉnh với sai thời điểm bán. [Notebook 09](../notebooks/09_business_insights_and_visualization.ipynb) · [Bốn chuỗi P10/P90](../reports/figures/demand_group_examples.png)
