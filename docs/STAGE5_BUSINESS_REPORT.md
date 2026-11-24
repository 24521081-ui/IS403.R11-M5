# Báo cáo phân tích kinh doanh Random Forest · v2

**Phương pháp.** Error = Forecast − Actual. MAE và RMSE tính ở cấp item–store–day trừ khi ghi rõ. WAPE = 100 × tổng |Error| / tổng Actual. Bias = 100 × tổng Error / tổng Actual. Tổng Actual bằng 0 trả về N/A. Tổng sai số tuyệt đối và độ lớn lệch tổng là hai đại lượng riêng.

**Nhóm biết trước.** Quy mô bán dùng tổng đơn vị d_1886–d_1913. Hai ngưỡng tercile tính theo nội suy tuyến tính trên các chuỗi có bán. Giá trị bằng ngưỡng thuộc nhóm thấp hơn, chuỗi không bán có nhóm riêng. ABC dùng tỷ trọng từ trọng số cấp 12 của evaluator, tương đương tỷ trọng doanh thu units × sell_price ở cùng cửa sổ. Xếp giảm theo trọng số, hòa theo series_id tăng dần. Chuỗi làm vượt 80% hoặc 95% vẫn ở lớp trước khi vượt. Khi đã chạm ngưỡng trong sai số số học 10⁻¹², chuỗi kế tiếp thuộc lớp sau. Doanh thu bằng 0 vào C. Evaluator gán giá 0 khi thiếu giá và lượng bán bằng 0, đồng thời dừng nếu thiếu giá tại dòng có bán.

**Các ví dụ.** Chọn theo thứ tự trình bày và loại chuỗi đã dùng. Ví dụ gián đoạn có ít nhất một ngày bán, ưu tiên số ngày 0 và sau đó là lượng bán. Ví dụ WAPE thấp chỉ xét chuỗi có Actual ít nhất bằng trung vị của nhóm có Actual dương. Các tiêu chí bằng nhau dùng series_id tăng dần. Đây là minh họa hồi cứu trên Test. RMSSE từng chuỗi và contribution lấy từ evaluator chính thức, với trọng số từ trước Test.

**Tệp phân tích.** Bảng bổ sung nằm trong reports/tables với tiền tố business_v2_. Có thể mở trực tiếp bằng Excel hoặc Power BI. Workbook chính thức và Forecast_Detail giữ dữ liệu niêm phong.

[Notebook phân tích](../notebooks/09_business_insights_and_visualization.ipynb) · [Danh mục 17 nhóm hình và đối chiếu 24 hình v1](../reports/FIGURE_CATALOG.md) · [Workbook chính thức](../reports/RandomForest_Forecast_Actual.xlsx) · [Manifest phiên bản](../reports/manifest_business_v2.json).

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
