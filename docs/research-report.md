# BÁO CÁO NGHIÊN CỨU THỰC NGHIỆM CÁC PHƯƠNG PHÁP ĐỐI CHỨNG THỐNG KÊ TRONG DỰ BÁO NHU CẦU BÁN LẺ DỮ LIỆU M5

**Chuyên đề:** Phân tích Dữ liệu Kinh doanh & Dự báo Chuỗi Thời gian Phân cấp  
**Bộ dữ liệu:** M5 Forecasting - Accuracy (Walmart Retail Dataset)  
**Phạm vi nghiên cứu:** Mô hình thống kê đối chứng (Seasonal Naïve, ES_bu, Croston, Croston–SBA)  

---

## 1. TỔNG QUAN BÀI TOÁN VÀ MỤC TIÊU NGHIÊN CỨU

### 1.1. Bối cảnh bài toán M5
Cuộc thi M5 Forecasting là một trong những chuẩn mực thực nghiệm quy mô lớn nhất trong lĩnh vực dự báo bán lẻ, khai thác dữ liệu từ tập đoàn Walmart bao gồm 30.490 chuỗi thời gian sản phẩm–cửa hàng (SKU-level) trải dài trên 10 siêu thị thuộc 3 bang (California, Texas, Wisconsin) qua hơn 5 năm lịch sử. 

Dữ liệu chuỗi thời gian bán lẻ tại cấp độ SKU sở hữu hai đặc tính cố hữu gây khó khăn nghiêm trọng cho công tác dự báo:
1. **Tính gián đoạn cao (Intermittent Demand):** 68,37% số quan sát bán hàng hằng ngày trên toàn bộ tập huấn luyện có giá trị bằng 0. Rất nhiều mặt hàng, đặc biệt là nhóm Hobbies và Household, chỉ phát sinh giao dịch sau nhiều ngày ngắt quãng.
2. **Tính mùa vụ chu kỳ ngắn (Weekly Seasonality):** Phân tích khám phá dữ liệu (EDA) ghi nhận doanh số bán lẻ vào hai ngày cuối tuần (Thứ Bảy và Chủ Nhật) bình quân cao hơn 31% so với các ngày thường trong tuần.

### 1.2. Mục tiêu và phạm vi của hợp phần đối chứng thống kê
Trong kiến trúc tổng thể của đề tài nghiên cứu so sánh mô hình, hợp phần này đảm nhiệm việc thiết lập các mốc đối chứng định lượng (statistical benchmarks) có cơ sở lý thuyết vững chắc, nhằm giải quyết các câu hỏi nghiên cứu cốt lõi:
- **Câu hỏi 1:** Các mô hình thống kê chuyên biệt cho dữ liệu gián đoạn (Croston, Croston–SBA) và dữ liệu có mùa vụ tuần kết hợp phân cấp (ES_bu) cải thiện được bao nhiêu phần trăm độ chính xác so với mốc cơ sở Seasonal Naïve?
- **Câu hỏi 2:** Hiệu quả tương đối giữa các phương pháp biến đổi như thế nào khi chuyển đổi giữa các ngành hàng có đặc tính tiêu dùng khác nhau (FOODS vs. HOBBIES vs. HOUSEHOLD) và giữa các nhóm chuỗi có mức độ bán ngắt quãng khác nhau (từ dưới 30% đến trên 70% số ngày bằng 0)?
- **Câu hỏi 3:** Hiệu chỉnh Syntetos-Boylan (SBA) có thực sự khắc phục được hiện tượng phóng đại dự báo (positive bias) của phương pháp Croston cổ điển trên dữ liệu thực tế hay không?
- **Câu hỏi 4:** Thiết lập các mốc sai số chuẩn tắc (MAE, RMSE, RMSSE) trên cùng một giao thức phân chia thời gian nghiêm ngặt để làm căn cứ thẩm định giá trị gia tăng của các mô hình học máy phức tạp hơn (như LightGBM-Tweedie, Random Forest và Two-part LightGBM).

---

## 2. CƠ SỞ LÝ THUYẾT VÀ THUẬT TOÁN DỰ BÁO

### 2.1. Mô hình Seasonal Naïve (Mốc đối chứng cơ sở)
Mô hình Seasonal Naïve giả định rằng nhu cầu tiêu thụ trong tương lai sẽ tái lập hoàn toàn quy luật của chu kỳ gần nhất. Với dữ liệu bán lẻ có chu kỳ tuần $m = 7$ ngày, dự báo cho bước $h$ ($h = 1, \dots, 28$) từ thời điểm gốc $T$ được xác định theo công thức:

$$\hat{y}_{T+h} = y_{T - m + ((h - 1) \pmod m) + 1}$$

- **Ưu điểm:** Không cần ước lượng tham số, chi phí tính toán bằng 0, nắm bắt được hiệu ứng chênh lệch giữa ngày trong tuần và cuối tuần.
- **Hạn chế:** Rất nhạy cảm với các đột biến ngẫu nhiên (outliers) hoặc các giá trị 0 ngẫu nhiên xuất hiện trong tuần cuối cùng của tập huấn luyện.

### 2.2. Phương pháp San bằng Mũ và Cơ chế Tổng hợp Bottom-Up (ES_bu)
Phương pháp San bằng Mũ Phân cấp Bottom-Up (ES_bu) kết hợp hai thành phần: mô hình san bằng mũ mùa vụ tại cấp chi tiết nhất và cơ chế cộng dồn nhất quán lên các cấp phân cấp cao hơn.

#### a) Mô hình San bằng Mũ Mùa vụ tại cấp sản phẩm–cửa hàng (Level 12)
Tại cấp độ cặp sản phẩm - cửa hàng, chuỗi được mô hình hóa thông qua phương trình phân rã mức (Level - $L_t$) và thành phần mùa vụ cộng (Seasonal - $S_t$) với chu kỳ $m = 7$:

$$L_t = \alpha (y_t - S_{t-m}) + (1 - \alpha) L_{t-1}$$

$$S_t = \gamma (y_t - L_t) + (1 - \gamma) S_{t-m}$$

Trong đó:
- $\alpha \in (0, 1)$ là hệ số làm mịn thành phần mức (chọn $\alpha = 0,2$).
- $\gamma \in (0, 1)$ là hệ số làm mịn thành phần mùa vụ (chọn $\gamma = 0,15$).
- Khởi tạo: Mức cơ sở ban đầu $L_0$ được tính bằng trung bình của chu kỳ đầu tiên kể từ ngày phát sinh lượng bán dương đầu tiên; các hệ số $S_1, \dots, S_7$ được chuẩn hóa xoay quanh 0.

Dự báo ngoài mẫu cho $h$ ngày tiếp theo:
$$\hat{y}_{T+h} = \max\left(0, L_T + S_{T + ((h - 1) \pmod m) + 1}\right)$$

#### b) Cơ chế Tổng hợp Phân cấp Bottom-Up
Dự báo được tạo lập độc lập tại cấp độ cơ sở (Level 12: cặp `item_id` và `store_id`). Khi cần phân tích hoặc báo cáo tại các cấp quản trị cao hơn (Level 3: Store, Level 5: Department, Level 1: Toàn quốc), dự báo được cộng dồn trực tiếp:

$$\hat{y}_{K, T+h} = \sum_{i \in \text{Nhóm } K} \hat{y}_{i, T+h}$$

Cơ chế Bottom-Up đảm bảo tính nhất quán đại số hoàn hảo (reconciliation coherence) giữa các cấp mà không gây ra hiện tượng mâu thuẫn số liệu.

### 2.3. Mô hình Croston và Hiệu chỉnh Syntetos-Boylan (Croston–SBA)

#### a) Mô hình Croston cổ điển (1972)
Đối với chuỗi thời gian gián đoạn, việc áp dụng trực tiếp các phương pháp làm mịn thông thường sẽ dẫn đến dự báo bị giảm mạnh sau các chuỗi ngày bằng 0 liên tiếp và tăng vọt khi xuất hiện đột biến giao dịch. 

Thuật toán Croston giải quyết vấn đề này bằng cách phân rã chuỗi quan sát $y_t$ thành hai biến ngẫu nhiên độc lập:
1. **Quy mô nhu cầu khi phát sinh ($z_t$):** Lượng hàng bán được tại các thời điểm $y_t > 0$.
2. **Khoảng cách giữa các lần phát sinh nhu cầu ($p_t$):** Số ngày tính từ lần bán dương liền trước.

Cơ chế cập nhật:
- Nếu $y_t = 0$:
  $$z'_t = z'_{t-1}, \quad p'_t = p'_{t-1}$$
- Nếu $y_t > 0$:
  $$z'_t = z'_{t-1} + \alpha (y_t - z'_{t-1})$$
  $$p'_t = p'_{t-1} + \alpha (t - t_{\text{prev}} - p'_{t-1})$$
  $$t_{\text{prev}} = t$$

Dự báo tỷ lệ nhu cầu trung bình hằng ngày cho mọi bước tương lai:
$$\hat{y}_{T+h}^{\text{Croston}} = \frac{z'_T}{p'_T}$$

#### b) Hiệu chỉnh Syntetos-Boylan (Croston–SBA, 2005)
Syntetos và Boylan (2005) đã chứng minh trên phương diện toán học rằng ước lượng tỷ số $\frac{z'_T}{p'_T}$ của Croston tồn tại độ chệch dương cố hữu: do $p'_t$ là một biến ngẫu nhiên, theo bất đẳng thức Jensen, kỳ vọng toán học $E[1/p'_t] > 1/E[p'_t]$, khiến mô hình Croston luôn có xu hướng dự báo vượt mức nhu cầu thực tế.

Hiệu chỉnh SBA đưa vào thừa số thu nhỏ $(1 - \frac{\alpha}{2})$ để triệt tiêu phần chệch này:

$$\hat{y}_{T+h}^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{z'_T}{p'_T}$$

Với hệ số $\alpha = 0,1$, dự báo của Croston–SBA sẽ giảm 5% so với Croston cổ điển: $\hat{y}^{\text{SBA}} = 0,95 \cdot \hat{y}^{\text{Croston}}$.

### 2.4. Đối chiếu với phương pháp tham khảo

| Mô hình | Giữ từ tài liệu tham khảo | Lựa chọn triển khai của nhóm |
| :--- | :--- | :--- |
| ES_bu | Dự báo ở cấp sản phẩm–cửa hàng rồi cộng lên 11 cấp còn lại của M5. | Benchmark M5 tự chọn mô hình san bằng mũ cho từng chuỗi. Nhóm dùng một cấu hình cố định gồm mùa vụ cộng chu kỳ 7 ngày, $\alpha = 0,2$ và $\gamma = 0,15$. Nhóm khởi tạo theo tuần đầu kể từ lần bán đầu tiên, dùng trung bình nếu lịch sử từ lần bán đầu dưới 7 ngày, trả về 0 cho chuỗi chưa bán và chặn dự báo âm. |
| Croston–SBA | Làm mịn riêng lượng bán dương và khoảng cách giữa các lần bán, rồi nhân tỷ số với hệ số hiệu chỉnh $1-\alpha/2$. | Nhóm cố định $\alpha = 0,1$ và dùng cùng một mức dự báo cho cả 28 ngày. Khoảng cách khởi tạo là 1 ngày. Nếu chỉ có một lần bán, nhóm dùng số ngày quan sát từ lần bán đó đến hết cửa sổ huấn luyện, tính cả ngày bán. Chuỗi chưa bán có dự báo bằng 0. |

Benchmark ES_bu của M5 dùng WRMSSE để đánh giá trên toàn bộ hệ phân cấp. Nhóm báo cáo MAE, RMSE và RMSSE trung bình ở cấp sản phẩm–cửa hàng trên hai cửa sổ 28 ngày. Tệp cộng gộp ES_bu lấy tổng trực tiếp từ các giá trị cấp sản phẩm–cửa hàng đã làm tròn 4 chữ số để đối chiếu được với tệp bàn giao.

Tham khảo [Makridakis, Spiliotis và Assimakopoulos (2022)](https://doi.org/10.1016/j.ijforecast.2021.11.013) và [Syntetos và Boylan (2005)](https://doi.org/10.1016/j.ijforecast.2004.10.001).

---

## 3. THIẾT KẾ THỰC NGHIỆM VÀ PHÂN CHIA TẬP DỮ LIỆU

### 3.1. Phân chia tập dữ liệu theo dòng thời gian
Nhằm ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (data leakage) và đảm bảo tính so sánh khách quan với các mô hình học máy của các thành viên khác trong nhóm, giao thức chia tập được thiết lập cố định như sau:

| Giai đoạn | Khoảng ngày | Điểm gốc ($T$) | Mục đích sử dụng |
| :--- | :---: | :---: | :--- |
| **Train ban đầu** | $d_1 \dots d_{1885}$ | $d_{1885}$ | Ước lượng tham số và trạng thái ban đầu của các mô hình |
| **Validation** | $d_{1886} \dots d_{1913}$ | $d_{1885}$ | Đánh giá sai số ngoài mẫu trên 28 ngày; so sánh cấu hình |
| **Huấn luyện lại** | $d_1 \dots d_{1913}$ | $d_{1913}$ | Cập nhật trạng thái mô hình trên toàn bộ dữ liệu trước kiểm định |
| **Test cuối** | $d_{1914} \dots d_{1941}$ | $d_{1913}$ | Đánh giá độc lập lần cuối cùng trên 28 ngày tương lai |

### 3.2. Tiêu chuẩn đánh giá sai số
Thực nghiệm sử dụng ba thước đo sai số bổ trợ cho nhau:

1. **Mean Absolute Error (MAE):** Đo lường mức độ sai lệch tuyệt đối trung bình (đơn vị: sản phẩm/ngày).
   $$\text{MAE} = \frac{1}{H} \sum_{h=1}^H |y_{T+h} - \hat{y}_{T+h}|$$

2. **Root Mean Squared Error (RMSE):** Đo lường độ lệch chuẩn của sai số, phạt nặng các sai số đột biến lớn.
   $$\text{RMSE} = \sqrt{\frac{1}{H} \sum_{h=1}^H (y_{T+h} - \hat{y}_{T+h})^2}$$

3. **Root Mean Squared Scaled Error (RMSSE - Chuẩn M5):** Thước đo chuẩn mực không phụ thuộc vào thứ nguyên hay quy mô của sản phẩm:
   $$\text{RMSSE} = \sqrt{\frac{\frac{1}{H} \sum_{h=1}^H (y_{T+h} - \hat{y}_{T+h})^2}{\frac{1}{T - t_{\text{first}} - 1} \sum_{t=t_{\text{first}}+1}^T (y_t - y_{t-1})^2}}$$
   - Mẫu số là sai số bình phương trung bình trong mẫu của phương pháp bước ngẫu nhiên (in-sample Naïve 1-step), tính từ ngày bán dương đầu tiên $t_{\text{first}}$.
   - Giá trị $\text{RMSSE} < 1,0$ chỉ ra rằng mô hình tạo ra dự báo tốt hơn phương pháp Naïve trong mẫu; giá trị càng nhỏ thì mô hình càng vượt trội.

### 3.3. Cấu trúc dữ liệu bàn giao (Standard Output Schema)
Toàn bộ dự báo ngoài mẫu được lưu trữ trong tệp CSV với 8 cột chuẩn hóa, đảm bảo nhóm có thể tích hợp trực tiếp vào bộ chấm tập trung:
- `model`: Tên mô hình (`Seasonal_Naive`, `Croston`, `Croston_SBA`, `ES_bu`)
- `split`: Giai đoạn thực nghiệm (`validation`, `test`)
- `origin_day`: Ngày mốc quan sát cuối cùng ($1885$ hoặc $1913$)
- `item_id`: Mã định danh sản phẩm (3.049 mã)
- `store_id`: Mã định danh cửa hàng (10 cửa hàng)
- `target_day`: Ngày được dự báo ($d_{1886} \dots d_{1913}$ hoặc $d_{1914} \dots d_{1941}$)
- `horizon`: Bước dự báo tương lai ($1 \dots 28$)
- `y_pred`: Lượng bán dự báo (làm tròn 4 chữ số thập phân)

---

## 4. KẾT QUẢ THỰC NGHIỆM ĐỊNH LƯỢNG

Thực nghiệm được triển khai song song trên hai quy mô:
1. Quy mô kiểm định quy trình trên cửa hàng đại diện **CA_1** (3.049 chuỗi).
2. Quy mô mở rộng toàn diện trên toàn bộ **10 cửa hàng** (30.490 chuỗi, tổng cộng 3.414.880 dòng dự báo cho mỗi giai đoạn).

### 4.1. Đánh giá tổng thể hiệu năng dự báo

Dưới đây là bảng tổng hợp sai số trung bình trên toàn bộ 30.490 chuỗi sản phẩm–cửa hàng cho cả hai giai đoạn:

| Giai đoạn | Mô hình dự báo | MAE trung bình | RMSE trung bình | RMSSE trung bình | Số chuỗi đánh giá |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation** | Seasonal_Naive | 1,2324 | 1,7400 | 0,9756 | 30.490 |
| | Croston | 1,1026 | 1,3986 | 0,7616 | 30.490 |
| | **Croston_SBA** | **1,0862** | **1,3909** | **0,7589** | 30.490 |
| | ES_bu | 1,0608 | 1,3983 | 0,7775 | 30.490 |
| **Test** | Seasonal_Naive | 1,2440 | 1,7519 | 0,9972 | 30.490 |
| | Croston | 1,0898 | 1,3969 | 0,7704 | 30.490 |
| | **Croston_SBA** | **1,0759** | **1,3926** | **0,7687** | 30.490 |
| | ES_bu | 1,0659 | 1,4121 | 0,7972 | 30.490 |

#### Nhận xét tổng thể:
1. **Vượt trội so với mốc cơ sở:** Cả ba phương pháp thống kê nâng cao (Croston, Croston-SBA, ES_bu) đều vượt qua Seasonal Naïve với khoảng cách rất lớn. Trên tập Validation, RMSSE giảm từ 0,9756 xuống mức xấp xỉ 0,7589–0,7775 (mức cải thiện hơn 22% về sai số co giãn RMSSE).
2. **Hiệu quả thực tế của hiệu chỉnh SBA:** Croston-SBA vượt qua Croston cổ điển trên cả 3 chỉ tiêu ở cả hai giai đoạn:
   - Trên Validation: MAE giảm từ 1,1026 xuống 1,0862; RMSE giảm từ 1,3986 xuống 1,3909; RMSSE giảm từ 0,7616 xuống 0,7589.
   - Trên Test: MAE giảm từ 1,0898 xuống 1,0759; RMSSE giảm từ 0,7704 xuống 0,7687.
   Kết quả này khẳng định độ chệch dương trong mô hình Croston đã được thu hẹp hiệu quả nhờ hệ số $(1 - \alpha/2)$.
3. **So sánh giữa ES_bu và Croston-SBA:** ES_bu đạt chỉ số MAE thấp nhất (1,0608 trên Validation và 1,0659 trên Test), trong khi Croston-SBA lại đạt RMSE và RMSSE thấp nhất (RMSE = 1,3909; RMSSE = 0,7589). Điều này bắt nguồn từ việc ES_bu nắm bắt tốt tính mùa vụ hằng ngày của các mặt hàng tiêu thụ đều, trong khi Croston-SBA ổn định hơn trên các chuỗi gián đoạn lớn.

---

### 4.2. Hiệu quả dự báo theo ngành hàng (Category Breakdown)

Bảng phân tích sai số trên tập Validation theo 3 ngành hàng:

| Ngành hàng | Mô hình | MAE | RMSE | RMSSE | Tỷ trọng chuỗi |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **FOODS** | Seasonal_Naive | 1,6179 | 2,2150 | 0,9779 | 47,1% (14.370 chuỗi) |
| | Croston | 1,4554 | 1,8159 | 0,7806 | |
| | Croston_SBA | 1,4324 | 1,8032 | 0,7771 | |
| | **ES_bu** | **1,3794** | **1,7831** | **0,7824** | |
| **HOBBIES** | Seasonal_Naive | 0,8312 | 1,3073 | 0,9462 | 18,5% (5.650 chuỗi) |
| | Croston | 0,7347 | 0,9975 | 0,7101 | |
| | **Croston_SBA** | **0,7241** | **0,9954** | **0,7085** | |
| | ES_bu | 0,7328 | 1,0444 | 0,7464 | |
| **HOUSEHOLD**| Seasonal_Naive | 0,9197 | 1,3217 | 0,9883 | 34,4% (10.470 chuỗi) |
| | Croston | 0,8170 | 1,0424 | 0,7633 | |
| | **Croston_SBA** | **0,8063** | **1,0386** | **0,7610** | |
| | ES_bu | 0,8006 | 1,0611 | 0,7875 | |

#### Phân tích chi tiết:
- **Ngành FOODS:** ES_bu thể hiện ưu thế áp đảo với MAE đạt 1,3794 và RMSE đạt 1,7831 (giảm 14,7% MAE và 19,5% RMSE so với Seasonal Naïve). Ngành hàng thực phẩm có tần suất mua sắm dày đặc và chịu ảnh hưởng mạnh bởi chu kỳ cuối tuần, do đó việc mô hình hóa trực tiếp thành phần mùa vụ tuần $S_t$ giúp ES_bu bám sát biến động doanh số.
- **Ngành HOBBIES:** Croston–SBA hoàn toàn thống trị với RMSSE đạt 0,7085 và RMSE đạt 0,9954 (thấp hơn rõ rệt so với mức 0,7464 và 1,0444 của ES_bu). Đồ chơi và đồ dùng sở thích là nhóm hàng có tính gián đoạn cao nhất trong hệ thống; khi áp dụng san bằng mũ mùa vụ, các hệ số mùa vụ bị nhiễu do có quá nhiều số 0, trong khi Croston–SBA phân tách rành mạch xác suất phát sinh nhu cầu và quy mô nhu cầu, tạo ra dự báo mượt mà và an toàn hơn.
- **Ngành HOUSEHOLD:** Thể hiện sự cân bằng thú vị: ES_bu đạt MAE nhỉnh hơn một chút (0,8006 vs. 0,8063), nhưng Croston–SBA lại vượt trội hơn ở RMSE (1,0386 vs. 1,0611) và RMSSE (0,7610 vs. 0,7875), hạn chế được các sai số cực đại.

---

### 4.3. Hiệu quả dự báo theo mức độ gián đoạn (Intermittency Analysis)

Phân loại 30.490 chuỗi thời gian thành 4 nhóm dựa trên tỷ lệ ngày có lượng bán bằng 0 (Zero Demand Ratio) tính từ ngày phát sinh lượng bán dương đầu tiên:

| Nhóm gián đoạn | Tiêu chí tỷ lệ số 0 | Số chuỗi | Mô hình | MAE | RMSE | RMSSE |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **1. Thường xuyên** | < 30% | 4.010 | Seasonal_Naive | 3,3304 | 4,3182 | 0,9515 |
| | | | Croston | 2,8691 | 3,5456 | 0,7668 |
| | | | Croston_SBA | 2,8172 | 3,5140 | 0,7618 |
| | | | **ES_bu** | **2,7734** | **3,4910** | **0,7643** |
| **2. Trung bình** | 30% – 50% | 5.894 | Seasonal_Naive | 1,7446 | 2,3706 | 0,9463 |
| | | | Croston | 1,5057 | 1,8978 | 0,7468 |
| | | | Croston_SBA | 1,4864 | 1,8901 | 0,7439 |
| | | | **ES_bu** | **1,4652** | 1,9051 | 0,7549 |
| **3. Gián đoạn** | 50% – 70% | 8.134 | Seasonal_Naive | 0,9963 | 1,4466 | 0,9613 |
| | | | Croston | 0,9352 | 1,1754 | 0,7602 |
| | | | Croston_SBA | 0,9234 | 1,1699 | 0,7572 |
| | | | **ES_bu** | **0,8715** | **1,1613** | 0,7676 |
| **4. Rất gián đoạn** | $\ge$ 70% | 12.452 | Seasonal_Naive | 0,4685 | 0,8030 | 1,0065 |
| | | | Croston | 0,4524 | 0,6168 | 0,7679 |
| | | | **Croston_SBA** | **0,4456** | **0,6154** | **0,7661** |
| | | | ES_bu | 0,4415 | 0,6393 | 0,7988 |

#### Phát hiện cốt lõi từ bảng phân tích:
1. **Sự sụp đổ của Seasonal Naïve tại nhóm Rất gián đoạn:** Tại nhóm 4 (chiếm hơn 40% tổng số chuỗi với 12.452 chuỗi), chỉ số RMSSE của Seasonal Naïve lên tới **1,0065** (> 1,0). Điều này chỉ ra rằng đối với các chuỗi có trên 70% số ngày không bán được hàng, việc lặp lại lượng bán của cùng thứ tuần trước thậm chí còn tạo ra sai số lớn hơn cả phương pháp bước ngẫu nhiên sơ đẳng trong mẫu!
2. **Vai trò then chốt của Croston–SBA trên chuỗi thưa:** Tại nhóm 4, Croston–SBA đạt RMSSE thấp nhất toàn diện (**0,7661**) và RMSE thấp nhất (**0,6154**), giảm gần 24% sai số RMSE so với Seasonal Naïve (0,8030). Khả năng duy trì mức dự báo phẳng xấp xỉ xác suất biên giúp Croston–SBA tránh được các dao động giả tạo.
3. **Sự dịch chuyển ưu thế:** Khi tỷ lệ số 0 giảm dần từ nhóm 4 về nhóm 1 (các chuỗi tiêu thụ thường xuyên), lợi thế dần chuyển dịch rõ ràng từ Croston–SBA sang ES_bu. Tại nhóm 1, ES_bu đạt MAE 2,7734 (giảm 16,7% so với mức 3,3304 của Seasonal Naïve).

---

### 4.4. Tính ổn định của mô hình qua 10 cửa hàng

Trích xuất chỉ số RMSSE trên tập Validation trên từng cửa hàng:

| Cửa hàng | Bang | Seasonal_Naive | Croston | Croston_SBA | ES_bu |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **CA_1** | California | 0,9360 | 0,7431 | **0,7396** | 0,7425 |
| **CA_2** | California | 1,0384 | 0,8201 | **0,8190** | 0,8279 |
| **CA_3** | California | 0,9584 | 0,7534 | **0,7502** | 0,7604 |
| **CA_4** | California | 0,9933 | 0,7638 | **0,7613** | 0,7845 |
| **TX_1** | Texas | 0,9496 | 0,7271 | **0,7243** | 0,7545 |
| **TX_2** | Texas | 0,9120 | 0,6903 | **0,6861** | 0,7139 |
| **TX_3** | Texas | 0,9743 | 0,7556 | **0,7523** | 0,7739 |
| **WI_1** | Wisconsin | 1,0142 | 0,7881 | **0,7853** | 0,8066 |
| **WI_2** | Wisconsin | 1,0207 | 0,8158 | **0,8139** | 0,8333 |
| **WI_3** | Wisconsin | 0,9590 | 0,7589 | **0,7569** | 0,7770 |

Croston–SBA đạt RMSSE thấp nhất trong bốn mô hình tại cả 10 cửa hàng trên tập validation. RMSSE theo cửa hàng là trung bình của các chuỗi sản phẩm–cửa hàng thuộc cửa hàng đó.

---

## 5. THẢO LUẬN CHUYÊN SÂU VÀ ĐỊNH VỊ VAI TRÒ BENCHMARK

### 5.1. Phân tích nguyên nhân chênh lệch hiệu năng giữa các phương pháp
1. **Tại sao Seasonal Naïve kém hiệu quả trên dữ liệu M5?**
   Seasonal Naïve dựa trên giả định tính tất định theo tuần. Tuy nhiên, ở cấp độ sản phẩm chi tiết (SKU), hành vi mua sắm mang tính ngẫu nhiên rất lớn: một sản phẩm bán được 2 đơn vị vào Thứ Bảy tuần này không đồng nghĩa Thứ Bảy tuần sau cũng sẽ bán được 2 đơn vị. Khi gặp các giá trị 0 ngẫu nhiên, Seasonal Naïve dự báo tương lai bằng 0, dẫn đến sai số nghiêm trọng khi khách hàng quay lại mua sắm.
2. **Cơ chế vượt trội của Croston và hiệu chỉnh SBA:**
   Croston đã chuyển hóa bài toán dự báo chuỗi thời gian gián đoạn thành hai bài toán ước lượng chuỗi liên tục: ước lượng quy mô trung bình khi có phát sinh giao dịch ($z$) và ước lượng tần suất xuất hiện giao dịch ($1/p$). Bằng cách đó, dự báo luôn là một giá trị dương nhỏ phản ánh kỳ vọng toán học của nhu cầu trong một đơn vị thời gian. Hiệu chỉnh SBA loại bỏ độ phóng đại của nghịch đảo khoảng cách, giúp đường dự báo sát với phân phối thực tế.
3. **Ưu thế và giới hạn của ES_bu:**
   ES_bu là mô hình lý tưởng cho phân khúc hàng hóa tiêu thụ nhanh (FMCG / FOODS), nơi tính mùa vụ tuần là một tín hiệu thực sự (signal) chứ không phải nhiễu (noise). Việc tổng hợp Bottom-Up từ cấp sản phẩm lên cấp cửa hàng giúp tích lũy chính xác doanh số của toàn bộ hệ thống.

### 5.2. Định vị vai trò làm mốc đối chứng (Benchmark) cho các mô hình học máy
Dựa trên kết quả thực nghiệm định lượng, nhóm nghiên cứu thiết lập bộ tiêu chuẩn đối chứng cho các nhánh mô hình tiếp theo như sau:

| Thước đo | Mốc cơ sở tối thiểu (Seasonal Naïve) | Mốc đối chứng thống kê (Croston_SBA / ES_bu) | Ngưỡng kỳ vọng cho Học máy (RF / LightGBM) |
| :--- | :---: | :---: | :---: |
| **MAE toàn thể (Validation)** | 1,2324 | **1,0608** (ES_bu) | Phải nhỏ hơn 1,0000 |
| **RMSE toàn thể (Validation)** | 1,7400 | **1,3909** (Croston_SBA) | Phải nhỏ hơn 1,3500 |
| **RMSSE toàn thể (Validation)** | 0,9756 | **0,7589** (Croston_SBA) | Phải nhỏ hơn 0,7200 |
| **RMSSE nhóm FOODS** | 0,9779 | **0,7771** (Croston_SBA) | Cần vượt mốc 0,7000 nhờ biến giá/lịch |
| **RMSSE nhóm HOBBIES** | 0,9462 | **0,7085** (Croston_SBA) | Thách thức rất lớn đối với cây quyết định |

- **Đối với Random Forest và LightGBM-Tweedie:** Mô hình học máy sử dụng rất nhiều đặc trưng phức tạp (lag, rolling, giá bán, sự kiện lịch, SNAP, ngày trong tuần). Nếu LightGBM không thể đạt RMSSE thấp hơn mốc **0,7589** của Croston–SBA hoặc MAE thấp hơn **1,0608** của ES_bu, điều đó chứng minh rằng độ phức tạp của mô hình học máy chưa mang lại giá trị tương xứng so với chi phí kỹ thuật.
- **Đối với Two-part LightGBM (Mô hình mở rộng):** Cơ chế của Two-part LightGBM (tách thành mô hình phân loại xác suất có bán và mô hình hồi quy khi có bán) thực chất chính là sự kế thừa triết lý phân tách của Croston nhưng được hiện đại hóa bằng cây tăng cường độ dốc (GBDT). Do đó, kết quả của Croston–SBA trong báo cáo này sẽ là điểm tựa lý thuyết trực tiếp nhất để đánh giá hiệu quả của nhánh Two-part LightGBM.

### 5.3. Đánh giá tính kinh tế và chi phí tài nguyên tính toán
Một phát hiện mang tính thực tiễn kinh doanh rất cao là sự chênh lệch khổng lồ về chi phí tính toán:
- **Tốc độ thực thi của mô hình thống kê:** Nhờ tối ưu hóa thuật toán bằng Numba, việc tạo lập dự báo cho toàn bộ 30.490 chuỗi thời gian của 10 cửa hàng chỉ mất:
  - Croston cổ điển: **2,51 giây**
  - Croston–SBA: **3,02 giây**
  - ES_bu: **6,43 giây**
  - Seasonal Naïve: **5,09 giây**
- **So sánh với Học máy:** Huấn luyện 10 mô hình LightGBM theo cửa hàng với bảng đặc trưng trượt (lag/rolling) thường đòi hỏi hàng chục phút đến vài giờ huấn luyện trên máy tính cấu hình cao và tiêu tốn hàng gigabyte RAM.
- **Ý nghĩa thực tiễn:** Trong môi trường vận hành bán lẻ với hàng trăm nghìn SKU, các mô hình thống kê như Croston–SBA hoặc ES_bu có thể chạy định kỳ hằng đêm chỉ trong vài giây với mức tiêu thụ tài nguyên gần như bằng 0, trong khi vẫn đạt được sai số RMSSE ở mức 0,7589 (rất ấn tượng). Đây là sự đánh đổi chi phí - hiệu năng (cost-accuracy trade-off) cần được phân tích kỹ lưỡng trong đồ án.

---

## 6. KẾT LUẬN VÀ BÀN GIAO THỰC NGHIỆM

### 6.1. Tóm tắt kết luận
1. Đã triển khai thành công 4 mô hình đối chứng thống kê trên toàn bộ 30.490 chuỗi thời gian dữ liệu M5, tuân thủ nghiêm ngặt mốc chia thời gian và không để xảy ra rò rỉ thông tin.
2. Chứng minh thực nghiệm rằng **Croston–SBA** là mô hình thống kê toàn diện nhất trên dữ liệu gián đoạn M5 (RMSSE = 0,7589), khẳng định giá trị của hiệu chỉnh Syntetos–Boylan.
3. Xác định **ES_bu** là phương pháp tối ưu cho nhóm hàng thực phẩm (FOODS) có tính mùa vụ tuần rõ nét (MAE = 1,3794).
4. Cung cấp bộ mốc đối chứng định lượng chuẩn xác để nhóm tiến hành so sánh với Random Forest, LightGBM-Tweedie và Two-part LightGBM.

### 6.2. Danh mục tài liệu và dữ liệu bàn giao cho nhóm

| Thư mục | Tên tệp | Nội dung bàn giao |
| :--- | :--- | :--- |
| `02_Ma_nguon/` | `config.py` | Cấu hình đường dẫn, mốc thời gian, tham số mô hình |
| | `models/croston.py` | Mã nguồn Croston và Croston-SBA tối ưu bằng Numba |
| | `models/exponential_smoothing.py` | Mã nguồn San bằng mũ mùa vụ tuần và cơ chế Bottom-Up |
| | `models/seasonal_naive.py` | Mã nguồn mô hình Seasonal Naïve chu kỳ 7 ngày |
| | `metrics.py` | Hàm tính MAE, RMSE, RMSSE theo chuẩn cuộc thi M5 |
| | `run_evaluation.py` | Kịch bản tự động hóa thực nghiệm từ nạp dữ liệu đến xuất bảng |
| | `test_sample_series.py` | Kịch bản kiểm thử đơn vị trên chuỗi đại diện minh họa |
| `03_Du_lieu_du_bao/` | `predictions_validation.csv` | Dự báo 28 ngày validation của 4 mô hình (đúng format chung) |
| | `predictions_test.csv` | Dự báo 28 ngày test của 4 mô hình (đúng format chung) |
| `04_Ket_qua_danh_gia/` | `summary_metrics_overall.csv` | Bảng tổng hợp sai số toàn thể (Validation & Test) |
| | `summary_metrics_by_category.csv` | Bảng sai số chi tiết theo ngành hàng FOODS, HOBBIES, HOUSEHOLD |
| | `summary_metrics_by_intermittency.csv`| Bảng sai số theo 4 mức độ gián đoạn chuỗi |
| | `summary_metrics_by_store.csv` | Bảng sai số chi tiết trên từng siêu thị trong 10 siêu thị |
