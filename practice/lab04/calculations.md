# BÀI TẬP TÍNH TOÁN VÀ ĐỊNH LƯỢNG (LAB 04 - CALCULATIONS)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [Bài 1 — Binary Classification: Logit, Sigmoid và Xác suất tiên nghiệm (Mục 6)](#1-bài-1--binary-classification-mục-6)
   - [1.1. Đề bài & Thông số giả thiết](#11-đề-bài--thông-số-giả-thiết)
   - [1.2. Cơ sở lý thuyết & Công thức toán học](#12-cơ-sở-lý-thuyết--công-thức-toán-học)
   - [1.3. Các bước tính toán chi tiết](#13-các-bước-tính-toán-chi-tiết)
   - [1.4. Ý nghĩa toán học của Logit và Sigmoid](#14-ý-nghĩa-toán-học-của-logit-và-sigmoid)
2. [Bài 2 — Prediction Threshold & Đánh đổi Precision - Recall (Mục 7)](#2-bài-2--prediction-threshold-mục-7)
   - [2.1. Đề bài](#21-đề-bài)
   - [2.2. Phân loại với ngưỡng quyết định tau = 0.5](#22-phân-loại-với-ngưỡng-quyết-định-tau--05)
   - [2.3. Phân loại với ngưỡng quyết định tau = 0.8](#23-phân-loại-với-ngưỡng-quyết-định-tau--08)
   - [2.4. Phân tích chuyên sâu: Cơ chế điều biến Precision và Recall theo Threshold](#24-phân-tích-chuyên-sâu-cơ-chế-điều-biến-precision-và-recall-theo-threshold)
3. [Bài 3 — Confusion Matrix & Đánh giá mô hình phân loại Spam (Mục 8)](#3-bài-3--confusion-matrix-mục-8)
   - [3.1. Đề bài & Ma trận nhầm lẫn chuẩn](#31-đề-bài--ma-trận-nhầm-lẫn-chuẩn)
   - [3.2. Tính toán các chỉ số Precision, Recall và F1-Score](#32-tính-toán-các-chỉ-số-precision-recall-và-f1-score)
   - [3.3. Ý nghĩa thực tiễn của False Positive và False Negative trong bài toán phát hiện Spam](#33-ý-nghĩa-thực-tiễn-của-false-positive-và-false-negative-trong-bài-toán-phát-hiện-spam)
4. [Bài 4 — Accuracy Trap & Phân tích dữ liệu mất cân bằng (Mục 9)](#4-bài-4--accuracy-trap-mục-9)
   - [4.1. Đề bài & Hiện tượng Dummy/Majority Classifier](#41-đề-bài--hiện-tượng-dummymajority-classifier)
   - [4.2. Trả lời câu hỏi: Mô hình này có thực sự tốt không?](#42-trả-lời-câu-hỏi-mô-hình-này-có-thực-sự-tốt-không)
   - [4.3. Các hệ metric bắt buộc xem thêm để chống bẫy Accuracy](#43-các-hệ-metric-bắt-buộc-xem-thêm-để-chống-bẫy-accuracy)

---

# 1. Bài 1 — Binary Classification (Mục 6)

## 1.1. Đề bài & Thông số giả thiết

Cho một mẫu văn bản được biểu diễn dưới dạng feature vector 3 chiều:
$$\vec{x} = [1, 2, 0]^T$$

Vector trọng số học được của mô hình tuyến tính (Linear Classifier / Logistic Regression):
$$\vec{w} = [0.5, -0.2, 0.4]^T$$

Hệ số chệch (bias / intercept):
$$b = 0.1$$

**Yêu cầu**:
1. Tính giá trị logit (linear combination):
   $$z = \vec{w}^T \vec{x} + b$$
2. Sử dụng hàm kích hoạt phi tuyến Sigmoid:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
3. Xác định xác suất hậu nghiệm thuộc lớp dương:
   $$P(y = 1 \mid \vec{x})$$

---

## 1.2. Cơ sở lý thuyết & Công thức toán học

Trong bài toán phân loại nhị phân (Binary Classification) với mô hình hồi quy logistic (Logistic Regression):
- **Tổ hợp tuyến tính (Logit score)**:
  Mỗi feature $x_i$ được gán một trọng số $w_i$. Trọng số $w_i > 0$ đóng góp tích cực vào khả năng mẫu thuộc lớp positive; ngược lại $w_i < 0$ kéo xác suất về lớp negative.
  $$z = \vec{w}^T \vec{x} + b = \sum_{i=1}^d w_i x_i + b$$
  Giá trị $z$ có miền xác định trên toàn bộ trục số thực: $z \in (-\infty, +\infty)$.
- **Hàm kích hoạt Sigmoid (Logistic function)**:
  Để chuyển đổi giá trị $z$ không bị chặn về một phân phối xác suất hợp lệ thỏa mãn tiên đề Kolmogorov ($P \in [0, 1]$), ta sử dụng hàm Sigmoid:
  $$\sigma(z) = \frac{1}{1 + \exp(-z)}$$
- **Xác suất dự báo**:
  $$P(y = 1 \mid \vec{x}) = \sigma(z)$$
  $$P(y = 0 \mid \vec{x}) = 1 - P(y = 1 \mid \vec{x}) = 1 - \sigma(z) = \frac{\exp(-z)}{1 + \exp(-z)} = \sigma(-z)$$

---

## 1.3. Các bước tính toán chi tiết

### Bước 1: Tính tích vô hướng $\vec{w}^T \vec{x}$
$$\vec{w}^T \vec{x} = (w_1 \cdot x_1) + (w_2 \cdot x_2) + (w_3 \cdot x_3)$$
$$\vec{w}^T \vec{x} = (0.5 \times 1) + (-0.2 \times 2) + (0.4 \times 0)$$
$$\vec{w}^T \vec{x} = 0.5 - 0.4 + 0 = 0.1$$

### Bước 2: Cộng hệ số bias $b$ để tìm Logit $z$
$$z = \vec{w}^T \vec{x} + b = 0.1 + 0.1 = \mathbf{0.2}$$

### Bước 3: Áp dụng hàm Sigmoid $\sigma(z)$
Thay $z = 0.2$ vào công thức hàm sigmoid:
$$\sigma(0.2) = \frac{1}{1 + e^{-0.2}}$$

Tính giá trị lũy thừa cơ số tự nhiên:
$$e^{-0.2} \approx 0.8187307530779818$$

Tính mẫu số:
$$1 + e^{-0.2} \approx 1 + 0.8187307530779818 = 1.8187307530779818$$

Thực hiện phép chia:
$$\sigma(0.2) = \frac{1}{1.8187307530779818} \approx \mathbf{0.5498339973} \approx \mathbf{0.5498}$$

### Bước 4: Kết luận xác suất
$$P(y = 1 \mid \vec{x}) = \sigma(0.2) \approx \mathbf{0.5498} \quad (54.98\%)$$
$$P(y = 0 \mid \vec{x}) = 1 - 0.5498 = \mathbf{0.4502} \quad (45.02\%)$$

---

## 1.4. Ý nghĩa toán học của Logit và Sigmoid

1. **Logit $z$ là Log-Odds**:
   Biểu thức $z$ chính là logarithm tự nhiên của tỷ lệ cược (log-odds ratio):
   $$z = \ln \left( \frac{P(y=1 \mid \vec{x})}{P(y=0 \mid \vec{x})} \right) = \ln \left( \frac{P(y=1 \mid \vec{x})}{1 - P(y=1 \mid \vec{x})} \right)$$
   Vì $z = 0.2 > 0$, tỷ số cược $\frac{P(y=1)}{P(y=0)} = e^{0.2} \approx 1.2214 > 1$, nghĩa là mẫu $\vec{x}$ có xu hướng nghiêng về lớp positive hơn là lớp negative.
2. **Đối xứng qua tâm $(0, 0.5)$**:
   - Khi $z = 0 \implies \sigma(0) = 0.5$ (mô hình hoàn toàn phân vân giữa hai lớp).
   - Khi $z > 0 \implies \sigma(z) > 0.5$.
   - Khi $z < 0 \implies \sigma(z) < 0.5$.
   Vì $z = 0.2$ chỉ hơi lớn hơn 0 một lượng nhỏ, xác suất $0.5498$ cho thấy độ tự tin (confidence) của mô hình đối với mẫu này là ở mức vừa phải.

---

# 2. Bài 2 — Prediction Threshold (Mục 7)

## 2.1. Đề bài

Giả sử mô hình đã ước lượng được xác suất dự báo:
$$P(y = 1 \mid \vec{x}) = 0.72$$

Quy tắc quyết định phân lớp (decision rule) với ngưỡng $\tau \in [0, 1]$ được định nghĩa:
$$\hat{y} = \begin{cases} 1 & \text{nếu } P(y = 1 \mid \vec{x}) \ge \tau \\ 0 & \text{nếu } P(y = 1 \mid \vec{x}) < \tau \end{cases}$$

**Yêu cầu**:
1. Với threshold $\tau = 0.5$, mô hình dự đoán class nào?
2. Với threshold $\tau = 0.8$, mô hình dự đoán class nào?
3. Trả lời: Thay đổi threshold có thể làm Precision và Recall thay đổi như thế nào?

---

## 2.2. Phân loại với ngưỡng quyết định $\tau = 0.5$

So sánh xác suất dự báo với ngưỡng $\tau = 0.5$:
$$P(y = 1 \mid \vec{x}) = 0.72 \ge 0.5$$

**Kết luận**: Mô hình dự đoán nhãn **Class 1 (Positive)**.

---

## 2.3. Phân loại với ngưỡng quyết định $\tau = 0.8$

So sánh xác suất dự báo với ngưỡng $\tau = 0.8$:
$$P(y = 1 \mid \vec{x}) = 0.72 < 0.8$$

**Kết luận**: Mô hình dự đoán nhãn **Class 0 (Negative)**.

---

## 2.4. Phân tích chuyên sâu: Cơ chế điều biến Precision và Recall theo Threshold

Việc dịch chuyển ngưỡng quyết định $\tau$ tạo ra sự đánh đổi kinh điển giữa **Precision (Độ chính xác)** và **Recall (Độ nhạy / Độ bao phủ)**:

```text
Threshold tau thấp (0.1 - 0.3) <======== Ngưỡng mặc định (0.5) ========> Threshold tau cao (0.7 - 0.9)
    Dễ gán nhãn Positive                                                      Khắt khe khi gán Positive
    Bắt trọn mọi ca dương                                                     Chỉ gán khi cực kỳ chắc chắn
    Recall TĂNG, Precision GIẢM                                               Precision TĂNG, Recall GIẢM
```

### Chi tiết toán học của sự dịch chuyển:

1. **Khi tăng ngưỡng quyết định ($\tau \uparrow$, ví dụ từ $0.5 \to 0.8$)**:
   - **Cơ chế**: Mô hình trở nên khắt khe, thận trọng hơn; chỉ những mẫu có độ tin cậy rất cao ($P \ge 0.8$) mới được phân loại là Positive.
   - **Biến động số lượng**:
     - Số lượng mẫu được dự đoán là Positive ($\hat{y} = 1$) giảm đi.
     - Số ca dự đoán nhầm dương tính (False Positive - $FP$) giảm mạnh.
     - Số ca dương tính thực tế bị bỏ sót (False Negative - $FN$) tăng lên.
   - **Hệ quả đối với metrics**:
     - $\text{Precision} = \frac{TP}{TP + FP} \uparrow$ (TĂNG vì mẫu số $TP + FP$ thu nhỏ và $FP$ bị triệt tiêu).
     - $\text{Recall} = \frac{TP}{TP + FN} \downarrow$ (GIẢM vì $FN$ tăng lên).
2. **Khi giảm ngưỡng quyết định ($\tau \downarrow$, ví dụ từ $0.5 \to 0.2$)**:
   - **Cơ chế**: Mô hình nới lỏng tiêu chuẩn; chỉ cần có dấu hiệu nghi vấn nhỏ ($P \ge 0.2$) là lập tức phân loại là Positive.
   - **Biến động số lượng**:
     - Bắt được tối đa các ca dương tính thực tế trong dữ liệu ($TP \uparrow$).
     - Hạn chế tối đa bỏ sót ($FN \downarrow$).
     - Tuy nhiên, kéo theo rất nhiều mẫu âm tính thực tế bị kết luận nhầm thành dương tính ($FP \uparrow$).
   - **Hệ quả đối với metrics**:
     - $\text{Recall} = \frac{TP}{TP + FN} \uparrow$ (TĂNG).
     - $\text{Precision} = \frac{TP}{TP + FP} \downarrow$ (GIẢM).

### Bảng đối chiếu ứng dụng thực tế:

| Bài toán ứng dụng | Mục tiêu ưu tiên | Chiến lược chọn $\tau$ | Lý do thực tiễn |
| :--- | :--- | :---: | :--- |
| **Phát hiện Email Spam** | Ưu tiên **Precision** cao | Tăng $\tau$ ($\tau \approx 0.7 - 0.8$) | Thà để lọt một vài email rác vào Inbox ($FN$) còn hơn là chuyển nhầm email công việc/học tập quan trọng của người dùng vào thùng rác Spam ($FP$). |
| **Chẩn đoán Y tế (Ung thư / Bệnh hiểm nghèo)** | Ưu tiên **Recall** cao | Giảm $\tau$ ($\tau \approx 0.2 - 0.3$) | Thà báo động nhầm để bệnh nhân đi xét nghiệm lại chuyên sâu ($FP$) còn hơn là bỏ sót một ca ung thư giai đoạn sớm khiến bệnh nhân tử vong ($FN$). |
| **Phát hiện gian lận tài chính (Fraud Detection)** | Cân bằng F1 hoặc Recall cao | Giảm nhẹ $\tau$ ($\tau \approx 0.35 - 0.45$) | Giữ an toàn dòng tiền giao dịch bằng cách tạm giữ các giao dịch đáng ngờ để kiểm duyệt thủ công. |

---

# 3. Bài 3 — Confusion Matrix (Mục 8)

## 3.1. Đề bài & Ma trận nhầm lẫn chuẩn

Cho bảng ma trận nhầm lẫn (Confusion Matrix) thực nghiệm:

| | **Predicted Positive** | **Predicted Negative** | **Tổng hàng (Actual)** |
| :--- | :---: | :---: | :---: |
| **Actual Positive** | $TP = 80$ | $FN = 20$ | $P_{\text{actual}} = 100$ |
| **Actual Negative** | $FP = 10$ | $TN = 90$ | $N_{\text{actual}} = 100$ |
| **Tổng cột (Predicted)** | $TP + FP = 90$ | $FN + TN = 110$ | **Tổng số mẫu $N = 200$** |

Trong đó:
- $TP$ (True Positive): Thực tế dương, mô hình đoán đúng là dương ($80$).
- $FN$ (False Negative): Thực tế dương, mô hình đoán sai thành âm ($20$).
- $FP$ (False Positive): Thực tế âm, mô hình đoán sai thành dương ($10$).
- $TN$ (True Negative): Thực tế âm, mô hình đoán đúng là âm ($90$).

---

## 3.2. Tính toán các chỉ số Precision, Recall và F1-Score

### 1. Độ chính xác (Precision):
Tỷ lệ các mẫu thực sự là Positive trong số tất cả các mẫu được mô hình dự đoán là Positive:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{80}{80 + 10} = \frac{80}{90} = \frac{8}{9} \approx \mathbf{0.8889} \quad (88.89\%)$$

### 2. Độ nhạy / Độ bao phủ (Recall):
Tỷ lệ các mẫu Positive được mô hình nhận diện thành công so với tổng số mẫu Positive thực tế:
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{80}{80 + 20} = \frac{80}{100} = \mathbf{0.8000} \quad (80.00\%)$$

### 3. F1-Score (Trung bình điều hòa giữa Precision và Recall):
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Thay các giá trị đã tính:
$$F_1 = 2 \cdot \frac{\frac{8}{9} \cdot \frac{8}{10}}{\frac{8}{9} + \frac{8}{10}} = 2 \cdot \frac{\frac{64}{90}}{\frac{80 + 72}{90}} = 2 \cdot \frac{64}{152} = \frac{128}{152} = \frac{16}{19} \approx \mathbf{0.8421} \quad (84.21\%)$$

*(Có thể kiểm tra lại qua công thức tương đương theo số đếm)*:
$$F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2 \times 80}{(2 \times 80) + 10 + 20} = \frac{160}{160 + 30} = \frac{160}{190} = \frac{16}{19} \approx \mathbf{0.8421}$$

---

## 3.3. Ý nghĩa thực tiễn của False Positive và False Negative trong bài toán phát hiện Spam

Quy ước trong bài toán lọc thư rác (Spam Detection):
- **Class 1 (Positive)**: Thư rác / Độc hại (**Spam**).
- **Class 0 (Negative)**: Thư hợp lệ / Thư bình thường (**Ham / Non-Spam**).

### 1. Ý nghĩa và hệ quả của False Positive ($FP = 10$):
- **Bản chất**: Một email thực sự là thư quan trọng/hợp lệ ($Actual = 0$), nhưng bộ lọc classifier lại gán nhãn nhầm là thư rác ($Predicted = 1$).
- **Hành vi hệ thống**: Email này bị đẩy thẳng vào mục `Spam / Junk` hoặc bị xóa tự động.
- **Hệ quả thực tế**: **CỰC KỲ NGHIÊM TRỌNG**. Người dùng bị bỏ lỡ các email tối quan trọng: thư mời phỏng vấn xin việc, thông báo trúng tuyển đại học, hợp đồng thương mại với đối tác, mã OTP xác thực ngân hàng, hay biên lai thanh toán viện phí. Thiệt hại có thể đo đếm bằng cơ hội nghề nghiệp, tài chính và pháp lý.
- **Yêu cầu đối với bài toán**: Cần tối đa hóa **Precision**, kiểm soát để tỷ lệ $FP$ tiến sát về 0 tuyệt đối.

### 2. Ý nghĩa và hệ quả của False Negative ($FN = 20$):
- **Bản chất**: Một email thực tế là thư rác lừa đảo ($Actual = 1$), nhưng bộ lọc lại để lọt và dự đoán là thư bình thường ($Predicted = 0$).
- **Hành vi hệ thống**: Email spam xuất hiện bình thường trong hộp thư đến chính (`Inbox`).
- **Hệ quả thực tế**: **Ít nghiêm trọng hơn so với FP**. Người dùng bị làm phiền, mất vài giây lướt qua và bấm nút xóa thủ công (trừ trường hợp email mang mã độc hoặc lừa đảo phi tinh vi đánh lừa được người dùng không cảnh giác).
- **Đánh giá so sánh**: Trong thiết kế hệ thống Spam Filter thực tế của Gmail hay Outlook, các kỹ sư luôn chọn ngưỡng $\tau$ cao để chấp nhận một lượng nhỏ $FN$ nhằm bảo vệ triệt để chỉ số $FP = 0$.

---

# 4. Bài 4 — Accuracy Trap (Mục 9)

## 4.1. Đề bài & Hiện tượng Dummy/Majority Classifier

Cho một tập dữ liệu gồm $N = 1000$ mẫu với phân phối nhãn bị lệch nghiêm trọng (Imbalanced Dataset):
- Lớp âm tính (Negative): $950$ mẫu ($95\%$)
- Lớp dương tính (Positive): $50$ mẫu ($5\%$)

Một mô hình ngây thơ (Trivial / Majority Baseline Classifier) đưa ra chiến lược:
$$\text{Với mọi mẫu } x \implies \hat{y} = \text{Negative}$$

Độ chính xác tổng thể (Accuracy) đạt được:
$$\text{Accuracy} = \frac{\text{Số mẫu đoán đúng}}{\text{Tổng số mẫu}} = \frac{950}{1000} = \mathbf{95\%}$$

---

## 4.2. Trả lời câu hỏi: Mô hình này có thực sự tốt không?

### Kết luận dứt khoát:
**Mô hình này HOÀN TOÀN KHÔNG TỐT, thậm chí hoàn toàn VÔ DỤNG trên thực tế (Statistically Trivial & Practically Useless).**

### Luận điểm giải thích khoa học:
1. **Mô hình không học được tri thức phân biệt**:
   Mô hình không khai thác bất kỳ feature vector nào từ văn bản. Nó chỉ đơn giản đoán nhãn của lớp đa số (majority class). Hàm quyết định không hề có tính phân loại:
   $$f(x) \equiv 0 \quad \forall x$$
2. **Triệt tiêu hoàn toàn khả năng phát hiện lớp thiểu số**:
   Trong các bài toán thực tế, lớp thiểu số (Positive) luôn là lớp mục tiêu mang giá trị kinh doanh hoặc an toàn sống còn:
   - Nếu Positive là phát hiện giao dịch gian lận thẻ tín dụng: Mô hình bỏ sót $100\%$ các vụ trộm tiền.
   - Nếu Positive là phát hiện khối u ác tính: Mô hình để $100\%$ bệnh nhân ung thư ra về mà không được điều trị.
   - Nếu Positive là cảnh báo tấn công mạng (Cyber attack): Hệ thống phòng thủ bị xuyên thủng hoàn toàn.
3. **Sự sụp đổ của các chỉ số hiệu năng thực sự**:
   - $TP = 0, \quad FN = 50, \quad TN = 950, \quad FP = 0$.
   - $\text{Recall}_{\text{Positive}} = \frac{0}{0 + 50} = \mathbf{0.00\%}$
   - $\text{Precision}_{\text{Positive}} = \frac{0}{0 + 0} \implies \text{Không xác định / Bằng } \mathbf{0.00\%}$
   - $F1_{\text{Positive}} = \mathbf{0.00\%}$

Con số **95% Accuracy chỉ là một "ảo ảnh thống kê" (Statistical Illusion)** do tỷ lệ mất cân bằng dữ liệu tạo ra. Đây chính là hiện tượng kinh điển được gọi là **Accuracy Trap (Bẫy độ chính xác)**.

---

## 4.3. Các hệ metric bắt buộc xem thêm để chống bẫy Accuracy

Để đánh giá một mô hình một cách khách quan, toàn diện và không bị lừa bởi con số Accuracy khi làm việc với dữ liệu thực tế, sinh viên và kỹ sư NLP bắt buộc phải xem xét hệ thống các metric sau:

1. **Ma trận nhầm lẫn (Confusion Matrix)**:
   - Trực quan hóa số lượng cụ thể của $TP, TN, FP, FN$.
   - Cho phép nhìn trực tiếp xem mô hình có thực sự phân loại được lớp nào hay chỉ đoán toàn bộ vào một phía.
2. **Precision, Recall và F1-Score riêng cho từng Class (Class-wise Metrics)**:
   - Báo cáo rõ ràng $F1_{\text{Positive}}$ và $F1_{\text{Negative}}$.
   - Trong ví dụ trên: $F1_{\text{Negative}} \approx 0.974$, nhưng $F1_{\text{Positive}} = 0.000$, lập tức phơi bày bản chất yếu kém của mô hình.
3. **Macro-averaged F1-Score**:
   - Tính trung bình số học không trọng số của F1 giữa các lớp:
     $$\text{Macro-F1} = \frac{F1_{\text{Negative}} + F1_{\text{Positive}}}{2} = \frac{0.974 + 0.000}{2} = \mathbf{0.487} \quad (48.7\%)$$
   - Con số $48.7\%$ phản ánh trung thực rằng mô hình hoạt động kém hơn cả việc tung đồng xu ngẫu nhiên.
4. **Độ chính xác cân bằng (Balanced Accuracy)**:
   - Trung bình cộng của Recall từng lớp (True Positive Rate và True Negative Rate):
     $$\text{Balanced Accuracy} = \frac{\text{TPR} + \text{TNR}}{2} = \frac{0.0 + 1.0}{2} = \mathbf{50.0\%}$$
   - Một mô hình có Balanced Accuracy $= 50\%$ đồng nghĩa với việc không có giá trị dự đoán nào hơn một phép đoán ngẫu nhiên.
5. **Đường cong ROC-AUC và PR-AUC (Precision-Recall AUC)**:
   - **ROC-AUC**: Đánh giá năng lực xếp hạng của mô hình qua mọi ngưỡng $\tau$. Với mô hình đoán cố định 1 lớp, $\text{ROC-AUC} = 0.5$.
   - **PR-AUC**: Đường cong diện tích dưới Precision-Recall đặc biệt nhạy bén và chuẩn xác đối với các bài toán mất cân bằng dữ liệu nghiêm trọng.
