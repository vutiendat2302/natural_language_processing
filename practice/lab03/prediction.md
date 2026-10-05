# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (LAB 03 - PREDICTION)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên**: 23000111  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
- [9. Prediction trước experiment](#9-prediction-trước-experiment)
  - [9.1. Prediction 1 — Những từ nào gần nhau nhất?](#91-prediction-1--những-từ-nào-gần-nhau-nhất)
  - [9.2. Prediction 2 — Nếu context window tăng từ 2 lên 5](#92-prediction-2--nếu-context-window-tăng-từ-2-lên-5)
  - [9.3. Prediction 3 — Nếu embedding dimension tăng 50 -> 100 -> 300](#93-prediction-3--nếu-embedding-dimension-tăng-50---100---300)
  - [9.4. Prediction 4 — Doctor và physician có chắc chắn gần nhau nếu corpus chỉ có 100 câu?](#94-prediction-4--doctor-và-physician-có-chắc-chắn-gần-nhau-nếu-corpus-chỉ-có-100-câu)

---

# 9. Prediction trước experiment

Theo triết lý khoa học thực nghiệm của học phần NLP, sinh viên bắt buộc phải hình thành các giả thuyết khoa học (hypotheses) và đưa ra các dự đoán định tính/định lượng **trước khi tiến hành huấn luyện và chạy mô hình (pre-experiment predictions)**. 

Mỗi dự đoán dưới đây tuân thủ nghiêm ngặt cấu trúc 3 phần bắt buộc:
1. **Prediction**: Dự đoán kết quả thực nghiệm một cách cụ thể, rõ ràng.
2. **Reason**: Lập luận và cơ sở lý thuyết ngôn ngữ học / khoa học máy tính giải thích cho dự đoán.
3. **Confidence**: Mức độ tin cậy khoa học của dự đoán (*High / Medium / Low*).

---

## 9.1. Prediction 1 — Những từ nào gần nhau nhất?

### Câu hỏi nghiên cứu
> *Trong tập từ vựng gồm: `doctor`, `physician`, `hospital`, `banana`, `car`, những từ nào gần nhau nhất trong không gian embedding?*

### Lời giải & Dự đoán chính thức

- **Prediction (Dự đoán)**:
  1. Cặp từ có độ tương đồng cao nhất (gần nhau nhất) là:
     $$\mathbf{(doctor, physician)}$$
  2. Kế tiếp là cặp từ có liên kết chủ đề mạnh trong cùng lĩnh vực y tế:
     $$(doctor, hospital) \quad \text{và} \quad (physician, hospital)$$
  3. Các từ `banana` (thực phẩm) và `car` (phương tiện) sẽ nằm ở những phân vùng không gian tách biệt hoàn toàn, có độ tương đồng rất thấp hoặc phân kỳ so với nhóm y tế:
     $$\cos(doctor, physician) > \cos(doctor, hospital) \gg \cos(doctor, car) \approx \cos(doctor, banana)$$

- **Reason (Cơ sở lý thuyết)**:
  - **Giả thuyết phân phối (Distributional Hypothesis)**: Các mô hình embedding (Word2Vec CBOW / Skip-gram) tối ưu hóa biểu diễn vector dựa trên ngữ cảnh xuất hiện.
  - **Quan hệ thay thế đồng nghĩa (Paradigmatic Relation / True Synonyms)**: `doctor` và `physician` cùng mang nghĩa "bác sĩ, thầy thuốc", có vai trò cú pháp và trường kết hợp ngữ nghĩa hoàn toàn tương đương (cùng làm chủ ngữ cho các động từ *treat, prescribe, examine*, cùng đi sau giới từ hoặc tính từ chuyên môn). Do đó, hàm phân phối ngữ cảnh $P(c \mid doctor)$ và $P(c \mid physician)$ gần như trùng khớp, buộc quá trình tối ưu hóa vector phải kéo chúng về cùng một vị trí lân cận trong không gian tiềm ẩn.
  - **Quan hệ liên tưởng chủ đề (Syntagmatic Relation / Topical Association)**: `hospital` là địa điểm làm việc của `doctor`. Chúng thường đồng xuất hiện trong cùng văn bản (topical co-occurrence), nhưng `hospital` là danh từ chỉ nơi chốn/tổ chức nên có những cấu trúc ngữ pháp riêng biệt (ví dụ: *in the hospital, go to the hospital*). Do đó, chúng gần nhau về mặt chủ đề nhưng độ tương đồng vector vẫn xếp sau cặp từ đồng nghĩa trực tiếp.
  - **Khác biệt trường ngữ nghĩa (Semantic Disjointness)**: `banana` và `car` hoàn toàn không có sự tương đồng về ngữ cảnh sử dụng với `doctor`, dẫn đến tích vô hướng vector tiến về 0 hoặc nhận giá trị âm.

- **Confidence (Độ tin cậy)**: **High (Rất cao)**

---

## 9.2. Prediction 2 — Nếu context window tăng từ 2 lên 5

### Câu hỏi nghiên cứu
> *Nếu kích thước cửa sổ ngữ cảnh (context window) tăng từ $2 \to 5$, độ tương đồng (similarity) giữa các từ có thay đổi không? Bản chất thay đổi như thế nào?*

### Lời giải & Dự đoán chính thức

- **Prediction (Dự đoán)**:
  - **CÓ THAY ĐỔI RÕ RỆT VỀ BẢN CHẤT HỌC NGỮ NGHĨA.**
  - Khi tăng cửa sổ từ $window = 2$ lên $window = 5$:
    1. Độ tương đồng giữa các từ liên quan về mặt **chủ đề rộng (Topical / Syntagmatic Relatedness)** như `(doctor, hospital)`, `(doctor, disease)`, `(cat, pet)` sẽ **tăng lên đáng kể**.
    2. Độ tập trung vào quan hệ **đồng nghĩa thay thế thuần túy (Paradigmatic / Syntactic Similarity)** như `(doctor, physician)` sẽ **bị pha loãng tương đối** so với các từ cùng chủ đề.

- **Reason (Cơ sở lý thuyết)**:
  - **Cửa sổ hẹp ($window = 2$) — Học đặc trưng cú pháp và thay thế (Syntactic / Functional Roles)**:
    - Khi chỉ xét 1–2 từ lân cận trực tiếp, ngữ cảnh chủ yếu phản ánh ràng buộc ngữ pháp cục bộ (từ loại, cấu trúc động từ - tân ngữ, giới từ đi kèm).
    - Các từ có thể thay thế trực tiếp vào cùng một vị trí mà không làm hỏng cấu trúc câu (như hai danh từ đồng nghĩa `doctor` và `physician`) sẽ đạt độ tương đồng cao nhất.
  - **Cửa sổ rộng ($window = 5$) — Học đặc trưng chủ đề / miền lĩnh vực (Topical / Semantic Domain)**:
    - Cửa sổ kích thước 5 bao quát từ 10 đến 11 token trong câu (5 từ trước + từ mục tiêu + 5 từ sau), mở rộng ra toàn bộ mệnh đề.
    - Lúc này, các từ cùng xuất hiện trong một ngữ cảnh chủ đề y tế (như *doctor, patient, nurse, treatment, hospital, medicine, surgery*) sẽ liên tục nằm trong cửa sổ của nhau. Mô hình xem các từ này là ngữ cảnh của nhau, dẫn đến việc vector của các từ cùng chủ đề bị kéo lại gần nhau.
  - Tóm lại: $window$ nhỏ ưu tiên **tính chất cú pháp và từ đồng nghĩa (interchangeable words)**; $window$ lớn ưu tiên **quan hệ chủ đề và liên tưởng ngữ nghĩa (domain association)**.

- **Confidence (Độ tin cậy)**: **High (Rất cao)**

---

## 9.3. Prediction 3 — Nếu embedding dimension tăng 50 -> 100 -> 300

### Câu hỏi nghiên cứu
> *Nếu số chiều embedding (embedding dimension) tăng từ $50 \to 100 \to 300$, chất lượng của vector biểu diễn có chắc chắn tăng không?*

### Lời giải & Dự đoán chính thức

- **Prediction (Dự đoán)**:
  - **KHÔNG CHẮC CHẮN TĂNG TRONG MỌI TRƯỜNG HỢP.**
  - Chất lượng biểu diễn chỉ tăng khi kích thước ngữ liệu huấn luyện (corpus size) tương xứng với dung lượng mô hình. Nếu ngữ liệu nhỏ hoặc trung bình, việc tăng số chiều lên $300$ sẽ dẫn đến hiện tượng **bão hòa (plateau)** hoặc thậm chí **suy giảm chất lượng (degradation due to overfitting)** kèm theo lãng phí lớn về tài nguyên tính toán.

- **Reason (Cơ sở lý thuyết)**:
  - **Đánh đổi giữa Dung lượng mô hình và Kích thước dữ liệu (Model Capacity vs Data Volume)**:
    - Số chiều $d$ xác định số lượng bậc tự do của không gian tiềm ẩn. Số lượng tham số cần tối ưu trong Word2Vec là $2 \times |\mathcal{V}| \times d$.
    - Khi tăng $d = 50 \to 100$: Không gian biểu diễn mở rộng, giúp giải quyết hiện tượng chen chúc (crowding problem), cho phép mã hóa độc lập nhiều sắc thái ngữ nghĩa khác nhau (giới tính, thì động từ, quan hệ số ít - số nhiều, lĩnh vực chuyên môn). Đây thường là điểm cân bằng tốt cho các ngữ liệu vừa và nhỏ.
    - Khi tăng tiếp $d = 100 \to 300$:
      - Nếu huấn luyện trên corpus khổng lồ (hàng tỷ từ như Google News 100B tokens), $d = 300$ cho chất lượng vượt trội vì dữ liệu đủ để cập nhật hội tụ mọi tọa độ.
      - Nhưng nếu huấn luyện trên ngữ liệu học tập/thực hành hạn chế, số chiều $300$ là quá dư thừa (overparameterization). Mô hình rơi vào hiện tượng quá khớp (overfitting), không gian bị thưa thớt hình học (geometric sparsity), các chiều tự do không nhận đủ gradient hữu ích mà bị chi phối bởi nhiễu ngẫu nhiên (noise).
  - **Quy luật hiệu suất biên giảm dần (Diminishing Returns)**: Chi phí bộ nhớ và thời gian tính toán tăng tuyến tính $O(d)$, nhưng mức độ cải thiện điểm tương đồng (similarity correlation) hay điểm suy luận tương tự (analogy accuracy) sẽ tiệm cận giới hạn bão hòa.

- **Confidence (Độ tin cậy)**: **High (Rất cao)**

---

## 9.4. Prediction 4 — Doctor và physician có chắc chắn gần nhau nếu corpus chỉ có 100 câu?

### Câu hỏi nghiên cứu
> *Hai từ `doctor` và `physician` có chắc chắn gần nhau trong không gian embedding không nếu ngữ liệu huấn luyện chỉ có vỏn vẹn 100 câu?*

### Lời giải & Dự đoán chính thức

- **Prediction (Dự đoán)**:
  - **KHÔNG CHẮC CHẮN (RẤT NHIỀU KHẢ NĂNG LÀ HOÀN TOÀN KHÔNG GẦN NHAU).**
  - Trong một corpus cực nhỏ chỉ gồm 100 câu, xác suất cao là vector của `doctor` và `physician` sẽ không thể hiện được tính đồng nghĩa, thậm chí có thể trực giao ($\cos \approx 0$) hoặc bị phân kỳ ngẫu nhiên.

- **Reason (Cơ sở lý thuyết)**:
  - **Bản chất Thống kê quy mô lớn của Giả thuyết Phân phối**:
    - Mô hình Word2Vec không học bằng tri thức logic hay từ điển mà hoàn toàn dựa vào thống kê lặp lại của các cặp đồng xuất hiện (empirical co-occurrence patterns) qua hàng ngàn bước cập nhật gradient descent.
  - **Thực tế phân phối trong 100 câu**:
    - Trong 100 câu văn bản tự nhiên, tổng số token chỉ dao động từ $1,500$ đến $2,500$ từ.
    - Tần số xuất hiện của các từ nội dung cụ thể như `doctor` hay `physician` là cực kỳ thấp (thường chỉ 0, 1 hoặc 2 lần). Thậm chí, một trong hai từ có thể hoàn toàn vắng mặt hoặc bị loại bỏ thẳng tay bởi ngưỡng tần số tối thiểu (`min_count >= 2`).
    - Nếu cả hai từ cùng xuất hiện 1 lần, khả năng chúng xuất hiện trong cùng một cấu trúc câu với các từ ngữ cảnh giống hệt nhau là gần như bằng 0 (ví dụ: một câu viết *"The doctor examined the child"*, câu khác viết *"The physician signed the paper"*).
  - **Hệ quả huấn luyện**:
    - Không có ngữ cảnh giao thoa (zero context overlap), hàm mất mát không hề nhận được tín hiệu kéo hai vector lại gần nhau.
    - Trọng số vector của hai từ này phần lớn vẫn giữ nguyên giá trị khởi tạo ngẫu nhiên ban đầu (random initialization) bị nhiễu bởi vài bước cập nhật cục bộ đơn lẻ.
  - **Kết luận**: Khả năng phản ánh ngữ nghĩa của Word Embedding chỉ phát huy tác dụng khi thỏa mãn điều kiện tiên quyết về quy mô ngữ liệu (Large Data Regime).

- **Confidence (Độ tin cậy)**: **High (Rất cao)**
