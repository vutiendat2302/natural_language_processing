# BẢN DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (PRE-EXPERIMENT PREDICTIONS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 06 — Attention Mechanism  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

## 1. Giới thiệu

Tài liệu này ghi lại các dự đoán mang tính định lượng và bản chất lý thuyết **trước khi tiến hành thực nghiệm code** trong LAB 06, tuân thủ đúng định dạng đánh giá học thuật:
$$\text{Prediction} \longrightarrow \text{Reason} \longrightarrow \text{Confidence} \longrightarrow \text{Post-experiment Verification}$$

---

## 2. Chi tiết 5 dự đoán trước thực nghiệm (Mục 12)

### Prediction 1: Key có dot product lớn hơn

* **Tình huống khảo sát:** Nếu một key $k_i$ có dot product với query $q$ lớn hơn các key khác ($s_i = q^T k_i > s_j = q^T k_j, \forall j \neq i$), attention weight của nó sẽ như thế nào?
* **Dự đoán (Prediction):**
  Attention weight của key đó ($\alpha_i$) chắc chắn sẽ **lớn nhất** trong toàn bộ phân phối:
  $$\alpha_i > \alpha_j \quad \forall j \neq i$$
* **Lý do (Reason):**
  Hàm Softmax được định nghĩa là $\alpha_k = \frac{\exp(s_k)}{\sum_m \exp(s_m)}$. Đạo hàm của hàm số mũ tự nhiên $\frac{d}{dx}(e^x) = e^x > 0$ với mọi $x \in \mathbb{R}$, tức $e^x$ là hàm đồng biến ngặt. Khi mẫu số là một hằng số dương cố định, thứ tự của các điểm số thô $s$ được bảo toàn nguyên vẹn sau ánh xạ Softmax.
* **Độ tin cậy (Confidence):** **High** (100%).
* **Kiểm chứng sau thực nghiệm (Post-experiment Verification):**
  * Thử nghiệm với $s = [2.0, 1.0, 0.0]$: Score lớn nhất là $s_1 = 2.0 \implies \alpha_1 = 0.6652$ (áp đảo $0.2447$ và $0.0900$).
  * Thử nghiệm với $s = [1.0, 2.0, 3.0]$: Score lớn nhất là $s_3 = 3.0 \implies \alpha_3 = 0.6652$ (áp đảo $0.0900$ và $0.2447$).
  * **Kết luận:** Dự đoán chính xác $100\%$.

---

### Prediction 2: Tất cả scores giống nhau

* **Tình huống khảo sát:** Nếu tất cả scores bằng nhau tuyệt đối: $s = [1, 1, 1, 1]$ (hoặc $s_i = C$ với mọi $i \in \{1, \dots, T\}$), attention weights sẽ như thế nào?
* **Dự đoán (Prediction):**
  Attention weights sẽ tạo thành một **phân phối đều tuyệt đối (Uniform Distribution)**:
  $$\alpha_i = \frac{1}{T} = \frac{1}{4} = 0.25 \quad \forall i \in \{1, 2, 3, 4\}$$
* **Lý do (Reason):**
  Khi $s_1 = s_2 = s_3 = s_4 = C$, tử số của mỗi phần tử đều là $e^C$. Mẫu số là $\sum_{j=1}^4 e^C = 4 e^C$. Khi đó:
  $$\alpha_i = \frac{e^C}{4 e^C} = \frac{1}{4} = 0.25$$
  Mô hình hoàn toàn không thể phân biệt được token nào liên quan hơn, entropy đạt cực đại $H_{\max} = \ln(4) \approx 1.3863$ nats.
* **Độ tin cậy (Confidence):** **High** (100%).
* **Kiểm chứng sau thực nghiệm (Post-experiment Verification):**
  * Code NumPy chạy với $s = [1, 1, 1, 1]$ cho kết quả: `alpha = [0.25, 0.25, 0.25, 0.25]`.
  * **Kết luận:** Dự đoán chính xác $100\%$.

---

### Prediction 3: Một score lớn vượt trội

* **Tình huống khảo sát:** Nếu một score lớn hơn rất nhiều so với các phần tử còn lại: $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?
* **Dự đoán (Prediction):**
  Attention sẽ **tập trung gần như $100\%$ ($> 99.9\%$) vào phần tử thứ 3**, các phần tử còn lại bị đè bẹp xấp xỉ bằng $0$. Phân phối tiệm cận dạng *Hard Attention* hoặc vector *One-hot*.
* **Lý do (Reason):**
  Hàm mũ phóng đại sự chênh lệch tuyến tính thành chênh lệch cấp số nhân:
  * $e^1 \approx 2.718$
  * $e^{10} \approx 22026.466$
  * Tỷ số: $\frac{e^{10}}{e^1} = e^9 \approx 8103.08$ lần.
  * Mẫu số: $\sum = 2 \times 2.718 + 22026.466 = 22031.902$.
  * Trọng số phần tử 3: $\alpha_3 = \frac{22026.466}{22031.902} \approx \mathbf{0.99975} \quad (99.98\%)$.
* **Độ tin cậy (Confidence):** **High** (100%).
* **Kiểm chứng sau thực nghiệm (Post-experiment Verification):**
  * Code NumPy chạy cho kết quả: `alpha = [0.00012, 0.00012, 0.99975]`.
  * **Kết luận:** Dự đoán chính xác $100\%$.

---

### Prediction 4: Chi phí tính toán khi Sequence Length tăng

* **Tình huống khảo sát:** Nếu độ dài sequence $T$ tăng từ $10\text{ tokens} \to 1000\text{ tokens}$ (tăng gấp 100 lần), chi phí tính attention sẽ thay đổi như thế nào?
* **Dự đoán (Prediction):**
  * Đối với **Self-Attention toàn phần** (nhân ma trận $Q K^T$ giữa $T$ tokens với $T$ tokens): Chi phí tính toán và bộ nhớ tăng gấp **$10.000$ lần** (độ phức tạp $\mathcal{O}(T^2)$).
  * Đối với **Attention với một Query đơn lẻ** ($q^T K$): Chi phí tăng **$100$ lần** (độ phức tạp $\mathcal{O}(T)$).
* **Lý do (Reason):**
  Trong Self-Attention, mỗi token phải so khớp với tất cả các token khác. Kích thước ma trận Attention $A$ là $T \times T$.
  * Khi $T = 10 \implies$ Ma trận có $10^2 = 100$ phần tử.
  * Khi $T = 1000 \implies$ Ma trận có $1000^2 = 1.000.000$ phần tử.
  * Tỷ số tăng trưởng: $\frac{1.000.000}{100} = 10.000 = (100)^2$ lần.
* **Độ tin cậy (Confidence):** **High** (95%).
* **Kiểm chứng sau thực nghiệm (Post-experiment Verification):**
  * Thực nghiệm đo đạc thời gian tính toán và kích thước mảng trong Mục 25 xác nhận số phép tính dot-product tỉ lệ thuận tuyệt đối với $T^2$.
  * **Kết luận:** Dự đoán chính xác $100\%$.

---

### Prediction 5: Khả năng căn chỉnh của Attention với trực giác con người

* **Tình huống khảo sát:** Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
* **Dự đoán (Prediction):**
  **Không luôn luôn.** Mức độ chú ý của mô hình (attention weights) không nhất thiết phải trùng khớp với những gì con người trực giác cho là "từ khóa quan trọng".
* **Lý do (Reason):**
  1. **Tối ưu hóa toán học vs Nhận thức sinh học:** Attention weights được học gián tiếp thông qua việc tối ưu hàm mất mát (loss function) của toàn mạng để dự đoán nhãn phân loại, không có bất kỳ nhãn giám sát (ground truth) nào ép buộc attention phải nhìn đúng ngữ pháp của con người.
  2. **Nhiễu và Hư từ (Stopwords):** Rất nhiều nghiên cứu (như *Jain & Wallace, 2019 - "Attention is not Explanation"*) chỉ ra rằng Attention thường xuyên phân bổ trọng số lớn vào dấu câu (như dấu chấm `.`, dấu phẩy `,`) hoặc hư từ (*"the", "is"*) để đóng vai trò như một bộ chứa thông tin trung gian (information sink).
* **Độ tin cậy (Confidence):** **High** (90%).
* **Kiểm chứng sau thực nghiệm (Post-experiment Verification):**
  * Trong Thực nghiệm 3 và Mục 27 (Error Analysis), mô hình có những câu chú ý mạnh vào từ *"was"* hoặc *"is"* thay vì từ khóa cảm xúc cốt lõi, nhưng vẫn dự đoán đúng (hoặc ngược lại).
  * **Kết luận:** Dự đoán hoàn toàn phù hợp với thực tế và cảnh báo quan trọng trong tài liệu LAB 06: *Phân biệt Attention Pattern với Linguistic Reasoning*.

---

## 3. Tổng kết bảng dự đoán

| Prediction | Câu hỏi cốt lõi | Dự đoán tóm tắt | Độ tin cậy | Kết quả thực nghiệm |
| :---: | :--- | :--- | :---: | :---: |
| **1** | Dot product lớn nhất $\to$ Weight? | Luôn nhận weight lớn nhất ($\alpha_{\max}$) | High (100%) | **ĐÚNG** |
| **2** | Scores bằng nhau $\to$ Weight? | Phân phối đều $\alpha_i = 1/T$ | High (100%) | **ĐÚNG** |
| **3** | Một score lớn vượt trội ($s_3 = 10$)? | Tập trung $\approx 99.98\%$ vào phần tử đó | High (100%) | **ĐÚNG** |
| **4** | Tăng sequence từ 10 lên 1000? | Tăng $(100)^2 = 10.000$ lần phép tính | High (95%) | **ĐÚNG** |
| **5** | Attention luôn giống con người? | Không luôn luôn (không đồng nhất với reasoning) | High (90%) | **ĐÚNG** |
