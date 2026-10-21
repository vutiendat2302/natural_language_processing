# BÁO CÁO TỔNG KẾT VÀ TỰ ĐÁNH GIÁ (REFLECTION REPORT - LAB 05)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Sinh viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  

---

# 28. Individual learning check & Reflection

Báo cáo này tổng kết sâu sắc toàn bộ hành trình thực nghiệm Lab 05, đối chiếu trực tiếp giữa các giả thuyết ban đầu với kết quả thực tế, đồng thời trả lời toàn diện các câu hỏi sát hạch cá nhân theo đúng rubric yêu cầu.

---

## 28.1. Trả lời toàn diện các câu hỏi sát hạch cá nhân (Learning Check Questions)

### Câu 1: RNN khác Average Word2Vec ở điểm nào cốt lõi?
* **Average Word2Vec (LAB 04)** biểu diễn câu bằng trung bình cộng vector từ: $\vec{v} = \frac{1}{N} \sum_{i=1}^N x_i$. Phép cộng vector có tính chất **giao hoán** ($A+B=B+A$), do đó hoàn toàn làm mất trật tự từ và cấu trúc ngữ pháp. Hai câu có ngữ nghĩa đối nghịch như *"dog bites man"* và *"man bites dog"* có biểu diễn y hệt nhau.
* **RNN (LAB 05)** xử lý từ tuần tự qua các bước thời gian $t$. Quá trình cập nhật trạng thái ẩn $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ là một chuỗi **hàm hợp lồng nhau phi giao hoán** ($f(x_A, f(x_B, h)) \neq f(x_B, f(x_A, h))$). Do đó, RNN giữ trọn vẹn được thứ tự từ, ngữ cảnh trước/sau và phạm vi phủ định.

---

### Câu 2: Hidden state ($h_t$) có ý nghĩa bản chất là gì?
* $h_t$ không chỉ là biểu diễn của riêng token hiện tại $x_t$, mà đóng vai trò là một **vector bộ nhớ ngữ cảnh động (dynamic context vector)**.
* Nó nén và tích lũy toàn bộ lịch sử thông tin ngôn ngữ từ bước $1$ đến bước $t$, đồng thời được điều biến liên tục qua từng token mới được nạp vào. $h_t$ chính là trạng thái nội tại định hướng cách mô hình hiểu các từ tiếp theo.

---

### Câu 3: Tại sao Simple RNN lại gặp hiện tượng Vanishing Gradient?
* Khi thực hiện lan truyền ngược qua thời gian (BPTT), gradient từ bước cuối $T$ truyền về bước $t$ chứa chuỗi tích ma trận Jacobian:
  $$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \operatorname{diag}(1 - h_k^2) W_{hh}^T$$
* Đạo hàm của $\tanh$ luôn $\le 1$. Nếu các trị riêng của ma trận trọng số hồi quy $W_{hh} < 1$, việc nhân ma trận lặp liên tiếp $T - t$ lần sẽ làm tích co lại theo hàm mũ ($\gamma^{T-t} \to 0$).
* Khi chuỗi dài hơn $20 \sim 30$ từ, gradient truyền về các bước đầu tiên bị triệt tiêu hoàn toàn về $0$, khiến mô hình không nhận được tín hiệu lỗi để cập nhật trọng số cho các liên kết tầm xa.

---

### Câu 4: Forget gate của LSTM dùng để làm gì?
* Forget gate ($f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \in (0, 1)$) là một cơ chế van điều tiết:
  - **Khi $f_t \approx 1$**: Mô hình chủ động **giữ lại toàn bộ thông tin cũ** trong Cell State, tạo thành "đường cao tốc gradient" giúp gradient truyền thẳng qua hàng trăm bước mà không bị suy giảm.
  - **Khi $f_t \approx 0$**: Mô hình chủ động **xóa bỏ / reset ô nhớ** khi ngữ cảnh cũ đã kết thúc (như chuyển câu, gặp liên từ chuyển ý *"However"*) để tránh bị nhiễu bởi các thông tin lỗi thời.

---

### Câu 5: Tại sao tăng sequence length có thể làm training chậm hơn?
* **Độ phức tạp tính toán**: Thuật toán BPTT phải unroll mạng qua $T$ bước thời gian, độ phức tạp tính toán tỷ lệ thuận bậc nhất với $T$: $\mathcal{O}(T \cdot d_h^2)$.
* **Chi phí bộ nhớ GPU**: GPU phải lưu giữ toàn bộ activations của cả $T$ bước trên mỗi mini-batch để phục vụ đạo hàm ngược: $\mathcal{O}(\text{batch\_size} \times T \times d_h)$.
* **Tỷ lệ lãng phí do đệm (Padding)**: Khi nâng $T$ lên cao, hàng loạt câu ngắn phải độn thêm nhiều token `<PAD>`, buộc GPU phải tính toán hàng triệu phép nhân ma trận trên các token giả.

---

### Câu 6: GRU khác LSTM ở điểm nào?
1. **Số lượng trạng thái**: LSTM duy trì 2 trạng thái riêng biệt (Cell State $C_t$ cho dài hạn và Hidden State $h_t$ cho ngắn hạn). GRU chỉ duy trì **duy nhất 1 trạng thái ẩn $h_t$**.
2. **Cơ chế cổng**: LSTM có 3 cổng riêng biệt (*Forget, Input, Output*). GRU chỉ có 2 cổng (*Reset gate $r_t$ và Update gate $z_t$*).
3. **Cơ chế cập nhật**: GRU kết hợp cổng quên và cổng vào thành một cơ chế nội suy tuyến tính lồi:
   $$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$
4. **Chi phí tham số & Tốc độ**: GRU có ít hơn LSTM khoảng $25\%$ số lượng tham số, giúp huấn luyện nhanh hơn và ít bị quá khớp hơn trên dữ liệu vừa và nhỏ.

---

### Câu 7: Tại sao mô hình có training loss giảm nhưng validation loss tăng?
* Đây là biểu hiện kinh điển của hiện tượng **Quá khớp (Overfitting)**.
* **Nguyên nhân**: Mô hình bắt đầu "học vẹt" (memorizing) các đặc trưng nhiễu, từ hiếm hoặc các chi tiết đặc thù chỉ có trong tập huấn luyện thay vì học các quy luật ngôn ngữ mang tính tổng quát.
* **Biện pháp khắc phục**: Áp dụng Dropout, Weight Decay ($L_2$ regularization), Early Stopping, hoặc giảm bớt dung lượng mô hình (giảm hidden dimension / số layer).

---

### Câu 8: Nếu một review bị truncate ở 30 tokens, thông tin gì có thể bị mất?
* Mất **câu kết luận tổng kết cảm xúc** (Conclusion) thường được người xem đặt ở cuối bài viết.
* Mất **liên từ chuyển ý tương phản** (*"However", "Although"*) trong cấu trúc khen trước chê sau.
* Mất **phạm vi phủ định tầm xa** và các cấu trúc câu điều kiện phức tạp.

---

## 28.2. Đối chiếu thực nghiệm với Dự đoán ban đầu (Prediction vs Reality)

1. **Prediction 1 (Word Order)**: ✅ **Hoàn toàn chính xác**. Thực nghiệm chứng minh TF-IDF đưa ra xác suất giống hệt nhau trên `"dog bites man"` và `"man bites dog"`, trong khi RNN phân biệt rõ ràng hai câu với độ chênh xác suất $0.2281$.
2. **Prediction 2 (RNN vs LSTM)**: ✅ **Hoàn toàn chính xác**. LSTM đạt Accuracy $76.86\%$ và F1 $78.12\%$, vượt trội hoàn toàn so với Simple RNN chỉ đạt $57.93\%$ và F1 $54.66\%$.
3. **Prediction 3 (Sequence Length)**: ✅ **Hoàn toàn chính xác**. Khi tăng $T$ từ 50 lên 200, thời gian train LSTM tăng từ 3.65s lên 9.60s (gần 3 lần), F1 tăng từ $70.13\%$ lên $78.12\%$.
4. **Prediction 5 (LSTM vs mọi dataset)**: ✅ **Chính xác**. Thực nghiệm trên IMDb cho thấy GRU thậm chí đạt kết quả cao hơn LSTM ($81.94\%$ vs $76.86\%$) do ít tham số hơn và hội tụ nhanh hơn trong 5 epochs.

---

## 28.3. Mạch nối tri thức từ LAB 01 đến LAB 05

```text
LAB 01: Vector Không Gian Rời Rạc (TF-IDF, Bag-of-Words, Khái niệm Vector Space)
   │
   ▼
LAB 02: Mô hình Xác Suất Thống Kê Chuỗi (N-gram Language Modeling, Perplexity)
   │
   ▼
LAB 03: Biểu Diễn Phân Tán Dày Đặc (Word2Vec, Skip-gram, CBOW, Semantic Regularities)
   │
   ▼
LAB 04: Biểu Diễn Văn Bản Cố Định (Average Word2Vec vs TF-IDF, Logistic Regression Classifier)
   │
   ▼
LAB 05: Mô Hình Chuỗi Hồi Quy & Cổng Bộ Nhớ (Simple RNN, LSTM, GRU, Gating Highway, BPTT)
```

Qua chuỗi bài thực hành, ta thấy rõ sự tiến hóa tất yếu của NLP: Từ các mô hình túi từ tĩnh không thứ tự (BoW) $\to$ không gian nhúng dày đặc $\to$ mô hình chuỗi động lực học theo thời gian (RNN/LSTM/GRU). Mặc dù LSTM/GRU đã giải quyết xuất sắc bài toán vanishing gradient ở độ dài vài trăm tokens, bản chất tuần tự từng bước một (sequential bottleneck) khiến chúng không thể tính toán song song hoàn toàn. Đây chính là tiền đề dẫn đến bước ngoặt vĩ đại tiếp theo: **Cơ chế Tự chú ý (Self-Attention) và Kiến trúc Transformer**.
