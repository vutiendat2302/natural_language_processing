# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM VÀ KIỂM CHỨNG (LAB 07 - PREDICTIONS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 07 — Transformer Architecture (From Attention to Transformer)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 8. Prediction trước experiment

Trước khi tiến hành chạy thực nghiệm code, toàn bộ 6 dự đoán được ghi nhận theo đúng khung chuẩn: **Prediction – Reason – Confidence**.

### Prediction 1
* **Câu hỏi:** *Một Transformer có thể xử lý các token trong sequence song song hơn RNN không?*
* **Prediction:** **Có, song song hóa vượt trội hơn hẳn RNN.**
* **Reason:** RNN bị ràng buộc bởi công thức đệ quy $h_t = f(h_{t-1}, x_t)$, buộc phải tính tuần tự từng bước thời gian $t = 1 \to T$. Ngược lại, Transformer tính ma trận tương quan $Q K^T$ và mạng FFN đồng thời cho toàn bộ $T$ vị trí bằng các phép nhân ma trận trên GPU trong đúng 1 bước lan truyền tiến.
* **Confidence:** **High** ($100\%$).

### Prediction 2
* **Câu hỏi:** *Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?*
* **Prediction:** **Không, mô hình hoàn toàn mất nhận thức về vị trí.**
* **Reason:** Phép tính Self-Attention ($q_i^T k_j$ và $\sum_j \alpha_{ij} v_j$) và FFN áp dụng tại từng vị trí đều có tính chất **bất biến tương đương với hoán vị (Permutation Equivariance)**. Nếu không cộng thêm Positional Encoding, mô hình chỉ coi đầu vào như một tập hợp từ không có thứ tự (Bag of Tokens).
* **Confidence:** **High** ($100\%$).

### Prediction 3
* **Câu hỏi:** *Nếu hai câu có cùng tập token nhưng khác thứ tự, Transformer không có positional information có thể phân biệt chúng không?*
* **Prediction:** **Hoàn toàn không thể phân biệt được.**
* **Reason:** Với hai câu hoán vị như `"dog bites man"` và `"man bites dog"`, nếu không có Positional Encoding, tập các vector ngữ cảnh đầu ra của hai câu là giống hệt nhau (chỉ đổi chỗ các hàng). Sau khi qua bước Mean Pooling để lấy biểu diễn toàn câu, hai vector câu trùng khớp $100\%$ (khoảng cách L2 bằng $0$).
* **Confidence:** **High** ($100\%$).

### Prediction 4
* **Câu hỏi:** *Hai attention heads có nhất thiết học cùng một attention pattern không?*
* **Prediction:** **Không nhất thiết (và thông thường sẽ học các pattern khác nhau).**
* **Reason:** Mỗi head có bộ ma trận chiếu $(W_i^Q, W_i^K, W_i^V)$ riêng được khởi tạo ngẫu nhiên khác nhau và chiếu vào các không gian con (subspaces) độc lập. Khi tối ưu hóa bằng Gradient Descent, các heads có xu hướng phân hóa để bắt các mối quan hệ khác nhau (cục bộ, phủ định, thực thể - vị ngữ), dù đôi khi vẫn có thể xuất hiện một số head dư thừa (redundant).
* **Confidence:** **High** ($95\%$).

### Prediction 5
* **Câu hỏi:** *Tăng số attention heads có chắc chắn làm accuracy tăng không?*
* **Prediction:** **Không chắc chắn.**
* **Reason:** Khi giữ cố định $d_{\text{model}} = 128$, việc tăng số heads $n_{\text{heads}}$ sẽ làm giảm số chiều của mỗi head theo công thức $d_k = d_{\text{model}} / n_{\text{heads}}$. Nếu tăng số heads quá lớn (ví dụ $8$ hoặc $16$ heads), chiều không gian con $d_k$ trở nên quá nhỏ ($16$ hoặc $8$), làm giảm năng lực biểu diễn của từng head và có thể gây bão hòa hoặc giảm nhẹ Accuracy/F1.
* **Confidence:** **High** ($95\%$).

### Prediction 6
* **Câu hỏi:** *Transformer có luôn nhanh hơn RNN trên mọi sequence length không?*
* **Prediction:** **Không luôn luôn.**
* **Reason:** Ở chuỗi trung bình ($T \le 256$), Transformer nhanh hơn nhờ song song hóa trên GPU. Tuy nhiên, chi phí bộ nhớ và tính toán của Self-Attention tăng theo hàm bậc hai $\mathcal{O}(T^2)$ (ma trận $T \times T$), trong khi RNN chỉ tăng tuyến tính $\mathcal{O}(T)$. Khi $T$ rất lớn, chi phí $T^2$ của Transformer sẽ bùng nổ và chậm hơn RNN.
* **Confidence:** **High** ($95\%$).

---

# 28. Prediction verification

Đối chiếu 6 dự đoán ban đầu với kết quả thực nghiệm thu được từ Mục 12, 19, 20, 21, 22:

| Prediction | Dự đoán ban đầu | Kết quả thực nghiệm | Đúng/Sai | Giải thích & Đối chiếu định lượng |
| :---: | :--- | :--- | :---: | :--- |
| **1** | Transformer xử lý song song nhanh hơn RNN | Thời gian train Transformer ($23.2\text{s}$) nhanh gấp đôi LSTM ($49.8\text{s}$) ở $T=128$ | **Đúng** | Toàn bộ $T$ tokens được tính toán đồng thời qua phép nhân ma trận thay vì vòng lặp tuần tự $T$ bước của LSTM. |
| **2** | Bỏ PE thì Transformer không biết thứ tự | Khi đảo câu `"I love NLP"` $\to$ `"NLP love I"`, các hàng đầu ra chỉ hoán vị vị trí, không đổi giá trị | **Đúng** | Self-Attention thuần túy là phép toán tập hợp (Permutation Equivariant), hoàn toàn mù vị trí nếu thiếu PE. |
| **3** | Không có PE không phân biệt được 2 câu hoán vị | Ở Mục 12 (Exp A), khoảng cách L2 giữa `"dog bites man"` và `"man bites dog"` sau Mean Pooling bằng $0.000000$ | **Đúng** | Chỉ khi cộng PE (Exp B), khoảng cách L2 mới tăng lên $0.109214$, giúp phân biệt chủ ngữ và tân ngữ. |
| **4** | Hai heads không nhất thiết học cùng pattern | Ở Mục 20 (`attention_heads.png`), 4 heads tạo ra 4 bản đồ chú ý khác biệt (cục bộ, phủ định, thực thể, phân tán) | **Đúng** | Các ma trận chiếu độc lập giúp mỗi head trích xuất một quan hệ tương tác khác nhau trong không gian con. |
| **5** | Tăng số heads không chắc chắn làm tăng Accuracy | Ở Mục 19, Accuracy tăng từ $1\text{ head } (87.12\%) \to 4\text{ heads } (88.60\%)$, nhưng giảm nhẹ ở $8\text{ heads } (88.32\%)$ | **Đúng** | Khi $n_{\text{heads}}=8$, chiều $d_k$ giảm xuống chỉ còn $16$, làm hạn chế độ phân giải tích vô hướng của từng head. |
| **6** | Transformer không luôn nhanh hơn ở mọi độ dài $T$ | Ở Mục 22, khi $T$ tăng từ $128 \to 256$, kích thước ma trận tăng $4\times$ ($16,384 \to 65,536$), thời gian tăng vọt lên $58.9\text{s}$ | **Đúng** | Quy luật bậc hai $\mathcal{O}(T^2)$ khiến thời gian và bộ nhớ của Transformer tăng gấp 4 lần mỗi khi nhân đôi chiều dài chuỗi. |
