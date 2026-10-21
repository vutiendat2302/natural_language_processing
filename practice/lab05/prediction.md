# BÁO CÁO DỰ ĐOÁN KHOA HỌC TRƯỚC THỰC NGHIỆM (PREDICTION REPORT - LAB 05)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Sinh viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  

---

# 10. Prediction trước experiment

Trước khi tiến hành huấn luyện các mô hình trong thực nghiệm, sinh viên ghi nhận đầy đủ 5 dự đoán khoa học kèm theo lập luận cơ chế mô hình và độ tin cậy theo đúng chuẩn quy trình khoa học:

---

### Prediction 1: Giữa TF-IDF + Logistic Regression và RNN + Embedding, model nào có khả năng xử lý word order trực tiếp hơn?
* **Prediction**: **RNN + Embedding** có khả năng xử lý thứ tự từ (word order) trực tiếp và vượt trội hoàn toàn.
* **Reason**: 
  - **TF-IDF + Logistic Regression** là mô hình túi từ (Bag-of-Words). Biểu diễn văn bản dựa trên tần suất từ ngữ có tính chất giao hoán: $\vec{v}(S_1) \equiv \vec{v}(S_2)$ khi hai câu cùng từ vựng. Hai câu đối nghịch nhau như `"dog bites man"` và `"man bites dog"` có vector TF-IDF y hệt nhau, khiến Logistic Regression không thể phân biệt được chủ ngữ và tân ngữ.
  - **RNN + Embedding** xử lý đầu vào tuần tự từng bước thời gian $t$. Trạng thái ẩn $h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$ là một chuỗi các hàm hợp lồng nhau phi giao hoán ($f(x_A, f(x_B, h)) \neq f(x_B, f(x_A, h))$). Thứ tự xuất hiện của token trực tiếp làm thay đổi quỹ đạo trạng thái ẩn, giúp mô hình nắm bắt cấu trúc ngữ pháp và phạm vi phủ định (negation scope) trực tiếp.
* **Confidence**: **Cao (High - 95%)**.

---

### Prediction 2: Giữa RNN và LSTM, model nào có khả năng xử lý dependency dài tốt hơn?
* **Prediction**: **LSTM** có khả năng nắm bắt và duy trì dependency dài tốt hơn đáng kể so với **Vanilla/Simple RNN**.
* **Reason**: 
  - **Simple RNN** gặp hiện tượng triệt tiêu đạo hàm nghiêm trọng (Vanishing Gradient) do ma trận Jacobian lặp $\prod_{k=t+1}^T \operatorname{diag}(1-h_k^2) W_{hh}^T$. Khi $T - t > 20 \sim 30$ tokens, tích này tiến dần về 0 theo hàm mũ, khiến tín hiệu lỗi không truyền ngược về đầu chuỗi và RNN bị "mất trí nhớ tầm xa".
  - **LSTM** giải quyết triệt để nhờ cơ chế **Cell State ($C_t$) cộng tính (Additive Update)**:
    $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
    Đạo hàm $\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^T f_k$. Khi Forget Gate $f_k \approx 1$, gradient có thể truyền thẳng qua hàng trăm bước như một "đường cao tốc" (gradient highway) mà không bị nhân liên tiếp với ma trận trọng số, ngăn chặn hiện tượng vanishing gradient.
* **Confidence**: **Cao (High - 95%)**.

---

### Prediction 3: Nếu sequence length tăng từ 50 lên 200, training của RNN sẽ thay đổi như thế nào?
* **Prediction**:
  1. Thời gian huấn luyện (Training time) mỗi epoch sẽ tăng tuyến tính khoảng $3 \sim 4$ lần do độ phức tạp tính toán BPTT là $\mathcal{O}(T \cdot d_h^2)$.
  2. Hiện tượng Vanishing Gradient trong Simple RNN trở nên cực kỳ trầm trọng, gradient chuẩn tại các timestep đầu tiên gần như bằng 0.
  3. Hiệu năng (Accuracy/F1) của Simple RNN gần như không tăng, thậm chí có thể giảm do nhiễu và memory decay; trong khi LSTM/GRU sẽ tận dụng tốt hơn ngữ cảnh dài.
* **Reason**: Việc nhân ma trận Jacobian qua 199 bước thời gian khiến gradient suy giảm mạnh hơn gấp hàng triệu lần so với 49 bước ($\gamma^{199} \ll \gamma^{49}$).
* **Confidence**: **Cao (High - 90%)**.

---

### Prediction 4: Nếu truncate toàn bộ review về 30 tokens, thông tin classification có thể bị ảnh hưởng không?
* **Prediction**: **CÓ, ẢNH HƯỞNG RẤT NẶNG NỀ (Độ chính xác suy giảm từ 10% đến 15%)**.
* **Reason**:
  - Dữ liệu IMDb có độ dài trung vị là 173 từ, trung bình 231 từ. Cắt tại 30 tokens làm vứt bỏ tới hơn $85\%$ dung lượng bài viết.
  - Cấu trúc bài phê bình phim thường mở đầu bằng tóm tắt cốt truyện và giới thiệu diễn viên (thông tin trung lập). Câu chốt hạ khen/chê (conclusion) quyết định nhãn thường nằm ở cuối bài viết. Việc cắt cụt ở 30 tokens đầu tiên sẽ tước đoạt mất tín hiệu phân loại quan trọng nhất.
* **Confidence**: **Cao (High - 90%)**.

---

### Prediction 5: LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?
* **Prediction**: **KHÔNG CHẮC CHẮN.**
* **Reason**:
  1. Trên các tập dữ liệu chuỗi rất ngắn ($T < 10 \sim 15$ từ như tiêu đề báo, tweet ngắn), Simple RNN không gặp vấn đề vanishing gradient nên năng lực biểu diễn là hoàn toàn tương đương.
  2. LSTM có tới 4 cổng/tầng nội bộ ($f_t, i_t, o_t, \tilde{c}_t$), số tham số gấp gần 4 lần Simple RNN. Trên các tập dữ liệu ít mẫu huấn luyện, số lượng tham số lớn khiến LSTM rất dễ bị **quá khớp (overfitting)**, trong khi Simple RNN lại tổng quát hóa tốt hơn.
* **Confidence**: **Tuyệt đối (Absolute - 98%)**.
