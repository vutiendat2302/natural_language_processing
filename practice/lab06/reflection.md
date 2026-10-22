# BÁO CÁO TỔNG KẾT VÀ TỰ ĐÁNH GIÁ (REFLECTION)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 06 — Attention Mechanism (From Recurrent State to Context-Aware Representation)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [Mạch nối chuyển giao từ LAB 05 sang LAB 06 (Mục 35)](#1-mạch-nối-chuyển-giao-từ-lab-05-sang-lab-06-mục-35)
2. [So sánh toàn diện Recurrent Model vs Attention Model (Mục 28)](#2-so-sánh-toàn-diện-recurrent-model-vs-attention-model-mục-28)
3. [Hai câu hỏi cốt lõi của bài Lab (Mục 28)](#3-hai-câu-hỏi-cốt-lõi-của-bài-lab-mục-28)
   - [3.1. Attention giải quyết vấn đề gì của Recurrent Processing?](#31-attention-giải-quyết-vấn-đề-gì-của-recurrent-processing)
   - [3.2. Attention tạo ra vấn đề mới nào?](#32-attention-tạo-ra-vấn-đề-mới-nào)
4. [Thực nghiệm bản chất: "Attention không miễn phí" (Mục 25 & 29)](#4-thực-nghiệm-bản-chất-attention-không-miễn-phí-mục-25--29)
5. [Sự khác biệt về Representation giữa LSTM và Attention (Mục 24)](#5-sự-khác-biệt-về-representation-giữa-lstm-và-attention-mục-24)
6. [Individual Learning Check — Trả lời 9 câu hỏi vấn đáp (Mục 31)](#6-individual-learning-check--trả-lời-9-câu-hỏi-vấn-đáp-mục-31)
7. [Bài học kinh nghiệm và Hướng phát triển tới Transformer](#7-bài-học-kinh-nghiệm-và-hướng-phát-triển-tới-transformer)

---

# 1. Mạch nối chuyển giao từ LAB 05 sang LAB 06 (Mục 35)

* **Ở LAB 05 (Recurrent State):**
  $$\text{Text} \to \text{Tokens} \to \text{Embedding} \to \text{RNN / LSTM / GRU} \to h_T \to \text{Classification}$$
  * *Câu hỏi trung tâm của LAB 05:* **Làm thế nào để duy trì thông tin qua thời gian/chuỗi?**
  * Mô hình giải quyết bằng cách tích lũy thông tin từng bước qua hàm chuyển trạng thái $h_t = f(h_{t-1}, x_t)$.
* **Ở LAB 06 (Attention Mechanism):**
  $$\text{Text} \to \text{Tokens} \to Q, K, V \to \text{Attention} \to \text{Context Representation } c \to \text{Classification}$$
  * *Câu hỏi trung tâm của LAB 06:* **Làm thế nào để chọn lọc đúng thông tin liên quan trong chuỗi?**
  * Mô hình giải quyết bằng cơ chế đánh trọng số động:
    $$c_i = \sum_{j=1}^T \alpha_{ij} v_j \quad \text{với} \quad \alpha_{ij} = \text{softmax}\left(\frac{q_i^T k_j}{\sqrt{d_k}}\right)$$
  * Đây là bước chuyển hóa căn bản: từ **bộ nhớ tuần tự bị suy hao** sang **truy vấn nội dung trực tiếp (Content-based Retrieval)**.

---

# 2. So sánh toàn diện Recurrent Model vs Attention Model (Mục 28)

Bảng đối chiếu chuẩn hóa theo Mục 28 của tài liệu:

| Đặc tính (Property) | RNN / LSTM / GRU (LAB 05) | Attention Mechanism (LAB 06) |
| :--- | :---: | :---: |
| **Bảo toàn thứ tự từ (Sequence order)** | Có ($\checkmark$ tự nhiên qua thứ tự bước $t$) | Không tự nhiên (cần thêm *Positional Encoding*) |
| **Duy trì Hidden state** | Bắt buộc ($\checkmark$) | Không bắt buộc (tùy chọn) |
| **Tính toán tuần tự (Sequential computation)** | Bắt buộc từng bước một ($\mathcal{O}(T)$ bước) | Không bắt buộc (xử lý song song ma trận) |
| **Tương tác trực tiếp giữa mọi token (Direct token-to-token)** | Rất hạn chế (phải truyền qua các trạm trung gian) | **Trực tiếp $100\%$ ($\checkmark$)** |
| **Xử lý phụ thuộc tầm xa (Long-range interaction)** | Rất khó khăn (*Vanishing Gradient*) | **Dễ dàng (Độ dài đường dẫn gradient $\mathcal{O}(1)$)** |
| **Khả năng song song hóa (Parallel computation)** | Rất hạn chế (GPU phải đợi timestep trước) | **Tối ưu vượt trội trên GPU/TPU ($\checkmark$)** |
| **Độ phức tạp tính toán theo độ dài chuỗi** | Tuyến tính $\mathcal{O}(T)$ | **Bậc hai $\mathcal{O}(T^2)$** cho full Self-Attention |
| **Biểu diễn ngữ cảnh (Context representation)** | Recurrent hidden state nén $h_T$ | **Tổ hợp có trọng số động (Weighted combination)** |

---

# 3. Hai câu hỏi cốt lõi của bài Lab (Mục 28)

## 3.1. Attention giải quyết vấn đề gì của Recurrent Processing?
1. **Xóa bỏ Nút thắt cổ chai thông tin (Information Bottleneck):**
   * Trong RNN, câu 100 từ bị ép phải nén vào đúng một vector ẩn kích thước $d$. Attention cho phép giữ lại toàn bộ $T$ vector và tạo ra Context Vector riêng biệt linh hoạt theo từng truy vấn.
2. **Khắc phục triệt để khoảng cách đường dẫn gradient (Gradient Path Length):**
   * Trong RNN, gradient từ bước $T$ muốn cập nhật cho từ ở bước $1$ phải nhân qua chuỗi ma trận $T$ lần $\implies$ triệt tiêu đạo hàm.
   * Trong Attention, đường dẫn gradient giữa bất kỳ cặp từ nào $(i, j)$ luôn là **$\mathcal{O}(1)$ (bước nhảy trực tiếp)**, bất kể khoảng cách cách nhau bao xa.
3. **Mở khóa năng lực tính toán song song (Parallelization):**
   * Toàn bộ ma trận điểm số $QK^T$ được tính bằng đúng một phép nhân ma trận (GEMM) trên GPU, không còn vòng lặp tuần tự `for t in range(T)`.

---

## 3.2. Attention tạo ra vấn đề mới nào?
1. **Bùng nổ chi phí tính toán và bộ nhớ ($\mathcal{O}(T^2)$ Complexity):**
   * Khi chuỗi dài $T$, ma trận Attention $T \times T$ đòi hỏi bộ nhớ và số phép tính tăng theo hàm bậc hai.
2. **Đánh mất nhận thức thứ tự từ tự nhiên (Loss of Positional Awareness):**
   * Do phép cộng có trọng số $c = \sum \alpha_i v_i$ và phép nhân tích vô hướng có tính giao hoán theo tập hợp (permutation equivariant), nếu xáo trộn thứ tự các từ trong câu, Attention cho ra cùng một tập hợp vector ngữ cảnh nếu không có **Positional Encoding**.
3. **Hiện tượng quá tự tin hoặc phân tán sai lệch:**
   * Attention dễ bị chi phối bởi các từ xuất hiện tần suất cao hoặc các mối tương quan giả (spurious correlations) trong dữ liệu huấn luyện nếu không có cơ chế điều hòa (regularization).

---

# 4. Thực nghiệm bản chất: "Attention không miễn phí" (Mục 25 & 29)

Một ngộ nhận cực kỳ phổ biến trong cộng đồng học máy là:
> *"Attention vượt trội hơn RNN nên luôn luôn thay thế hoàn toàn RNN trong mọi tình huống."*

Thực nghiệm định lượng tại **Mục 25 & 29** chứng minh điều ngược lại:

| Độ dài chuỗi ($T$) | Kích thước ma trận Attention ($T \times T$) | Số phần tử ma trận | Dung lượng bộ nhớ (float32) | Chi phí tương đối so với $T=10$ |
| :---: | :---: | :---: | :---: | :---: |
| **$10$** | $10 \times 10$ | $100$ | $0.4\text{ KB}$ | $1\times$ |
| **$100$** | $100 \times 100$ | $10.000$ | $40\text{ KB}$ | $100\times$ |
| **$500$** | $500 \times 500$ | $250.000$ | $1.0\text{ MB}$ | $2.500\times$ |
| **$1.000$** | $1.000 \times 1.000$ | **$1.000.000$ (1 triệu)** | $4.0\text{ MB}$ | **$10.000\times$** |
| **$10.000$** | $10.000 \times 10.000$ | **$100.000.000$ (100 triệu)** | **$400\text{ MB}$** | **$1.000.000\times$** |
| **$100.000$** | $100.000 \times 100.000$ | **$10.000.000.000$ (10 tỷ)** | **$40\text{ GB}$ (OOM)** | **$100.000.000\times$** |

### Nhận định then chốt:
* Khi chuỗi đạt tới kích thước tài liệu dài hoặc sách ($T \ge 10.000$), chỉ riêng việc lưu trữ ma trận Attention của **một tầng đơn lẻ** đã chiếm hàng trăm Megabytes; trên mô hình sâu nhiều tầng (Multi-layer) và nhiều đầu (Multi-head), bộ nhớ GPU lập tức bị tràn (**CUDA Out of Memory**).
* Trong khi đó, RNN chỉ tốn chi phí bộ nhớ tuyến tính $\mathcal{O}(T)$ khi inference. Đây chính là lý do các kiến trúc lai hiện đại như **Mamba, RWKV, Linear Attention** đang quay trở lại kết hợp ưu điểm của Recurrent State cho văn bản siêu dài.

---

# 5. Sự khác biệt về Representation giữa LSTM và Attention (Mục 24)

> **Câu hỏi trọng tâm:** *Representation của hai model khác nhau ở đâu?*

1. **Bản chất của Vector đại diện:**
   * **LSTM Representation ($h_T$):** Là một vector **đơn nhất mang tính tích lũy lịch sử (chronological accumulation)**. Mỗi chiều số học của $h_T$ là kết quả sau khi qua hàng loạt cổng quên ($f_t$), cổng vào ($i_t$), cổng ra ($o_t$). Vector này bị thiên lệch rất mạnh vào các token ở cuối chuỗi (Recency bias).
   * **Attention Representation ($c$):** Là một **tổ hợp lồi trực tiếp (convex combination)** từ các vector Value: $c = \sum \alpha_i v_i$. Mỗi chiều trong $c$ là sự đóng góp tường minh của các từ quan trọng mà không bị lọc qua thời gian.
2. **Khả năng giải thích (Interpretability):**
   * Vector $h_T$ của LSTM là một chiếc "hộp đen" hoàn toàn không thể bóc tách từ nào đóng góp bao nhiêu phần trăm.
   * Vector $c$ của Attention đi kèm ma trận trọng số $\alpha$, cho phép trực quan hóa Heatmap để quan sát phân phối chú ý của mô hình.

---

# 6. Individual Learning Check — Trả lời 9 câu hỏi vấn đáp (Mục 31)

### Câu 1: Attention score được tính như thế nào?
* **Trả lời:** Attention score đo mức độ tương đồng giữa Query $q$ và Key $k$. Trong dạng cơ bản, nó được tính bằng tích vô hướng: $s_i = q^T k_i$. Điểm càng lớn chứng tỏ hai vector càng chỉ về cùng một hướng trong không gian ngữ nghĩa.

### Câu 2: Tại sao phải dùng Softmax?
* **Trả lời:** Có 3 lý do: (1) Chuẩn hóa các điểm số thô bất kỳ $(-\infty, +\infty)$ thành phân phối xác suất hợp lệ có giá trị $0 < \alpha_i < 1$ và tổng $\sum \alpha_i = 1$; (2) Khuếch đại hàm mũ ($e^x$) giúp làm nổi bật token quan trọng nhất; (3) Hàm Softmax khả vi liên tục, cho phép tính đạo hàm để huấn luyện bằng lan truyền ngược (backpropagation).

### Câu 3: Query, Key và Value có vai trò gì?
* **Trả lời:** Lấy cảm hứng từ cơ sở dữ liệu:
  * **Query ($Q$):** Yêu cầu truy vấn ("Tôi đang cần tìm kiếm loại thông tin gì?").
  * **Key ($K$):** Nhãn nhận diện của từng token ("Tôi chứa đặc trưng gì để khớp với truy vấn?").
  * **Value ($V$):** Nội dung thông tin thực sự được trích xuất ("Đây là thông tin ngữ nghĩa tôi sẽ đóng góp vào biểu diễn cuối cùng").

### Câu 4: Context vector được tính như thế nào?
* **Trả lời:** Context vector được tính bằng tổng có trọng số (weighted sum) giữa các vector Value với các trọng số Attention tương ứng: $c = \sum_{i=1}^T \alpha_i v_i$ (hoặc dạng ma trận $C = A V$).

### Câu 5: Tại sao Scaled Dot-Product Attention phải chia cho $\sqrt{d_k}$?
* **Trả lời:** Khi chiều $d_k$ lớn, tích vô hướng $q^T k = \sum_{i=1}^{d_k} q_i k_i$ có phương sai tăng tuyến tính bằng $d_k$. Điểm số quá lớn sẽ đẩy Softmax vào vùng bão hòa có độ dốc cực nhỏ, gây ra triệt tiêu đạo hàm (vanishing gradient). Phép chia cho $\sqrt{d_k}$ đưa phương sai trở về $1$, đảm bảo gradient khỏe mạnh và ổn định.

### Câu 6: Self-Attention khác Attention thông thường ở đâu?
* **Trả lời:** Attention thông thường (Cross-Attention) có Query lấy từ một nguồn khác (Decoder) so với Key/Value (Encoder). Trong Self-Attention, cả 3 đại lượng $Q, K, V$ đều được sinh ra từ **cùng một chuỗi đầu vào $X$** thông qua các phép chiếu tuyến tính ($X W_Q, X W_K, X W_V$), cho phép các từ trong cùng một câu tự tương tác với nhau.

### Câu 7: Nếu sequence dài gấp đôi thì Attention Matrix thay đổi như thế nào?
* **Trả lời:** Nếu chiều dài chuỗi tăng gấp đôi ($T \to 2T$), kích thước ma trận Attention $T \times T$ sẽ tăng gấp **4 lần** ($(2T)^2 = 4T^2$), tức số phần tử tính toán và bộ nhớ tăng theo hàm bậc hai.

### Câu 8: Attention có chứng minh model đang suy luận (reasoning) không?
* **Trả lời:** **Không.** Trọng số Attention chỉ là các con số tính toán thống kê thuần túy thể hiện độ tương thích hình học giữa Query và Key trong không gian vector để giảm thiểu hàm mất mát. Không thể xem việc attention tập trung vào một từ là bằng chứng mô hình có nhận thức ngôn ngữ hay suy luận logic như con người.

### Câu 9: Tại sao Attention xử lý long-range interaction tốt hơn recurrent state?
* **Trả lời:** Vì đường dẫn truyền tín hiệu và gradient giữa hai từ bất kỳ trong Attention luôn có chiều dài là **$\mathcal{O}(1)$ (bước nhảy trực tiếp qua tích vô hướng)**, không phải đi qua $T$ bước đệ quy tuần tự như RNN. Nhờ đó thông tin tầm xa không bị suy hao hay quên lãng.

---

# 7. Bài học kinh nghiệm và Hướng phát triển tới Transformer

Qua LAB 06, sinh viên đã nắm vững:
1. Bản chất toán học từ tích vô hướng, hàm Softmax ổn định số học, phép chia tỷ lệ Scaling cho tới Context Vector.
2. Nắm rõ ưu điểm vượt trội về song song hóa và giải quyết bài toán tầm xa, nhưng đồng thời hiểu rõ cái giá phải trả về độ phức tạp $\mathcal{O}(T^2)$.
3. Đây là bước chuẩn bị hoàn hảo nhất để bước sang nghiên cứu **Multi-Head Attention**, **Positional Encoding**, và **Kiến trúc Transformer hoàn chỉnh** trong các tuần tiếp theo.
