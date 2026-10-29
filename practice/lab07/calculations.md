# BÀI TẬP TÍNH TOÁN, LÝ THUYẾT VÀ PHÂN TÍCH ĐỊNH LƯỢNG (LAB 07)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 07 — Transformer Architecture (From Attention to Transformer)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 1. Bối cảnh

* **LAB 05 (Recurrent Models):**
  $$\text{Sequence} \to \text{RNN / LSTM / GRU} \to \text{Hidden state} \to \text{Classification}$$
  Mô hình hóa chuỗi bằng cách xử lý tuần tự qua từng bước thời gian $t = 1 \to T$.
* **LAB 06 (Attention Mechanism):**
  $$\text{Query} + \text{Key / Value} \to \text{Attention weights} \to \text{Context vector}$$
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
* **LAB 07 (Transformer Architecture):**
  Trả lời câu hỏi lớn: *Làm thế nào ghép cơ chế Self-Attention thành một kiến trúc hoàn chỉnh để xử lý văn bản?*
  $$\text{Input embeddings} \to \text{Positional Encoding} \to \text{Multi-Head Self-Attention} \to \text{Residual + LayerNorm} \to \text{Feed-Forward Network} \to \text{Residual + LayerNorm} \to \text{Transformer Encoder}$$

---

# 2. Mục tiêu học tập

Hoàn thành đầy đủ 13 mục tiêu đầu ra (**LO1 – LO13**): từ việc giải thích và tính toán Multi-Head Attention, Positional Encoding, Feed-Forward Network, Residual + LayerNorm, cho tới tự ghép một `TransformerBlock` tối giản, kiểm tra Tensor Shapes và đánh giá thực nghiệm trên bài toán phân loại văn bản.

---

# 3. Cấu trúc lab

Bao gồm 11 giai đoạn liên hoàn: Theory recap $\to$ Calculation $\to$ Prediction $\to$ Experiment 1 (Positional encoding) $\to$ Core implementation (Transformer block) $\to$ Experiment 2 (Attention heads) $\to$ Experiment 3 (Transformer vs RNN) $\to$ Application (Text classification) $\to$ Error analysis $\to$ Reflection.

---

# 4. Từ Self-Attention đến Transformer

* **Self-attention của LAB 06:**
  $$X \to Q, K, V \to \text{Attention} \to Z$$
* **Một Transformer Encoder block hoàn chỉnh bổ sung:**
  $$Z \to \text{Residual + Norm} \to \text{FFN} \to \text{Residual + Norm}$$

```text
Input X
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Add & Norm  (LayerNorm(X + MHA(X)))
  │
  ▼
Feed-Forward Network
  │
  ▼
Add & Norm  (LayerNorm(H_1 + FFN(H_1)))
  │
  ▼
Output
```

---

# 5. Tại sao cần Multi-Head Attention?

Một attention head đơn lẻ có thể chỉ tập trung vào một kiểu quan hệ (ví dụ: `subject <-> verb`, `negation <-> sentiment word`, hoặc `pronoun <-> entity`).  
*(Lưu ý học thuật: Đây là động lực trực quan; không giả định cứng nhắc rằng mỗi head bắt buộc phải ánh xạ 1-1 với một quy tắc ngôn ngữ cố định).*

**Công thức toán học của Multi-Head Attention:**
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

---

# 6. Bài tập tính toán — Một attention head

### Đề bài
Cho ma trận đầu vào $X \in \mathbb{R}^{3 \times 4}$ (3 tokens, $d_{\text{model}} = 4$).  
Một head sử dụng $d_k = 2$ với các ma trận chiếu $W^Q, W^K, W^V \in \mathbb{R}^{4 \times 2}$:

$$X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix}, \quad W^Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 1 \end{bmatrix}, \quad W^V = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

### Các bước tính toán chi tiết

#### Bước 1: Tính $Q = XW^Q, \ K = XW^K, \ V = XW^V$
$$Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

$$K = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

$$V = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 1 & 2 \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

#### Bước 2: Tính Scaled Scores $S = \frac{QK^T}{\sqrt{2}}$
$$QK^T = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 2 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 & 2 \\ 0 & 4 & 2 \\ 1 & 2 & 2 \end{bmatrix}$$

$$S = \frac{QK^T}{\sqrt{2}} \approx \begin{bmatrix} 1.4142 & 0.0000 & 1.4142 \\ 0.0000 & 2.8284 & 1.4142 \\ 0.7071 & 1.4142 & 1.4142 \end{bmatrix}$$

#### Bước 3: Tính Attention Weights $A = \text{softmax}(S)$
* Hàng 1: $\text{softmax}([1.4142, 0.0000, 1.4142]) = [\mathbf{0.4458}, \mathbf{0.1084}, \mathbf{0.4458}]$
* Hàng 2: $\text{softmax}([0.0000, 2.8284, 1.4142]) = [\mathbf{0.0454}, \mathbf{0.7679}, \mathbf{0.1867}]$
* Hàng 3: $\text{softmax}([0.7071, 1.4142, 1.4142]) = [\mathbf{0.1978}, \mathbf{0.4011}, \mathbf{0.4011}]$

$$A = \begin{bmatrix} 0.4458 & 0.1084 & 0.4458 \\ 0.0454 & 0.7679 & 0.1867 \\ 0.1978 & 0.4011 & 0.4011 \end{bmatrix}$$

#### Bước 4: Tính Context Representation $Z = AV$
$$Z = \begin{bmatrix} 0.4458 & 0.1084 & 0.4458 \\ 0.0454 & 0.7679 & 0.1867 \\ 0.1978 & 0.4011 & 0.4011 \end{bmatrix} \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} \mathbf{1.3374} & \mathbf{1.2168} \\ \mathbf{0.2775} & \mathbf{2.6771} \\ \mathbf{0.7967} & \mathbf{2.0056} \end{bmatrix} \in \mathbb{R}^{3 \times 2}$$

**Nhận xét:** Một attention head chính là cơ chế Self-Attention của LAB 06 được gắn thêm các ma trận chiếu giảm chiều từ $d_{\text{model}} = 4$ xuống $d_k = 2$.

---

# 7. Bài tập — Multi-head

Cho 2 attention heads: $\text{head}_1 \in \mathbb{R}^{T \times 2}$ và $\text{head}_2 \in \mathbb{R}^{T \times 2}$.  
Sau khi ghép nối:
$$\text{Concat}(\text{head}_1, \text{head}_2) \in \mathbb{R}^{T \times 4}$$

### Trả lời câu hỏi:
> **Vì sao cần concatenation trước khi áp dụng $W^O$?**

1. **Khôi phục chiều biểu diễn gốc ($d_{\text{model}} = 4$):** Mỗi head hoạt động trên không gian con $d_v = 2$. Phép `Concat` nối $2$ heads cạnh nhau để gom lại đủ $2 + 2 = 4$ chiều, giúp đầu ra khớp kích thước với đầu vào $X \in \mathbb{R}^{T \times 4}$ để thực hiện phép cộng Residual Connection ($X + \text{MHA}(X)$).
2. **Tổng hợp và hòa trộn thông tin liên-head (Cross-head mixing):** Các heads được tính toán hoàn toàn độc lập. Khi đặt cạnh nhau thành một vector $4$ chiều rồi nhân với ma trận $W^O \in \mathbb{R}^{4 \times 4}$, mỗi chiều đầu ra sẽ là tổ hợp tuyến tính của tất cả các chiều từ cả $\text{head}_1$ và $\text{head}_2$, cho phép hợp nhất các góc nhìn ngữ nghĩa khác nhau.
3. **Bảo toàn đặc trưng trước khi chiếu:** Khác với phép cộng trực tiếp ($\text{head}_1 + \text{head}_2$) dễ làm triệt tiêu đặc trưng đối dấu, `Concat` giữ nguyên vẹn mọi thành phần của từng head để $W^O$ tự học trọng số kết hợp tối ưu.

---

# 8. Prediction trước experiment

*(Chi tiết toàn bộ 6 dự đoán và bảng kiểm chứng được trình bày đầy đủ tại file `prediction.md`).*

---

# 9. Positional Encoding

Công thức Sinusoidal Positional Encoding:
$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
Đầu vào đưa vào Transformer:
$$X' = X + PE$$

---

# 10. Bài tập tính positional encoding

### Đề bài
Cho $d = 4$ và $pos = 0, 1, 2$. Tính vector $PE(pos)$ cho từng vị trí.

### Lời giải chi tiết
Với $d = 4$:
* Khi $i = 0$ (chiều $0, 1$): Mẫu số $= 10000^{0/4} = 1 \implies PE(pos, 0) = \sin(pos), \ PE(pos, 1) = \cos(pos)$.
* Khi $i = 1$ (chiều $2, 3$): Mẫu số $= 10000^{2/4} = 100 \implies PE(pos, 2) = \sin(pos/100), \ PE(pos, 3) = \cos(pos/100)$.

1. **Vị trí $pos = 0$:**
   $$PE(0) = [\sin(0), \ \cos(0), \ \sin(0), \ \cos(0)] = [\mathbf{0.0000}, \ \mathbf{1.0000}, \ \mathbf{0.0000}, \ \mathbf{1.0000}]$$
2. **Vị trí $pos = 1$:**
   $$PE(1) = [\sin(1), \ \cos(1), \ \sin(0.01), \ \cos(0.01)] \approx [\mathbf{0.8415}, \ \mathbf{0.5403}, \ \mathbf{0.0100}, \ \mathbf{1.0000}]$$
3. **Vị trí $pos = 2$:**
   $$PE(2) = [\sin(2), \ \cos(2), \ \sin(0.02), \ \cos(0.02)] \approx [\mathbf{0.9093}, \ \mathbf{-0.4161}, \ \mathbf{0.0200}, \ \mathbf{0.9998}]$$

**Nhận xét:** Mỗi vị trí $pos = 0, 1, 2$ nhận một vector biểu diễn vị trí hoàn toàn khác biệt nhau.

---

# 11. Experiment 1 — Positional encoding

Khảo sát trên sequence `["I", "love", "NLP"]` và câu đảo thứ tự `["NLP", "love", "I"]`.

### Trả lời câu hỏi:
> **Nếu chỉ dùng self-attention và không có positional information, tại sao thứ tự trở thành vấn đề?**
* **Trả lời:** Trong Self-Attention thuần túy, điểm tương đồng $s_{ij} = \frac{q_i^T k_j}{\sqrt{d_k}}$ chỉ phụ thuộc vào nội dung của hai vector $x_i$ và $x_j$ mà không hề phụ thuộc vào khoảng cách hay chỉ số vị trí $i, j$. Do đó, khi hoán vị câu `"I love NLP"` thành `"NLP love I"`, vector ngữ cảnh của từ `"I"`, `"love"`, `"NLP"` giữ nguyên giá trị số học tuyệt đối (chỉ đổi hàng). Mô hình không thể biết ai là chủ thể yêu cái gì.

---

# 12. Permutation experiment

Thử nghiệm định lượng trên hai câu có cùng từ vựng:
* $S_1 =$ `"dog bites man"`
* $S_2 =$ `"man bites dog"`

| Thực nghiệm | Biểu diễn Mean Pooling của $S_1$ | Biểu diễn Mean Pooling của $S_2$ | Khoảng cách Euclidean $\|r(S_1) - r(S_2)\|_2$ | Kết luận thực nghiệm |
| :--- | :---: | :---: | :---: | :--- |
| **Experiment A (Không PE)** | $[0.6777, 0.6777, 0.3223, 0.3223]$ | $[0.6777, 0.6777, 0.3223, 0.3223]$ | **$0.000000$** | Hai câu cho ra biểu diễn trùng khít $100\%$, mô hình hoàn toàn thất bại trong việc phân biệt thứ tự. |
| **Experiment B (Có PE)** | $[1.2244, 1.0392, 0.3123, 1.3971]$ | $[1.2972, 0.9639, 0.2900, 1.3755]$ | **$0.109214$** | Nhờ cộng $PE$, hai câu tạo ra hai vector khác biệt rõ rệt, phân biệt được chủ ngữ và tân ngữ. |

---

# 13. Feed-Forward Network

Công thức Position-wise Feed-Forward Network:
$$\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2$$
* **Vai trò:** Trong khi Self-Attention cho phép các token tương tác và trao đổi thông tin với nhau (cross-token), FFN áp dụng phép biến đổi phi tuyến độc lập trên từng vị trí (position-wise) để trích xuất đặc trưng mức cao.

---

# 14. Bài tập tính FFN

### Đề bài
Cho:
$$x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad W_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0 \\ -1 \end{bmatrix}$$

### Các bước tính toán chi tiết
1. **Tính $h = \text{ReLU}(W_1 x + b_1)$:**
   $$W_1 x + b_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix} + \begin{bmatrix} 0 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
   $$h = \text{ReLU}\left(\begin{bmatrix} 1 \\ 1 \end{bmatrix}\right) = \begin{bmatrix} \max(0, 1) \\ \max(0, 1) \end{bmatrix} = \begin{bmatrix} \mathbf{1} \\ \mathbf{1} \end{bmatrix}$$
2. **Tính $y = W_2 h + b_2$:**
   $$y = W_2 \begin{bmatrix} 1 \\ 1 \end{bmatrix} + b_2$$
   *(Ví dụ với $W_2 = \begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}, b_2 = \begin{bmatrix} 0 \\ 0 \end{bmatrix} \implies y = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$).*
3. **Nhận xét mục tiêu:** FFN là mạng MLP truyền thẳng tác động lên từng vector đơn lẻ, hoàn toàn không phải là một cơ chế Attention thứ hai.

---

# 15. Residual connection

Công thức:
$$y = x + F(x)$$

### Trả lời câu hỏi:
> **Nếu chỉ dùng $F(x)$, điều gì khác với việc sử dụng $x + F(x)$?**
1. **Truyền dẫn đạo hàm (Optimization):** Với $y = x + F(x)$, đạo hàm là $\frac{\partial y}{\partial x} = \mathbf{I} + \frac{\partial F(x)}{\partial x}$. Hạng tử đơn vị $\mathbf{I}$ tạo đường truyền trực tiếp cho gradient chạy về các tầng đầu mà không bị triệt tiêu (vanishing gradient) như khi chỉ dùng $F(x)$.
2. **Giữ thông tin gốc:** Nếu chỉ dùng $F(x)$, biểu diễn gốc của token dễ bị biến dạng qua nhiều tầng. Kết nối $x + F(x)$ bảo toàn thông tin đầu vào $x$ và chỉ yêu cầu $F(x)$ học phần hiệu chỉnh bổ sung (residual).

---

# 16. Layer Normalization

LayerNorm chuẩn hóa từng vector token độc lập theo chiều đặc trưng ($d_{\text{model}}$) về trung bình bằng $0$ và phương sai bằng $1$:
$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
Giúp ổn định phân phối kích hoạt bên trong khối Transformer và tăng tốc độ hội tụ.

---

# 17. Core implementation — Transformer block

*(Mã nguồn hoàn chỉnh được cài đặt tại file `transformer_block.py`).*

---

# 18. Kiểm tra shape

Cho $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$ với:
$$B = 8, \quad T = 100, \quad d_{\text{model}} = 128, \quad n_{\text{heads}} = 4 \implies d_k = \frac{128}{4} = 32, \quad d_{\text{ff}} = 512$$

### Chuỗi biến đổi Shape chuẩn xác:
```text
X                 : (8, 100, 128)
  ↓
Q                 : (8, 4, 100, 32)
K                 : (8, 4, 100, 32)
V                 : (8, 4, 100, 32)
  ↓
Attention scores  : (8, 4, 100, 100)   [do (8, 4, 100, 32) @ (8, 4, 32, 100)]
  ↓
Attention output  : (8, 100, 128)      [sau khi nhân V -> (8, 4, 100, 32), Concat 4 heads & nhân W_O]
  ↓
FFN               : (8, 100, 128)      [Linear1 -> (8, 100, 512) -> ReLU -> Linear2 -> (8, 100, 128)]
  ↓
Output            : (8, 100, 128)      [sau Add & LayerNorm]
```

---

# 19. Experiment 2 — Number of attention heads

Giữ cố định $d_{\text{model}} = 128$, khảo sát $n_{\text{heads}} \in \{1, 2, 4, 8\}$:

| Heads ($n_{\text{heads}}$) | $d_k = d_{\text{model}} / n_{\text{heads}}$ | Parameters | Training time | Accuracy | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $128$ | $198,274$ | $21.4\text{s}$ | $87.12\%$ | $0.8712$ |
| **2** | $64$ | $198,274$ | $22.1\text{s}$ | $88.04\%$ | $0.8803$ |
| **4** | $32$ | $198,274$ | $23.2\text{s}$ | **$88.60\%$** | **$0.8856$** |
| **8** | $16$ | $198,274$ | $25.0\text{s}$ | $88.32\%$ | $0.8830$ |

### Trả lời câu hỏi:
1. **Tăng số heads có đồng nghĩa với tăng representation dimension không?**
   * **Không.** Tổng chiều biểu diễn $d_{\text{model}} = 128$ và tổng số tham số của khối ($198,274$) giữ nguyên không đổi, vì ma trận chiếu được chia nhỏ cho các heads.
2. **Khi $d_{\text{model}}$ cố định, tăng số heads làm $d_k$ thay đổi như thế nào?**
   * $d_k$ giảm tỉ lệ nghịch với số heads: $d_k = \frac{d_{\text{model}}}{n_{\text{heads}}}$ ($128 \to 64 \to 32 \to 16$). Khi $d_k$ quá nhỏ (như ở $8$ heads, $d_k=16$), dung lượng biểu diễn của từng không gian con bị thu hẹp làm hiệu năng bắt đầu bão hòa.

---

# 20. Experiment 3 — Attention head diversity

Quan sát biểu đồ 4 heads tại file `attention_heads.png`:
* **Head 1 (Tập trung mạnh trên đường chéo):** Chú ý cục bộ vào bản thân token và từ liền kề.
* **Head 2 (Tập trung mạnh theo cặp từ cảm xúc):** Tập trung nhọn vào cặp phủ định `not <-> good` và chuyển ý `but <-> shone`.
* **Head 3 (Tập trung vừa phải theo cụm thực thể):** Liên kết danh từ với vị ngữ (`movie <-> was`, `acting <-> shone`).
* **Head 4 (Phân tán - Diffuse):** Phân phối trọng số trải rộng gần như đồng đều trên toàn câu để thu thập ngữ cảnh nền.

**Trả lời câu hỏi:** *Multi-head attention thực sự tạo ra các attention patterns khác nhau rõ rệt trong thực nghiệm, giúp mô hình nắm bắt đồng thời cả thông tin cục bộ lẫn tương tác tầm xa.*

---

# 21. Experiment 4 — Transformer vs RNN/LSTM

So sánh trên cùng tập dữ liệu phân loại cảm xúc IMDb với cùng cấu hình huấn luyện:

| Model | Parameters | Training time | Accuracy | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | $199,810$ | $49.8\text{s}$ | $85.24\%$ | $0.8540$ | $0.8504$ | $0.8522$ |
| **Transformer** | **$198,274$** | **$23.2\text{s}$** | **$88.60\%$** | **$0.8888$** | **$0.8824$** | **$0.8856$** |

**Trả lời câu hỏi:** *Sự khác biệt về hiệu năng và tốc độ xuất phát từ: (1) Cơ chế Self-Attention cho phép kết nối trực tiếp $\mathcal{O}(1)$ giữa từ phủ định/liên từ phản đề với tính từ cảm xúc mà không bị suy hao qua chuỗi cổng đệ quy như LSTM; (2) Tính toán ma trận song song toàn chuỗi giúp thời gian huấn luyện của Transformer nhanh hơn gấp $2.1$ lần.*

---

# 22. Experiment 5 — Sequence length

Khảo sát ảnh hưởng của `max_length` $T \in \{32, 64, 128, 256\}$:

| Length ($T$) | Tokens retained (%) | Attention matrix ($T^2$) | Attention memory (batch=32) | Training time | F1 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **32** | $24.5\%$ | $1,024$ | $0.5\text{ MB}$ | $8.4\text{s}$ | $0.8134$ |
| **64** | $46.8\%$ | $4,096$ | $2.0\text{ MB}$ | $13.6\text{s}$ | $0.8545$ |
| **128** | $78.2\%$ | $16,384$ | $8.0\text{ MB}$ | $23.2\text{s}$ | $0.8856$ |
| **256** | $94.6\%$ | **$65,536$** | **$32.0\text{ MB}$** | **$58.9\text{s}$** | **$0.8910$** |

**Trả lời câu hỏi:** *Transformer không thể giải quyết mọi vấn đề chỉ bằng cách tăng sequence length vì kích thước ma trận Attention tăng theo hàm bậc hai $\mathcal{O}(T^2)$ ($128 \to 256$ làm số phần tử tăng gấp 4 lần từ $16,384 \to 65,536$). Khi $T$ lên tới hàng chục nghìn token, bộ nhớ GPU sẽ bị tràn (OOM) và thời gian tính toán tăng vọt.*

---

# 23. Application — Transformer text classification

Kiểm thử mô hình phân loại văn bản trên 5 dạng câu điển hình:

| Dạng câu kiểm thử | Văn bản đầu vào (Input) | Dự đoán | Xác suất |
| :--- | :--- | :---: | :---: |
| **1. Câu ngắn (Short)** | `"I really enjoyed this movie."` | **Positive** | $96.4\%$ |
| **2. Câu dài (Long sequence)** | `"From the opening cinematography to the final orchestral score, every single scene kept the audience completely captivated."` | **Positive** | $95.1\%$ |
| **3. Phủ định (Negation)** | `"I did not enjoy the plot at all, it was neither funny nor engaging."` | **Negative** | $93.8\%$ |
| **4. Tương phản (Contrast)** | `"The trailer looked terrible and cheap, but the actual film turned out to be a masterpiece."` | **Positive** | $91.2\%$ |
| **5. Mơ hồ (Ambiguous)** | `"It is a visually strange film that some viewers might love while others will walk out."` | **Positive** | $53.4\%$ |

---

# 24. Pooling & 25. Experiment 6 — Pooling strategy

So sánh hai chiến lược tổng hợp biểu diễn câu từ ma trận $H \in \mathbb{R}^{T \times d}$:

| Pooling Strategy | Công thức | Accuracy | F1 | Training time |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Pooling** | $h_{\text{mean}} = \frac{1}{T}\sum_{t=1}^T h_t$ | **$88.60\%$** | **$0.8856$** | $23.2\text{s}$ |
| **First-token / CLS-style** | $h_{\text{cls}} = h_1$ | $87.88\%$ | $0.8786$ | $23.0\text{s}$ |

**Trả lời câu hỏi:** *Cách lấy sequence representation có ảnh hưởng rõ rệt tới kết quả phân loại. Với mô hình Transformer nông (1-2 tầng) huấn luyện từ đầu, **Mean Pooling** cho kết quả tốt hơn vì nó tổng hợp trực tiếp tín hiệu từ mọi vị trí trong câu, trong khi CLS-style pooling đòi hỏi mạng nhiều tầng sâu hoặc pretraining (như BERT) để token đầu tiên học cách gom đủ toàn bộ thông tin câu.*
