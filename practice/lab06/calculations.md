# BÀI TẬP TÍNH TOÁN VÀ LÝ THUYẾT ĐỊNH LƯỢNG (LAB 06)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: From Recurrent State to Context-Aware Representation  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục

- [4. Theory recap](#4-theory-recap)
  - [4.1 Từ hidden state sang context vector](#41-từ-hidden-state-sang-context-vector)
- [5. Attention score](#5-attention-score)
- [6. Softmax attention weights](#6-softmax-attention-weights)
- [7. Context vector](#7-context-vector)
- [8. Bài tập tính toán (Bài 1 — Attention scores)](#8-bài-tập-tính-toán-bài-1--attention-scores)
- [9. Bài 2 — Softmax](#9-bài-2--softmax)
- [10. Bài 3 — Context vector](#10-bài-3--context-vector)
- [11. Bài 4 — Toàn bộ attention computation](#11-bài-4--toàn-bộ-attention-computation)
- [12. Prediction trước experiment](#12-prediction-trước-experiment)
  - [Prediction 1](#prediction-1)
  - [Prediction 2](#prediction-2)
  - [Prediction 3](#prediction-3)
  - [Prediction 4](#prediction-4)
  - [Prediction 5](#prediction-5)
- [13. Scaled dot-product attention](#13-scaled-dot-product-attention)
- [14. Bài tập — Scaling](#14-bài-tập--scaling)
- [15. Temperature](#15-temperature)
- [16. Core implementation](#16-core-implementation)
- [17. Numerical stability](#17-numerical-stability)
- [20. Self-attention](#20-self-attention)
- [21. Bài tập self-attention](#21-bài-tập-self-attention)
- [22. Attention matrix](#22-attention-matrix)

---

# 4. Theory recap

## 4.1 Từ hidden state sang context vector

Ở LAB 05, một cách đơn giản để classification là sử dụng hidden state cuối cùng:
$$x_1 \to x_2 \to x_3 \to \dots \to x_T \implies h_T \implies \text{classifier}$$

### Vấn đề:
> *Liệu $h_T$ có phải luôn là representation tốt nhất của toàn bộ sequence?*

* **Câu trả lời: Không.**
* **Lý do:**
  1. **Nút thắt cổ chai thông tin (Information Bottleneck):** Mọi ngữ nghĩa của chuỗi dài $T$ token bị ép buộc phải nén chặt vào một vector có kích thước cố định $h_T \in \mathbb{R}^d$.
  2. **Triệt tiêu gradient (Vanishing Gradient):** Do thông tin phải đi qua chuỗi đệ quy $h_t = f(h_{t-1}, x_t)$, các token quan trọng nằm ở đầu hoặc giữa câu (ví dụ trong câu *"The movie was extremely long and boring, but the final scene was absolutely fantastic"*) sẽ bị suy giảm hoặc lấn át khi truyền tới $h_T$.

### Giải pháp của Attention:
Thay vì chỉ sử dụng $h_T$, mô hình xem xét toàn bộ tập hợp:
$$\{h_1, h_2, \dots, h_T\}$$
và tự động quyết định xem hidden state nào quan trọng hơn đối với mục tiêu dự đoán:

```text
h1 ──┐
h2 ──┤
h3 ──┤ ───> Attention ───> Context vector (c)
h4 ──┤
h5 ──┘
```

---

# 5. Attention score

Cho vector query: $q$ và các key vectors: $k_1, k_2, \dots, k_T$.  
Công thức **Dot-product score**:
$$s_i = q^T k_i$$
* **Ý nghĩa:** Điểm số $s_i$ càng lớn thể hiện Query $q$ và Key $k_i$ càng tương đồng trong không gian biểu diễn (góc giữa 2 vector càng nhỏ).

### Ví dụ minh họa trong tài liệu:
Cho:
$$q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$

Tính toán:
$$s_1 = q^T k_1 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = (1 \times 1) + (0 \times 0) = \mathbf{1}$$
$$s_2 = q^T k_2 = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = (1 \times 0) + (0 \times 1) = \mathbf{0}$$

> **Nhận xét:** Do $s_1 = 1 > s_2 = 0$, query có xu hướng chú ý tới $k_1$ nhiều hơn $k_2$ (vì $q$ và $k_1$ hoàn toàn cùng hướng).

---

# 6. Softmax attention weights

Các điểm số thô $s_i$ được chuyển đổi thành các trọng số dạng xác suất (probability-like weights):
$$\alpha_i = \text{softmax}(s)_i = \frac{\exp(s_i)}{\sum_{j} \exp(s_j)}$$
Do đó luôn bảo toàn:
$$\sum_{i} \alpha_i = 1, \quad 0 < \alpha_i < 1$$

### Bài tập ví dụ trong tài liệu:
Cho vector điểm số:
$$s = [2, 1, 0]$$

#### 1. Tính $\alpha = \text{softmax}(s)$:
* Lũy thừa:
  * $e^{s_1} = e^2 \approx 7.3891$
  * $e^{s_2} = e^1 \approx 2.7183$
  * $e^{s_3} = e^0 = 1.0000$
* Mẫu số chuẩn hóa:
  $$\sum_{j=1}^3 e^{s_j} = 7.3891 + 2.7183 + 1.0000 = \mathbf{11.1074}$$
* Từng trọng số:
  * $\alpha_1 = \frac{7.3891}{11.1074} \approx \mathbf{0.6652} \quad (66.52\%)$
  * $\alpha_2 = \frac{2.7183}{11.1074} \approx \mathbf{0.2447} \quad (24.47\%)$
  * $\alpha_3 = \frac{1.0000}{11.1074} \approx \mathbf{0.0900} \quad (9.00\%)$

$$\alpha = [\mathbf{0.6652}, \mathbf{0.2447}, \mathbf{0.0900}]$$
Kiểm tra tổng: $0.6652 + 0.2447 + 0.0900 = \mathbf{1.0000}$.

#### 2. Trả lời câu hỏi:
> **Token nào nhận attention lớn nhất?**
* **Trả lời:** **Token 1** (vị trí index 0) nhận attention lớn nhất với trọng số $\mathbf{\alpha_1 \approx 66.52\%}$.

---

# 7. Context vector

Sau khi có attention weights $\alpha_1, \alpha_2, \dots, \alpha_T$, context vector $c$ được tính bằng tổng có trọng số:
$$c = \sum_{i=1}^{T} \alpha_i v_i$$
Trong đó $v_i$ là vector Value tương ứng với key $k_i$.

### Toàn bộ quá trình tính toán cốt lõi:
```text
Query
│
├── dot(K)
↓
Scores
│
├── softmax
↓
Attention weights
│
├── weighted sum(V)
↓
Context vector
```

---

# 8. Bài tập tính toán (Bài 1 — Attention scores)

### Đề bài
Cho query:
$$q = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
Và ma trận key $K \in \mathbb{R}^{3 \times 2}$:
$$K = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \implies k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

**Yêu cầu:** Tính $s_i = q^T k_i$ và viết toàn bộ score vector.

### Lời giải chi tiết
* $s_1 = q^T k_1 = 1 \times 1 + 2 \times 0 = \mathbf{1}$
* $s_2 = q^T k_2 = 1 \times 0 + 2 \times 1 = \mathbf{2}$
* $s_3 = q^T k_3 = 1 \times 1 + 2 \times 1 = 1 + 2 = \mathbf{3}$

**Dưới dạng nhân ma trận:**
$$s = K q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$$

**Kết luận:** Toàn bộ score vector là:
$$s = [1, 2, 3]^T$$

---

# 9. Bài 2 — Softmax

### Đề bài
Cho vector scores:
$$s = [1, 2, 3]$$

**Yêu cầu:**
1. Tính $\alpha = \text{softmax}(s)$.
2. Chỉ ra rằng $\sum_{i} \alpha_i = 1$.
3. Trả lời: *Vì sao phần tử có score lớn nhất nhận weight lớn nhất?*

### Lời giải chi tiết

#### 1. Tính $\alpha = \text{softmax}(s)$
* Lũy thừa:
  * $e^1 \approx 2.7183$
  * $e^2 \approx 7.3891$
  * $e^3 \approx 20.0855$
* Mẫu số: $\sum = 2.7183 + 7.3891 + 20.0855 = \mathbf{30.1929}$
* Trọng số attention:
  * $\alpha_1 = \frac{2.7183}{30.1929} \approx \mathbf{0.0900} \quad (9.00\%)$
  * $\alpha_2 = \frac{7.3891}{30.1929} \approx \mathbf{0.2447} \quad (24.47\%)$
  * $\alpha_3 = \frac{20.0855}{30.1929} \approx \mathbf{0.6652} \quad (66.52\%)$

$$\alpha = [\mathbf{0.0900}, \mathbf{0.2447}, \mathbf{0.6652}]$$

#### 2. Chỉ ra $\sum_i \alpha_i = 1$
* **Số học:** $0.0900 + 0.2447 + 0.6652 = \mathbf{1.0000}$.
* **Đại số:** $\sum_{i=1}^3 \alpha_i = \frac{e^1 + e^2 + e^3}{\sum_{j=1}^3 e^j} = \frac{\sum_{j=1}^3 e^j}{\sum_{j=1}^3 e^j} = 1$.

#### 3. Trả lời: Vì sao phần tử có score lớn nhất nhận weight lớn nhất?
1. **Tính đơn điệu tăng ngặt:** Hàm số mũ $f(x) = e^x$ có đạo hàm $f'(x) = e^x > 0, \forall x \in \mathbb{R}$. Khi $s_3 > s_2 > s_1 \implies e^{s_3} > e^{s_2} > e^{s_1}$. Mẫu số chung là hằng số dương nên thứ tự độ lớn được giữ nguyên: $\alpha_3 > \alpha_2 > \alpha_1$.
2. **Khuếch đại hàm mũ:** Sự chênh lệch tuyến tính ở đầu vào ($s_3 - s_2 = 1$) được khuếch đại theo cấp số nhân $\frac{\alpha_3}{\alpha_2} = e \approx 2.72$ lần, làm cho phần tử cao nhất luôn áp đảo trọng số.

---

# 10. Bài 3 — Context vector

### Đề bài
Cho trọng số attention:
$$\alpha = [0.1, 0.2, 0.7]$$
Các vector giá trị:
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$

**Yêu cầu:**
1. Tính context vector: $c = 0.1 v_1 + 0.2 v_2 + 0.7 v_3$.
2. Trả lời và giải thích: *Value nào đóng góp nhiều nhất vào context vector?*

### Lời giải chi tiết
$$c = 0.1 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.2 \begin{bmatrix} 0 \\ 1 \end{bmatrix} + 0.7 \begin{bmatrix} 2 \\ 2 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.1 \times 1 + 0.2 \times 0 + 0.7 \times 2 \\ 0.1 \times 0 + 0.2 \times 1 + 0.7 \times 2 \end{bmatrix} = \begin{bmatrix} 0.1 + 0 + 1.4 \\ 0 + 0.2 + 1.4 \end{bmatrix} = \begin{bmatrix} \mathbf{1.5} \\ \mathbf{1.6} \end{bmatrix}$$

### Trả lời và giải thích:
> **Value nào đóng góp nhiều nhất vào context vector?**
* **Trả lời:** **$v_3$ đóng góp nhiều nhất.**
* **Giải thích:**
  1. Trọng số $\alpha_3 = 0.7$ lớn nhất (chiếm tới $70\%$ tổng trọng số chú ý).
  2. Thành phần đóng góp của $v_3$ là $0.7 v_3 = [1.4, 1.4]^T$. So với vector tổng $c = [1.5, 1.6]^T$:
     * Chiều thứ nhất: $v_3$ chiếm $1.4 / 1.5 \approx \mathbf{93.33\%}$.
     * Chiều thứ hai: $v_3$ chiếm $1.4 / 1.6 = \mathbf{87.50\%}$.
     * Độ dài đóng góp: $\|0.7 v_3\|_2 \approx 1.98$, vượt trội so với $\|0.1 v_1\|_2 = 0.1$ và $\|0.2 v_2\|_2 = 0.2$.

---

# 11. Bài 4 — Toàn bộ attention computation

### Đề bài
Cho query:
$$q = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$
Các key:
$$k_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad k_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad k_3 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$
Các value:
$$v_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 0 \\ 2 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$

**Yêu cầu:** Thực hiện toàn bộ: $q \to \text{scores} \to \text{softmax} \to \text{attention weights} \to \text{weighted sum} \to \text{context vector}$ (không sử dụng thư viện attention).

### Lời giải chi tiết

#### Bước 1: Tính scores ($s_i = q^T k_i$)
* $s_1 = q^T k_1 = 1 \times 1 + 0 \times 0 = \mathbf{1}$
* $s_2 = q^T k_2 = 1 \times 0 + 0 \times 1 = \mathbf{0}$
* $s_3 = q^T k_3 = 1 \times 1 + 0 \times 1 = \mathbf{1}$

$$\implies s = [1, 0, 1]$$

#### Bước 2: Softmax $\to$ Attention weights ($\alpha$)
* $e^1 \approx 2.71828, \quad e^0 = 1.00000, \quad e^1 \approx 2.71828$
* Mẫu số: $\sum = 2.71828 + 1.00000 + 2.71828 = \mathbf{6.43656}$
* Các trọng số:
  * $\alpha_1 = \frac{2.71828}{6.43656} \approx \mathbf{0.4223} \quad (42.23\%)$
  * $\alpha_2 = \frac{1.00000}{6.43656} \approx \mathbf{0.1554} \quad (15.54\%)$
  * $\alpha_3 = \frac{2.71828}{6.43656} \approx \mathbf{0.4223} \quad (42.23\%)$

$$\implies \alpha = [\mathbf{0.4223}, \mathbf{0.1554}, \mathbf{0.4223}]$$

#### Bước 3: Weighted sum(V) $\to$ Context vector ($c$)
$$c = 0.4223 \begin{bmatrix} 1 \\ 0 \end{bmatrix} + 0.1554 \begin{bmatrix} 0 \\ 2 \end{bmatrix} + 0.4223 \begin{bmatrix} 2 \\ 1 \end{bmatrix}$$
$$c = \begin{bmatrix} 0.4223 \times 1 + 0.1554 \times 0 + 0.4223 \times 2 \\ 0.4223 \times 0 + 0.1554 \times 2 + 0.4223 \times 1 \end{bmatrix} = \begin{bmatrix} 0.4223 + 0 + 0.8446 \\ 0 + 0.3108 + 0.4223 \end{bmatrix} = \begin{bmatrix} \mathbf{1.2669} \\ \mathbf{0.7331} \end{bmatrix}$$

---

# 12. Prediction trước experiment

### Prediction 1
* **Câu hỏi:** Nếu một key có dot product lớn hơn các key khác, attention weight của nó sẽ như thế nào?
* **Prediction:** Attention weight của nó sẽ **lớn nhất**.
* **Reason:** Hàm số mũ $e^x$ đồng biến ngặt trên $\mathbb{R}$. Chia cho mẫu số chung dương giữ nguyên thứ tự bất đẳng thức.
* **Confidence:** **High** (100%).

### Prediction 2
* **Câu hỏi:** Nếu tất cả scores giống nhau: $s = [1, 1, 1, 1]$, attention weights sẽ như thế nào?
* **Prediction:** Attention weights sẽ **chia đều bằng nhau tuyệt đối**: $\alpha = [0.25, 0.25, 0.25, 0.25]$ (phân phối đều).
* **Reason:** Khi $s_i$ bằng nhau, mọi tử số $e^{s_i}$ bằng nhau $\implies \alpha_i = \frac{e^1}{4 \cdot e^1} = \frac{1}{4} = 0.25$.
* **Confidence:** **High** (100%).

### Prediction 3
* **Câu hỏi:** Nếu một score lớn hơn rất nhiều: $s = [1, 1, 10]$, attention sẽ tập trung như thế nào?
* **Prediction:** Attention sẽ tập trung **gần như 100% vào phần tử thứ 3** (tiến tới *Hard Attention* / one-hot).
* **Reason:** $e^{10} \approx 22026.47 \gg e^1 \approx 2.72$. Trọng số $\alpha_3 \approx \frac{22026.47}{22031.91} \approx 99.98\%$.
* **Confidence:** **High** (100%).

### Prediction 4
* **Câu hỏi:** Nếu sequence tăng từ $10\text{ tokens} \to 1000\text{ tokens}$, chi phí tính attention sẽ thay đổi như thế nào?
* **Prediction:** Tăng gấp **$10.000$ lần** đối với Self-Attention toàn phần ($\mathcal{O}(T^2)$), hoặc tăng **$100$ lần** nếu chỉ có 1 Query đơn lẻ ($\mathcal{O}(T)$).
* **Reason:** Kích thước ma trận tương quan giữa $T$ tokens với nhau là $T \times T$. Khi $T$ tăng từ $10$ lên $1000$ (gấp 100 lần), số phép tính tăng gấp $(100)^2 = 10.000$ lần.
* **Confidence:** **High** (95%).

### Prediction 5
* **Câu hỏi:** Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?
* **Prediction:** **Không luôn luôn.**
* **Reason:** Attention được tối ưu để cực tiểu hóa hàm mất mát (loss function) của bài toán, không được giám sát bởi sự chú ý của con người. Nó có thể chú ý vào hư từ (stopwords) hoặc mối tương quan giả (spurious correlations).
* **Confidence:** **High** (90%).

---

# 13. Scaled dot-product attention

Quy trình:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```text
QKᵀ ───> divide √d_k ───> softmax ───> weights ───> multiply V ───> Output
```

### Bản chất vì sao cần chia cho $\sqrt{d_k}$:
Khi chiều vector key $d_k$ lớn, tích vô hướng $q^T k = \sum_{i=1}^{d_k} q_i k_i$ có phương sai bằng $\text{Var}(q^T k) = d_k$.  
Độ lệch chuẩn là $\sqrt{d_k}$ làm giá trị score dao động rất lớn, đẩy Softmax vào vùng bão hòa có đạo hàm triệt tiêu $\approx 0$ (**Vanishing Gradient**).  
Phép chia cho $\sqrt{d_k}$ kéo phương sai trở về $1$:
$$\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$
giúp gradient lan truyền ngược ổn định.

---

# 14. Bài tập — Scaling

### Đề bài
Cho $d_k = 4$ và $Q K^T = [4, 2, 0]$.
1. Tính scaled scores: $\frac{Q K^T}{\sqrt{4}}$.
2. Tính Softmax cho scaled scores.
3. So sánh với trường hợp không scaling.
4. Trả lời: *Scaling làm thay đổi độ tập trung của softmax như thế nào?*

### Lời giải chi tiết
1. **Scaled scores** ($\sqrt{d_k} = \sqrt{4} = 2$):
   $$s_{\text{scaled}} = \frac{[4, 2, 0]}{2} = [\mathbf{2.0}, \mathbf{1.0}, \mathbf{0.0}]$$
2. **Softmax có Scaling:**
   * $e^2 \approx 7.3891, \quad e^1 \approx 2.7183, \quad e^0 = 1.0000 \implies \sum = 11.1074$
   * $\alpha_{\text{scaled}} = [\mathbf{0.6652}, \mathbf{0.2447}, \mathbf{0.0900}]$
3. **Softmax KHÔNG Scaling ($s = [4, 2, 0]$):**
   * $e^4 \approx 54.5982, \quad e^2 \approx 7.3891, \quad e^0 = 1.0000 \implies \sum = 62.9873$
   * $\alpha_{\text{unscaled}} = [\mathbf{0.8668}, \mathbf{0.1173}, \mathbf{0.0159}]$

### Bảng so sánh đối chiếu:

| Chế độ | Scores | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | Mức độ tập trung |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Không Scaling** | $[4, 2, 0]$ | **$86.68\%$** | $11.73\%$ | $1.59\%$ | Rất nhọn, cực đoan, áp đảo |
| **Có Scaling** | $[2, 1, 0]$ | **$66.52\%$** | $24.47\%$ | $9.00\%$ | Mềm mại (softer), mượt hơn |

### Trả lời câu hỏi:
> **Scaling làm thay đổi độ tập trung của softmax như thế nào?**
* **Trả lời:** Scaling làm **giảm độ nhọn (làm mềm / phẳng hóa phân phối)** của Softmax. Nó ngăn chặn hiện tượng một phần tử độc chiếm toàn bộ trọng số quá sớm, cho phép các token ngữ cảnh phụ vẫn đóng góp thông tin và giữ gradient không bị triệt tiêu khi kích thước $d_k$ lớn.

---

# 15. Temperature

Sử dụng công thức:
$$\text{softmax}\left(\frac{s}{T}\right)$$
Với cùng score vector $s = [2.0, 1.0, 0.0]$, thử với $T = 0.5, 1.0, 2.0$:

### Bảng kết quả hoàn chỉnh theo mẫu tài liệu:

| Temperature ($T$) | Vector điều chỉnh ($s / T$) | Attention Weights ($\alpha$) | Max weight | Entropy ($H$ - nats) | Nhận xét |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.5** | $[4.0, 2.0, 0.0]$ | $[0.8668, 0.1173, 0.0159]$ | **0.8668** | **0.4411** | Temperature thấp làm phân phối **tập trung cực nhọn**, token lớn nhất chiếm gần $87\%$, entropy thấp (ít bất định). |
| **1.0** | $[2.0, 1.0, 0.0]$ | $[0.6652, 0.2447, 0.0900]$ | **0.6652** | **0.8324** | Phân phối tiêu chuẩn ở mức cân bằng, token cao nhất vẫn nổi trội rõ rệt. |
| **2.0** | $[1.0, 0.5, 0.0]$ | $[0.5065, 0.3072, 0.1863]$ | **0.5065** | **1.0202** | Temperature cao làm phân phối **phẳng hơn rất nhiều (dàn trải đều)**, khoảng cách giữa các phần tử bị thu hẹp, entropy cao. |

---

# 16. Core implementation

Cài đặt hàm chuẩn bằng NumPy theo đúng đặc tả yêu cầu của đề bài:

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Thực hiện Scaled Dot-Product Attention từ đầu bằng NumPy:
      S = Q K^T / sqrt(d_k)
      A = softmax(S)
      C = A V
    Trả về: scores, attention_weights, context
    """
    d_k = K.shape[-1]
    
    # 1. Tính scores đã scale
    scores = (Q @ K.T) / np.sqrt(d_k)
    
    # 2. Softmax ổn định số trị (Numerical stability) theo hàng cuối cùng
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 3. Tính context vector
    context = attention_weights @ V
    
    return scores, attention_weights, context
```

---

# 17. Numerical stability

Công thức Softmax ổn định số học:
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_{j} e^{x_j - \max(x)}}$$

### Test với $x = [1000, 1001, 1002]$:
* **Cách trực tiếp (`np.exp(x)`):** $e^{1000} \to \infty$ (`inf`). Phép chia $\frac{\text{inf}}{\text{inf}} = \mathbf{NaN}$ $\implies$ **Lỗi sụp đổ**.
* **Cách ổn định ($\max(x) = 1002$):**
  $$x - \max(x) = [1000 - 1002, 1001 - 1002, 1002 - 1002] = [\mathbf{-2}, \mathbf{-1}, \mathbf{0}]$$
  * $e^{-2} \approx 0.1353, \quad e^{-1} \approx 0.3679, \quad e^0 = 1.0000 \implies \sum = 1.5032$
  * $\alpha = [\mathbf{0.0900}, \mathbf{0.2447}, \mathbf{0.6652}]$ (tổng $= 1.0$).

### Trả lời câu hỏi:
> **Tại sao phép trừ $\max(x)$ không làm thay đổi kết quả softmax?**
* **Chứng minh toán học:**
  $$\frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}} = \frac{e^{x_i} \cdot e^{-\max(x)}}{\sum_j (e^{x_j} \cdot e^{-\max(x)})} = \frac{e^{-\max(x)} \cdot e^{x_i}}{e^{-\max(x)} \cdot \sum_j e^{x_j}} = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
* Do $e^{-\max(x)}$ là thừa số chung khác $0$ ở cả tử số và toàn bộ các số hạng ở mẫu số, nên chúng triệt tiêu nhau hoàn toàn. Tỉ lệ xác suất được bảo toàn tuyệt đối.

---

# 20. Self-attention

Ở attention tổng quát: Query có thể đến từ một nguồn khác với Key/Value (như trong Seq2Seq machine translation).  
Trong **Self-attention**, $Q, K, V$ được tạo ra từ cùng một sequence $X \in \mathbb{R}^{T \times d}$:
$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
Sau đó tính:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```text
X
│
├── WQ → Q
├── WK → K
└── WV → V
↓
Self-Attention
↓
Contextual representations
```

---

# 21. Bài tập self-attention

### Đề bài
Cho sequence gồm 3 token: `["I", "like", "cats"]`.  
Giả sử embedding matrix: $X \in \mathbb{R}^{3 \times 2}$. Cho các ma trận $W_Q, W_K, W_V \in \mathbb{R}^{2 \times 2}$.

### Dữ liệu số học mẫu minh họa:
$$X = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix} \begin{matrix} \text{("I")} \\ \text{("like")} \\ \text{("cats")} \end{matrix}, \quad W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}, \quad W_V = \begin{bmatrix} 1 & 2 \\ 2 & 0 \end{bmatrix}$$

#### 1. Tính $Q, K, V$:
* $Q = X W_Q = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix}$
* $K = X W_K = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 0 & 2 \end{bmatrix}$
* $V = X W_V = \begin{bmatrix} 1 & 2 \\ 3 & 2 \\ 4 & 0 \end{bmatrix}$

#### 2. Tính $Q K^T$:
$$Q K^T = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 & 0 \\ 1 & 2 & 2 \end{bmatrix} = \begin{bmatrix} 1 & 1 & 0 \\ 2 & 3 & 2 \\ 2 & 4 & 4 \end{bmatrix}$$

#### 3. Tính $\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$ với $d_k = 2$ ($\sqrt{2} \approx 1.4142$):
* Scaled scores:
  $$S = \begin{bmatrix} 0.7071 & 0.7071 & 0.0000 \\ 1.4142 & 2.1213 & 1.4142 \\ 1.4142 & 2.8284 & 2.8284 \end{bmatrix}$$
* Ma trận Attention $A = \text{softmax}(S)$:
  $$A = \begin{bmatrix} 0.4011 & 0.4011 & 0.1978 \\ 0.2483 & 0.5035 & 0.2483 \\ 0.1084 & 0.4458 & 0.4458 \end{bmatrix}$$
* Contextual representations $C = A V$:
  $$C = \begin{bmatrix} 0.4011 & 0.4011 & 0.1978 \\ 0.2483 & 0.5035 & 0.2483 \\ 0.1084 & 0.4458 & 0.4458 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 2 \\ 4 & 0 \end{bmatrix} = \begin{bmatrix} \mathbf{2.3956} & \mathbf{1.6044} \\ \mathbf{2.7517} & \mathbf{1.5035} \\ \mathbf{3.2290} & \mathbf{1.1084} \end{bmatrix}$$

> **Mục tiêu học tập đạt được:**
> Một token trong câu tương tác đồng thời với tất cả các token khác trong cùng chuỗi, biến đổi vector tĩnh thành vector phụ thuộc ngữ cảnh.

---

# 22. Attention matrix

Với sequence gồm $T$ tokens, ma trận attention:
$$A \in \mathbb{R}^{T \times T}$$
* **Ý nghĩa:** Mỗi hàng $i$ biểu diễn token thứ $i$ đang phân bổ sự chú ý tới các token $j$ nào trong chuỗi, với ràng buộc $\sum_{j=1}^T A_{ij} = 1$.

### Ví dụ ma trận trong tài liệu:
```text
           I     like   cats
I       [ 0.6    0.2    0.2 ]   --> Tổng = 1.0
like    [ 0.1    0.7    0.2 ]   --> Tổng = 1.0
cats    [ 0.1    0.3    0.6 ]   --> Tổng = 1.0
```

### Phân tích biểu diễn Heatmap:
* **Đường chéo chính ($A_{ii} = [0.6, 0.7, 0.6]$):** Có giá trị lớn nhất, thể hiện các token luôn giữ sự chú ý đáng kể vào thông tin của chính mình.
* **Tương tác từ chéo:** Động từ `"like"` hướng $20\%$ sự chú ý sang tân ngữ `"cats"` và $10\%$ sang chủ ngữ `"I"`. Trong khi đó, `"cats"` tương tác mạnh ngược lại với động từ `"like"` ($30\%$) để bổ sung mối quan hệ cú pháp vị ngữ - tân ngữ.
