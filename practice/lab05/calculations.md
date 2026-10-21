# BÀI TẬP TÍNH TOÁN VÀ ĐỊNH LƯỢNG (LAB 05 - CALCULATIONS)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [Bài 1 — Một bước tính toán thuận của RNN (Mục 6)](#1-bài-1--một-bước-tính-toán-thuận-của-rnn-mục-6)
   - [1.1. Đề bài & Thông số giả thiết](#11-đề-bài--thông-số-giả-thiết)
   - [1.2. Cơ sở lý thuyết & Công thức cập nhật](#12-cơ-sở-lý-thuyết--công-thức-cập-nhật)
   - [1.3. Các bước tính toán chi tiết](#13-các-bước-tính-toán-chi-tiết)
   - [1.4. Kết quả & Ý nghĩa trạng thái ẩn](#14-kết-quả--ý-nghĩa-trạng-thái-ẩn)
2. [Bài 2 — Dòng chảy thông tin (Information Flow) qua Sequence (Mục 7)](#2-bài-2--dòng-chảy-thông-tin-information-flow-qua-sequence-mục-7)
   - [2.1. Đề bài & Chuỗi sequence](#21-đề-bài--chuỗi-sequence)
   - [2.2. Mô tả chi tiết trạng thái ẩn sau từng bước thời gian](#22-mô-tả-chi-tiết-trạng-thái-ẩn-h_t-sau-từng-bước-thời-gian)
   - [2.3. Cơ chế ảnh hưởng của từ 'not' lên trạng thái đọc từ 'like'](#23-phân-tích-cơ-chế-thông-tin-not-ảnh-hưởng-tới-h_4-khi-đọc-like-như-thế-nào)
   - [2.4. So sánh toàn diện với Average Word2Vec](#24-so-sánh-toàn-diện-với-average-word2vec)
3. [Bài 3 — Thứ tự từ (Sequence Order) trong câu (Mục 8)](#3-bài-3--thứ-tự-từ-sequence-order-trong-câu-mục-8)
4. [Bài 4 — Cơ chế điều khiển cổng của LSTM (LSTM Gates Calculation) (Mục 17)](#4-bài-4--cơ-chế-điều-khiển-cổng-của-lstm-lstm-gates-calculation-mục-17)

---

# 1. Bài 1 — Một bước tính toán thuận của RNN (Mục 6)

## 1.1. Đề bài & Thông số giả thiết

Cho một mô hình Recurrent Neural Network (RNN) đơn giản tại bước thời gian $t$.

* **Vector đầu vào tại timestep $t$** ($x_t \in \mathbb{R}^{2 \times 1}$):
  $$x_t = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

* **Hidden state từ timestep trước đó** ($h_{t-1} \in \mathbb{R}^{2 \times 1}$):
  $$h_{t-1} = \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}$$

* **Ma trận trọng số liên kết Input $\to$ Hidden** ($W_{xh} \in \mathbb{R}^{2 \times 2}$):
  $$W_{xh} = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix}$$

* **Ma trận trọng số hồi quy Hidden $\to$ Hidden** ($W_{hh} \in \mathbb{R}^{2 \times 2}$):
  $$W_{hh} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix}$$

* **Vector hệ số chệch (bias)** ($b_h \in \mathbb{R}^{2 \times 1}$):
  $$b_h = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

**Yêu cầu**:
1. Tính vector kích hoạt trước phi tuyến (pre-activation vector):
   $$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
2. Tính hidden state tại timestep $t$ qua hàm kích hoạt $\tanh$:
   $$h_t = \tanh(a_t)$$
3. Trình bày đầy đủ từng bước tính toán đại số ma trận.

---

## 1.2. Cơ sở lý thuyết & Công thức cập nhật

Khác với mạng truyền thẳng (Feed-Forward Neural Network) chỉ nhận đầu vào hiện tại, mạng nơ-ron hồi quy (Simple/Vanilla RNN) duy trì một bộ nhớ nội tại dưới dạng trạng thái ẩn $h_t$.

Trạng thái $h_t$ đóng vai trò là "vector ngữ cảnh" (context vector) tích lũy toàn bộ thông tin từ timestep $1$ đến timestep $t$:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
$$h_t = \tanh(a_t)$$

Trong đó:
- $W_{xh} x_t$: Đóng góp từ token đầu vào hiện tại tại vị trí $t$.
- $W_{hh} h_{t-1}$: Đóng góp từ ngữ cảnh quá khứ được truyền lại từ vị trí $t-1$.
- $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$: Hàm kích hoạt phi tuyến ánh xạ giá trị về miền $(-1, 1)$, giúp chuẩn hóa biên độ giá trị truyền qua thời gian.

---

## 1.3. Các bước tính toán chi tiết

### Bước 1: Tính thành phần Input-to-Hidden ($W_{xh} x_t$)

Nhân ma trận $W_{xh}$ với vector cột $x_t$:
$$W_{xh} x_t = \begin{bmatrix} 0.5 & 0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

Thực hiện nhân từng hàng với vector cột:
- Phần tử 1:
  $$(W_{xh} x_t)_1 = 0.5 \times 1 + 0.2 \times 0 = 0.5 + 0.0 = 0.5$$
- Phần tử 2:
  $$(W_{xh} x_t)_2 = 0.1 \times 1 + 0.4 \times 0 = 0.1 + 0.0 = 0.1$$

Kết quả:
$$W_{xh} x_t = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix}$$

---

### Bước 2: Tính thành phần Hidden-to-Hidden ($W_{hh} h_{t-1}$)

Nhân ma trận $W_{hh}$ với hidden state quá khứ $h_{t-1}$:
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.3 & 0.1 \\ 0.2 & 0.5 \end{bmatrix} \begin{bmatrix} 0.5 \\ 0.2 \end{bmatrix}$$

Thực hiện nhân từng hàng với vector cột:
- Phần tử 1:
  $$(W_{hh} h_{t-1})_1 = 0.3 \times 0.5 + 0.1 \times 0.2 = 0.15 + 0.02 = 0.17$$
- Phần tử 2:
  $$(W_{hh} h_{t-1})_2 = 0.2 \times 0.5 + 0.5 \times 0.2 = 0.10 + 0.10 = 0.20$$

Kết quả:
$$W_{hh} h_{t-1} = \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix}$$

---

### Bước 3: Tính tổng Pre-activation ($a_t$)

Cộng hai thành phần đóng góp và vector bias $b_h$:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$

$$a_t = \begin{bmatrix} 0.5 \\ 0.1 \end{bmatrix} + \begin{bmatrix} 0.17 \\ 0.20 \end{bmatrix} + \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

Cộng từng phần tử tương ứng:
- Phần tử 1:
  $$a_{t, 1} = 0.5 + 0.17 + 0 = 0.67$$
- Phần tử 2:
  $$a_{t, 2} = 0.1 + 0.20 + 0 = 0.30$$

Kết quả:
$$a_t = \begin{bmatrix} 0.67 \\ 0.30 \end{bmatrix}$$

---

### Bước 4: Áp dụng hàm kích hoạt phi tuyến $\tanh$ để tìm $h_t$

$$h_t = \tanh(a_t) = \begin{bmatrix} \tanh(a_{t, 1}) \\ \tanh(a_{t, 2}) \end{bmatrix} = \begin{bmatrix} \tanh(0.67) \\ \tanh(0.30) \end{bmatrix}$$

Áp dụng định nghĩa hàm $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$:
- Với $z_1 = 0.67$:
  $$e^{0.67} \approx 1.954237, \quad e^{-0.67} \approx 0.511709$$
  $$\tanh(0.67) = \frac{1.954237 - 0.511709}{1.954237 + 0.511709} = \frac{1.442528}{2.465946} \approx 0.584980$$

- Với $z_2 = 0.30$:
  $$e^{0.30} \approx 1.349859, \quad e^{-0.30} \approx 0.740818$$
  $$\tanh(0.30) = \frac{1.349859 - 0.740818}{1.349859 + 0.740818} = \frac{0.609041}{2.090677} \approx 0.291313$$

Kết quả cuối cùng:
$$h_t = \begin{bmatrix} 0.584980 \\ 0.291313 \end{bmatrix} \approx \begin{bmatrix} 0.5850 \\ 0.2913 \end{bmatrix}$$

---

## 1.4. Kết quả & Ý nghĩa trạng thái ẩn

* **Bảng tổng hợp giá trị**:

| Đại lượng | Ký hiệu toán học | Chiều vector | Giá trị |
| :--- | :---: | :---: | :---: |
| Input vector | $x_t$ | $2 \times 1$ | $[1.0, 0.0]^T$ |
| Previous hidden state | $h_{t-1}$ | $2 \times 1$ | $[0.5, 0.2]^T$ |
| Input contribution | $W_{xh} x_t$ | $2 \times 1$ | $[0.50, 0.10]^T$ |
| Context contribution | $W_{hh} h_{t-1}$ | $2 \times 1$ | $[0.17, 0.20]^T$ |
| Pre-activation | $a_t$ | $2 \times 1$ | $[0.67, 0.30]^T$ |
| **Current hidden state** | $\mathbf{h_t}$ | $2 \times 1$ | $\mathbf{[0.5850, 0.2913]^T}$ |

* **Nhận xét chuyên sâu**:
  1. Trạng thái ẩn $h_t$ vừa tiếp nhận thông tin mới từ token $x_t$ (đóng góp qua $W_{xh} x_t$), vừa duy trì thông tin bộ nhớ từ bước trước $h_{t-1}$ (đóng góp qua $W_{hh} h_{t-1}$).
  2. Giá trị của $h_t$ luôn bị chặn trong khoảng $(-1, 1)$ nhờ tính chất của hàm $\tanh$, ngăn hiện tượng giá trị kích hoạt bị bùng nổ không kiểm soát qua từng timestep forward.

---

# 2. Bài 2 — Dòng chảy thông tin (Information Flow) qua Sequence (Mục 7)

## 2.1. Đề bài & Chuỗi sequence

Cho câu văn bản:
$$\mathcal{S} = \text{"I did not like this movie"}$$

Giả sử mô hình đọc tuần tự từ trái sang phải qua $T = 6$ bước thời gian ($t = 1, 2, \dots, 6$):
* $x_1 = \text{"I"}$
* $x_2 = \text{"did"}$
* $x_3 = \text{"not"}$
* $x_4 = \text{"like"}$
* $x_5 = \text{"this"}$
* $x_6 = \text{"movie"}$

Khởi tạo trạng thái ẩn ban đầu: $h_0 = \vec{0}$.  
Công thức chuyển trạng thái tại mỗi bước:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

---

## 2.2. Mô tả chi tiết trạng thái ẩn ($h_t$) sau từng bước thời gian

```
[h_0 = 0]
   │
   ▼
[t=1: "I"]       ───► h_1: Nhận diện chủ ngữ ngôi thứ nhất
   │
   ▼
[t=2: "did"]     ───► h_2: Kết hợp trợ động từ thời quá khứ ("I did...")
   │
   ▼
[t=3: "not"]     ───► h_3: Kích hoạt tín hiệu phủ định mạnh (Negation Trigger)
   │
   ▼
[t=4: "like"]    ───► h_4: Điều biến ngữ nghĩa "like" dưới ngữ cảnh "not" => Mang nghĩa tiêu cực ("dislike")
   │
   ▼
[t=5: "this"]    ───► h_5: Định vị đối tượng xác định ("this...")
   │
   ▼
[t=6: "movie"]   ───► h_6: Trạng thái kết thúc h_T: Đại diện toàn bộ câu mang sắc thái chê (Negative)
```

### Chi tiết quá trình biến đổi của từng $h_t$:

1. **Bước 1: Đọc $x_1 = \text{"I"}$**
   $$h_1 = \tanh(W_{xh} x_{\text{"I"}} + W_{hh} h_0 + b_h) = \tanh(W_{xh} x_{\text{"I"}} + b_h)$$
   * *Ý nghĩa ngữ cảnh*: Mô hình ghi nhận chủ ngữ của câu là ngôi thứ nhất ("tôi"). Không gian vector của $h_1$ thuần túy phản ánh thực thể người nói.

2. **Bước 2: Đọc $x_2 = \text{"did"}$**
   $$h_2 = \tanh(W_{xh} x_{\text{"did"}} + W_{hh} h_1 + b_h)$$
   * *Ý nghĩa ngữ cảnh*: $h_1$ được truyền qua ma trận chuyển tiếp $W_{hh}$ và kết hợp với embedding của trợ động từ "did". Trạng thái $h_2$ lưu giữ ngữ cảnh thời thì quá khứ của người nói ("tôi đã...").

3. **Bước 3: Đọc $x_3 = \text{"not"}$**
   $$h_3 = \tanh(W_{xh} x_{\text{"not"}} + W_{hh} h_2 + b_h)$$
   * *Ý nghĩa ngữ cảnh*: **Đây là bước chuyển biến quan trọng nhất về mặt cảm xúc.**
   * Từ phủ định "not" được đưa vào. Trạng thái $h_3$ tích lũy tín hiệu phủ định (negation signal) đi liền sau chủ ngữ ("I did not..."). Bộ nhớ ẩn lúc này rơi vào trạng thái "chờ" (primed) để sẵn sàng đảo ngược phân cực của động từ / tính từ tiếp theo.

4. **Bước 4: Đọc $x_4 = \text{"like"}$**
   $$h_4 = \tanh(W_{xh} x_{\text{"like"}} + W_{hh} h_3 + b_h)$$
   * *Ý nghĩa ngữ cảnh*: **Tâm điểm xử lý ngữ nghĩa.**
   * Bản thân từ "like" mang embedding $x_{\text{"like"}}$ hướng tới cảm xúc tích cực (positive). Tuy nhiên, thành phần $W_{hh} h_3$ mang theo thông tin ngữ cảnh "I did not" từ bước trước.
   * Ma trận trọng số $W_{hh}$ đã được huấn luyện để khi kết hợp một trạng thái mang tính phủ định ($h_3$) với một từ cảm xúc tích cực ($x_{\text{"like"}}$), tổng kích hoạt $a_4$ sẽ bị kéo ngược về miền giá trị tiêu cực (negative polarity).
   * Do đó, $h_4$ không mang nghĩa "thích", mà biểu diễn ngữ nghĩa của cụm **"không thích" (dislike / negative sentiment)**.

5. **Bước 5: Đọc $x_5 = \text{"this"}$**
   $$h_5 = \tanh(W_{xh} x_{\text{"this"}} + W_{hh} h_4 + b_h)$$
   * *Ý nghĩa ngữ cảnh*: Bổ sung từ chỉ định, giữ nguyên trạng thái cảm xúc tiêu cực từ $h_4$ và chuẩn bị tiếp nhận đối tượng bị đánh giá.

6. **Bước 6: Đọc $x_6 = \text{"movie"}$**
   $$h_6 = \tanh(W_{xh} x_{\text{"movie"}} + W_{hh} h_5 + b_h)$$
   * *Ý nghĩa ngữ cảnh*: Xác định đối tượng của hành động "không thích" chính là bộ phim ("movie").
   * $h_6 = h_T$ là vector đại diện cho toàn bộ câu (sentence representation vector). Khi đưa qua tầng phân loại $z = W_{hy} h_6 + b_y \to \sigma(z)$, mô hình sẽ tự tin dự đoán nhãn **Negative**.

---

## 2.3. Phân tích cơ chế: Thông tin "not" ảnh hưởng tới $h_4$ khi đọc "like" như thế nào?

Về mặt toán học và cơ chế mô hình:
1. **Cơ chế điều biến trạng thái (State Modulation)**:
   Tại bước $t=4$:
   $$a_4 = \underbrace{W_{xh} x_{\text{"like"}}}_{\text{Tín hiệu Positive gốc}} + \underbrace{W_{hh} h_3}_{\text{Ngữ cảnh phủ định ("I did not")}} + b_h$$
   Thông tin "not" nằm gọn trong $h_3$. Phép biến đổi tuyến tính $W_{hh} h_3$ hoạt động như một toán tử dịch chuyển không gian (spatial shift operator) hoặc toán tử đảo cực. Nó triệt tiêu hoặc lấn át hoàn toàn thành phần dương của $W_{xh} x_{\text{"like"}}$.
2. **Khái niệm Context-dependent State**:
   Trạng thái $h_4$ không còn là biểu diễn của riêng từ "like" (như trong static embedding), mà là biểu diễn của từ "like" **đặt trong bối cảnh đã xuất hiện "not" ngay trước đó**. Đây chính là năng lực mô hình hóa thứ tự và phạm vi phủ định (scope of negation).

---

## 2.4. So sánh toàn diện với Average Word2Vec

| Tiêu chí | Average Word2Vec (LAB 04) | Recurrent Sequence Model - RNN (LAB 05) |
| :--- | :--- | :--- |
| **Công thức biểu diễn câu** | $\vec{x}_{\text{avg}} = \frac{1}{6} \sum_{i=1}^6 \vec{w}_i$ | $h_6 = \tanh(W_{xh} x_6 + W_{hh} h_5 + b_h)$ |
| **Tính chất phép toán** | Phép cộng có tính chất **giao hoán** ($A + B = B + A$). | Phép cập nhật tuần tự **phi giao hoán** ($f(x_t, h_{t-1}) \neq f(x_{t-1}, h_t)$). |
| **Xử lý phạm vi phủ định (Negation Scope)** | **Không thể**: Từ "not" và từ "like" bị ném vào cùng một tổng số học. Trọng số đóng góp của "like" là một vector dương độc lập, "not" là một vector tĩnh độc lập. Mô hình không biết "not" áp dụng lên từ nào. | **Rất tốt ở khoảng cách ngắn**: "not" xuất hiện ở $t=3$ trực tiếp biến đổi trạng thái ngữ cảnh $h_3$, truyền thẳng vào bước $t=4$ để biến đổi ý nghĩa của "like". |
| **Phân biệt cấu trúc câu** | Không phân biệt được câu gốc với câu đảo từ: `"I did not like this movie"` và `"I did like this movie, not that one"` sẽ có vector trung bình gần như giống hệt nhau. | Phân biệt rõ rệt hai câu vì thứ tự xuất hiện của từ làm thay đổi toàn bộ quỹ đạo của chuỗi trạng thái ẩn $h_1 \to h_2 \to \dots \to h_T$. |
| **Bản chất vector** | Static, Bag-of-Words (mất hoàn toàn cấu trúc cú pháp). | Sequential, Dynamic, Context-dependent (giữ được cú pháp và thời gian). |

* **Kết luận**:
  Average Word2Vec thất bại trong việc mô hình hóa phủ định cục bộ vì làm phẳng (flatten) văn bản thành một vector cố định ngay lập tức. Ngược lại, RNN duy trì dòng chảy trạng thái theo thời gian, cho phép từ đi trước thiết lập ngữ cảnh để định nghĩa lại ý nghĩa của từ đi sau.
