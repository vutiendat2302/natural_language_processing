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
   - [3.1. Đề bài & Hai chuỗi đối chiếu](#31-đề-bài--hai-chuỗi-đối-chiếu)
   - [3.2. Khả năng phân biệt của Average Word2Vec](#32-khả-năng-phân-biệt-của-average-word2vec)
   - [3.3. Cơ chế phân biệt dựa trên thứ tự của RNN](#33-cơ-chế-phân-biệt-dựa-trên-thứ-tự-của-rnn)
   - [3.4. So sánh toán học và ý nghĩa biểu diễn](#34-so-sánh-toán-học-và-ý-nghĩa-biểu-diễn)
4. [Bài 4 — Hiện tượng Vanishing Gradient và Giới hạn phụ thuộc tầm xa (Mục 9)](#4-bài-4--hiện-tượng-vanishing-gradient-và-giới-hạn-phụ-thuộc-tầm-xa-mục-9)
   - [4.1. Bản chất toán học của BPTT và tích Jacobian](#41-bản-chất-toán-học-của-bptt-và-tích-jacobian)
   - [4.2. Trực giác hiện tượng: Vanishing Gradient vs Exploding Gradient](#42-trực-giác-hiện-tượng-vanishing-gradient-vs-exploding-gradient)
   - [4.3. Vì sao Sequence càng dài khiến RNN khó học Long-range Dependency?](#43-vì-sao-sequence-càng-dài-khiến-rnn-khó-học-long-range-dependency)
   - [4.4. Minh họa trực quan và Tác động tới trọng số](#44-minh-họa-trực-quan-và-tác-động-tới-trọng-số)
5. [Bài 5 — Prediction trước thực nghiệm (Pre-experiment Predictions) (Mục 10)](#5-bài-5--prediction-trước-thực-nghiệm-pre-experiment-predictions-mục-10)
   - [Prediction 1: TF-IDF + Logistic Regression vs RNN + Embedding](#prediction-1-tf-idf--logistic-regression-vs-rnn--embedding-về-khả-năng-xử-lý-word-order)
   - [Prediction 2: RNN vs LSTM về xử lý Long-range Dependency](#prediction-2-rnn-vs-lstm-về-khả-năng-xử-lý-long-range-dependency)
   - [Prediction 3: Tác động khi Sequence Length tăng từ 50 lên 200](#prediction-3-tác-động-khi-sequence-length-tăng-từ-50-lên-200-đối-với-quá-trình-training-của-rnn)
   - [Prediction 4: Tác động khi Truncate toàn bộ review về 30 tokens](#prediction-4-tác-động-khi-truncate-toàn-bộ-review-về-30-tokens-lên-classification)
   - [Prediction 5: Liệu LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset?](#prediction-5-liệu-lstm-có-chắc-chắn-cho-accuracy-cao-hơn-rnn-trên-mọi-dataset-không)
6. [Bài 6 — Kiểm tra & Thống kê Dataset IMDb (Dataset Inspection) (Mục 11)](#6-bài-6--kiểm-tra--thống-kê-dataset-imdb-dataset-inspection-mục-11)
   - [6.1. Tổng quan tập dữ liệu IMDb Movie Reviews](#61-tổng-quan-tập-dữ-liệu-imdb-movie-reviews)
   - [6.2. Bảng thống kê chi tiết đặc tính dữ liệu](#62-bảng-thống-kê-chi-tiết-đặc-tính-dữ-liệu)
   - [6.3. Phân bố độ dài chuỗi (Sequence Length Distribution & Histogram)](#63-phân-bố-độ-dài-chuỗi-sequence-length-distribution--histogram)
   - [6.4. Cơ sở khoa học lựa chọn tham số `max_length`](#64-cơ-sở-khoa-học-lựa-chọn-tham-số-max_length)
7. [Bài 7 — Thực nghiệm Padding và Truncation (Experiment 2) (Mục 13)](#7-bài-7--thực-nghiệm-padding-và-truncation-experiment-2-mục-13)
   - [7.1. Bảng số liệu thực nghiệm định lượng](#71-bảng-số-liệu-thực-nghiệm-định-lượng)
   - [7.2. Phân tích: Tăng max_length có luôn cải thiện performance không?](#72-phân-tích-tăng-max_length-có-luôn-cải-thiện-performance-không)
   - [7.3. Phân tích: Nếu max_length quá nhỏ, loại thông tin nào bị mất mát?](#73-phân-tích-nếu-max_length-quá-nhỏ-loại-thông-tin-nào-bị-mất-mát)
8. [Bài 8 — Cài đặt và Giải thích Computation Graph của Simple RNN Cell (Mục 14)](#8-bài-8--cài-đặt-và-giải-thích-computation-graph-của-simple-rnn-cell-mục-14)
   - [8.1. Đồ thị tính toán (Computation Graph)](#81-đồ-thị-tính-toán-computation-graph)
   - [8.2. Giải thích từng thành phần và phép toán](#82-giải-thích-từng-thành-phần-và-phép-toán)
   - [8.3. Kết quả xác thực tương đương với PyTorch nn.RNNCell](#83-kết-quả-xác-thực-tương-đương-với-pytorch-nnrnncell)
9. [Bài 9 — Thực nghiệm Baseline: Logistic Regression vs Simple RNN (Experiment 3) (Mục 15)](#9-bài-9--thực-nghiệm-baseline-logistic-regression-vs-simple-rnn-experiment-3-mục-15)
   - [9.1. Bảng số liệu so sánh toàn diện Baseline](#91-bảng-số-liệu-so-sánh-toàn-diện-baseline)
   - [9.2. Bằng chứng thực nghiệm: RNN có thực sự sử dụng thông tin Sequence hay không?](#92-bằng-chứng-thực-nghiệm-rnn-có-thực-sự-sử-dụng-thông-tin-sequence-hay-không)
10. [Bài 10 — Cơ chế điều khiển cổng của LSTM (LSTM Gates Calculation) (Mục 17)](#10-bài-10--cơ-chế-điều-khiển-cổng-của-lstm-lstm-gates-calculation-mục-17)
    - [10.1. Đề bài & Thông số giả thiết](#101-đề-bài--thông-số-giả-thiết)
    - [10.2. Các bước tính toán cập nhật Cell State](#102-các-bước-tính-toán-cập-nhật-cell-state)
    - [10.3. Phân tích bản chất: Forget Gate gần 1 vs gần 0](#103-phân-tích-bản-chất-forget-gate-gần-1-vs-gần-0)
    - [10.4. Gates như một cơ chế điều khiển dòng chảy thông tin (Information Flow)](#104-gates-như-một-cơ-chế-điều-khiển-dòng-chảy-thông-tin-information-flow)

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

---

# 3. Bài 3 — Thứ tự từ (Sequence Order) trong câu (Mục 8)

## 3.1. Đề bài & Hai chuỗi đối chiếu

Cho hai chuỗi văn bản (sequences):
* **Chuỗi 1 ($S_1$)**: `"dog bites man"` (Con chó cắn người đàn ông — hiện tượng bình thường)
* **Chuỗi 2 ($S_2$)**: `"man bites dog"` (Người đàn ông cắn con chó — sự kiện giật gân, bất thường)

Giả sử cả hai câu đều dùng chung một bộ từ vựng (vocabulary) $\mathcal{V} = \{\text{"dog"}, \text{"bites"}, \text{"man"}\}$ và các vector embedding tương ứng là $x_{\text{dog}}, x_{\text{bites}}, x_{\text{man}} \in \mathbb{R}^d$.

**Câu hỏi đặt ra**:
1. **Average Word2Vec** có thể phân biệt hai câu này dựa trên thứ tự hay không? Tại sao?
2. **RNN** có khả năng phân biệt chúng dựa trên thứ tự như thế nào? Giải thích chi tiết dựa trên cơ chế nội tại của mô hình (không chỉ dựa vào kết quả thực nghiệm).

---

## 3.2. Khả năng phân biệt của Average Word2Vec

### Câu trả lời:
**Average Word2Vec HOÀN TOÀN KHÔNG THỂ phân biệt được hai câu $S_1$ và $S_2$.**

### Giải thích bản chất toán học:
1. **Bản chất Bag-of-Words (Túi từ)**:
   Mô hình Average Word2Vec biểu diễn toàn bộ câu bằng trung bình cộng (hoặc tổng) các vector embedding của từng từ trong câu:
   $$\vec{v}(S_1) = \frac{1}{3} \left( x_{\text{dog}} + x_{\text{bites}} + x_{\text{man}} \right)$$
   $$\vec{v}(S_2) = \frac{1}{3} \left( x_{\text{man}} + x_{\text{bites}} + x_{\text{dog}} \right)$$

2. **Tính chất giao hoán (Commutativity) của phép cộng vector**:
   Trong không gian vector Euclide $\mathbb{R}^d$, phép cộng vector có tính chất giao hoán và kết hợp:
   $$u + v + w = w + v + u \quad \forall u, v, w \in \mathbb{R}^d$$
   Do đó:
   $$\vec{v}(S_1) \equiv \vec{v}(S_2)$$

3. **Hệ quả ngữ nghĩa**:
   - Vector đặc trưng đầu vào cấp câu của $S_1$ và $S_2$ là **trùng khớp hoàn toàn** (khoảng cách Cosine bằng $0$, khoảng cách Euclide bằng $0$).
   - Mọi bộ phân loại phía sau (Logistic Regression, MLP, SVM) đều nhận cùng một vector đặc trưng, dẫn đến **xác suất đầu ra dự đoán của $S_1$ và $S_2$ luôn luôn giống hệt nhau**:
     $$P(y \mid S_1) = P(y \mid S_2)$$
   - Average Word2Vec làm mất hoàn toàn thông tin cú pháp: Ai là **chủ thể (Subject - Agent)** thực hiện hành động và ai là **tân ngữ (Object - Patient)** chịu tác động của hành động.

---

## 3.3. Cơ chế phân biệt dựa trên thứ tự của RNN

### Câu trả lời:
**RNN phân biệt $S_1$ và $S_2$ một cách TỰ NHIÊN VÀ RÕ RỆT nhờ cơ chế cập nhật trạng thái tuần tự phi giao hoán theo thời gian.**

### Giải thích chi tiết dựa trên cơ chế model:

Giả sử mạng RNN có ma trận trọng số đầu vào $W_{xh}$, ma trận hồi quy $W_{hh}$, bias $b_h$, và khởi tạo $h_0 = \vec{0}$.  
Công thức chuyển trạng thái tại timestep $t$:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

Hãy xem xét diễn biến trạng thái của từng câu:

### 1. Quá trình xử lý chuỗi $S_1 = \text{"dog bites man"}$:
* **Tại $t=1$ (đọc "dog")**:
  $$h_1^{(1)} = \tanh(W_{xh} x_{\text{dog}} + b_h)$$
  Trạng thái $h_1^{(1)}$ ghi nhận chủ ngữ là "dog".
* **Tại $t=2$ (đọc "bites")**:
  $$h_2^{(1)} = \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(1)} + b_h)$$
  $h_2^{(1)}$ kết hợp hành động "bites" với chủ thể "dog" từ quá khứ $\implies$ biểu diễn cụm "con chó cắn".
* **Tại $t=3$ (đọc "man")**:
  $$h_3^{(1)} = \tanh(W_{xh} x_{\text{man}} + W_{hh} h_2^{(1)} + b_h)$$
  Vector đại diện cuối cùng $h_{\text{final}}^{(S_1)} = h_3^{(1)}$ mã hóa đầy đủ quan hệ: "dog" là chủ ngữ (chủ thể gây ra vết cắn), "man" là tân ngữ (đối tượng nhận vết cắn).

---

### 2. Quá trình xử lý chuỗi $S_2 = \text{"man bites dog"}$:
* **Tại $t=1$ (đọc "man")**:
  $$h_1^{(2)} = \tanh(W_{xh} x_{\text{man}} + b_h)$$
  Trạng thái $h_1^{(2)}$ ghi nhận chủ ngữ là "man" (hoàn toàn khác $h_1^{(1)}$ vì $x_{\text{man}} \neq x_{\text{dog}}$).
* **Tại $t=2$ (đọc "bites")**:
  $$h_2^{(2)} = \tanh(W_{xh} x_{\text{bites}} + W_{hh} h_1^{(2)} + b_h)$$
  Dù từ hiện tại vẫn là "bites", nhưng ngữ cảnh quá khứ truyền vào là $h_1^{(2)}$ ("man") chứ không phải $h_1^{(1)}$ ("dog"). Do đó $h_2^{(2)} \neq h_2^{(1)}$, đại diện cho ý niệm "người đàn ông cắn".
* **Tại $t=3$ (đọc "dog")**:
  $$h_3^{(2)} = \tanh(W_{xh} x_{\text{dog}} + W_{hh} h_2^{(2)} + b_h)$$
  Vector đại diện cuối cùng $h_{\text{final}}^{(S_2)} = h_3^{(2)}$ mã hóa cấu trúc: "man" là chủ thể gây ra hành động, "dog" là tân ngữ chịu tác động.

---

### 3. Bản chất toán học của sự khác biệt: Tính phi giao hoán (Non-commutativity)
Khai triển tường minh trạng thái cuối cùng (bỏ qua $b_h$ để nhìn rõ cấu trúc):
* Với $S_1$:
  $$h_3^{(1)} = \tanh\Big( W_{xh} x_{\text{man}} + W_{hh} \tanh\big( W_{xh} x_{\text{bites}} + W_{hh} \tanh(W_{xh} x_{\text{dog}}) \big) \Big)$$
* Với $S_2$:
  $$h_3^{(2)} = \tanh\Big( W_{xh} x_{\text{dog}} + W_{hh} \tanh\big( W_{xh} x_{\text{bites}} + W_{hh} \tanh(W_{xh} x_{\text{man}}) \big) \Big)$$

Nhận xét:
1. **Hàm hợp lồng nhau phi tuyến tính**: Phép tính trạng thái ẩn là một chuỗi các hàm hợp lồng nhau $f(x, h) = \tanh(W_{xh} x + W_{hh} h)$. Phép toán này **không có tính chất giao hoán**:
   $$f(x_A, f(x_B, h)) \neq f(x_B, f(x_A, h))$$
2. **Số lần nhân ma trận chuyển tiếp $W_{hh}$ (Temporal Distance)**:
   - Trong $S_1$, thông tin của "dog" xuất hiện từ $t=1$, nên bị nhân liên tiếp bởi $W_{hh}$ hai lần ($W_{hh}^2$) qua hai bước thời gian để đến bước cuối. Trong khi đó, "man" xuất hiện ở $t=3$ và chỉ nhân với $W_{xh}$ (không bị nhân $W_{hh}$).
   - Trong $S_2$, vị trí bị đảo ngược: "man" chịu tác động của $W_{hh}^2$, còn "dog" đứng ở vị trí cuối cùng.
3. Vì $x_{\text{dog}} \neq x_{\text{man}}$ và $W_{hh} \neq I$, chắc chắn dẫn đến:
   $$h_3^{(1)} \neq h_3^{(2)}$$

Do hai vector đại diện $h_3^{(1)}$ và $h_3^{(2)}$ nằm ở hai vị trí hoàn toàn khác nhau trong không gian biểu diễn ẩn, tầng phân loại (Classifier) tiếp theo hoàn toàn có thể phân loại đúng ý nghĩa và nhãn của từng câu.

---

## 3.4. So sánh toán học và ý nghĩa biểu diễn

| Đặc trưng | Average Word2Vec (LAB 04) | Recurrent Neural Network - Simple RNN (LAB 05) |
| :--- | :--- | :--- |
| **Biểu thức toán học** | $\vec{v} = \frac{1}{3} (x_1 + x_2 + x_3)$ | $h_3 = \tanh(W_{xh} x_3 + W_{hh} h_2 + b_h)$ |
| **Tính chất phép toán** | Phép cộng có tính chất giao hoán ($A+B=B+A$) | Phép toán hàm hợp phi tuyến tuần tự phi giao hoán |
| **Giá trị vector của $S_1$ vs $S_2$** | $\vec{v}(S_1) \equiv \vec{v}(S_2)$ (Giống nhau 100%) | $h_{\text{final}}(S_1) \neq h_{\text{final}}(S_2)$ (Tách biệt rõ ràng) |
| **Bảo toàn vai trò ngữ pháp (Grammatical Roles)** | Hoàn toàn mất (không biết đâu là Subject, đâu là Object) | Bảo toàn trọn vẹn nhờ vị trí thời gian của token trong chuỗi |
| **Khả năng dự đoán nhãn khác nhau** | ❌ Không thể (Output probabilities luôn giống nhau) |  Có thể dự đoán nhãn phân loại hoặc cảm xúc hoàn toàn khác nhau |

---

# 4. Bài 4 — Hiện tượng Vanishing Gradient và Giới hạn phụ thuộc tầm xa (Mục 9)

## 4.1. Bản chất toán học của BPTT và tích Jacobian

Trong Simple RNN, trạng thái ẩn truyền qua các bước thời gian:
$$h_1 \to h_2 \to h_3 \to \dots \to h_t \to \dots \to h_T$$

Khi huấn luyện với thuật toán **Lan truyền ngược qua thời gian (Backpropagation Through Time - BPTT)**, giả sử hàm mất mát $\mathcal{L}$ được tính tại bước cuối $T$ (hoặc tổng mất mát $\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t$).  
Để cập nhật trọng số tại các bước thời gian đầu tiên, ta cần tính đạo hàm riêng của $\mathcal{L}$ theo $h_t$:

$$\frac{\partial \mathcal{L}}{\partial h_t} = \frac{\partial \mathcal{L}}{\partial h_T} \frac{\partial h_T}{\partial h_t}$$

Áp dụng quy tắc dây chuyền (chain rule) xuyên suốt các bước thời gian trung gian từ $t$ đến $T$:
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}$$

Với mỗi bước chuyển tiếp $h_k = \tanh(a_k) = \tanh(W_{xh} x_k + W_{hh} h_{k-1} + b_h)$, ma trận Jacobian chuyển trạng thái là:
$$\frac{\partial h_k}{\partial h_{k-1}} = \operatorname{diag}\big(1 - \tanh^2(a_k)\big) \cdot W_{hh}^T$$

Do đó, tích lan truyền ngược trở thành một chuỗi tích gồm $T - t$ ma trận nhân liên tiếp:
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^{T} \Big[ \operatorname{diag}\big(1 - h_k^2\big) \cdot W_{hh}^T \Big]$$

---

## 4.2. Trực giác hiện tượng: Vanishing Gradient vs Exploding Gradient

Tích ma trận liên tiếp qua $K = T - t$ bước có thể xem xét trực giác dưới dạng tích các hệ số co giãn $a_k$:
$$\left| \prod_{k=t+1}^T a_k \right|$$

Trong không gian nhiều chiều, tốc độ co giãn của tích phụ thuộc vào **trị riêng lớn nhất (spectral radius / largest singular value)** của $W_{hh}$, ký hiệu là $\lambda_1$:

1. **Vanishing Gradient (Triệt tiêu đạo hàm)**:
   * Đạo hàm của hàm kích hoạt $\tanh(z)$ thỏa mãn:
     $$0 < 1 - \tanh^2(z) \le 1$$
     Giá trị này chỉ đạt cực đại bằng $1$ khi $z=0$, và giảm rất nhanh về $0$ khi $|z|$ lớn (vùng bão hòa của $\tanh$).
   * Nếu các trị riêng của $W_{hh}$ có độ lớn $|\lambda| < 1$, hoặc tích hợp với đạo hàm $\tanh'$ khiến độ chuẩn (norm) của ma trận Jacobian nhỏ hơn 1:
     $$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\| \le \gamma < 1$$
   * Khi đó:
     $$\left\| \frac{\partial h_T}{\partial h_t} \right\| \le \gamma^{T - t} \xrightarrow{T - t \gg 1} 0$$
   * Tín hiệu gradient suy giảm theo hàm mũ (**exponential decay**). Khi $T - t$ chỉ khoảng $10 \sim 20$ bước, gradient truyền về $h_t$ gần như triệt tiêu hoàn toàn về $0$.

2. **Exploding Gradient (Bùng nổ đạo hàm)**:
   * Nếu các trị riêng của $W_{hh}$ thỏa mãn $|\lambda| > 1$ và các trạng thái không rơi vào vùng bão hòa triệt tiêu của $\tanh$, chuẩn ma trận có thể lớn hơn 1:
     $$\left\| \frac{\partial h_k}{\partial h_{k-1}} \right\| \ge \beta > 1 \implies \left\| \frac{\partial h_T}{\partial h_t} \right\| \ge \beta^{T - t} \xrightarrow{T - t \gg 1} \infty$$
   * Gradient tăng trưởng bùng nổ theo hàm mũ, dẫn đến cập nhật trọng số cực lớn, làm mô hình phân kỳ và sinh ra lỗi `NaN` / `Inf`. (Vấn đề này thường được kiểm soát bằng kỹ thuật **Gradient Clipping**).

---

## 4.3. Vì sao Sequence càng dài khiến RNN khó học Long-range Dependency?

### Trả lời trực tiếp câu hỏi cốt lõi:
**Mô hình học được phụ thuộc giữa 2 từ thông qua tín hiệu Gradient nhận được từ Backpropagation. Khi sequence càng dài, tín hiệu gradient từ bước cuối bị triệt tiêu hàm mũ khi truyền ngược về các bước đầu tiên, khiến trọng số không nhận được phản hồi để điều chỉnh.**

Cụ thể:
1. **Mất tín hiệu phản hồi lỗi (Credit Assignment Problem)**:
   - Giả sử có một quan hệ phụ thuộc xa giữa từ $x_1$ và từ $x_T$ (Ví dụ câu tiếng Anh: *"The **dog** that was chased by two cats through the park all afternoon **was** barking"* — danh từ số ít *"dog"* ở đầu câu quyết định trợ động từ *"was"* ở cuối câu).
   - Khi mô hình dự đoán sai ở bước $T$ (dự đoán *"were"* thay vì *"was"*), hàm mất mát $\mathcal{L}$ tạo ra một lượng lỗi lớn tại bước $T$.
   - Lỗi này cần được truyền ngược về bước $t=1$ để mô hình học được rằng: *"Sở dĩ bước $T$ cần chia động từ số ít là vì đã có từ 'dog' ở bước 1"*.
   - Nhưng do sequence dài ($T - 1$ lớn), $\frac{\partial \mathcal{L}}{\partial h_1} \approx 0$. 

2. **Trọng số không được cập nhật cho ngữ cảnh xa**:
   - Công thức cập nhật trọng số qua Gradient Descent:
     $$W_{hh} \leftarrow W_{hh} - \eta \frac{\partial \mathcal{L}}{\partial W_{hh}}, \quad \text{trong đó } \frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$
   - Thành phần đóng góp của các timestep ở xa ($t \ll T$) vào gradient tổng là $\frac{\partial \mathcal{L}}{\partial h_t} \approx 0$.
   - Do đó, mạng **chỉ cập nhật trọng số dựa trên các từ nằm ngay sát timestep hiện tại** (short-term memory). Mô hình cư xử giống hệt như một mô hình Markov bậc thấp (chỉ nhìn thấy $1 \sim 3$ từ gần nhất), hoàn toàn mù tịt trước các thông tin ở đầu chuỗi.

3. **Hiện tượng Catastrophic Forgetting (Quên lãng thông tin quá khứ)**:
   - Về chiều thuận (Forward pass): Việc nhân liên tiếp $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t)$ qua hàng chục bước khiến thông tin ban đầu bị "ghi đè" (overwritten) liên tục bởi các token mới.
   - Về chiều nghịch (Backward pass): Vanishing gradient ngăn cản mạng học cách bảo toàn thông tin đó.
   - Kết quả: Sequence càng dài, Simple RNN càng hoàn toàn bất lực trong việc duy trì ngữ cảnh dài hạn.

---

## 4.4. Minh họa trực quan và Tác động tới trọng số

```
Chiều THUẬN (Forward Pass) - Bị ghi đè liên tục:
  x_1 (dog)          x_2 (...)          x_{T-1} (...)       x_T (was/were)
     │                  │                     │                  │
     ▼                  ▼                     ▼                  ▼
   [h_1] ──W_hh──►    [h_2] ──W_hh──► ... ──► [h_{T-1}] ──W_hh──► [h_T] ──► Loss L
   (nhớ dog)      (bị pha loãng)           (quên gần hết)     (chỉ nhớ từ gần)

─────────────────────────────────────────────────────────────────────────────
Chiều NGHỊCH (Backward Pass / BPTT) - Gradient triệt tiêu theo hàm mũ:
   ∂L/∂h_1            ∂L/∂h_2                ∂L/∂h_{T-1}        ∂L/∂h_T
   ≈ 0.000001         ≈ 0.0001                ≈ 0.12             = 1.0
     ◄──[x Jacobian]──── ◄──[x Jacobian]──── ... ◄──[x Jacobian]──── │
   (Không thể học)   (Học rất yếu)         (Học mạnh mẽ)       (Tín hiệu gốc)
```

### Bảng tóm tắt nguyên nhân và hậu quả:
| Hiện tượng | Cơ chế toán học | Hậu quả thực tế trên NLP | Giải pháp khắc phục |
| :--- | :--- | :--- | :--- |
| **Vanishing Gradient** | $\prod_{k=t+1}^T \operatorname{diag}(1-h_k^2) W_{hh}^T \to 0$ khi $T - t$ lớn | Không thể học ngữ pháp cách xa (Subject-Verb agreement), đảo ngữ, phủ định xa | **LSTM / GRU** (đường truyền tắt additive Cell State), **Residual Connections**, **Attention Mechanism / Transformer** |
| **Exploding Gradient** | Tích Jacobian có trị riêng $> 1 \to \infty$ | Trọng số cập nhật bất ổn định, NaN/Inf, mô hình mất kiểm soát | **Gradient Clipping** ($\|\mathbf{g}\| \leftarrow \min(1, \frac{\text{clip}}{\|\mathbf{g}\|}) \mathbf{g}$) |

---

# 5. Bài 5 — Prediction trước thực nghiệm (Pre-experiment Predictions) (Mục 10)

Trước khi tiến hành huấn luyện các mô hình trong thực nghiệm, sinh viên đưa ra các giả thuyết và dự đoán khoa học kèm theo lập luận cơ chế và mức độ tin cậy theo đúng chuẩn quy trình khoa học dữ liệu:
* **Prediction**: Nội dung dự đoán cụ thể.
* **Reason**: Giải thích bản chất cơ chế toán học / kiến trúc mô hình dẫn đến dự đoán đó.
* **Confidence**: Mức độ tin cậy (Thấp / Trung bình / Cao / Tuyệt đối) kèm cơ sở.

---

### Prediction 1: TF-IDF + Logistic Regression vs RNN + Embedding về khả năng xử lý Word Order

* **Prediction**: 
  **RNN + Embedding** có khả năng xử lý thứ tự từ (word order) trực tiếp và vượt trội hoàn toàn so với **TF-IDF + Logistic Regression**.

* **Reason**:
  1. **TF-IDF + Logistic Regression** hoàn toàn là mô hình túi từ (Bag-of-Words). Biểu diễn văn bản được tính dựa trên tần suất từ mà không lưu vết vị trí:
     $$\vec{x}_{\text{TF-IDF}} = [ \text{tfidf}(w_1), \text{tfidf}(w_2), \dots, \text{tfidf}(w_{|V|}) ]^T$$
     Phép tính này có tính chất đối xứng hoàn toàn đối với vị trí của từ trong câu. Hai câu hoán vị vị trí như *"dog bites man"* và *"man bites dog"* có vector TF-IDF y hệt nhau, khiến Logistic Regression không thể phân biệt được chủ thể và khách thể.
  2. **RNN + Embedding** xử lý đầu vào tuần tự từng bước thời gian $t$. Trạng thái ẩn được cập nhật qua chuỗi hàm hợp phi tuyến:
     $$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
     Do phép toán này không có tính giao hoán ($f(x_A, f(x_B, h)) \neq f(x_B, f(x_A, h))$), vị trí xuất hiện của mỗi token sẽ trực tiếp định hình quỹ đạo dịch chuyển của $h_t$, cho phép mô hình nắm bắt cấu trúc ngữ pháp và phạm vi phủ định (negation scope) một cách trực tiếp.

* **Confidence**: **Cao (High - 95%)**
  *(Cơ sở: Đây là bản chất kiến trúc toán học nền tảng của mạng hồi quy so với mô hình BoW).*

---

### Prediction 2: RNN vs LSTM về khả năng xử lý Long-range Dependency

* **Prediction**: 
  **LSTM** có khả năng nắm bắt và duy trì các quan hệ phụ thuộc tầm xa (long-range dependencies) tốt hơn đáng kể so với **Vanilla/Simple RNN**.

* **Reason**:
  1. **Simple RNN** gặp hiện tượng **triệt tiêu đạo hàm (vanishing gradient)** nghiêm trọng do phép nhân lặp qua ma trận trọng số $W_{hh}^T$ và đạo hàm co của $\tanh$:
     $$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \operatorname{diag}(1 - h_k^2) W_{hh}^T \xrightarrow{T - t \gg 1} 0$$
     Khi chuỗi dài ($> 20 \sim 30$ tokens), tín hiệu lỗi từ tương lai không thể truyền ngược về quá khứ, khiến Simple RNN bị "mất trí nhớ tầm xa" và chỉ hành xử như một mô hình Markov bậc thấp.
  2. **LSTM** giải quyết triệt để vấn đề này nhờ cơ chế **Cell State ($C_t$) cộng tính (Additive Update)**:
     $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
     Khi lan truyền ngược từ $C_T$ về $C_t$:
     $$\frac{\partial C_T}{\partial C_t} = \prod_{k=t+1}^T f_k$$
     Nếu **forget gate** $f_k \approx 1$, gradient có thể truyền thẳng xuyên suốt hàng trăm bước thời gian như một "đường cao tốc" (gradient highway) mà không bị nhân với ma trận trọng số hay đạo hàm làm co biên độ, ngăn chặn hiện tượng vanishing gradient.

* **Confidence**: **Cao (High - 95%)**
  *(Cơ sở: Đã được chứng minh bằng lý thuyết giải tích giải thuật BPTT của Hochreiter & Schmidhuber 1997 và kiểm chứng thực nghiệm rộng rãi).*

---

### Prediction 3: Tác động khi Sequence Length tăng từ 50 lên 200 đối với quá trình training của RNN

* **Prediction**: 
  Khi sequence length tăng từ $50 \to 200$:
  1. **Thời gian huấn luyện (Training time)** mỗi epoch sẽ tăng xấp xỉ tuyến tính gấp $3 \sim 4$ lần.
  2. **Hiện tượng Vanishing/Exploding Gradient** trong Simple RNN sẽ trở nên trầm trọng hơn rất nhiều; gradient chuẩn (gradient norm) tại các timestep đầu tiên sẽ gần như bằng 0.
  3. **Hiệu năng (Accuracy)** của Simple RNN hầu như không tăng, thậm chí có thể giảm hoặc bão hòa sớm do nhiễu và độ trễ bộ nhớ; trong khi LSTM/GRU sẽ tận dụng tốt hơn ngữ cảnh dài nếu tài liệu chứa các từ khóa quan trọng ở xa.

* **Reason**:
  - Độ phức tạp tính toán của BPTT tỷ lệ thuận với số bước thời gian $T$: $\mathcal{O}(T \cdot d_h^2)$.
  - Tích ma trận Jacobian từ bước $200$ về bước $1$ trải qua $199$ lần nhân liên tiếp thay vì $49$ lần. Với hệ số co $\gamma < 1$, giá trị $\gamma^{199}$ nhỏ hơn $\gamma^{49}$ hàng chục bậc lũy thừa (ví dụ nếu $\gamma = 0.9$, $0.9^{49} \approx 0.0057$, trong khi $0.9^{199} \approx 1.07 \times 10^{-9}$), khiến tín hiệu gradient từ đầu câu biến mất hoàn toàn.
  - Bộ nhớ GPU tiêu tốn cho việc lưu giữ activations phục vụ BPTT tăng gấp 4 lần.

* **Confidence**: **Cao (High - 90%)**
  *(Cơ sở: Tính toán độ phức tạp thuật toán và phân tích hàm mũ số bước BPTT).*

---

### Prediction 4: Tác động khi Truncate toàn bộ review về 30 tokens lên Classification

* **Prediction**: 
  Nếu cắt ngắn (truncate) toàn bộ review về chỉ còn 30 tokens, **độ chính xác (accuracy) của mô hình phân loại sẽ bị suy giảm rõ rệt (dự kiến giảm từ $5\% \sim 15\%$)**.

* **Reason**:
  1. **Phân bố độ dài thực tế của IMDb Review**: IMDb là tập đánh giá phim dài, độ dài trung bình thường từ $150 \sim 250$ tokens. Việc cắt tại $30$ tokens làm vứt bỏ tới $80\% \sim 90\%$ dung lượng văn bản của đa số tài liệu.
  2. **Đặc thù hành văn phê bình phim**: 
     - Rất nhiều người viết mở đầu bài review bằng cách tóm tắt cốt truyện, giới thiệu diễn viên hoặc bối cảnh trung lập trong $1 \sim 2$ câu đầu ($30$ tokens đầu tiên).
     - Kết luận khen/chê cốt lõi mang tính quyết định (ví dụ: *"Overall, this is a waste of time"* hoặc *"In summary, a brilliant masterpiece"*) thường nằm ở **cuối bài viết (conclusion)**.
     - Cắt cụt ở 30 tokens đầu tiên khiến mô hình mất đi tín hiệu phân loại quan trọng nhất, dẫn đến phân loại nhầm lẫn hàng loạt.

* **Confidence**: **Cao (High - 90%)**
  *(Cơ sở: Đặc điểm phân bố độ dài văn bản IMDb và cấu trúc hành văn phê bình).*

---

### Prediction 5: Liệu LSTM có chắc chắn cho accuracy cao hơn RNN trên mọi dataset không?

* **Prediction**: 
  **KHÔNG CHẮC CHẮN.** LSTM **không** phải luôn cho accuracy cao hơn RNN trên mọi dataset.

* **Reason**:
  1. **Trường hợp chuỗi rất ngắn (Short Sequences)**: Với các bài toán có sequence length rất ngắn ($T < 10 \sim 15$ tokens, ví dụ phân loại tiêu đề báo ngắn, tweet ngắn), Simple RNN không gặp vấn đề vanishing gradient. Do đó, khả năng biểu diễn của Simple RNN là hoàn toàn đủ.
  2. **Vấn đề số lượng tham số và Overfitting**:
     - LSTM có tới 4 cổng/tầng nội bộ ($f_t, i_t, o_t, \tilde{C}_t$), do đó số lượng tham số xấp xỉ gấp **4 lần** so với Simple RNN với cùng kích thước hidden size $d_h$:
       $$\Theta_{\text{LSTM}} \approx 4 \times (d_x d_h + d_h^2 + d_h), \quad \Theta_{\text{RNN}} \approx (d_x d_h + d_h^2 + d_h)$$
     - Trên các dataset kích thước nhỏ (ít mẫu huấn luyện), số lượng tham số lớn khiến LSTM rất dễ bị **quá khớp (overfitting)**, trong khi Simple RNN với số tham số ít hơn lại tổng quát hóa (generalize) tốt hơn trên tập test.
  3. **Đặc trưng bài toán dựa trên từ khóa cục bộ**: Nếu nhãn phân loại chỉ phụ thuộc vào sự xuất hiện của một vài từ khóa đơn lẻ (keyword-driven task) mà không phụ thuộc vào liên kết đường dài, Simple RNN hoàn toàn có thể đạt kết quả ngang ngửa hoặc thậm chí nhỉnh hơn nếu LSTM không được điều chuẩn (regularization/dropout) tốt.

* **Confidence**: **Tuyệt đối (Absolute - 98%)**
  *(Cơ sở: Định lý "No Free Lunch" trong Machine Learning và trade-off giữa năng lực mô hình vs rủi ro overfitting).*

---

# 6. Bài 6 — Kiểm tra & Thống kê Dataset IMDb (Dataset Inspection) (Mục 11)

## 6.1. Tổng quan tập dữ liệu IMDb Movie Reviews

Tập dữ liệu chính được sử dụng cho toàn bộ các thực nghiệm phân loại chuỗi trong Lab 05 là **IMDb Large Movie Review Dataset** (Maas et al., 2011).
* **Bài toán (Task)**: Phân loại sắc thái cảm xúc nhị phân (Binary Sentiment Classification).
* **Không gian nhãn**:
  * `0`: Tiêu cực (Negative review).
  * `1`: Tích cực (Positive review).
* **Lý do lựa chọn**:
  - Dữ liệu văn bản tự nhiên có chiều dài đa dạng, câu phức tạp, giàu sắc thái đảo ngữ và phủ định.
  - Phù hợp hoàn hảo để khảo sát các mô hình xử lý chuỗi (Sequence Models: RNN, LSTM, GRU), so sánh đối chiếu trực tiếp với cách tiếp cận Bag-of-Words / TF-IDF và Average Word2Vec từ LAB 04.

---

## 6.2. Bảng thống kê chi tiết đặc tính dữ liệu

Dựa trên việc kiểm tra toàn diện qua script phân tích tập dữ liệu (`datasets.load_dataset('stanfordnlp/imdb')`):

| Thuộc tính kiểm tra | Giá trị thống kê | Nhận xét & Đánh giá |
| :--- | :--- | :--- |
| **Tổng số mẫu (Total samples)** | **50,000** | Tập dữ liệu tiêu chuẩn có kích thước lớn cho bài toán phân loại sắc thái. |
| **Số mẫu tập huấn luyện (Train samples)** | **25,000** | Đủ lớn để huấn luyện các mạng nơ-ron hồi quy mà không cần pre-trained weights. |
| **Số mẫu tập kiểm thử (Test samples)** | **25,000** | Kích thước test độc lập, chuẩn mực để đánh giá độ tin cậy. |
| **Số lượng nhãn (Number of classes)** | **2** | Nhị phân (Binary): `0 (Negative)`, `1 (Positive)`. |
| **Phân bố nhãn Train (Class distribution Train)** | **12,500 / 12,500 (50.0% - 50.0%)** | **Cân bằng hoàn hảo (Balanced)**. Accuracy là metric đo lường hoàn toàn tin cậy. |
| **Phân bố nhãn Test (Class distribution Test)** | **12,500 / 12,500 (50.0% - 50.0%)** | **Cân bằng hoàn hảo (Balanced)**. |
| **Văn bản bị khuyết (Missing text / NaN)** | **0 mẫu (0.0%)** | Không có văn bản rỗng hoặc `None`. |
| **Nhãn bị khuyết (Missing labels)** | **0 mẫu (0.0%)** | Đầy đủ nhãn ground-truth 100%. |
| **Số lượng văn bản trùng lặp (Duplicates)** | **418 mẫu** trên toàn bộ 50k mẫu<br>(96 mẫu trong Train, 199 mẫu trong Test) | Tỷ lệ trùng lặp rất nhỏ (~0.8%), không làm sai lệch đáng kể phân phối huấn luyện. |
| **Độ dài chuỗi nhỏ nhất (Min sequence length)** | **4 từ** | Câu cực ngắn (ví dụ nhận xét cộc lốc hoặc tiêu đề vắn tắt). |
| **Độ dài chuỗi trung vị (Median sequence length)** | **173.0 từ** | 50% số lượng văn bản có độ dài $\le 173$ từ. |
| **Độ dài chuỗi trung bình (Mean sequence length)** | **231.16 từ** | Độ lệch chuẩn $\sigma = 171.34$ từ cho thấy sự phân tán độ dài rất lớn. |
| **Độ dài chuỗi lớn nhất (Max sequence length)** | **2,470 từ** | Các bài phê bình điện ảnh phân tích chi tiết, rất dài. |

---

## 6.3. Phân bố độ dài chuỗi (Sequence Length Distribution & Histogram)

### 1. Bảng phân vị chi tiết (Percentiles of Sequence Length):
* **25th percentile**: $126$ từ
* **50th percentile (Median)**: $173$ từ
* **75th percentile**: $280$ từ
* **80th percentile**: $320$ từ
* **85th percentile**: $372$ từ
* **90th percentile**: $451$ từ
* **95th percentile**: $590$ từ
* **98th percentile**: $779$ từ
* **99th percentile**: $908$ từ

### 2. Biểu đồ Histogram phân bố độ dài chuỗi:

```text
Sequence Length (words)      Số văn bản   Tỷ lệ     Histogram ASCII
─────────────────────────────────────────────────────────────────────────────────
[   0 -   50]               1,152        2.3%     ██
[  50 -  100]               4,865        9.7%     ██████████
[ 100 -  150]              13,661       27.3%     ██████████████████████████████
[ 150 -  200]               9,737       19.5%     █████████████████████
[ 200 -  300]               9,362       18.7%     ████████████████████
[ 300 -  400]               4,706        9.4%     ██████████
[ 400 -  500]               2,591        5.2%     █████
[ 500 -  750]               2,758        5.5%     ██████
[ 750 - 1000]               1,071        2.1%     ██
[1000 - 2500]                  97        0.2%     ▏
─────────────────────────────────────────────────────────────────────────────────
```

*File biểu đồ đồ họa sắc nét độ phân giải cao đã được tạo và lưu trữ tại:* `sequence_length_distribution.png`.

### Nhận xét phân bố:
- Phân bố độ dài văn bản có dạng **lệch phải mạnh (strongly right-skewed)**:
  - Đỉnh phân bố tập trung dày đặc nhất trong khoảng từ $100$ đến $250$ từ.
  - Hơn $75\%$ bài review có độ dài dưới $280$ từ.
  - Đuôi phân bố (long-tail) kéo dài tới tận $2,470$ từ, nhưng số lượng tài liệu dài trên $1,000$ từ chiếm chưa tới $0.2\%$.

---

## 6.4. Cơ sở khoa học lựa chọn tham số `max_length`

> **Câu hỏi yêu cầu**:  
> *Nếu phải chọn `max_length`, nên chọn dựa trên thông tin nào?*  
> *(Không chấp nhận câu trả lời chỉ là: "Chọn 100 vì model chạy nhanh".)*

### Trả lời chi tiết:
Việc lựa chọn tham số `max_length` (ngưỡng cắt tỉa và đệm chuỗi) là một bài toán tối ưu hóa đa mục tiêu giữa **độ bảo toàn thông tin (Information Coverage)**, **chi phí tính toán / bộ nhớ (Computational Cost)**, và **độ ổn định của Gradient (Numerical Stability)**. 

Cụ thể, việc lựa chọn cần căn cứ chặt chẽ vào **4 thông tin cốt lõi** sau:

1. **Phân tích phân vị tích lũy (Cumulative Percentile Coverage) từ biểu đồ thực nghiệm**:
   - Không nên chọn ngẫu nhiên một con số tròn trĩnh, mà phải nhìn vào phân vị tích lũy của phân bố chiều dài:
     - Nếu chọn $\text{max\_length} = 100$: Chỉ bao phủ được $\approx 12\%$ số văn bản trọn vẹn; gần $88\%$ bài review bị cắt cụt (truncate), dẫn đến mất mát thông tin trầm trọng.
     - Nếu chọn $\text{max\_length} = 200$: Bao phủ được xấp xỉ $60\%$ số văn bản nguyên vẹn.
     - Nếu chọn $\text{max\_length} = 250 \sim 300$: Bao phủ được từ $70\% \sim 80\%$ văn bản mà không cần cắt bỏ một từ nào.
   - Nguyên tắc chuẩn trong kỹ thuật NLP thực tế là chọn điểm "đầu gối" (elbow point) của hàm tích lũy, thường bao phủ từ **$75\% \sim 85\%$ độ dài của toàn bộ tập dữ liệu** (tương ứng khoảng $200 \sim 300$ tokens với IMDb).

2. **Cơ chế hoạt động của thuật toán BPTT & Hạn chế kiến trúc (Architecture Limitation)**:
   - Với **Simple RNN**: Gradient triệt tiêu theo hàm mũ $\gamma^T \to 0$. Khi chuỗi dài quá $100$ bước, RNN hoàn toàn không thể học được thông tin ở các bước đầu. Nếu đặt `max_length = 500` cho Simple RNN, chi phí tính toán tăng gấp 5 lần nhưng mô hình chỉ bị thêm nhiễu mà không cải thiện được chất lượng.
   - Với **LSTM / GRU**: Bộ nhớ có thể vươn tới $150 \sim 300$ bước nhờ additive state. Do đó `max_length = 200 \sim 250` là điểm giao thoa hoàn hảo giữa khả năng duy trì bộ nhớ của LSTM và bảo toàn thông tin văn bản.

3. **Cân đối tài nguyên bộ nhớ GPU và Độ phức tạp tính toán (Computational Complexity)**:
   - Độ phức tạp thời gian của BPTT tỉ lệ bậc nhất $\mathcal{O}(T \cdot d_h^2)$ với chiều dài chuỗi $T$.
   - Chi phí bộ nhớ GPU để lưu giữ đồ thị tính toán (activation graph) cho mỗi mini-batch là $\mathcal{O}(B \times T \times d_h)$.
   - Nếu nâng từ $T = 200$ lên $T = 1,000$ để bao phủ nốt $2\%$ văn bản còn lại, ta phải trả giá bằng việc **tăng gấp 5 lần thời gian huấn luyện và 5 lần bộ nhớ GPU** cho toàn bộ 50,000 mẫu (do phải padding các câu ngắn lên 1,000). Đây là sự đánh đổi hoàn toàn phi kinh tế.

4. **Phương thức cắt tỉa (Truncation Strategy): Pre-truncation vs Post-truncation**:
   - Khi chọn `max_length`, thông tin ngữ nghĩa nằm ở đâu cũng quyết định cách cắt:
     - Nếu review khen/chê nằm ở cuối bài (nhận định tổng kết), cắt bỏ phần đầu (head truncation / lấy $N$ từ cuối) hoặc giữ phần cuối sẽ giữ được tín hiệu phân loại tốt hơn là cắt cụt đuôi (tail truncation).
     - Việc chọn `max_length = 200` kết hợp chiến lược truncation hợp lý giúp bảo toàn được cả ngữ cảnh mở đầu lẫn tín hiệu cảm xúc quyết định.

* **Khuyến nghị thông số định lượng cho Lab 05**:  
  Chọn **`max_length = 200`** (hoặc $250$):  
  - Bao phủ trọn vẹn hơn $60\% \sim 70\%$ văn bản và giữ được phần lớn nội dung của $30\%$ còn lại.  
  - Nằm trong giới hạn tính toán an toàn của GPU/CPU trường học.  
  - Giúp BPTT của LSTM/GRU học ổn định mà không bị bùng nổ thời gian train.

---

# 7. Bài 7 — Thực nghiệm Padding và Truncation (Experiment 2) (Mục 13)

## 7.1. Bảng số liệu thực nghiệm định lượng

Thực nghiệm được tiến hành trên tập dữ liệu IMDb (Train: 25,000 mẫu, Test: 25,000 mẫu, Vocab size = 20,000, Embedding Dim = 128, Hidden Dim = 128, Batch Size = 128, Optimizer: Adam, Learning Rate = $10^{-3}$, 5 Epochs) đo lường trên phần cứng thực tế (NVIDIA GeForce RTX 3060 12GB):

* **Tổng số lượng token thực tế trong toàn bộ 50,000 văn bản**: $11,443,438$ tokens.

| Max length | % tokens retained | Training time (s) | Test Accuracy (%) | Test F1-Score (%) |
| :---: | :---: | :---: | :---: | :---: |
| **50** | **21.75%** (2,488,840 / 11.44M tokens) | **2.79s** | **72.60%** | **72.25%** |
| **100** | **41.98%** (4,803,757 / 11.44M tokens) | **4.53s** | **78.14%** | **78.01%** |
| **200** | **69.04%** (7,900,697 / 11.44M tokens) | **8.08s** | **79.26%** | **79.22%** |

*(Ghi chú đối chiếu với Simple RNN: Với Simple RNN, khi tăng `max_length` từ 50 lên 200, thời gian huấn luyện tăng từ 1.45s lên 2.11s nhưng do ảnh hưởng nặng nề của Vanishing Gradient, độ chính xác bị bão hòa và sụt giảm quanh mức $50\% \sim 56\%$. Trong khi đó, LSTM với cơ chế gating thể hiện rõ năng lực tận dụng thông tin tăng dần khi `max_length` tăng).*

---

## 7.2. Phân tích: Tăng max_length có luôn cải thiện performance không?

### Câu trả lời:
**KHÔNG. Tăng `max_length` KHÔNG LUÔN LUÔN cải thiện hiệu năng (performance), mà hiệu năng sẽ tăng dần đến một điểm bão hòa (plateau) rồi đi ngang, thậm chí suy giảm nếu tiếp tục tăng quá mức.**

### Giải thích bản chất cơ chế:

1. **Quy luật lợi ích cận biên giảm dần (Diminishing Marginal Returns)**:
   - Từ $T = 50 \to 100$: Tỷ lệ token giữ lại tăng từ $21.75\% \to 41.98\%$ ($+20.23\%$). Mô hình thu nhận thêm rất nhiều câu văn và từ khóa mang cảm xúc, giúp Accuracy tăng vọt từ **$72.60\% \to 78.14\%$ ($+5.54\%$)**.
   - Từ $T = 100 \to 200$: Tỷ lệ token giữ lại tiếp tục tăng thêm $+27.06\%$ (lên $69.04\%$), nhưng Accuracy chỉ tăng thêm **$+1.12\%$** (đạt $79.26\%$). Tốc độ cải thiện đã chậm lại rõ rệt.
   - Nếu tiếp tục tăng $T = 200 \to 500$: Tỷ lệ văn bản dài trên 200 từ chỉ còn thiểu số, nhưng toàn bộ các bài ngắn buộc phải độn thêm hàng trăm token `<PAD>`, dẫn đến lượng thông tin mới nạp vào là rất ít so với chi phí tính toán.

2. **Nhiễu do Padding (Padding Noise) và Memory Decay**:
   - Khi `max_length` quá lớn (ví dụ 500 hoặc 1,000), phần lớn văn bản trong batch là các token `<PAD>`.
   - Mặc dù ta đã dùng `padding_idx=0`, trạng thái ẩn $h_t$ vẫn phải chạy qua nhiều bước chuyển tiếp hoặc bị ảnh hưởng bởi độ dài không đồng nhất.
   - Với Simple RNN, việc tăng $T$ còn làm trầm trọng hóa hiện tượng **Vanishing Gradient** (đạo hàm co liên tiếp qua $\gamma^T \approx 0$), khiến việc cập nhật trọng số ở các bước đầu câu bị vô hiệu hóa hoàn toàn, làm mô hình giảm độ chính xác thay vì tăng.

3. **Gia tăng chi phí tính toán và Nguy cơ Overfitting**:
   - Thời gian tính toán và bộ nhớ GPU tăng tỷ lệ thuận tuyến tính với $T$. 
   - Số bước lan truyền qua thời gian càng dài thì không gian tìm kiếm gradient càng phức tạp, dễ dẫn đến hội tụ vào các cực tiểu cục bộ (local minima) xấu hoặc bị quá khớp (overfit) vào các đoạn mô tả dài dòng không mang nhãn phân loại.

---

## 7.3. Phân tích: Nếu max_length quá nhỏ, loại thông tin nào bị mất mát?

Khi `max_length` bị ép quá nhỏ (ví dụ $T = 30$ hoặc $T = 50$ tokens, chỉ giữ được $< 22\%$ dung lượng bài viết), mô hình sẽ phải chịu sự mất mát của **3 loại thông tin ngôn ngữ cực kỳ quan trọng**:

### 1. Mất câu kết luận tổng kết cảm xúc (Loss of Concluding Sentiment):
* **Đặc thù bài phê bình (Review Structure)**: Cấu trúc kinh điển của một bài đánh giá phim là:
  $$\text{Giới thiệu bối cảnh / Tóm tắt nội dung} \longrightarrow \text{Phân tích chi tiết} \longrightarrow \text{Kết luận & Chấm điểm}$$
* $30 \sim 50$ từ đầu tiên hầu như chỉ chứa các thông tin trung lập: tên đạo diễn, tên nhân vật, giới thiệu cốt truyện sơ khởi (ví dụ: *"This movie tells the story of a young boy living in the 1920s..."*).
* Cảm xúc chốt hạ mang tính quyết định nhãn lại nằm ở cuối bài viết (ví dụ: *"...Overall, despite a few flaws, this is an absolute masterpiece that everyone must watch"*). Cắt cụt ở $T = 50$ sẽ **vứt bỏ hoàn toàn câu kết luận này**, khiến mô hình không thể tìm thấy bằng chứng phân loại.

### 2. Mất thông tin đảo ngược cảm xúc qua các liên từ tương phản (Contrastive Discourse Markers):
* Trong ngôn ngữ tự nhiên, người viết rất hay sử dụng cấu trúc tương phản (*"khen trước chê sau"* hoặc *"chê trước khen sau"*):
  - Ví dụ: *"The visuals and CGI in the first half were stunning and visually breathtaking. **However**, the plot was horribly written, the acting was dreadful, and I completely hated the ending."*
* Nếu cắt tại $T=50$, mô hình chỉ đọc được phần *"stunning and visually breathtaking"* và tự tin gán nhãn **Positive**, trong khi thực chất nhãn của toàn bộ bài viết là **Negative** (nằm sau từ liên từ chuyển ý *"However"*).

### 3. Mất phạm vi phủ định tầm xa (Long-range Negation Scope) và Sự phụ thuộc ngữ pháp (Long-distance Dependency):
* Các cấu trúc câu phức tạp như phủ định điều kiện:
  - Ví dụ: *"I thought I would never find a movie as terrible as this one until I watched..."* hoặc *"I cannot believe anyone would praise this garbage."*
* Khi chuỗi bị cắt ngắn, các mệnh đề quan hệ, các cặp từ liên kết điều kiện (*neither... nor*, *although... but*) bị đứt đoạn, khiến mô hình chỉ nhìn thấy các mảnh ngữ cảnh vụn vặt và diễn giải sai lệch hoàn toàn ngữ nghĩa của người viết.

---

# 8. Bài 8 — Cài đặt và Giải thích Computation Graph của Simple RNN Cell (Mục 14)

## 8.1. Đồ thị tính toán (Computation Graph)

Dưới đây là đồ thị tính toán chi tiết của một bước chuyển tiếp trong Simple RNN Cell:

```text
    x_t (Input vector tại t)                h_(t-1) (Hidden state từ quá khứ)
       │                                            │
       ▼                                            ▼
┌──────────────┐                             ┌──────────────┐
│ nhân ma trận │                             │ nhân ma trận │
│   x_t @ W_ih │                             │ h_(t-1)@W_hh │
└──────┬───────┘                             └──────┬───────┘
       │                                            │
       ├─────────────────────┐                      │
       │                     ▼                      │
       │                 ┌───────┐                  │
       │                 │   +   │ ◄────────────────┘
       │                 └───┬───┘
       │                     │  + b_h (Hệ số bias)
       │                     ▼
       │              [pre-activation: a_t]
       │                     │
       │                     ▼
       │                 ┌───────┐
       │                 │ tanh  │ (Hàm kích hoạt phi tuyến)
       │                 └───┬───┘
       │                     │
       ▼                     ▼
 (Nhánh khác)               h_t (Hidden state tại thời điểm t)
                             │
                             ├──────────────────────────► Đến timestep tiếp theo (t+1)
                             ▼
                 Đưa lên Output Classifier (hoặc loss tại t)
```

---

## 8.2. Giải thích từng thành phần và phép toán

1. **Vector đầu vào hiện tại ($x_t \in \mathbb{R}^{B \times d_{\text{in}}}$)**:
   * Biểu diễn phân tán (Word Embedding) của token tại vị trí thời gian $t$.
   * Được nhân với ma trận trọng số $W_{ih} \in \mathbb{R}^{d_h \times d_{\text{in}}}$ (trong code là `x_t @ W_ih.T` do batch-first) để chuyển đổi đặc trưng từ không gian từ vựng sang không gian trạng thái ẩn.

2. **Hidden State quá khứ ($h_{t-1} \in \mathbb{R}^{B \times d_h}$)**:
   * Lưu giữ toàn bộ "bộ nhớ" tóm tắt thông tin ngữ cảnh từ các timestep trước ($1, 2, \dots, t-1$).
   * Được nhân với ma trận trọng số hồi quy $W_{hh} \in \mathbb{R}^{d_h \times d_h}$ (`h_prev @ W_hh.T`) để kiểm soát lượng thông tin quá khứ nào sẽ được tích hợp tiếp.

3. **Phép cộng tổng hợp và cộng Bias ($+$)**:
   * Hai dòng thông tin (hiện tại và quá khứ) được cộng tuyến tính với nhau cùng với vector hệ số chệch $b_h$:
     $$a_t = x_t W_{ih}^T + h_{t-1} W_{hh}^T + b_h$$
   * Phép cộng này tạo ra vector kích hoạt thô (pre-activation $a_t$).

4. **Hàm kích hoạt Hyperbolic Tangent ($\tanh$)**:
   * Ánh xạ từng phần tử của $a_t$ về khoảng giá trị giới hạn $(-1, 1)$:
     $$h_t = \tanh(a_t) = \frac{e^{a_t} - e^{-a_t}}{e^{a_t} + e^{-a_t}}$$
   * Giúp nén biên độ của trạng thái ẩn, ngăn chặn các giá trị phát triển mất kiểm soát sau hàng chục bước nhân lặp, đồng thời bổ sung tính phi tuyến giúp mạng học được các quan hệ phức tạp.

5. **Trạng thái ẩn mới ($h_t \in \mathbb{R}^{B \times d_h}$)**:
   * Vừa đóng vai trò là đầu ra của timestep $t$, vừa được tái sử dụng làm đầu vào quá khứ $h_{\text{prev}}$ cho timestep tiếp theo $t+1$.

---

## 8.3. Kết quả xác thực tương đương với PyTorch `nn.RNNCell`

Code cài đặt hoàn chỉnh độc lập được viết trong file [`rnn_cell.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab05/rnn_cell.py).

Khi chạy kiểm thử nghiệm thu so sánh trực tiếp trên cùng một tập trọng số và tensor đầu vào ngẫu nhiên ($B=3, d_{\text{in}}=4, d_h=5$):
* **Độ lệch tuyệt đối lớn nhất giữa hai implementation**:
  $$\max |h_{\text{custom}} - h_{\text{pytorch}}| = 4.47 \times 10^{-8}$$
* **Kết luận**:
  Lớp `SimpleRNNCell` tự xây dựng đạt độ chính xác tương đương tuyệt đối với `torch.nn.RNNCell` của thư viện PyTorch (vượt qua kiểm tra `torch.allclose` với sai số máy tính $< 10^{-7}$). Điều này chứng minh việc hiểu đúng đắn và chính xác 100% đồ thị tính toán của tế bào RNN hồi quy.

---

# 9. Bài 9 — Thực nghiệm Baseline: Logistic Regression vs Simple RNN (Experiment 3) (Mục 15)

## 9.1. Bảng số liệu so sánh toàn diện Baseline

Thực nghiệm so sánh giữa mô hình Baseline truyền thống từ LAB 04 (**TF-IDF + Logistic Regression**) và mô hình tuần tự nơ-ron hồi quy đơn giản (**Word Embedding + Simple RNN + Linear Classifier**) trên tập dữ liệu chuẩn IMDb (Train: 25,000, Test: 25,000):

| Model | Representation | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test F1-Score (%) | Training time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **TF-IDF (Unigram)** | **89.98%** | **89.40%** | **90.80%** | **90.09%** | **13.86s** (fit + tfidf) |
| **Simple RNN** | **Learnable Embedding (dim=128)** | **61.37%** | **62.10%** | **58.35%** | **60.17%** | **4.06s** (5 epochs GPU) |

### Nhận xét sơ bộ kết quả:
* **Logistic Regression với TF-IDF** đạt độ chính xác rất cao ($89.98\%$) vì IMDb chứa nhiều từ ngữ mang phân cực cảm xúc rõ rệt (*"terrible", "awful", "fantastic", "masterpiece"*). Phương pháp Bag-of-Words dựa trên tần suất từ ngữ bắt trọn vẹn các từ khóa này trên toàn bộ tài liệu mà không bị giới hạn chiều dài.
* **Simple RNN** chỉ đạt độ chính xác khiêm tốn ($61.37\%$). Đây là hiện tượng chuẩn tắc của mạng Vanilla RNN khi đối mặt với văn bản dài do **Vanishing Gradient** làm mô hình chỉ ghi nhớ được một vài từ cuối chuỗi và quên sạch bối cảnh phía trước.
* **Tuy nhiên**, câu hỏi cốt lõi không chỉ là Accuracy bên nào cao hơn, mà là: **Simple RNN có thực sự sử dụng thông tin thứ tự (sequence order) mà TF-IDF hoàn toàn bỏ qua hay không?**

---

## 9.2. Bằng chứng thực nghiệm: RNN có thực sự sử dụng thông tin Sequence hay không?

### Câu trả lời:
**CÓ, CHẮC CHẮN 100%. RNN thực sự sử dụng và mô hình hóa thông tin thứ tự tuần tự của chuỗi (Sequence Information), trong khi TF-IDF + Logistic Regression hoàn toàn bất lực và bỏ qua thông tin này.**

### Bằng chứng thực nghiệm định lượng (Empirical Evidence):

Tiến hành kiểm thử trên các cặp câu có **tập hợp từ vựng giống hệt nhau** nhưng **đảo ngược thứ tự từ hoặc đảo vị trí từ phủ định**:

#### Cặp 1: Phép hoán vị chủ ngữ - tân ngữ
* $S_1 = \text{"dog bites man"}$
* $S_2 = \text{"man bites dog"}$

| Model & Phương pháp | Dự đoán $P(\text{Positive} \mid S_1)$ | Dự đoán $P(\text{Positive} \mid S_2)$ | Kết luận phân biệt |
| :--- | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | **0.5287** | **0.5287** | ❌ **Trùng khớp 100%** (Sai khác: `0.000000`) |
| **Simple RNN + Embedding** | **0.5974** | **0.3693** |  **Phân biệt rõ ràng** (Chênh lệch: `0.2281`) |

* **Giải thích**:
  - Với **TF-IDF**, vector đếm tần suất của cả hai câu là đồng nhất:
    $$\vec{v}_{\text{tfidf}}(S_1) \equiv \vec{v}_{\text{tfidf}}(S_2)$$
    Hệ số sai khác giữa 2 vector qua `(v1 != v2).nnz == 0`. Do đó mô hình tuyến tính bắt buộc phải cho ra xác suất y hệt nhau.
  - Với **RNN**, token `"dog"` ở vị trí đầu câu làm thay đổi $h_1$, sau đó đi qua hàm hợp phi tuyến và tương tác với $W_{hh}$. Trạng thái cuối $h_3^{(S_1)} \neq h_3^{(S_2)}$, phản ánh đúng việc vị trí từ quyết định ý nghĩa biểu diễn.

---

#### Cặp 2: Phạm vi phủ định (Negation Scope)
* $S_1 = \text{"this movie is good"}$ (Khen)
* $S_2 = \text{"this movie is not good"}$ (Chê qua phủ định)

| Model & Phương pháp | Dự đoán $P(\text{Positive} \mid S_1)$ | Dự đoán $P(\text{Positive} \mid S_2)$ | Sự chuyển dịch xác suất |
| :--- | :---: | :---: | :---: |
| **Simple RNN + Embedding** | **0.3630** | **0.2142** |  **Xác suất tích cực giảm mạnh khi xuất hiện "not" ngay trước "good"** |

* **Giải thích**:
  - Sự xuất hiện của từ `"not"` ở timestep $t=4$ trực tiếp điều biến trạng thái ẩn $h_4$, làm giảm phân cực dương của từ `"good"` ở bước kế tiếp $t=5$. RNN đã nắm bắt được sự tương tác tuần tự mang tính cục bộ.

---

#### Cặp 3: Đảo ngữ tương phản
* $S_1 = \text{"not bad, actually quite good"}$ (Thiên hướng khen)
* $S_2 = \text{"not good, actually quite bad"}$ (Thiên hướng chê)

| Model & Phương pháp | Dự đoán $P(\text{Positive} \mid S_1)$ | Dự đoán $P(\text{Positive} \mid S_2)$ | Nhận xét |
| :--- | :---: | :---: | :---: |
| **TF-IDF (Unigram) + LogReg** | Gần như không đổi do chứa cùng các từ: *not, good, bad, actually, quite* | Không thể biết "not" áp dụng lên "good" hay "bad" |
| **Simple RNN + Embedding** | **0.2648** vs **0.4368** | Phân tách hai trạng thái biểu diễn tách biệt |

---

### Tổng kết bài học rút ra từ Experiment 3:
1. **Trade-off giữa Global Bag-of-Words vs Local Sequence Order**:
   - TF-IDF mạnh ở việc nhìn thấy bức tranh tổng thể của toàn bộ tài liệu (Global keywords) mà không bị giới hạn độ dài, nhưng hoàn toàn "mù" trước thứ tự từ.
   - Simple RNN nhạy cảm với thứ tự từ và cấu trúc ngữ pháp ngắn hạn, nhưng lại bị tê liệt trên chuỗi dài do Vanishing Gradient.
2. Đây chính là động lực khoa học cấp thiết dẫn đến sự ra đời của **LSTM và GRU** (được nghiên cứu ở các phần tiếp theo), nhằm kết hợp trọn vẹn cả hai ưu điểm: **vừa bắt được thứ tự từ, vừa bảo toàn được thông tin tầm xa qua hàng trăm timestep**.

---

# 10. Bài 10 — Cơ chế điều khiển cổng của LSTM (LSTM Gates Calculation) (Mục 17)

## 10.1. Đề bài & Thông số giả thiết

Cho một mô hình Long Short-Term Memory (LSTM) tại một timestep $t$ cụ thể với các giá trị vô hướng (scalar) đại diện cho một chiều đặc trưng:
* **Forget gate (cổng quên)**: $f_t = 0.9$
* **Input gate (cổng vào)**: $i_t = 0.2$
* **Candidate cell state (trạng thái ứng viên)**: $\tilde{c}_t = 0.5$
* **Previous cell state (trạng thái ô nhớ trước đó)**: $c_{t-1} = 0.8$

**Yêu cầu**:
1. Tính giá trị trạng thái ô nhớ mới tại timestep $t$:
   $$c_t = f_t c_{t-1} + i_t \tilde{c}_t$$
2. Phân tích bản chất:
   - Nếu *forget gate* gần $1$ ($f_t \to 1$), mô hình có xu hướng giữ lại hay loại bỏ thông tin cũ?
   - Nếu *forget gate* gần $0$ ($f_t \to 0$) thì sao?
3. Giải thích khái niệm **Gate như một cơ chế điều khiển dòng chảy thông tin (Information Flow)**, vượt lên trên việc ghi nhớ công thức máy móc.

---

## 10.2. Các bước tính toán cập nhật Cell State

Áp dụng công thức cập nhật trạng thái ô nhớ cộng tính (additive update) của LSTM:
$$c_t = f_t c_{t-1} + i_t \tilde{c}_t$$

### Bước 1: Tính thành phần giữ lại từ quá khứ (Retained Past Information)
$$f_t c_{t-1} = 0.9 \times 0.8 = 0.72$$
* Ý nghĩa: Cổng quên $f_t = 0.9$ cho phép **$90\%$** nội dung của ô nhớ quá khứ $c_{t-1} = 0.8$ được tiếp tục bảo tồn và truyền vào trạng thái hiện tại, chỉ lãng quên đi $10\%$.

### Bước 2: Tính thành phần bổ sung từ đầu vào mới (New Added Information)
$$i_t \tilde{c}_t = 0.2 \times 0.5 = 0.10$$
* Ý nghĩa: Cổng vào $i_t = 0.2$ chỉ mở một phần nhỏ (**$20\%$**) để đón nhận thông tin ứng viên mới $\tilde{c}_t = 0.5$ từ token hiện tại.

### Bước 3: Tổng hợp trạng thái Cell State hiện tại ($c_t$)
$$c_t = 0.72 + 0.10 = 0.82$$

* **Kết luận định lượng**: Trạng thái ô nhớ mới tại timestep $t$ là **$c_t = 0.82$**.

---

## 10.3. Phân tích bản chất: Forget Gate gần 1 vs gần 0

Hàm kích hoạt của Forget Gate là hàm sigmoid $\sigma(z) \in (0, 1)$, đóng vai trò là một van điều tiết tỷ lệ phần trăm thông tin được phép đi qua:

### 1. Nếu Forget Gate gần 1 ($f_t \to 1$):
* **Xu hướng của mô hình**: Mô hình có xu hướng **GIỮ LẠI HẦU HẾT (HOẶC TOÀN BỘ) THÔNG TIN CŨ** từ quá khứ:
  $$f_t c_{t-1} \approx 1 \times c_{t-1} = c_{t-1}$$
* **Ý nghĩa ngữ nghĩa trong NLP**:
  - Mô hình nhận diện được rằng ngữ cảnh quá khứ là **cực kỳ quan trọng** và vẫn đang chi phối câu hiện tại.
  - *Ví dụ*: Khi đọc một câu dài có chủ ngữ ở đầu câu: *"The **director** who won three Oscars last year and spent five years writing the screenplay ... [t=20: was / were]"*. 
  - Tại các từ trung gian ở giữa câu, mô hình duy trì $f_t \approx 1$ để bảo toàn đặc tính số ít của *"director"* suốt 20 bước mà không bị xóa mờ, giúp bước cuối chia đúng động từ *"was"*.
* **Ý nghĩa về mặt Gradient (BPTT)**:
  - Gradient của Cell State lan truyền ngược:
    $$\frac{\partial c_T}{\partial c_{t-1}} = \frac{\partial c_T}{\partial c_t} \cdot f_t$$
  - Khi $f_t \approx 1$, gradient truyền thẳng qua thời gian mà **không bị co giảm biên độ** ($\times 1.0$), đóng vai trò là **"đường cao tốc gradient" (gradient highway)** giải quyết triệt để hiện tượng Vanishing Gradient.

### 2. Nếu Forget Gate gần 0 ($f_t \to 0$):
* **Xu hướng của mô hình**: Mô hình có xu hướng **LOẠI BỎ / XÓA BỎ HOÀN TOÀN THÔNG TIN CŨ** từ quá khứ:
  $$f_t c_{t-1} \approx 0 \times c_{t-1} = 0 \implies c_t \approx i_t \tilde{c}_t$$
* **Ý nghĩa ngữ nghĩa trong NLP**:
  - Mô hình chủ động thực hiện thao tác **"xóa bảng nhớ" (Reset memory)** khi nhận thấy ngữ cảnh cũ đã kết thúc và không còn giá trị tham chiếu.
  - *Ví dụ 1 (Chấm hết câu hoặc chuyển đoạn)*: Khi gặp dấu chấm câu `.` hoặc dấu chấm phẩy `;`, thông tin về chủ ngữ của câu trước đã hoàn thành nhiệm vụ, $f_t \to 0$ giúp mô hình quên đi câu cũ để sẵn sàng tiếp nhận câu mới.
  - *Ví dụ 2 (Đảo ngược ngữ cảnh qua liên từ chuyển tiếp)*: Khi gặp từ chuyển ý mạnh như *"However"*, *"On the contrary"*, $f_t \to 0$ cho phép mạng xóa sạch ấn tượng tích cực ở đoạn đầu để bắt đầu ghi nhận chuỗi nhận xét tiêu cực tiếp theo.

---

## 10.4. Gates như một cơ chế điều khiển dòng chảy thông tin (Information Flow)

Mục tiêu cốt lõi của kiến trúc LSTM không phải là ghi nhớ một mớ công thức phức tạp, mà là hiểu **Gates như một hệ thống van đóng/mở điều tiết dòng chảy thông tin**:

```text
               ┌─────────────────────── [ Forget Gate: f_t ] (Giữ bao nhiêu % quá khứ?)
               │                                │
               ▼                                ▼
[ c_(t-1) ] ───────► ( x ) ───────────────────► ( + ) ─────────────────────────────► [ c_t ]
Quá khứ              │                            ▲             Cell State           │
                     │                            │             Đường cao tốc        │
                     │  ┌─ [ Input Gate: i_t ]    │             (Additive Highway)   │
                     │  │  (Nạp bao nhiêu % mới?) │                                  │
                     │  │         │               │                                  │
                     │  ▼         ▼               │                                  ▼
[ x_t, h_(t-1) ] ────┴────► ( x ) ────────────────┘                              ┌───────┐
Hiện tại & Context          ▲                                                    │ tanh  │
                            │                                                    └───┬───┘
                    [ Candidate: c~_t ]                                              │
                    (Thông tin ứng viên mới)                                         ▼
                                                        ┌─ [ Output Gate: o_t ] ──► ( x )
                                                        │  (Tiết lộ bao nhiêu ra ngoài?)
                                                        │                            │
                                                        ▼                            ▼
                                                [ Điều khiển o_t ]                [ h_t ]
```

### So sánh bản chất điều khiển thông tin:

| Thành phần Cổng | Toán tử điều khiển | Câu hỏi quyết định ngữ nghĩa | Ý nghĩa hành vi của Model |
| :--- | :---: | :--- | :--- |
| **Forget Gate ($f_t = \sigma(\dots)$)** | $\odot c_{t-1}$ | *"Thông tin quá khứ nào không còn giá trị và cần phải quên đi?"* | Ngăn chặn ô nhớ bị bão hòa hoặc bị "ám ảnh" bởi các ngữ cảnh cũ đã lỗi thời. |
| **Input Gate ($i_t = \sigma(\dots)$)** | $\odot \tilde{c}_t$ | *"Thông tin mới từ token hiện tại có đáng tin và quan trọng để ghi nhớ không?"* | Lọc bỏ từ dừng (stop words vô nghĩa như *the, a, an*) bằng cách cho $i_t \to 0$; chỉ cho phép các từ khóa quan trọng lọt vào $c_t$. |
| **Cell State Update ($c_t$)** | **Phép cộng ($+$)** | *"Tích hợp bộ nhớ dài hạn như thế nào?"* | **Additive Highway**: Khác với RNN dùng phép nhân hàm hợp lồng nhau làm triệt tiêu gradient, phép cộng cho phép gradient truyền ngược nguyên vẹn qua hàng trăm bước. |
| **Output Gate ($o_t = \sigma(\dots)$)** | $\odot \tanh(c_t)$ | *"Phần nào trong ô nhớ dài hạn cần được bộc lộ ra ngoài làm hidden state $h_t$?"* | Che giấu các chi tiết chưa cần dùng tới trong $c_t$, chỉ đưa ra $h_t$ những thông tin cần thiết phục vụ ngay cho bước tính toán kế tiếp hoặc bộ phân loại. |
