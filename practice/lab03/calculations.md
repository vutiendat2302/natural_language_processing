# BÀI TẬP TÍNH TOÁN (LAB 03 - CALCULATIONS)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên**: 23000111  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
- [6. Bài tập tính toán](#6-bài-tập-tính-toán)
  - [6.1. Bài 1 — Co-occurrence Matrix (Ma trận đồng xuất hiện)](#61-bài-1--co-occurrence-matrix-ma-trận-đồng-xuất-hiện)
  - [6.2. Bài 2 — Cosine Similarity & Phân tích Độ lớn Vector](#62-bài-2--cosine-similarity--phân-tích-độ-lớn-vector)
- [7. Bài 3 — So sánh semantic similarity](#7-bài-3--so-sánh-semantic-similarity)
- [8. Bài 4 — Sparse vs dense](#8-bài-4--sparse-vs-dense)
- [16. Bài tập prediction — CBOW vs Skip-gram](#16-bài-tập-prediction--cbow-vs-skip-gram)
- [23. Bài tập tính analogy](#23-bài-tập-tính-analogy)

---

# 6. Bài tập tính toán

## 6.1. Bài 1 — Co-occurrence Matrix (Ma trận đồng xuất hiện)

### Đề bài
Cho tập ngữ liệu (corpus) gồm 4 câu:
- $S_1$: `"the cat eats fish"`
- $S_2$: `"the dog eats fish"`
- $S_3$: `"the cat likes milk"`
- $S_4$: `"the dog likes meat"`

Cửa sổ ngữ cảnh đối xứng (context window):
$$k = 1$$
(xét 1 từ liền trước và 1 từ liền sau trong cùng một câu).

**Yêu cầu**: Hãy xây dựng word-context vector cho 4 từ mục tiêu (target words):
$$\text{cat}, \quad \text{dog}, \quad \text{eats}, \quad \text{likes}$$
*(Thực hiện tính toán thủ công, không sử dụng Python).*

---

### Cơ sở lý thuyết & Định nghĩa toán học
Theo **Giả thuyết phân phối (Distributional Hypothesis)** của Zellig Harris (1954) và J.R. Firth (1957):
> *"Words that occur in similar contexts tend to have similar meanings."*  
> (Những từ xuất hiện trong ngữ cảnh tương tự nhau thường có ý nghĩa gần nhau).

Ma trận đồng xuất hiện từ - ngữ cảnh (Word-Context Matrix) $\mathbf{X} \in \mathbb{R}^{|V_{\text{target}}| \times |V_{\text{context}}|}$ đếm số lần từ mục tiêu $w_i$ xuất hiện trong khoảng cách cửa sổ $k$ đối với từ ngữ cảnh $c_j$:
$$X_{ij} = \sum_{S \in \mathcal{D}} \sum_{t=1}^{|S|} \mathbb{I}(S[t] = w_i) \sum_{\substack{1 \le |p| \le k \\ 1 \le t+p \le |S|}} \mathbb{I}(S[t+p] = c_j)$$

Trong bài toán này, xét $k = 1$: với mỗi từ ở vị trí $t$, từ ngữ cảnh là các từ ở vị trí $t - 1$ (liền trước) và $t + 1$ (liền sau) trong cùng câu.

---

### Xác định Tập từ vựng (Vocabulary)
Theo thiết kế chuẩn của Lab 03 (Mục 5 & Mục 6):
Tập từ vựng loại bỏ từ dừng (`the` - stopword) gồm 7 từ nội dung (content words):
$$\mathcal{V} = \{\text{cat}, \text{dog}, \text{eats}, \text{likes}, \text{fish}, \text{milk}, \text{meat}\} \quad (|\mathcal{V}| = 7)$$

*(Ghi chú: Mục 6.1.5 bên dưới trình bày thêm trường hợp giữ lại cả từ dừng `"the"`).*

---

### Phân tích chi tiết từng cặp đồng xuất hiện ($k = 1$)

Ta duyệt qua từng câu trong ngữ liệu với cửa sổ $k = 1$:

1. **Câu 1 ($S_1$)**: `["the", "cat", "eats", "fish"]`
   - `the`: từ sau là `cat`.
   - `cat`: từ trước là `the` (bỏ qua nếu không trong $\mathcal{V}$), từ sau là `eats`. $\to$ Cặp: `(cat, eats): 1`.
   - `eats`: từ trước là `cat`, từ sau là `fish`. $\to$ Cặp: `(eats, cat): 1`, `(eats, fish): 1`.
   - `fish`: từ trước là `eats`. $\to$ Cặp: `(fish, eats): 1`.

2. **Câu 2 ($S_2$)**: `["the", "dog", "eats", "fish"]`
   - `the`: từ sau là `dog`.
   - `dog`: từ trước là `the`, từ sau là `eats`. $\to$ Cặp: `(dog, eats): 1`.
   - `eats`: từ trước là `dog`, từ sau là `fish`. $\to$ Cặp: `(eats, dog): 1`, `(eats, fish): 1`.
   - `fish`: từ trước là `eats`. $\to$ Cặp: `(fish, eats): 1`.

3. **Câu 3 ($S_3$)**: `["the", "cat", "likes", "milk"]`
   - `the`: từ sau là `cat`.
   - `cat`: từ trước là `the`, từ sau là `likes`. $\to$ Cặp: `(cat, likes): 1`.
   - `likes`: từ trước là `cat`, từ sau là `milk`. $\to$ Cặp: `(likes, cat): 1`, `(likes, milk): 1`.
   - `milk`: từ trước là `likes`. $\to$ Cặp: `(milk, likes): 1`.

4. **Câu 4 ($S_4$)**: `["the", "dog", "likes", "meat"]`
   - `the`: từ sau là `dog`.
   - `dog`: từ trước là `the`, từ sau là `likes`. $\to$ Cặp: `(dog, likes): 1`.
   - `likes`: từ trước là `dog`, từ sau là `meat`. $\to$ Cặp: `(likes, dog): 1`, `(likes, meat): 1`.
   - `meat`: từ trước là `likes`. $\to$ Cặp: `(meat, likes): 1`.

---

### Kết quả xây dựng Word-Context Vectors

#### Bảng ma trận đồng xuất hiện đầy đủ (với $\mathcal{V}$ gồm 7 từ nội dung):

| Word | cat | dog | eats | likes | fish | milk | meat |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **cat** | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **dog** | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **eats** | 1 | 1 | 0 | 0 | 2 | 0 | 0 |
| **likes** | 1 | 1 | 0 | 0 | 0 | 1 | 1 |

#### Chi tiết 4 vector theo yêu cầu đề bài:
Thứ tự các chiều: $[\text{cat}, \text{dog}, \text{eats}, \text{likes}, \text{fish}, \text{milk}, \text{meat}]$

1. **Vector cho từ `cat`**:
   $$\vec{v}_{\text{cat}} = [0, \; 0, \; 1, \; 1, \; 0, \; 0, \; 0]$$
   - Xuất hiện cạnh `eats` 1 lần ($S_1$) và cạnh `likes` 1 lần ($S_3$).

2. **Vector cho từ `dog`**:
   $$\vec{v}_{\text{dog}} = [0, \; 0, \; 1, \; 1, \; 0, \; 0, \; 0]$$
   - Xuất hiện cạnh `eats` 1 lần ($S_2$) và cạnh `likes` 1 lần ($S_4$).

3. **Vector cho từ `eats`**:
   $$\vec{v}_{\text{eats}} = [1, \; 1, \; 0, \; 0, \; 2, \; 0, \; 0]$$
   - Xuất hiện cạnh `cat` 1 lần ($S_1$), cạnh `dog` 1 lần ($S_2$), và cạnh `fish` 2 lần ($S_1, S_2$).

4. **Vector cho từ `likes`**:
   $$\vec{v}_{\text{likes}} = [1, \; 1, \; 0, \; 0, \; 0, \; 1, \; 1]$$
   - Xuất hiện cạnh `cat` 1 lần ($S_3$), cạnh `dog` 1 lần ($S_4$), cạnh `milk` 1 lần ($S_3$), và cạnh `meat` 1 lần ($S_4$).

---

#### Mở rộng: Trường hợp tập từ vựng giữ lại cả từ dừng `"the"`
Nếu $\mathcal{V}' = \{\text{the}, \text{cat}, \text{dog}, \text{eats}, \text{likes}, \text{fish}, \text{milk}, \text{meat}\}$ (8 chiều):
- Cả `cat` và `dog` đều đứng ngay sau `the` trong 2 câu của mỗi từ:
  $$\vec{v}'_{\text{cat}} = [2, \; 0, \; 0, \; 1, \; 1, \; 0, \; 0, \; 0]$$
  $$\vec{v}'_{\text{dog}} = [2, \; 0, \; 0, \; 1, \; 1, \; 0, \; 0, \; 0]$$
  $$\vec{v}'_{\text{eats}} = [0, \; 1, \; 1, \; 0, \; 0, \; 2, \; 0, \; 0]$$
  $$\vec{v}'_{\text{likes}} = [0, \; 1, \; 1, \; 0, \; 0, \; 0, \; 1, \; 1]$$

---

### Nhận xét học thuật sâu sắc
- **Hiện tượng $\vec{v}_{\text{cat}} \equiv \vec{v}_{\text{dog}}$**:
  Trong cả 2 cách chọn từ vựng, vector của `cat` và `dog` **hoàn toàn trùng khít nhau 100%**:
  $$\cos(\vec{v}_{\text{cat}}, \vec{v}_{\text{dog}}) = 1.0$$
  Điều này phản ánh trực tiếp nguyên lý nền tảng của NLP: dù `cat` và `dog` là hai sinh vật khác nhau và chưa từng xuất hiện cùng nhau trong một câu, nhưng vì chúng chia sẻ **phân phối ngữ cảnh tương đồng tuyệt đối** (đều đứng sau `the`, đều thực hiện hành động `eats` và `likes`), không gian biểu diễn phân phối đã tự động ánh xạ chúng về cùng một vị trí.
- **Sự tương đồng giữa `eats` và `likes`**:
  Cả hai động từ này đều nhận `cat` và `dog` làm chủ ngữ ở bên trái, và các danh từ thức ăn (`fish`, `milk`, `meat`) làm tân ngữ ở bên phải. Do đó chúng cũng có độ tương đồng ngữ cảnh rất cao.

---

## 6.2. Bài 2 — Cosine Similarity & Phân tích Độ lớn Vector

### Đề bài
Cho hai vector:
$$x = [1, 2, 1]$$
$$y = [2, 4, 2]$$

**Yêu cầu**:
1. Tính $\cos(x, y)$.
2. Giải thích: *Hai vector có độ lớn khác nhau nhưng cosine similarity bằng 1 có ý nghĩa gì?*

---

### Chi tiết tính toán

#### Bước 1: Tính tích vô hướng (Dot Product) $x \cdot y$
$$x \cdot y = \sum_{i=1}^3 x_i y_i = (1 \times 2) + (2 \times 4) + (1 \times 2) = 2 + 8 + 2 = 12$$

#### Bước 2: Tính chuẩn Euclid ($L_2$ Norm) của từng vector
- Chuẩn của $x$:
  $$\|x\|_2 = \sqrt{1^2 + 2^2 + 1^2} = \sqrt{1 + 4 + 1} = \sqrt{6} \approx 2.4494897$$
- Chuẩn của $y$:
  $$\|y\|_2 = \sqrt{2^2 + 4^2 + 2^2} = \sqrt{4 + 16 + 4} = \sqrt{24} = \sqrt{4 \times 6} = 2\sqrt{6} \approx 4.8989795$$

#### Bước 3: Tính Cosine Similarity
$$\cos(x, y) = \frac{x \cdot y}{\|x\|_2 \cdot \|y\|_2} = \frac{12}{\sqrt{6} \cdot (2\sqrt{6})} = \frac{12}{2 \times 6} = \frac{12}{12} = \mathbf{1.0}$$

*(Góc giữa hai vector: $\theta = \arccos(1.0) = 0^\circ$)*.

---

### Ý nghĩa bản chất trong NLP & Hình học không gian

1. **Ý nghĩa hình học**:
   - Hai vector $x$ và $y$ có mối quan hệ phụ thuộc tuyến tính:
     $$y = 2 \cdot x \quad (\text{hệ số tỉ lệ } c = 2 > 0)$$
   - Tỉ số giữa các thành phần tương ứng hoàn toàn đồng nhất: $\frac{y_1}{x_1} = \frac{y_2}{x_2} = \frac{y_3}{x_3} = 2$.
   - Hai vector chỉ cùng một hướng duy nhất trong không gian 3 chiều, góc lệch $\theta = 0^\circ$. Độ lớn (magnitude) của $y$ gấp đôi độ lớn của $x$ ($\|y\|_2 = 2\|x\|_2$), nhưng Cosine Similarity hoàn toàn triệt tiêu độ lớn thông qua phép chuẩn hóa (normalization).

2. **Ý nghĩa trong Xử lý ngôn ngữ tự nhiên (NLP)**:
   - **Độ lớn vector ($\|v\|$) đại diện cho TẦN SUẤT (Frequency / Document Length)**: Vector $y$ có thể là biểu diễn của một từ xuất hiện nhiều gấp đôi từ $x$, hoặc biểu diễn của một tài liệu có độ dài gấp đôi nhưng có cùng nội dung chủ đề.
   - **Hướng của vector đại diện cho HỒ SƠ NGỮ NGHĨA (Semantic Profile / Distributional Meaning)**: Hướng phản ánh tỉ lệ tương đối giữa các đặc trưng ngữ cảnh. Ở đây, tỉ lệ xuất hiện giữa ngữ cảnh 1, ngữ cảnh 2 và ngữ cảnh 3 của cả hai từ đều là $1 : 2 : 1$.
   - **Kết luận**: $\cos(x, y) = 1$ chứng minh rằng hai từ/văn bản có **ngữ nghĩa tương đồng tuyệt đối** về mặt phân phối chủ đề, không bị ảnh hưởng bởi sự thiên lệch tần suất xuất hiện tuyệt đối. Đây chính là lý do vì sao **Cosine Similarity** được chọn làm độ đo chuẩn mực thay cho khoảng cách Euclid ($L_2$ distance) trong không gian vector từ và tìm kiếm thông tin.

---

# 7. Bài 3 — So sánh semantic similarity

### Đề bài
Cho các vector embedding giả định:
$$v_{\text{doctor}} = [0.8, \; 0.1, \; 0.7]$$
$$v_{\text{physician}} = [0.7, \; 0.2, \; 0.8]$$
$$v_{\text{banana}} = [-0.2, \; 0.9, \; -0.1]$$

**Yêu cầu**:
1. Dự đoán từ nào gần `doctor` hơn trước khi tính.
2. Tính chính xác:
   - $\cos(\text{doctor}, \text{physician})$
   - $\cos(\text{doctor}, \text{banana})$

---

### Dự đoán trước khi tính toán (Hypothesis & Intuition)
- **Về mặt ngôn ngữ học**:
  - `doctor` và `physician` là hai từ đồng nghĩa chân thực (true synonyms), cùng chỉ nghề nghiệp thầy thuốc y khoa, xuất hiện trong các ngữ cảnh điều trị, bệnh viện, khám chữa bệnh.
  - `banana` (quả chuối) thuộc trường ngữ nghĩa nông sản/thực phẩm, hoàn toàn dị biệt với trường y tế.
- **Về mặt hình thức số học của vector**:
  - Vector $v_{\text{doctor}}$ và $v_{\text{physician}}$ đều có giá trị lớn ở chiều 1 ($\approx 0.7 - 0.8$) và chiều 3 ($\approx 0.7 - 0.8$), và giá trị rất nhỏ ở chiều 2 ($\approx 0.1 - 0.2$). Các chiều này đại diện cho các latent features tương thích nhau.
  - Ngược lại, $v_{\text{banana}}$ tập trung hầu hết trọng số vào chiều 2 ($0.9$), trong khi chiều 1 và chiều 3 mang giá trị âm ($-0.2, -0.1$).
- **Dự đoán**:
  `physician` sẽ gần `doctor` hơn rất nhiều so với `banana`. Kỳ vọng $\cos(\text{doctor}, \text{physician}) \approx 1$ (dương lớn, rất gần 1), trong khi $\cos(\text{doctor}, \text{banana}) < 0$ (âm, phân kỳ hoặc trực giao).

---

### Chi tiết tính toán

#### Bước 1: Tính chuẩn Euclid ($L_2$ norm) của 3 vector

- Chuẩn của $v_{\text{doctor}}$:
  $$\|v_{\text{doctor}}\|_2 = \sqrt{0.8^2 + 0.1^2 + 0.7^2} = \sqrt{0.64 + 0.01 + 0.49} = \sqrt{1.14} \approx 1.0677078$$

- Chuẩn của $v_{\text{physician}}$:
  $$\|v_{\text{physician}}\|_2 = \sqrt{0.7^2 + 0.2^2 + 0.8^2} = \sqrt{0.49 + 0.04 + 0.64} = \sqrt{1.17} \approx 1.0816654$$

- Chuẩn của $v_{\text{banana}}$:
  $$\|v_{\text{banana}}\|_2 = \sqrt{(-0.2)^2 + 0.9^2 + (-0.1)^2} = \sqrt{0.04 + 0.81 + 0.01} = \sqrt{0.86} \approx 0.9273618$$

---

#### Bước 2: Tính $\cos(\text{doctor}, \text{physician})$

- Tích vô hướng:
  $$v_{\text{doctor}} \cdot v_{\text{physician}} = (0.8 \times 0.7) + (0.1 \times 0.2) + (0.7 \times 0.8) = 0.56 + 0.02 + 0.56 = 1.14$$

- Cosine similarity:
  $$\cos(\text{doctor}, \text{physician}) = \frac{v_{\text{doctor}} \cdot v_{\text{physician}}}{\|v_{\text{doctor}}\|_2 \cdot \|v_{\text{physician}}\|_2} = \frac{1.14}{\sqrt{1.14} \cdot \sqrt{1.17}} = \frac{\sqrt{1.14}}{\sqrt{1.17}} = \sqrt{\frac{1.14}{1.17}}$$
  $$\cos(\text{doctor}, \text{physician}) = \sqrt{\frac{38}{39}} \approx \sqrt{0.97435897} \approx \mathbf{0.987096} \quad (\approx \mathbf{0.9871})$$

---

#### Bước 3: Tính $\cos(\text{doctor}, \text{banana})$

- Tích vô hướng:
  $$v_{\text{doctor}} \cdot v_{\text{banana}} = (0.8 \times (-0.2)) + (0.1 \times 0.9) + (0.7 \times (-0.1)) = -0.16 + 0.09 - 0.07 = -0.14$$

- Cosine similarity:
  $$\cos(\text{doctor}, \text{banana}) = \frac{-0.14}{\sqrt{1.14} \cdot \sqrt{0.86}} = \frac{-0.14}{\sqrt{0.9804}} \approx \frac{-0.14}{0.9901515} \approx \mathbf{-0.141392} \quad (\approx \mathbf{-0.1414})$$

---

### Tổng hợp so sánh & Kết luận

| Cặp so sánh | Dot Product | Norm tích | Cosine Similarity | Nhận xét ngữ nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| `(doctor, physician)` | $+1.14$ | $1.1549$ | **$+0.9871$** | Cực kỳ gần gũi (Đồng nghĩa cao độ) |
| `(doctor, banana)` | $-0.14$ | $0.9902$ | **$-0.1414$** | Khác biệt hoàn toàn (Không liên quan / hơi âm) |

**Kết luận**: Kết quả tính toán thực nghiệm định lượng khớp chính xác 100% với dự đoán ngôn ngữ học ban đầu: từ `physician` gần `doctor` hơn áp đảo so với `banana`.

---

# 8. Bài 4 — Sparse vs dense

### Đề bài
Cho tập từ vựng kích thước:
$$V = 10,000$$
- Biểu diễn thứ nhất: Một **word-context representation** có $10,000$ chiều nhưng chỉ có $30$ giá trị khác không (non-zero entries).
- Biểu diễn thứ hai: Một **embedding** có $300$ chiều và hầu hết các thành phần đều khác 0.

Sinh viên trả lời 4 câu hỏi:
1. Biểu diễn nào sparse?
2. Biểu diễn nào dense?
3. Vì sao dense representation có thể thuận lợi hơn cho semantic similarity?
4. Dense representation có chắc chắn “tốt hơn” trong mọi bài toán không?

---

### Lời giải chi tiết & Phân tích chuyên sâu

#### 8.1. Biểu diễn nào sparse?
- **Trả lời**: **Word-context representation (10,000 chiều)** là biểu diễn **sparse (thưa)**.
- **Căn cứ định lượng**:
  - Số chiều: $D_{\text{sparse}} = 10,000$.
  - Số phần tử khác 0: $k = 30$.
  - Độ đặc (Density):
    $$\text{Density} = \frac{30}{10,000} = 0.003 = 0.3\%$$
  - Độ thưa (Sparsity):
    $$\text{Sparsity} = 1 - \text{Density} = 99.7\%$$
  - $99.7\%$ không gian của vector là các số 0 vô nghĩa, phản ánh đặc trưng kinh điển của biểu diễn one-hot hoặc word-context count thô.

---

#### 8.2. Biểu diễn nào dense?
- **Trả lời**: **Word embedding (300 chiều)** là biểu diễn **dense (dày đặc)**.
- **Căn cứ định lượng**:
  - Số chiều được nén lại đáng kể: $d = 300 \ll 10,000$ (giảm $33.3$ lần về số chiều).
  - Hầu hết các thành phần đều là các số thực liên tục khác 0 ($\text{Density} \approx 100\%$, $\text{Sparsity} \approx 0\%$).
  - Mỗi chiều trong vector không còn đại diện cho một từ rời rạc duy nhất, mà là một **chiều đặc trưng tiềm ẩn (latent semantic feature)** mang tính phân tán (distributed representation).

---

#### 8.3. Vì sao dense representation có thể thuận lợi hơn cho semantic similarity?

Dense representation mang lại 4 ưu thế vượt trội khi tính toán độ tương đồng ngữ nghĩa:

1. **Khắc phục Vấn đề Trực giao (Orthogonality Problem) & Thiếu hụt Ngữ cảnh trùng khớp**:
   - Trong biểu diễn sparse, hai từ đồng nghĩa (ví dụ: *doctor* và *physician*) nếu xuất hiện trong các câu có từ ngữ cảnh khác nhau từng chữ (exact word mismatch) thì tích vô hướng giữa chúng bằng 0:
     $$v_{\text{sparse}}^{(\text{doctor})} \cdot v_{\text{sparse}}^{(\text{physician})} = 0 \implies \cos = 0$$
   - Ngược lại, dense embedding ánh xạ từ vào không gian liên tục thấp chiều, nơi các ngữ cảnh tương tự đã được gom cụm (cluster) trong quá trình huấn luyện nơ-ron, giúp nhận diện được sự đồng nghĩa ngay cả khi chúng không dùng chung chính xác các từ xung quanh.

2. **Năng lực Khái quát hóa Ngữ nghĩa Tiềm ẩn (Latent Semantic Generalization)**:
   - Thay vì lưu trữ tần suất cơ học, dense vectors nén thông tin bậc cao (second-order co-occurrence / transitive relationships). Nếu từ $A$ đi với $B$, và từ $C$ cũng đi với $B$, mô hình dense embedding sẽ kéo vector của $A$ và $C$ lại gần nhau, tạo nên khả năng suy diễn bắc cầu ngữ nghĩa.

3. **Hiệu năng Tính toán và Tối ưu Tài nguyên (Computational Efficiency)**:
   - Tính toán Cosine similarity giữa hai vector 300 chiều tốn chi phí $O(d = 300)$ phép tính, nhanh hơn gấp hàng chục đến hàng trăm lần so với duyệt qua vector $10,000$ hoặc $100,000$ chiều ($O(|V|)$).
   - Tiết kiệm dung lượng RAM và bộ nhớ đệm (cache), cho phép mở rộng (scale) lên các tập dữ liệu triệu văn bản.

4. **Tránh Lời nguyền Số chiều (Curse of Dimensionality)**:
   - Trong không gian 10,000 chiều, khoảng cách giữa các điểm dữ liệu trở nên cực kỳ loãng và đồng đều (distance concentration phenomenon). Đưa về 300 chiều giúp các thuật toán phân cụm (k-means), phân loại (SVM, MLP) hoạt động ổn định và tránh hiện tượng quá khớp (overfitting).

---

#### 8.4. Dense representation có chắc chắn “tốt hơn” trong mọi bài toán không?

> **Khẳng định cốt lõi**: **KHÔNG CHẮC CHẮN.** Tuyệt đối không được biến embedding thành khẩu hiệu giáo điều *"dense = luôn tốt hơn"*. Sự lựa chọn phụ thuộc hoàn toàn vào bản chất bài toán, kích thước dữ liệu và yêu cầu hệ thống.

Các trường hợp cụ thể mà biểu diễn **Sparse** vượt trội hoặc bắt buộc sử dụng:

1. **Khả năng Giải thích và Minh bạch (Interpretability & Explainability)**:
   - Trong biểu diễn sparse (TF-IDF, Bag-of-Words), mỗi chiều tương ứng đúng 1 từ ngữ cảnh cụ thể. Ta biết chính xác từ nào đóng góp bao nhiêu trọng số vào quyết định phân loại.
   - Trong dense embedding, các chiều là đặc trưng ẩn (latent features) trừu tượng, mô hình hoạt động như một "hộp đen" (black box). Trong các hệ thống y tế pháp lý, kiểm toán, sparse representation thường được ưu tiên để đảm bảo khả năng giải trình.

2. **Tìm kiếm Từ khóa Chính xác & Thực thể Hiếm (Exact Keyword Matching & Rare Entities)**:
   - Trong bài toán Tìm kiếm thông tin (Information Retrieval), khi người dùng tìm kiếm mã lỗi kỹ thuật (ví dụ: `ERR_CONNECTION_TIMED_OUT`), mã định danh sản phẩm (ví dụ: `RTX-4090-TI`), hoặc tên riêng hiếm gặp (tên người, địa danh cụ thể):
     - **Sparse (BM25, TF-IDF)**: Trả về kết quả chính xác 100% nhờ cơ chế khớp từ khóa nguyên bản.
     - **Dense**: Có xu hướng "làm mờ" (blur) các từ hiếm và kéo về các từ phổ biến lân cận trong không gian tiềm ẩn (semantic drift / hallucination), dẫn đến kết quả tìm kiếm sai lệch.

3. **Ngữ liệu Huấn luyện Giới hạn (Low-Resource / Small Corpus)**:
   - Huấn luyện dense embedding (Word2Vec, FastText) cần lượng dữ liệu đủ lớn (hàng triệu token) để các trọng số hội tụ. Nếu dữ liệu quá nhỏ (ví dụ vài trăm câu), dense embedding sẽ học rất tệ, bị nhiễu và overfit.
   - Ngược lại, phương pháp đếm thống kê tần suất sparse vẫn hoạt động hoàn hảo và cực kỳ ổn định trên tập dữ liệu nhỏ.

4. **Chi phí Tính toán, Huấn luyện và Cập nhật (Training & Incremental Updates)**:
   - Sparse matrix xây dựng cực nhanh bằng một lượt duyệt đếm từ, có thể cập nhật tăng dần (incremental updates) theo thời gian thực mà không cần huấn luyện lại từ đầu.
   - Dense embedding cần quá trình tối ưu hóa gradient descent phức tạp, tốn thời gian và đòi hỏi tài nguyên tính toán (GPU).

5. **Xu thế Hiện đại trong Công nghiệp: Tìm kiếm Kết hợp (Hybrid Search)**:
   - Trong các hệ thống tìm kiếm hiện đại và kiến trúc **RAG (Retrieval-Augmented Generation)**, giải pháp tối ưu nhất luôn là **kết hợp Sparse + Dense** (ví dụ: BM25 kết hợp Dense Retriever qua thuật toán Reciprocal Rank Fusion - RRF) để tận dụng đồng thời độ chính xác từ khóa của sparse và chiều sâu ngữ nghĩa của dense.

---

# 16. Bài tập prediction — CBOW vs Skip-gram

### Đề bài
Cho câu:
$$\text{"the cat eats fish"}$$
với kích thước cửa sổ ngữ cảnh:
$$window = 1 \quad (k = 1)$$

**Yêu cầu**: Liệt kê đầy đủ các training examples (mẫu huấn luyện) cho:
1. Mô hình **CBOW** (Continuous Bag-of-Words)
2. Mô hình **Skip-gram**
*(Không chạy code, tự tạo thủ công các cặp dữ liệu huấn luyện).*

---

### Cơ sở lý thuyết về kiến trúc huấn luyện

| Đặc tính | CBOW (Continuous Bag-of-Words) | Skip-gram |
| :--- | :--- | :--- |
| **Bản chất bài toán** | Dự đoán từ mục tiêu dựa vào ngữ cảnh xung quanh | Dự đoán các từ ngữ cảnh xung quanh dựa vào từ mục tiêu |
| **Đầu vào (Input)** | Các từ ngữ cảnh trong cửa sổ: $\{w_{t-k}, \dots, w_{t+k}\} \setminus \{w_t\}$ | Từ mục tiêu hiện tại: $w_t$ |
| **Đầu ra (Target/Label)** | Từ mục tiêu ở vị trí trung tâm: $w_t$ | Từng từ ngữ cảnh riêng lẻ: $w_{t+j}$ ($j \in [-k, k], j \neq 0$) |
| **Cơ chế biểu diễn input** | Tổng (sum) hoặc trung bình (average) vector embedding của các context words | Duy nhất một vector embedding của target word |

---

### Xác định vị trí các token trong câu
Câu gồm $T = 4$ tokens:
$$w_0 = \text{"the"}, \quad w_1 = \text{"cat"}, \quad w_2 = \text{"eats"}, \quad w_3 = \text{"fish"}$$

Với $window = 1$, tại vị trí $t$, cửa sổ lân cận là $[t-1, t+1]$ (giới hạn trong phạm vi câu $0 \le \text{index} < 4$).

---

### 16.1. Liệt kê Training Examples cho CBOW
Ở mỗi vị trí $t$, tập hợp các từ ngữ cảnh làm đầu vào, từ $w_t$ làm nhãn dự đoán:

1. **Vị trí $t = 0$ ($w_0 = \text{"the"}$)**:
   - Ngữ cảnh bên phải ($t+1$): `cat` (bên trái không có).
   - **CBOW Example 1**: $\text{Context: } [\text{"cat"}] \longrightarrow \text{Target: } \text{"the"}$

2. **Vị trí $t = 1$ ($w_1 = \text{"cat"}$)**:
   - Ngữ cảnh bên trái ($t-1$): `the`; Ngữ cảnh bên phải ($t+1$): `eats`.
   - **CBOW Example 2**: $\text{Context: } [\text{"the"}, \text{"eats"}] \longrightarrow \text{Target: } \text{"cat"}$

3. **Vị trí $t = 2$ ($w_2 = \text{"eats"}$)**:
   - Ngữ cảnh bên trái ($t-1$): `cat`; Ngữ cảnh bên phải ($t+1$): `fish`.
   - **CBOW Example 3**: $\text{Context: } [\text{"cat"}, \text{"fish"}] \longrightarrow \text{Target: } \text{"eats"}$

4. **Vị trí $t = 3$ ($w_3 = \text{"fish"}$)**:
   - Ngữ cảnh bên trái ($t-1$): `eats` (bên phải không có).
   - **CBOW Example 4**: $\text{Context: } [\text{"eats"}] \longrightarrow \text{Target: } \text{"fish"}$

**Tổng số mẫu CBOW**: **4 training examples**.

---

### 16.2. Liệt kê Training Examples cho Skip-gram
Với mỗi từ trung tâm $w_t$, tạo ra các cặp huấn luyện độc lập $(\text{Input: } w_t, \text{Target: } c)$ với từng từ ngữ cảnh $c$:

1. **Từ mục tiêu $w_0 = \text{"the"}$**:
   - Từ ngữ cảnh lân cận: `cat`
   - Cặp huấn luyện:
     $$\mathbf{(\text{"the"}, \text{"cat"})}$$

2. **Từ mục tiêu $w_1 = \text{"cat"}$**:
   - Từ ngữ cảnh lân cận: `the` (bên trái), `eats` (bên phải)
   - Các cặp huấn luyện:
     $$\mathbf{(\text{"cat"}, \text{"the"})}$$
     $$\mathbf{(\text{"cat"}, \text{"eats"})}$$

3. **Từ mục tiêu $w_2 = \text{"eats"}$**:
   - Từ ngữ cảnh lân cận: `cat` (bên trái), `fish` (bên phải)
   - Các cặp huấn luyện:
     $$\mathbf{(\text{"eats"}, \text{"cat"})}$$
     $$\mathbf{(\text{"eats"}, \text{"fish"})}$$

4. **Từ mục tiêu $w_3 = \text{"fish"}$**:
   - Từ ngữ cảnh lân cận: `eats`
   - Cặp huấn luyện:
     $$\mathbf{(\text{"fish"}, \text{"eats"})}$$

**Tổng số mẫu Skip-gram**: **6 training pairs**.

---

# 23. Bài tập tính analogy

### Đề bài
Cho các vector biểu diễn giả định trong không gian 3 chiều:
$$\vec{v}_{\text{king}} = [8, \; 2, \; 7]$$
$$\vec{v}_{\text{man}} = [5, \; 1, \; 5]$$
$$\vec{v}_{\text{woman}} = [5, \; 3, \; 5]$$

**Yêu cầu**:
1. Tính giá trị vector số học: $\vec{v}_{\text{result}} = \vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$.
2. Giải thích vector mới này có thể đại diện cho loại quan hệ nào trong ngữ nghĩa học.

---

### Chi tiết các bước tính toán

#### Bước 1: Tính vector hiệu số $\vec{v}_{\text{king}} - \vec{v}_{\text{man}}$
$$\vec{v}_{\text{relation}} = \vec{v}_{\text{king}} - \vec{v}_{\text{man}} = [8 - 5, \; 2 - 1, \; 7 - 5] = [3, \; 1, \; 2]$$

#### Bước 2: Cộng với vector $\vec{v}_{\text{woman}}$
$$\vec{v}_{\text{result}} = (\vec{v}_{\text{king}} - \vec{v}_{\text{man}}) + \vec{v}_{\text{woman}} = [3 + 5, \; 1 + 3, \; 2 + 5] = \mathbf{[8, \; 4, \; 7]}$$

---

### Phân tích ý nghĩa hình học & Bản chất Ngữ nghĩa học

1. **Phân tích từng chiều đặc trưng (Feature Dimensions)**:
   - **Chiều 1 và Chiều 3 (Đặc tính Uy quyền / Hoàng gia - "Royalty / Monarch")**:
     - $\vec{v}_{\text{king}}$ có tọa độ $[8, \cdot, 7]$, trong khi người bình thường $\vec{v}_{\text{man}}$ và $\vec{v}_{\text{woman}}$ chỉ có $[5, \cdot, 5]$.
     - Vector hiệu số $[3, \cdot, 2]$ đóng vai trò là **vector dịch chuyển quan hệ (relational translation vector)**, mã hóa thuộc tính *"hoàng gia / nắm giữ vương quyền"*.
   - **Chiều 2 (Đặc tính Giới tính - "Gender Dimension")**:
     - Giữa $\vec{v}_{\text{man}} = [5, \mathbf{1}, 5]$ và $\vec{v}_{\text{woman}} = [5, \mathbf{3}, 5]$, chiều 2 tăng thêm $2$ đơn vị ($1 \to 3$), phản ánh sự chuyển dịch từ nam tính sang nữ tính.
     - Khi thực hiện phép tính, chiều 2 của kết quả là: $2 - 1 + 3 = 4$. Giá trị $4$ này lớn hơn giá trị của $\text{king}$ ($2$), hoàn toàn đồng pha với đặc tính nữ giới của $\text{woman}$.

2. **Vector mới đại diện cho khái niệm gì?**:
   - Vector kết quả $\vec{v}_{\text{result}} = [8, 4, 7]$ sở hữu đồng thời hai thuộc tính cốt lõi:
     - Tính chất hoàng gia cao cấp (chiều 1 bằng 8, chiều 3 bằng 7 — giống hệt `king`).
     - Tính chất nữ giới (chiều 2 bằng 4 — cao hơn cả `king` và mang đặc trưng của `woman`).
   - Do đó, vector này đại diện hoàn hảo cho khái niệm:
     $$\mathbf{\text{queen} \quad (\text{Nữ hoàng / Vương hậu})}$$
   - Đây chính là công thức tương tự kinh điển của Word2Vec:
     $$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$

3. **Lưu ý phương pháp luận quan trọng**:
   - Hiện tượng này là hệ quả của tính **tuyến tính cục bộ (local linearity)** trong không gian vector tối ưu hóa bởi Skip-gram/CBOW, chứ **không phải** bằng chứng cho thấy mô hình nơ-ron có "nhận thức hay tư duy như con người". Nó phản ánh việc mô hình bảo toàn được các quan hệ song song (parallel relational vectors) tồn tại dưới dạng cấu trúc thống kê trong ngôn ngữ.
