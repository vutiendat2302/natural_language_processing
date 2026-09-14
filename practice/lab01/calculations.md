# Part B — Calculation Exercises

Họ và tên sinh viên: Sinh viên MAT3561  
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
Bài tập: LAB 01 — From Text Processing to Search  

---

## Exercise 1 — Count Vector

### 1. Đề bài
Cho corpus gồm 3 văn bản:
- $D_1$ = `"cat eats fish"`
- $D_2$ = `"dog eats fish"`
- $D_3$ = `"cat likes fish"`

Vocabulary được sắp xếp theo thứ tự:
$$\text{Vocabulary} = [\text{"cat"}, \text{"dog"}, \text{"eats"}, \text{"fish"}, \text{"likes"}]$$
Kích thước từ vựng: $V = 5$.

Yêu cầu: Tính count vector của $D_1, D_2, D_3$.

### 2. Phương pháp tính
Count vector biểu diễn số lần xuất hiện của mỗi từ trong vocabulary trong văn bản:
$$\mathbf{c}(D) = [c(t_0, D), c(t_1, D), \dots, c(t_{V-1}, D)]^\top$$
Thứ tự index:
- Index 0: `cat`
- Index 1: `dog`
- Index 2: `eats`
- Index 3: `fish`
- Index 4: `likes`

### 3. Chi tiết tính toán
- **$D_1$ = "cat eats fish":**
  - `cat`: 1 lần $\to$ vị trí 0 = 1
  - `dog`: 0 lần $\to$ vị trí 1 = 0
  - `eats`: 1 lần $\to$ vị trí 2 = 1
  - `fish`: 1 lần $\to$ vị trí 3 = 1
  - `likes`: 0 lần $\to$ vị trí 4 = 0
  $$\mathbf{c}(D_1) = [1, 0, 1, 1, 0]$$

- **$D_2$ = "dog eats fish":**
  - `cat`: 0 lần $\to$ vị trí 0 = 0
  - `dog`: 1 lần $\to$ vị trí 1 = 1
  - `eats`: 1 lần $\to$ vị trí 2 = 1
  - `fish`: 1 lần $\to$ vị trí 3 = 1
  - `likes`: 0 lần $\to$ vị trí 4 = 0
  $$\mathbf{c}(D_2) = [0, 1, 1, 1, 0]$$

- **$D_3$ = "cat likes fish":**
  - `cat`: 1 lần $\to$ vị trí 0 = 1
  - `dog`: 0 lần $\to$ vị trí 1 = 0
  - `eats`: 0 lần $\to$ vị trí 2 = 0
  - `fish`: 1 lần $\to$ vị trí 3 = 1
  - `likes`: 1 lần $\to$ vị trí 4 = 1
  $$\mathbf{c}(D_3) = [1, 0, 0, 1, 1]$$

### 4. Bảng tổng hợp Count Matrix

| Document | cat (0) | dog (1) | eats (2) | fish (3) | likes (4) | Count Vector |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D_1$** | 1 | 0 | 1 | 1 | 0 | $[1, 0, 1, 1, 0]$ |
| **$D_2$** | 0 | 1 | 1 | 1 | 0 | $[0, 1, 1, 1, 0]$ |
| **$D_3$** | 1 | 0 | 0 | 1 | 1 | $[1, 0, 0, 1, 1]$ |

---

## Exercise 2 — Term Frequency (TF)

### 1. Đề bài
Với văn bản $D_1 =$ `"cat eats fish"`, hãy tính:
- $tf(\text{cat}, D_1)$
- $tf(\text{eats}, D_1)$
- $tf(\text{fish}, D_1)$
Kiểm tra điều kiện: $\sum_t tf(t, D_1) = 1$.

### 2. Công thức
$$tf(t, d) = \frac{c(t, d)}{\sum_{t'} c(t', d)}$$
Trong đó:
- $c(t, d)$ là số lần xuất hiện của từ $t$ trong document $d$.
- Mẫu số $\sum_{t'} c(t', d)$ là tổng số từ (tổng độ dài token) trong document $d$.

### 3. Chi tiết tính toán
Với $D_1$: tổng số từ là $1 + 1 + 1 = 3$.
- $tf(\text{cat}, D_1) = \frac{c(\text{cat}, D_1)}{\text{total words in } D_1} = \frac{1}{3} \approx 0.3333$
- $tf(\text{eats}, D_1) = \frac{c(\text{eats}, D_1)}{\text{total words in } D_1} = \frac{1}{3} \approx 0.3333$
- $tf(\text{fish}, D_1) = \frac{c(\text{fish}, D_1)}{\text{total words in } D_1} = \frac{1}{3} \approx 0.3333$

Đối với các từ không xuất hiện trong $D_1$ (`dog`, `likes`):
- $tf(\text{dog}, D_1) = \frac{0}{3} = 0$
- $tf(\text{likes}, D_1) = \frac{0}{3} = 0$

### 4. Kiểm tra
$$\sum_{t \in \text{Vocab}} tf(t, D_1) = \frac{1}{3} + 0 + \frac{1}{3} + \frac{1}{3} + 0 = \frac{3}{3} = 1.0 \quad (\text{Thỏa mãn})$$

---

## Exercise 3 — Inverse Document Frequency (IDF)

### 1. Đề bài
Corpus có tổng số tài liệu $N = 3$, và tần số tài liệu ($df$):
- $df(\text{cat}) = 2$ ($D_1, D_3$)
- $df(\text{dog}) = 1$ ($D_2$)
- $df(\text{eats}) = 2$ ($D_1, D_2$)
- $df(\text{fish}) = 3$ ($D_1, D_2, D_3$)
- $df(\text{likes}) = 1$ ($D_3$)

Sử dụng công thức chuẩn trong slide:
$$idf(t) = \log\left(\frac{N}{df(t)}\right)$$
(Áp dụng logarit tự nhiên $\ln$ - chuẩn thông dụng trong NLP).

### 2. Chi tiết tính toán
- $idf(\text{cat}) = \ln\left(\frac{3}{2}\right) = \ln(1.5) \approx 0.4055$
- $idf(\text{dog}) = \ln\left(\frac{3}{1}\right) = \ln(3) \approx 1.0986$
- $idf(\text{eats}) = \ln\left(\frac{3}{2}\right) = \ln(1.5) \approx 0.4055$
- $idf(\text{fish}) = \ln\left(\frac{3}{3}\right) = \ln(1) = 0.0000$
- $idf(\text{likes}) = \ln\left(\frac{3}{1}\right) = \ln(3) \approx 1.0986$

*(Ghi chú: Nếu dùng $\log_{10}$: $idf(\text{cat}) = \log_{10}(1.5) \approx 0.1761$; $idf(\text{dog}) \approx 0.4771$; $idf(\text{fish}) = 0$; $idf(\text{likes}) \approx 0.4771$. Thứ tự tương đối giữa các từ là không đổi).*

### 3. Câu hỏi và trả lời
**Câu hỏi:** *Term nào có IDF thấp nhất? Vì sao?*  
**Trả lời:**  
Term có IDF thấp nhất là **`fish`** ($idf = 0$).  
**Lý do:** Từ `fish` xuất hiện trong tất cả các văn bản trong corpus ($df(\text{fish}) = 3 = N$). Khi một từ xuất hiện ở mọi văn bản, tỷ số $\frac{N}{df(t)} = \frac{3}{3} = 1$, do đó $idf(t) = \ln(1) = 0$. Về mặt ý nghĩa thông tin, từ này mang tính phổ quát giống như stopword, không giúp phân biệt hay xếp hạng độ đặc trưng giữa các tài liệu.

---

## Exercise 4 — TF-IDF

### 1. Đề bài
Tính TF-IDF của $D_1 =$ `"cat eats fish"` cho cả 3 terms: `cat`, `eats`, `fish`.

### 2. Công thức
$$tfidf(t, d) = tf(t, d) \times idf(t)$$

### 3. Chi tiết tính toán
Dựa trên kết quả từ Exercise 2 và Exercise 3 (dùng log tự nhiên $\ln$):
- **Term `cat`**:
  $$tfidf(\text{cat}, D_1) = \frac{1}{3} \times \ln(1.5) \approx 0.3333 \times 0.4055 \approx 0.1352$$
- **Term `eats`**:
  $$tfidf(\text{eats}, D_1) = \frac{1}{3} \times \ln(1.5) \approx 0.3333 \times 0.4055 \approx 0.1352$$
- **Term `fish`**:
  $$tfidf(\text{fish}, D_1) = \frac{1}{3} \times 0 = 0.0000$$

Vector TF-IDF đầy đủ cho $D_1$ (theo thứ tự `[cat, dog, eats, fish, likes]`):
$$\mathbf{tfidf}(D_1) = [0.1352, 0, 0.1352, 0, 0]$$

### 4. Câu hỏi và trả lời
**Câu hỏi:** *Tại sao `fish` xuất hiện trong mọi document nhưng TF-IDF của nó bằng 0 theo công thức trên?*  
**Trả lời:**  
Theo công thức $tfidf(t, d) = tf(t, d) \times idf(t)$, nếu $idf(t) = 0$ thì tích số lập tức bằng 0 bất kể $tf(t, d)$ có lớn thế nào. Vì `fish` xuất hiện ở mọi document ($df = N \implies idf = \log(1) = 0$).  
Ý nghĩa lý thuyết: TF-IDF đo lường mức độ quan trọng và **tính đặc trưng riêng biệt** của một từ đối với một văn bản so với toàn bộ kho ngữ liệu. Nếu một từ có mặt ở tất cả các tài liệu, việc một tài liệu chứa từ này hoàn toàn không cung cấp thông tin phân biệt nào giữa tài liệu này với tài liệu khác.

---

## Exercise 5 — Cosine Similarity

### 1. Đề bài
Cho hai vector:
$$x = [1, 1, 1]^\top$$
$$y = [1, 1, 0]^\top$$

Yêu cầu:
1. Tính $\cos(x, y)$.
2. Giải thích bằng trực giác: Hai documents có hai term giống nhau trên ba term tổng cộng. Tại sao cosine similarity không bằng $2/3$?

### 2. Chi tiết tính toán
Công thức Cosine Similarity:
$$\cos(x, y) = \frac{x^\top y}{\|x\|_2 \|y\|_2}$$

- **Tích vô hướng (Dot product):**
  $$x^\top y = 1 \times 1 + 1 \times 1 + 1 \times 0 = 1 + 1 + 0 = 2$$
- **Chuẩn $L_2$ (Euclidean norm) của $x$:**
  $$\|x\|_2 = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3} \approx 1.7320$$
- **Chuẩn $L_2$ của $y$:**
  $$\|y\|_2 = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2} \approx 1.4142$$
- **Cosine similarity:**
  $$\cos(x, y) = \frac{2}{\sqrt{3} \times \sqrt{2}} = \frac{2}{\sqrt{6}} = \frac{\sqrt{6}}{3} \approx 0.8165$$

### 3. Giải thích trực giác
Giá trị $\frac{2}{3} \approx 0.6667$ là độ đo tương đồng tập hợp (Jaccard similarity: $\frac{|X \cap Y|}{|X \cup Y|} = \frac{2}{3}$) hoặc tỷ lệ số chiều trùng nhau trên tổng độ dài vector theo chuẩn $L_1$.

Tuy nhiên, **Cosine Similarity đo góc hình học $\theta$ giữa hai vector trong không gian vector đa chiều**:
1. Chuẩn hóa chiều dài được thực hiện bằng chuẩn $L_2$ (căn bậc hai của tổng bình phương). Do hiệu ứng căn bậc hai, sự chênh lệch ở một chiều (chiều thứ 3 bị lệch 1 đơn vị) không làm góc bị kéo dãn tuyến tính theo tỷ lệ phần trăm $1/3$.
2. Góc giữa $x$ và $y$ chỉ là $\theta = \arccos(0.8165) \approx 35.26^\circ$. Góc này khá nhỏ, phản ánh việc hai vector cùng hướng về phía góc phần tám thứ nhất với 2 trên 3 thành phần trùng khít nhau hoàn toàn. Vì vậy $\cos(\theta) \approx 0.8165 > 0.6667$.

---

## Exercise 6 — Prediction

### 1. Đề bài
Cho corpus 3 câu:
- $D_1$ = `"medical image classification"`
- $D_2$ = `"medical image analysis"`
- $D_3$ = `"natural language processing"`

Query:
- $Q$ = `"medical image classification"`

### 2. Các câu hỏi dự đoán (Không dùng code)
1. **Document nào có similarity cao nhất?**  
   - **Trả lời:** $D_1$.  
   - **Giải thích:** $D_1$ trùng khớp hoàn toàn 100% các từ với $Q$ (`medical`, `image`, `classification`). Vector của $D_1$ và $Q$ hoàn toàn cùng hướng, cosine similarity sẽ đạt giá trị tối đa là $1.0$.

2. **Document nào có similarity thấp nhất?**  
   - **Trả lời:** $D_3$.  
   - **Giải thích:** $D_3$ không chia sẻ bất kỳ từ nào với $Q$ (tập từ vựng của $D_3$ là `{natural, language, processing}`, giao với `{medical, image, classification}` là rỗng $\emptyset$). Do đó tích vô hướng bằng 0 $\implies$ Cosine similarity = $0.0$.

3. **Term nào có thể có giá trị IDF thấp?**  
   - **Trả lời:** Hai từ `"medical"` và `"image"`.  
   - **Giải thích:** Trong tập dữ liệu này, cả 2 từ `"medical"` và `"image"` đều xuất hiện ở cả $D_1$ và $D_2$ ($df = 2$). Trong khi đó, các từ khác (`classification`, `analysis`, `natural`, `language`, `processing`) chỉ xuất hiện đúng 1 lần ($df = 1$). Theo công thức $idf = \log(N / df)$, $df$ càng lớn thì IDF càng nhỏ, vì vậy `"medical"` và `"image"` có giá trị IDF thấp nhất trong corpus.

4. **Nếu bỏ IDF và chỉ sử dụng count vector thì ranking có thay đổi không?**  
   - **Trả lời:** Thứ tự ranking **không thay đổi**.  
   - **Giải thích:**  
     - Với count vector:  
       - $D_1$ trùng 3/3 từ với $Q \implies$ similarity = 1.0 (Rank 1).  
       - $D_2$ trùng 2/3 từ (`medical`, `image`) với $Q \implies$ similarity = $\frac{2}{\sqrt{3}\sqrt{2}} \approx 0.8165$ (Rank 2).  
       - $D_3$ không trùng từ nào $\implies$ similarity = 0.0 (Rank 3).  
     - Dù có nhân trọng số IDF hay không thì $D_1$ vẫn là match tuyệt đối (Rank 1), $D_2$ là partial match (Rank 2), và $D_3$ là no match (Rank 3).
