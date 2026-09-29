# Calculations
---
# 7. Bài tập tính toán
## 1. Bài 1 — Mô hình ngôn ngữ Unigram (Mục 7 — Bài 1)

### 1.1. Đề bài
Cho tập ngữ liệu (corpus) đồ chơi gồm 3 câu:
- $S_1$: `"the cat eats fish"`
- $S_2$: `"the cat likes fish"`
- $S_3$: `"the dog eats meat"`

Yêu cầu:
1. Xác định tập từ vựng (Vocabulary) $\mathcal{V}$ và kích thước $|\mathcal{V}|$.
2. Tính tổng số lượng token $N$ trong corpus.
3. Tính xác suất cực đại hợp lý (Maximum Likelihood Estimation - MLE) $P(w)$ cho các từ:
   - `the`
   - `cat`
   - `fish`
   - `dog`
4. Kiểm tra tiên đề xác suất chuẩn hóa: $\sum_{w \in \mathcal{V}} P(w) = 1$.

---

### 1.2. Phương pháp & Công thức
Trong mô hình Unigram, các từ được giả định xuất hiện hoàn toàn độc lập với nhau mà không phụ thuộc vào ngữ cảnh xung quanh. Ước lượng xác suất MLE của một từ $w$ được tính bằng tỷ số giữa tần số xuất hiện của từ đó với tổng số token:
$$P_{MLE}(w) = \frac{C(w)}{N}$$
Trong đó:
- $C(w)$ là số lần xuất hiện (count/frequency) của từ $w$ trong toàn bộ corpus.
- $N = \sum_{w \in \mathcal{V}} C(w)$ là tổng số token của corpus.
- $\mathcal{V}$ là tập hợp tất cả các loại từ phân biệt (unique word types).

---

### 1.3. Chi tiết tính toán

Phân rã token theo từng câu:
- $S_1$: `["the", "cat", "eats", "fish"]` (4 tokens)
- $S_2$: `["the", "cat", "likes", "fish"]` (4 tokens)
- $S_3$: `["the", "dog", "eats", "meat"]` (4 tokens)

Bảng thống kê tần số xuất hiện trên toàn bộ 3 câu:

| Từ vựng $w$ | Các câu xuất hiện | Tần số $C(w)$ |
| :--- | :--- | :---: |
| `the` | $S_1 (1), S_2 (1), S_3 (1)$ | **3** |
| `cat` | $S_1 (1), S_2 (1)$ | **2** |
| `eats` | $S_1 (1), S_3 (1)$ | **2** |
| `fish` | $S_1 (1), S_2 (1)$ | **2** |
| `dog` | $S_3 (1)$ | **1** |
| `likes` | $S_2 (1)$ | **1** |
| `meat` | $S_3 (1)$ | **1** |

1. **Tập từ vựng $\mathcal{V}$**:
   $$\mathcal{V} = \{\text{"cat"}, \text{"dog"}, \text{"eats"}, \text{"fish"}, \text{"likes"}, \text{"meat"}, \text{"the"}\}$$
   $$\text{Kích thước từ vựng } |\mathcal{V}| = V = 7$$

2. **Tổng số lượng token $N$**:
   $$N = 3 + 2 + 2 + 2 + 1 + 1 + 1 = 12 \text{ tokens}$$

- **$P(\text{the})$**:
  $$P(\text{the}) = \frac{C(\text{the})}{N} = \frac{3}{12} = \frac{1}{4} = \mathbf{0.25} \quad (25.0\%)$$

- **$P(\text{cat})$**:
  $$P(\text{cat}) = \frac{C(\text{cat})}{N} = \frac{2}{12} = \frac{1}{6} \approx \mathbf{0.1667} \quad (16.67\%)$$

- **$P(\text{fish})$**:
  $$P(\text{fish}) = \frac{C(\text{fish})}{N} = \frac{2}{12} = \frac{1}{6} \approx \mathbf{0.1667} \quad (16.67\%)$$

- **$P(\text{dog})$**:
  $$P(\text{dog}) = \frac{C(\text{dog})}{N} = \frac{1}{12} \approx \mathbf{0.0833} \quad (8.33\%)$$

*(Các từ còn lại: $P(\text{eats}) = \frac{2}{12} = \frac{1}{6}$, $P(\text{likes}) = \frac{1}{12}$, $P(\text{meat}) = \frac{1}{12}$).*


$$\sum_{w \in \mathcal{V}} P(w) = \frac{3 + 2 + 2 + 2 + 1 + 1 + 1}{12} = \frac{12}{12} = \mathbf{1.0}$$
Phân phối xác suất Unigram thỏa mãn đầy đủ các tiên đề Kolmogorov.

---

## 2. Bài 2 — Mô hình ngôn ngữ Bigram (Mục 7 — Bài 2)

### 2.1. Đề bài
Với cùng corpus 3 câu trên:
1. Tính các xác suất có điều kiện:
   - $P(\text{cat} \mid \text{the})$
   - $P(\text{dog} \mid \text{the})$
   - $P(\text{eats} \mid \text{cat})$
   - $P(\text{likes} \mid \text{cat})$
2. Trả lời câu hỏi bản chất:
   > *Tại sao tổng xác suất của các từ đứng sau `"the"` phải bằng 1 nếu từ vựng và ngữ cảnh được xử lý đầy đủ?*

---

### 2.2. Phương pháp & Công thức
Dưới giả định Markov bậc nhất (First-order Markov assumption), xác suất xuất hiện của từ hiện tại $w_t$ chỉ phụ thuộc duy nhất vào một từ đứng liền trước nó $w_{t-1}$. Công thức ước lượng MLE cho xác suất chuyển tiếp Bigram là:
$$P_{MLE}(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$
Trong đó:
- $C(w_{t-1}, w_t)$ là số lần cụm bigram $(w_{t-1}, w_t)$ xuất hiện trong corpus.
- $C(w_{t-1})$ là số lần xuất hiện của từ ngữ cảnh (context) $w_{t-1}$.

---

### 2.3. Trích xuất Bigram và Thống kê tần số
Các cặp bigram trích xuất từ từng câu:
- Từ $S_1$ (`"the cat eats fish"`):
  - `("the", "cat")`, `("cat", "eats")`, `("eats", "fish")`
- Từ $S_2$ (`"the cat likes fish"`):
  - `("the", "cat")`, `("cat", "likes")`, `("likes", "fish")`
- Từ $S_3$ (`"the dog eats meat"`):
  - `("the", "dog")`, `("dog", "eats")`, `("eats", "meat")`

Thống kê theo các từ ngữ cảnh liên quan:
- Ngữ cảnh $w_{t-1} = \text{"the"}$ (tổng $C(\text{"the"}) = 3$):
  - $C(\text{"the", "cat"}) = 2$
  - $C(\text{"the", "dog"}) = 1$
  - Các bigram khác bắt đầu bằng `"the"` có tần số bằng $0$.
- Ngữ cảnh $w_{t-1} = \text{"cat"}$ (tổng $C(\text{"cat"}) = 2$):
  - $C(\text{"cat", "eats"}) = 1$
  - $C(\text{"cat", "likes"}) = 1$
  - Các bigram khác bắt đầu bằng `"cat"` có tần số bằng $0$.
- Ngữ cảnh $w_{t-1} = \text{"dog"}$ (tổng $C(\text{"dog"}) = 1$):
  - $C(\text{"dog", "eats"}) = 1$
- Ngữ cảnh $w_{t-1} = \text{"eats"}$ (tổng $C(\text{"eats"}) = 2$):
  - $C(\text{"eats", "fish"}) = 1$
  - $C(\text{"eats", "meat"}) = 1$
- Ngữ cảnh $w_{t-1} = \text{"likes"}$ (tổng $C(\text{"likes"}) = 1$):
  - $C(\text{"likes", "fish"}) = 1$

---

### 2.4. Tính toán xác suất

1. **Điều kiện theo ngữ cảnh `"the"`:**
   $$P(\text{cat} \mid \text{the}) = \frac{C(\text{"the", "cat"})}{C(\text{"the"})} = \frac{2}{3} \approx \mathbf{0.6667} \quad (66.67\%)$$
   $$P(\text{dog} \mid \text{the}) = \frac{C(\text{"the", "dog"})}{C(\text{"the"})} = \frac{1}{3} \approx \mathbf{0.3333} \quad (33.33\%)$$

2. **Điều kiện theo ngữ cảnh `"cat"`:**
   $$P(\text{eats} \mid \text{cat}) = \frac{C(\text{"cat", "eats"})}{C(\text{"cat"})} = \frac{1}{2} = \mathbf{0.5000} \quad (50.0\%)$$
   $$P(\text{likes} \mid \text{cat}) = \frac{C(\text{"cat", "likes"})}{C(\text{"cat"})} = \frac{1}{2} = \mathbf{0.5000} \quad (50.0\%)$$

---

### 2.5. Giải thích bản chất: Tại sao $\sum_{w} P(w \mid \text{"the"}) = 1$?

**Cơ sở toán học:**
Với một ngữ cảnh lịch sử cố định $h$ (ở đây $h = \text{"the"}$), phân phối xác suất có điều kiện xác định một không gian mẫu rời rạc gồm tất cả các khả năng tiếp nối có thể $w \in \mathcal{V}$.
Lấy tổng xác suất trên toàn bộ từ vựng:
$$\sum_{w \in \mathcal{V}} P(w \mid h) = \sum_{w \in \mathcal{V}} \frac{C(h, w)}{C(h)} = \frac{\sum_{w \in \mathcal{V}} C(h, w)}{C(h)}$$

Trong mọi corpus mà mỗi lần từ $h$ xuất hiện đều được theo sau bởi chính xác một từ $w \in \mathcal{V}$ (hoặc ký hiệu kết thúc câu $\langle/s\rangle$):
$$\sum_{w \in \mathcal{V}} C(h, w) \equiv C(h)$$
Do đó:
$$\sum_{w \in \mathcal{V}} P(w \mid \text{"the"}) = \frac{C(\text{"the"})}{C(\text{"the"})} = \frac{3}{3} = \mathbf{1.0}$$
Cụ thể với từ `"the"`:
$$\sum_{w \in \mathcal{V}} P(w \mid \text{"the"}) = P(\text{cat} \mid \text{the}) + P(\text{dog} \mid \text{the}) + \sum_{w \notin \{\text{cat}, \text{dog}\}} \frac{0}{3} = \frac{2}{3} + \frac{1}{3} + 0 = \mathbf{1.0}$$
Điều này đảm bảo toàn bộ khối lượng xác suất được bảo toàn nguyên vẹn mà không bị thất thoát.

---

## 3. Bài 3 — Xác suất câu & Tính đơn điệu của chuỗi (Mục 7 — Bài 3)

### 3.1. Đề bài
Cho câu mục tiêu:
$$S = \text{"the cat eats fish"}$$
Giả sử sử dụng mô hình Bigram theo công thức được xác định trong đề bài:
$$P(S) = P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{eats} \mid \text{cat}) \cdot P(\text{fish} \mid \text{eats})$$

Yêu cầu:
1. Tính giá trị xác suất chính xác của câu $P(S)$.
2. Trả lời câu hỏi lý thuyết:
   > *Nếu thêm một từ vào câu, xác suất của cả câu có thể tăng không?*

---

### 3.2. Tính toán xác suất câu
Từ các kết quả đã tính ở các bước trước:
- $P(\text{the}) = \frac{3}{12} = \frac{1}{4}$
- $P(\text{cat} \mid \text{the}) = \frac{2}{3}$
- $P(\text{eats} \mid \text{cat}) = \frac{1}{2}$
- $P(\text{fish} \mid \text{eats}) = \frac{C(\text{"eats", "fish"})}{C(\text{"eats"})} = \frac{1}{2}$

Thay vào công thức tích chuỗi xác suất:
$$P(S) = \frac{1}{4} \times \frac{2}{3} \times \frac{1}{2} \times \frac{1}{2} = \frac{2}{48} = \frac{1}{24} \approx \mathbf{0.04167} \quad (4.167\%)$$

---

### 3.3. Giải thích bản chất: Tính đơn điệu không tăng của xác suất chuỗi

**Câu trả lời:**
**KHÔNG THỂ TĂNG.** (Xác suất của cả câu chỉ có thể giảm, hoặc giữ nguyên trong trường hợp suy biến lý thuyết).

**Chứng minh toán học:**
Theo quy tắc nhân chuỗi (Chain rule), khi ta mở rộng chuỗi $W_{1:T} = (w_1, \dots, w_T)$ thêm một từ mới $w_{T+1}$:
$$P(W_{1:T+1}) = P(W_{1:T}) \cdot P(w_{T+1} \mid W_{1:T})$$
Theo tiên đề xác suất Kolmogorov, mọi giá trị xác suất có điều kiện luôn bị chặn trong đoạn $[0, 1]$:
$$0 \le P(w_{T+1} \mid W_{1:T}) \le 1$$
Nhân cả hai vế với $P(W_{1:T}) \ge 0$:
$$0 \le P(W_{1:T+1}) \le P(W_{1:T})$$

- **Trường hợp giảm nghiêm ngặt ($<$):** Trong ngôn ngữ tự nhiên thực tế, xác suất của từ tiếp theo luôn nhỏ hơn 1 ($P(w_{T+1} \mid \cdot) < 1$). Do đó, nhân thêm một phân số trong khoảng $(0, 1)$ làm xác suất của chuỗi **đơn điệu giảm dần**:
  $$P(W_{1:T+1}) < P(W_{1:T})$$
- **Trường hợp giữ nguyên ($=$):** Chỉ xảy ra khi $P(w_{T+1} \mid W_{1:T}) = 1$ (sự kiện chắc chắn $100\%$).
- **Trường hợp triệt tiêu về $0$:** Nếu từ thêm vào chưa từng đi sau $w_T$, $P(w_{T+1} \mid W_{1:T}) = 0$, kéo toàn bộ $P(W_{1:T+1})$ về $0$.

**Ý nghĩa quan trọng đối với NLP:**
Xác suất chuỗi thô $P(S)$ luôn có xu hướng phạt câu dài (Length penalty)—câu càng dài thì $P(S)$ càng tiến sát về 0 do tích lũy nhiều phân số nhỏ. Vì vậy, ta **không thể dùng trực tiếp $P(S)$** để so sánh chất lượng ngữ pháp của các câu có độ dài khác nhau. Đây là động lực trực tiếp giải thích vì sao ta cần chuẩn hóa độ dài thông qua thước đo **Perplexity** ($PP(S) = P(S)^{-1/N}$) hoặc trung bình log-likelihood trên mỗi từ.

---

## 4. Bài 4 — Xếp hạng câu (Sentence Ranking) (Mục 7 — Bài 4)

### 4.1. Đề bài
Cho hai câu ứng viên:
- $S_1 = \text{"the cat eats fish"}$
- $S_2 = \text{"the dog eats fish"}$

Yêu cầu: Dựa trên corpus đã cho, hãy dự đoán và giải thích câu nào có xác suất cao hơn mà không chạy code trước.

---

### 4.2. Phân tích so sánh giải tích

Khai triển xác suất Bigram của cả hai câu:
$$P(S_1) = P(\text{the}) \cdot P(\text{cat} \mid \text{the}) \cdot P(\text{eats} \mid \text{cat}) \cdot P(\text{fish} \mid \text{eats})$$
$$P(S_2) = P(\text{the}) \cdot P(\text{dog} \mid \text{the}) \cdot P(\text{eats} \mid \text{dog}) \cdot P(\text{fish} \mid \text{eats})$$

Nhận xét:
- Thành phần đầu $P(\text{the}) = \frac{1}{4}$ và thành phần cuối $P(\text{fish} \mid \text{eats}) = \frac{1}{2}$ hoàn toàn giống nhau ở cả hai câu.
- Việc so sánh quy về so sánh tích chuyển tiếp ở giữa $\Pi_{mid}$:
  $$\Pi_{mid}(S_1) = P(\text{cat} \mid \text{the}) \cdot P(\text{eats} \mid \text{cat})$$
  $$\Pi_{mid}(S_2) = P(\text{dog} \mid \text{the}) \cdot P(\text{eats} \mid \text{dog})$$

Tính giá trị cụ thể:
- Với $S_1$:
  $$P(\text{cat} \mid \text{the}) = \frac{2}{3}, \quad P(\text{eats} \mid \text{cat}) = \frac{1}{2}$$
  $$\Pi_{mid}(S_1) = \frac{2}{3} \times \frac{1}{2} = \frac{2}{6} = \frac{1}{3}$$
  $$P(S_1) = \frac{1}{4} \times \frac{1}{3} \times \frac{1}{2} = \frac{1}{24} \approx \mathbf{0.04167}$$

- Với $S_2$:
  $$P(\text{dog} \mid \text{the}) = \frac{1}{3}, \quad P(\text{eats} \mid \text{dog}) = \frac{C(\text{"dog", "eats"})}{C(\text{"dog"})} = \frac{1}{1} = 1.0$$
  $$\Pi_{mid}(S_2) = \frac{1}{3} \times 1.0 = \frac{1}{3}$$
  $$P(S_2) = \frac{1}{4} \times \frac{1}{3} \times \frac{1}{2} = \frac{1}{24} \approx \mathbf{0.04167}$$

### 4.3. Kết luận & Phát hiện bản chất
1. **Dưới mô hình Bigram MLE**:
   $$P_{bigram}(S_1) = P_{bigram}(S_2) = \frac{\mathbf{1}}{\mathbf{24}}$$
   Hai câu có xác suất **chính xác bằng nhau**. Mặc dù từ `"cat"` xuất hiện nhiều gấp đôi từ `"dog"` ($C=2$ so với $C=1$), nhưng sau `"dog"` thì bước chuyển tiếp sang `"eats"` là tất định $100\%$ ($P(\text{eats} \mid \text{dog}) = 1.0$), trong khi sau `"cat"` thì xác suất bị chia đôi ($50\%$ `"eats"`, $50\%$ `"likes"`). Tính tất định này đã bù trừ chính xác cho tần suất ban đầu thấp hơn của `"the dog"`.

2. **So sánh với Unigram MLE**:
   $$P_{unigram}(S_1) = P(\text{the}) P(\text{cat}) P(\text{eats}) P(\text{fish}) = \frac{3}{12} \times \frac{2}{12} \times \frac{2}{12} \times \frac{2}{12} = \frac{1}{864} \approx 0.001157$$
   $$P_{unigram}(S_2) = P(\text{the}) P(\text{dog}) P(\text{eats}) P(\text{fish}) = \frac{3}{12} \times \frac{1}{12} \times \frac{2}{12} \times \frac{2}{12} = \frac{1}{1728} \approx 0.000579$$
   Ở mô hình Unigram, $P(S_1) = 2 \times P(S_2)$ vì Unigram hoàn toàn bỏ qua thứ tự từ và sự liên kết ngữ pháp.

---

# 9. Bài tập suy luận trước khi làm mịn
## 5. Bài 5

### 5.1. Đề bài
Cho tập corpus gồm 3 câu:
- $D_1$: `"I like NLP"`
- $D_2$: `"I like AI"`
- $D_3$: `"I study NLP"`

Giả sử cần tính xác suất chuyển tiếp: $P(\text{AI} \mid \text{study})$. Trả lời 4 câu hỏi định hướng:
1. Số lần đếm (count) của bigram `study AI` là bao nhiêu?
2. Xác suất MLE $P_{MLE}(\text{AI} \mid \text{study})$ bằng bao nhiêu?
3. Điều gì sẽ xảy ra khi tính xác suất của một câu chứa bigram này?
4. Điều này có đồng nghĩa với việc mô hình “biết” rằng câu đó chắc chắn không thể xảy ra trong thực tế không?

---

### 5.2. Lời giải & Phân tích chi tiết

#### Câu hỏi 1: Tần số đếm của `study AI`
Trong tập corpus huấn luyện, các bigram xuất hiện gồm:
- $D_1$: `("I", "like")`, `("like", "NLP")`
- $D_2$: `("I", "like")`, `("like", "AI")`
- $D_3$: `("I", "study")`, `("study", "NLP")`

Cụm từ `("study", "AI")` không xuất hiện bất kỳ lần nào trong tập train:
$$C(\text{"study", "AI"}) = \mathbf{0}$$

#### Câu hỏi 2: Xác suất MLE
Vì $C(\text{"study"}) = 1$ và $C(\text{"study", "AI"}) = 0$:
$$P_{MLE}(\text{AI} \mid \text{study}) = \frac{C(\text{"study", "AI"})}{C(\text{"study"})} = \frac{0}{1} = \mathbf{0.0}$$

#### Câu hỏi 3: Hậu quả đối với xác suất câu
Nếu một câu mới $S$ chứa cụm `study AI` (ví dụ: $S = \text{"I study AI"}$):
$$P(S) = P(\text{I}) \cdot P(\text{study} \mid \text{I}) \cdot P(\text{AI} \mid \text{study})$$
Do có một thừa số $P(\text{AI} \mid \text{study}) = 0.0$, toàn bộ tích bị triệt tiêu:
$$P(S) = P(\text{I}) \cdot P(\text{study} \mid \text{I}) \cdot 0.0 = \mathbf{0.0}$$
Khi chuyển sang không gian log-probability:
$$\ln P(S) = \ln P(\text{I}) + \ln P(\text{study} \mid \text{I}) + \ln(0) = -\infty$$
Hậu quả kéo theo là chỉ số Perplexity ($PP$) bùng nổ vô hạn:
$$PP(S) = \exp\left(-\frac{1}{N} \ln P(S)\right) = \exp(+\infty) = +\infty$$
Hệ thống bị phá hủy hoàn toàn khả năng đánh giá, phân loại hoặc xếp hạng câu.

#### Câu hỏi 4: Mô hình có thực sự "biết" câu đó không thể xảy ra?
**HOÀN TOÀN KHÔNG.**
Đây là sự khác biệt cơ bản giữa hai khái niệm:
- **Chưa quan sát thấy trong tập mẫu hữu hạn (Unobserved in finite sample):** Tập ngữ liệu huấn luyện chỉ là một lát cắt rất nhỏ của ngôn ngữ loài người. Cụm từ *"I study AI"* hoàn toàn đúng ngữ pháp, tự nhiên và có ý nghĩa thực tế sâu sắc.
- **Biến cố bất khả thi trong ngôn ngữ (Impossible event):** Các chuỗi vô nghĩa hoặc phản tự nhiên (ví dụ: *"airplane eats cloud"*).

Ước lượng MLE thuần túy mắc phải sai lầm "Thiên nga đen" (Black Swan fallacy)—đánh đồng việc chưa từng nhìn thấy trong tập mẫu nhỏ với việc xác suất thực tế bằng 0 tuyệt đối. Đây chính là lý do bắt buộc phải sử dụng **Kỹ thuật làm mịn (Smoothing)** để tái phân bổ một phần khối lượng xác suất cho các sự kiện chưa quan sát được.

---

# 11. Bài tập tính smoothing
## 6. Bài 6 — Kỹ thuật làm mịn Laplace

### 6.1. Đề bài
Cho các thông số:
- Tần số từ ngữ cảnh: $C(\text{cat}) = 10$
- Tần số bigram: $C(\text{cat eats}) = 0$
- Kích thước từ vựng: $V = |\mathcal{V}| = 5$

Yêu cầu:
1. Tính xác suất làm mịn Laplace $P_{Laplace}(\text{eats} \mid \text{cat})$.
2. Tính lại giá trị trên nếu $C(\text{cat eats}) = 3$.
3. Trả lời: Kỹ thuật smoothing đã làm thay đổi xác suất của những bigram khác như thế nào?

---

### 6.2. Công thức
Làm mịn Laplace (Add-one smoothing) bổ sung một tần số ảo giả định (pseudo-count) bằng $1$ cho mọi từ tiếp nối khả dĩ $w \in \mathcal{V}$:
$$P_{Laplace}(w \mid h) = \frac{C(h, w) + 1}{C(h) + V}$$
Hằng số $V$ ở mẫu số đảm bảo chuẩn hóa tổng xác suất:
$$\sum_{w \in \mathcal{V}} (C(h, w) + 1) = \sum_{w \in \mathcal{V}} C(h, w) + \sum_{w \in \mathcal{V}} 1 = C(h) + V$$

---

### 6.3. Chi tiết tính toán

#### Trường hợp 1: Bigram chưa từng xuất hiện ($C(\text{cat eats}) = 0$)
$$P_{Laplace}(\text{eats} \mid \text{cat}) = \frac{0 + 1}{10 + 5} = \frac{1}{15} \approx \mathbf{0.0667} \quad (6.67\%)$$
*(Từ chỗ MLE gán bằng $0.0$, làm mịn Laplace đã cấp cho nó một sàn xác suất an toàn dương $6.67\%$).*

#### Trường hợp 2: Bigram xuất hiện thường xuyên ($C(\text{cat eats}) = 3$)
$$P_{Laplace}(\text{eats} \mid \text{cat}) = \frac{3 + 1}{10 + 5} = \frac{4}{15} \approx \mathbf{0.2667} \quad (26.67\%)$$
*(So với ước lượng MLE ban đầu là $\frac{3}{10} = 0.3000$ hay $30\%$, Laplace đã chiết khấu giảm xuống còn $26.67\%$).*

---

### 6.4. Phân tích: Smoothing đã tái phân bổ xác suất như thế nào?
1. **Bảo toàn tổng xác suất:**
   Tổng xác suất trên toàn bộ $V=5$ từ vựng vẫn bảo toàn tuyệt đối bằng $1.0$:
   $$\sum_{w \in \mathcal{V}} P_{Laplace}(w \mid \text{cat}) = \frac{\sum C(\text{cat}, w) + 5}{10 + 5} = \frac{10 + 5}{15} = \mathbf{1.0}$$

2. **Cơ chế chiết khấu xác suất (Probability Discounting):**
   - Để cấp một lượng xác suất dương $\frac{1}{15}$ cho các từ chưa xuất hiện, Laplace bắt buộc phải **chiết khấu (lấy bớt)** xác suất từ các bigram đã xuất hiện nhiều lần ($C \ge 1$).
   - Với một bigram có tần số $c > 0$:
     Xác suất MLE ban đầu là $\frac{c}{10}$, xác suất Laplace là $\frac{c + 1}{15}$.
     Khi $c > \frac{10}{5} = 2$, ta luôn có $\frac{c + 1}{15} < \frac{c}{10}$.
     Ví dụ:
     - Với $c = 3$: giảm từ $0.3000 \to 0.2667$ (bị lấy mất $0.0333$).
     - Với $c = 7$: giảm từ $0.7000 \to 0.5333$ (bị lấy mất $0.1667$).
   - Khối lượng xác suất bị lấy đi này hợp thành một **nguồn ngân sách xác suất dự phòng** chia đều cho các trường hợp chưa từng xuất hiện.

3. **Hạn chế lớn của Laplace trong thực tế:**
   Khi từ vựng thực tế rất lớn ($V \approx 50.000$ đến $100.000$), mẫu số $C(h) + V$ bị chi phối áp đảo bởi $V$. Điều này khiến các từ phổ biến bị phạt quá nặng và một lượng xác suất khổng lồ bị phân bổ lãng phí cho hàng chục ngàn từ không bao giờ đi sau ngữ cảnh $h$. Đây là lý do thúc đẩy các phương pháp làm mịn cao cấp hơn như **Absolute Discounting** hay **Kneser-Ney Smoothing**.

---

# 18. Bài tập tính Perplexity
## 7. Bài 7 — Tính Perplexity & Độ nhạy cảm của mô hình 

### 7.1. Đề bài
Cho một chuỗi kiểm thử $W = (w_1, w_2, w_3)$ có độ dài $N = 3$ token.
Các xác suất thành phần:
- $P(w_1) = 0.5$
- $P(w_2 \mid w_1) = 0.25$
- $P(w_3 \mid w_2) = 0.5$

Yêu cầu:
1. Tính xác suất của chuỗi $P(W)$ và Perplexity $PP(W)$.
2. Tính lại cả hai chỉ số nếu $P(w_2 \mid w_1)$ giảm xuống còn $0.1$.
3. Trả lời: Vì sao chỉ một xác suất nhỏ cũng có thể làm Perplexity thay đổi đáng kể?

---

### 7.2. Công thức
$$\text{Xác suất chuỗi: } P(W) = P(w_1) \cdot P(w_2 \mid w_1) \cdot P(w_3 \mid w_2)$$
$$\text{Perplexity: } PP(W) = P(W)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(W)}} = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid \text{context}_i)\right)$$

---

### 7.3. Chi tiết tính toán

#### Thiết lập 1: $P(w_2 \mid w_1) = 0.25$
1. **Xác suất chuỗi $P(W)$**:
   $$P(W) = 0.5 \times 0.25 \times 0.5 = 0.0625 = \frac{1}{16}$$
2. **Perplexity $PP(W)$**:
   $$PP(W) = \left(\frac{1}{16}\right)^{-\frac{1}{3}} = 16^{\frac{1}{3}} = \sqrt[3]{16} = 2\sqrt[3]{2} \approx \mathbf{2.5198}$$

#### Thiết lập 2: $P(w_2 \mid w_1) = 0.10$
1. **Xác suất chuỗi $P(W)$**:
   $$P(W) = 0.5 \times 0.10 \times 0.5 = 0.0250 = \frac{1}{40}$$
2. **Perplexity $PP(W)$**:
   $$PP(W) = \left(\frac{1}{40}\right)^{-\frac{1}{3}} = 40^{\frac{1}{3}} = \sqrt[3]{40} \approx \mathbf{3.4200}$$

---

### 7.4. Phân tích: Vì sao Perplexity cực kỳ nhạy cảm với các xác suất nhỏ?

1. **Hiệu ứng nghịch đảo phi tuyến (Geometric Inversion):**
   Perplexity là nghịch đảo của trung bình nhân các xác suất:
   $$PP(W) = \frac{1}{\sqrt[N]{\prod_{i=1}^N P(w_i \mid h_i)}}$$
   Vì xác suất nằm ở mẫu số, khi bất kỳ xác suất thành phần nào suy giảm tiến dần về 0, mẫu số co cụm lại rất nhanh, khiến phép nghịch đảo bùng nổ theo cấp số nhân.

2. **Độ dốc cực lớn của hàm Log-loss gần điểm 0:**
   Trong không gian logarit, tổn thất cross-entropy của token thứ $i$ là:
   $$\mathcal{L}_i = -\log_2 P(w_i \mid h_i)$$
   Đạo hàm của hàm phạt là $\frac{d\mathcal{L}}{dp} = -\frac{1}{p \ln 2}$. Khi $p \to 0$, độ dốc này tiến tới vô cực:
   - Với $p = 0.25$: $-\log_2(0.25) = 2.00$ bits.
   - Với $p = 0.10$: $-\log_2(0.10) \approx 3.32$ bits (tăng tới $+66\%$ điểm phạt).
   - Nếu $p = 0.01$: $-\log_2(0.01) \approx 6.64$ bits (tăng tới $+232\%$ điểm phạt).

3. **Ý nghĩa trực quan (Số nhánh phân vân hiệu dụng):**
   Perplexity đo số lượng từ ứng viên mà mô hình đang "bối rối" lựa chọn tại mỗi bước dự đoán. Khi xác suất một từ giảm từ $0.25 \to 0.10$, độ phân vân của mô hình tăng từ $\approx 2.52$ từ lên $\approx 3.42$ từ. Sự nhạy cảm này giúp Perplexity trở thành thước đo cực kỳ nghiêm khắc để trừng phạt những mô hình quá tự tin hoặc bất ngờ trước dữ liệu thực tế.
