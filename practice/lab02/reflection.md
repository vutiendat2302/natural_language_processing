# BÁO CÁO PHẢN TƯ & CẦU NỐI SANG NEURAL LANGUAGE MODELS
---

## Mục 24: Trả lời chi tiết 7 câu hỏi Reflection

### Câu 24.1: Nếu tăng $n$, mô hình nhận thêm thông tin gì?
- **Mở rộng cửa sổ điều kiện hóa cục bộ**: Khi tăng bậc $n$ từ $1 \to 2 \to 3 \dots$, mô hình chuyển từ việc coi các từ độc lập thống kê $P(w)$ sang mô hình hóa xác suất có điều kiện dựa trên $(n-1)$ từ liền trước:
  $$P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$$
- **Các dạng thông tin mô hình tiếp thu thêm:**
  1. **Ràng buộc ngữ pháp và cú pháp ngắn hạn (Short-range Syntax):** Nắm bắt sự hòa hợp chủ vị (ví dụ: `"he eats"` thay vì `"he eat"`), cấu trúc mạo từ (`"a book"` vs `"an apple"`), và các cụm động từ - giới từ (`"depend on"`, `"interested in"`).
  2. **Cụm từ cố định và thuật ngữ (Collocations & Idioms):** Nhận diện các kết hợp từ mang ý nghĩa đặc thù không thể tách rời như `"machine learning"`, `"natural language processing"`, `"New York"`.
  3. **Giảm độ bất định thông tin (Entropy Reduction):** Dưới góc độ lý thuyết thông tin của Shannon, bổ sung thêm biến điều kiện luôn làm giảm hoặc giữ nguyên entropy:
     $$H(W_t \mid W_{t-2}, W_{t-1}) \le H(W_t \mid W_{t-1}) \le H(W_t)$$
     Phân phối xác suất của từ tiếp theo trở nên tập trung hơn, giúp mô hình dự đoán tự tin hơn.

---

### Câu 24.2: Tại sao tăng $n$ lại làm sparsity tăng?
- **Bùng nổ không gian trạng thái (Curse of Dimensionality):**
  Với từ vựng $|\mathcal{V}|$, số cấu hình n-gram lý thuyết tăng theo hàm mũ $|\mathcal{V}|^n$:
  - Unigram: $|\mathcal{V}| \approx 4.7 \times 10^4$
  - Bigram: $|\mathcal{V}|^2 \approx 2.3 \times 10^9$
  - Trigram: $|\mathcal{V}|^3 \approx 1.1 \times 10^{14}$
- **Ngữ liệu thực tế luôn hữu hạn:**
  Tổng số token trong corpus chỉ khoảng $3 \times 10^6$, chiếm một tỷ lệ vô cùng nhỏ bé so với không gian tổ hợp của Trigram.
- **Hiện tượng đuôi dài (Heavy-tailed Zipfian Distribution & Hapax Legomena):**
  Khi $n$ càng lớn, cụm từ càng mang tính cá biệt cao. Thống kê thực tế trên 10.000 documents cho thấy:
  - Tỷ lệ cụm từ chỉ xuất hiện đúng 1 lần (*hapax legomena*) tăng vọt từ **$44.87\%$** (Unigram) lên **$73.12\%$** (Bigram) và đạt tới **$88.14\%$** (Trigram).
  - Phần lớn các ô trong ma trận chuyển tiếp đều bằng $0$, dẫn tới hiện tượng dữ liệu cực thưa (data sparsity).

---

### Câu 24.3: Tại sao smoothing cần thiết?
- **Triệt tiêu thảm họa xác suất bằng 0 (Zero-Frequency Problem):**
  Theo ước lượng MLE thuần túy, mọi n-gram chưa xuất hiện trong tập huấn luyện đều bị gán $P(w \mid context) = 0.0$. Do xác suất câu là tích chuỗi:
  $$P(S) = \prod_{t=1}^T P(w_t \mid context_t)$$
  Chỉ cần một n-gram có xác suất 0, toàn bộ $P(S) = 0.0 \implies \ln P(S) = -\infty \implies \text{PPL} = +\infty$, làm hệ thống hoàn toàn sụp đổ.
- **Phân biệt giữa "Chưa thấy trong mẫu" và "Bất khả thi trong ngôn ngữ":**
  Ngữ liệu dù lớn đến đâu cũng chỉ là mẫu hữu hạn của ngôn ngữ tự nhiên vô hạn. Câu *"the cat likes meat"* hoàn toàn tự nhiên dù corpus chỉ có *"the cat eats meat"*. Smoothing giúp trích một phần xác suất từ các từ quen thuộc để phân bổ lại cho các trường hợp chưa thấy, tạo sàn xác suất dương an toàn cho hệ thống.

---

### Câu 24.4: Perplexity đo điều gì?
- **Định nghĩa toán học:** Perplexity là nghịch đảo trung bình nhân của xác suất có điều kiện, tương đương hàm mũ của cross-entropy:
  $$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid context_i)\right) = \left(\prod_{i=1}^N P(w_i \mid context_i)\right)^{-\frac{1}{N}}$$
- **Ý nghĩa trực giác — Số nhánh phân vân hiệu dụng (Effective Branching Factor):**
  Perplexity đo mức độ "bối rối", "ngạc nhiên" của mô hình khi đọc chuỗi văn bản:
  - Một mô hình có $\text{PPL} = K$ nghĩa là tại mỗi bước dự đoán, mô hình đang phân vân tương đương với việc **chọn ngẫu nhiên đều giữa $K$ từ có xác suất bằng nhau**.
  - PPL càng thấp nghĩa là mô hình gán xác suất càng cao cho câu văn thực tế, mức độ ngạc nhiên càng nhỏ.

---

### Câu 24.5: Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.
**KHÔNG LUÔN LUÔN.** Lý do:
1. **Perplexity đo khả năng đánh giá xác suất (Scoring), không đo giải thuật sinh văn bản (Decoding Strategy):**
   Một mô hình có perplexity thấp vẫn có thể bị kẹt trong vòng lặp lặp từ nhàm chán (ví dụ: *"the company said that the company said that..."*) khi sử dụng Greedy Search.
2. **Thiên lệch từ chức năng (Function Word Bias):**
   Các hư từ phổ biến (*the, of, in, and*) chiếm tỷ trọng rất lớn trong số lượng token. Mô hình đoán rất chuẩn các hư từ này sẽ có PPL thấp, nhưng nếu đoán sai danh từ/động từ chính chứa nội dung ngữ nghĩa thì con người vẫn thấy câu văn hoàn toàn vô nghĩa.
3. **Phụ thuộc vào từ vựng $|\mathcal{V}|$:**
   Một mô hình có từ vựng rất nhỏ (lọc bớt từ hiếm thành `<unk>`) sẽ dễ đoán hơn và có PPL thấp giả tạo so với một mô hình có từ vựng phong phú bao gồm nhiều từ chuyên ngành.

---

### Câu 24.6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
1. **Hoàn toàn thiếu vắng tính tương đồng ngữ nghĩa (Semantic Blindness):**
   N-gram coi các từ như các ký hiệu rời rạc (one-hot). Đối với Bigram MLE, *"cat"* và *"feline"* hay *"dog"* là các thực thể hoàn toàn xa lạ; việc quan sát thấy *"the cat purrs"* không giúp ích gì cho việc tính xác suất của *"the feline purrs"*.
2. **Bất lực trước sự phụ thuộc xa (Long-range Dependencies):**
   Do giả định Markov, Trigram chỉ nhìn lại 2 từ trước đó. Nó không thể duy trì ngữ cảnh qua câu dài, không nhớ được chủ ngữ ở đầu câu để hòa hợp vị ngữ ở cuối câu, và không duy trì được mạch chủ đề của đoạn văn.
3. **Không mô hình hóa được cấu trúc phân cấp cú pháp (Hierarchical Structure):**
   Ngôn ngữ tự nhiên có cấu trúc cây cú pháp phân cấp, trong khi N-gram chỉ nhìn chuỗi tuyến tính cục bộ bề mặt.

---

### Câu 24.7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
- **HOÀN TOÀN KHÔNG.**
- **Lý do:** Theo định nghĩa của giả thuyết Markov bậc 2:
  $$P(w_{100} \mid w_1, w_2, \dots, w_{99}) \equiv P(w_{100} \mid w_{98}, w_{99})$$
- Mô hình Trigram hoàn toàn vứt bỏ toàn bộ 97 từ đầu tiên ($w_1, \dots, w_{97}$).
- **Cầu nối sang Neural NLP:** Đây chính là nhược điểm chí tử thúc đẩy sự ra đời của **Recurrent Neural Networks (RNN/LSTM)** với trạng thái ẩn liên tục, cơ chế **Attention**, và kiến trúc **Transformer (Self-Attention)** cho phép mô hình nhìn lại toàn bộ lịch sử ngữ cảnh dài hàng ngàn từ.

---

## Mục 26: Chuẩn bị vấn đáp nhanh (Individual Learning Check)

### Câu 26.1: Vì sao Bigram có Zero Probability?
- **Trả lời:** Bigram tính xác suất theo $P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$. Khi gặp một cặp hai từ đi liền nhau chưa từng xuất hiện trong tập huấn luyện ($C(w_{t-1}, w_t) = 0$), tử số bằng 0 khiến xác suất bằng 0. Khi nhân chuỗi trong câu, chỉ một bigram có xác suất 0 sẽ kéo toàn bộ xác suất câu về 0.

### Câu 26.2: Tại sao phải dùng Log Probability?
- **Trả lời:** Xác suất của một câu là tích của hàng chục xác suất thành phần nhỏ ($p \in (0, 1)$). Khi nhân trực tiếp trên máy tính, tích số sẽ nhỏ hơn giới hạn biểu diễn nhỏ nhất của số thực dấu phẩy động 64-bit ($< 10^{-324}$), dẫn tới **tràn số dưới (numerical underflow)** và bị ép về đúng `0.0`. Chuyển sang không gian log biến phép nhân thành phép cộng: $\ln P(S) = \sum \ln P(w_t \mid context_t)$, giữ giá trị luôn là số âm hữu hạn ổn định.

### Câu 26.3: Perplexity thấp nghĩa là gì?
- **Trả lời:** Perplexity tỉ lệ nghịch với xác suất gán cho chuỗi văn bản. Perplexity thấp nghĩa là mô hình gán xác suất cao cho câu văn tự nhiên, độ bất ngờ (surprisal) thấp, tương đương với việc tại mỗi bước phân vân giữa một số lượng ít các từ ứng viên hợp lý.

### Câu 26.4: Tại sao Trigram không nhất thiết tốt hơn Bigram trên dữ liệu mới?
- **Trả lời:** Do hiện tượng dữ liệu cực thưa (data sparsity) và đánh đổi bias-variance. Trigram có không gian trạng thái $|\mathcal{V}|^3$ quá lớn; với corpus hữu hạn, rất nhiều bộ 3 từ chưa từng xuất hiện trong tập train. Khi kiểm thử trên dữ liệu mới, Trigram bị phạt rất nặng bởi các n-gram chưa thấy hoặc bị làm mịn quá mức (Laplace phạt nặng các context hiếm), khiến Perplexity của Trigram trên tập kiểm thử thường cao hơn nhiều so với Bigram.

### Câu 26.5: Nếu 'cat eats' chưa xuất hiện trong training corpus thì model xử lý thế nào?
- **Trả lời:** 
  - Nếu dùng **MLE**: $C(\text{"cat eats"}) = 0 \implies P(\text{"eats"} \mid \text{"cat"}) = 0.0$.
  - Nếu dùng **Laplace Smoothing**: $P(\text{"eats"} \mid \text{"cat"}) = \frac{0 + 1}{C(\text{"cat"}) + |\mathcal{V}|} > 0$.
  - Nếu dùng **Fallback/Backoff**: Khi không tìm thấy bigram `("cat", "eats")`, mô hình lùi xuống dùng Unigram $P(\text{"eats"})$.

---

## Mục 30: Mối liên hệ tiến hóa từ LAB 01 sang LAB 02 và định hướng LAB 03

```
LAB 01 (TF-IDF)                 LAB 02 (N-gram LMs)                 LAB 03 (Word Embeddings)
Text -> Count -> Vector         Text -> Count -> Probability       Count -> Dense Distributed Vectors
Túi từ (Bag-of-Words)           Mô hình hóa chuỗi rời rạc          Học biểu diễn ngữ nghĩa liên tục
Không xét trật tự trước sau     Có trật tự cục bộ (Markov)         Hiểu tính tương đồng ngữ nghĩa
Cực thưa (Sparse Vectors)       Cực thưa (Data Sparsity)           Không gian vector dày đặc (Dense)
```
- **LAB 01**: Dùng tần số để tạo vector biểu diễn tài liệu phục vụ tìm kiếm thông tin.
- **LAB 02**: Dùng tần số để mô hình hóa xác suất có điều kiện của chuỗi ngôn ngữ.
- **LAB 03**: Chuyển từ biểu diễn đếm rời rạc (Count-based discrete representation) sang biểu diễn phân tán liên tục (Dense distributed representation - Word2Vec / GloVe / FastText), khắc phục triệt để điểm yếu "mù ngữ nghĩa" của N-gram.
