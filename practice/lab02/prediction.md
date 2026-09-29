# 12. Prediction trước experiment
---

## 1. Dự đoán 1 — Tính Bất Biến Của Kích Thước Từ Vựng Qua Các Bậc N-Gram

### Câu hỏi nghiên cứu
> *Khi chuyển từ Unigram $\to$ Bigram $\to$ Trigram, kích thước từ vựng (Vocabulary size) có tăng lên không?*

### Lời giải & Dự đoán chính thức
- **Prediction (Dự đoán)**: **KHÔNG TĂNG.** Kích thước từ vựng giữ nguyên không đổi ($|\mathcal{V}_{uni}| = |\mathcal{V}_{bi}| = |\mathcal{V}_{tri}| = |\mathcal{V}|$).
- **Reason (Cơ sở lý thuyết)**: 
  - Trong xử lý ngôn ngữ tự nhiên thống kê, cần phân biệt rõ hai khái niệm:
    1. **Từ vựng (Vocabulary $\mathcal{V}$)**: Là tập hợp tất cả các đơn vị từ đơn phân biệt (unique word types / unigrams) được xác định từ ngữ liệu huấn luyện:
       $$\mathcal{V} = \{w \mid w \text{ xuất hiện trong } \mathcal{D}_{\text{train}}\}$$
    2. **Số lượng vị trí token n-gram (n-gram tokens / positions)**: Trong một chuỗi văn bản gồm $T$ từ, số vị trí xuất hiện n-gram là $T - n + 1$.
  - Khi chuyển từ Unigram sang Bigram hay Trigram, ta chỉ mở rộng độ dài cửa sổ lịch sử điều kiện ($n-1$ từ) dùng để tính xác suất cho từ tiếp theo $P(w_t \mid w_{t-n+1}^{t-1})$. Không gian các từ ứng viên đầu ra $w_t \in \mathcal{V}$ không thay đổi. Do đó, kích thước từ vựng $|\mathcal{V}|$ hoàn toàn độc lập với bậc $n$.
- **Confidence (Độ tin cậy)**: **High** 

---

## 2. Dự đoán 2 — Sự Khác Biệt Giữa Vị Trí Token Và Số Lượng N-Gram Phân Biệt

### Câu hỏi nghiên cứu
> *Số lượng các n-gram phân biệt (unique n-grams) sẽ biến đổi như thế nào khi bậc $n$ tăng lên?*

### Lời giải & Dự đoán chính thức
- **Prediction (Dự đoán)**: **Số lượng n-gram phân biệt (unique n-grams) quan sát được sẽ tăng mạnh khi $n$ tăng từ 1 lên 2 và 3:**
  $$\#\text{unique unigrams} \ll \#\text{unique bigrams} < \#\text{unique trigrams}$$
  *(trong khi tổng số vị trí n-gram token $T - n + 1$ gần như không đổi).*
- **Reason (Cơ sở lý thuyết)**:
  - Cần phân biệt rõ:
    - **Tổng số vị trí n-gram (tokens/instances)**: Trong ngữ liệu gồm $T$ token từ, số vị trí n-gram là $T - n + 1$. Với $T$ rất lớn so với $n$, con số này xấp xỉ bằng $T$ trên mọi bậc $n$.
    - **Số lượng n-gram phân biệt (unique n-grams / distinct types)**: Là số lượng chuỗi $n$ từ khác nhau thực tế xuất hiện trong văn bản.
  - Không gian kết hợp lý thuyết của n-gram tăng theo cấp số nhân là $|\mathcal{V}|^n$. Khi độ dài cụm từ $n$ tăng, tính đặc thù (specificity) của ngữ cảnh tăng lên, làm giảm khả năng một chuỗi $n$ từ cụ thể lặp lại nhiều lần. Đa số các n-gram bậc cao chỉ xuất hiện một vài lần hoặc đúng 1 lần (*hapax legomena*), khiến số lượng n-gram phân biệt quan sát được tăng vọt so với unigram.
- **Confidence (Độ tin cậy)**: **High** 

---

## 3. Dự đoán 3 — Nguy Cơ Mắc Phải Vấn Đề Xác Suất Bằng 0 (Zero Probability)

### Câu hỏi nghiên cứu
> *Mô hình nào có xác suất cao nhất gặp phải vấn đề xác suất bằng 0 khi đánh giá trên dữ liệu mới?*

### Lời giải & Dự đoán chính thức
- **Prediction (Dự đoán)**: **Mô hình Trigram có nguy cơ gặp xác suất bằng 0 cao nhất, kế tiếp là Bigram, và Unigram có nguy cơ thấp nhất:**
  $$\text{Nguy cơ Zero Probability: } \text{Trigram} > \text{Bigram} > \text{Unigram}$$
- **Reason (Cơ sở lý thuyết)**:
  - Giả sử mô hình sử dụng ước lượng cực đại hợp lý (MLE) thuần túy và **không áp dụng kỹ thuật làm mịn (no smoothing)**:
    $$P_{MLE}(w_t \mid w_{t-n+1}^{t-1}) = \frac{C(w_{t-n+1}^t)}{C(w_{t-n+1}^{t-1})}$$
  - Trên tập dữ liệu mới (validation/test set), xác suất bằng 0 xuất phát từ hai nguyên nhân chính:
    1. **Từ ngoài từ vựng (Out-Of-Vocabulary - OOV)**: Từ $w_t$ chưa từng xuất hiện trong tập huấn luyện ($C_{\text{train}}(w_t) = 0$). Đây là nguyên nhân duy nhất khiến mô hình Unigram MLE bị zero probability. Nếu tất cả các từ trong test set đều thuộc từ vựng train (closed vocabulary), Unigram MLE sẽ không bao giờ gặp zero probability.
    2. **Hiện tượng thưa dữ liệu (Data Sparsity / Unseen N-grams)**: Ngay cả khi **hoàn toàn không có từ OOV** (mọi từ đơn lẻ đều đã nằm trong từ vựng train), mô hình Bigram và Trigram vẫn gặp zero probability nếu các từ đã biết đó ghép thành một chuỗi chưa từng đi liền nhau trong tập train ($C_{\text{train}}(w_{t-1}, w_t) = 0$ hoặc $C_{\text{train}}(w_{t-2}, w_{t-1}, w_t) = 0$).
  - Không gian các chuỗi n-gram phân biệt mở rộng theo $|\mathcal{V}|^n$, trong khi tổng số token huấn luyện là hữu hạn. Trigram đòi hỏi sự đồng xuất hiện đồng thời của 3 từ liên tiếp, nên khả năng bắt gặp một cụm n-gram chưa từng thấy trên tập kiểm thử cao hơn nhiều so với cặp 2 từ của Bigram. Dưới ước lượng MLE không làm mịn, chỉ cần một n-gram có tần số bằng 0 thì tích xác suất của cả câu sẽ lập tức bằng 0.
- **Confidence (Độ tin cậy)**: **High** 

---

## 4. Dự đoán 4 — Xu Hướng Biến Đổi Của Perplexity Trên Tập Huấn Luyện (Train Set)

### Câu hỏi nghiên cứu
> *Mô hình nào được kỳ vọng sẽ đạt chỉ số Perplexity thấp nhất trên tập huấn luyện?*

### Lời giải & Dự đoán chính thức
- **Prediction (Dự đoán)**: **Trên tập huấn luyện (train set), mô hình Trigram được kỳ vọng đạt Perplexity thấp nhất, kế tiếp là Bigram, và Unigram có Perplexity cao nhất:**
  $$\text{PPL}_{\text{train}}(\text{Trigram}) \le \text{PPL}_{\text{train}}(\text{Bigram}) \le \text{PPL}_{\text{train}}(\text{Unigram})$$
- **Reason (Cơ sở lý thuyết)**:
  - Perplexity tỉ lệ nghịch với trung bình nhân xác suất của chuỗi văn bản:
    $$\text{PPL}(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid context_i)\right)$$
  - **Trên tập huấn luyện**: Mô hình bậc cao ($n=3$) điều kiện hóa trên lịch sử dài hơn ($w_{t-2}, w_{t-1}$). Theo lý thuyết thông tin, việc bổ sung thêm biến điều kiện không làm tăng độ bất định (Entropy):
    $$H(W_t \mid W_{t-2}, W_{t-1}) \le H(W_t \mid W_{t-1}) \le H(W_t)$$
    Dưới ước lượng MLE, phân phối có điều kiện của Trigram khớp sát (fit) vào các chuỗi cụ thể trong tập train, gán xác suất cao cho các chuyển tiếp đã quan sát, làm giảm cross-entropy và kéo training perplexity xuống mức thấp nhất.
  - **Phân biệt với khả năng tổng quát hóa (Generalization)**: Training perplexity thấp phản ánh khả năng ghi nhớ (memorization) tập train của mô hình, **hoàn toàn không đồng nghĩa với việc mô hình sẽ tổng quát hóa tốt hơn trên dữ liệu mới (validation/test set)**. Khi đánh giá trên dữ liệu kiểm thử, nếu không có smoothing, Trigram sẽ sụp đổ (Perplexity bằng $\infty$) ngay khi gặp n-gram chưa từng thấy. Ngay cả khi có smoothing cơ bản, mô hình bậc cao vẫn có thể bị quá khớp (overfitting) và cho test perplexity kém hơn mô hình bậc thấp.
- **Confidence (Độ tin cậy)**: **High** 

---

## 5. Dự đoán 5 — Khả Năng Khái Quát Hóa Của Trigram vs Bigram Trên Ngữ Liệu Nhỏ

### Câu hỏi nghiên cứu
> *Nếu kích thước corpus huấn luyện rất nhỏ, liệu Trigram có chắc chắn mang lại kết quả tốt hơn Bigram không?*

### Lời giải & Dự đoán chính thức
- **Prediction (Dự đoán)**: **KHÔNG CHẮC CHẮN.** Khi corpus huấn luyện rất nhỏ, mô hình Trigram thường có xu hướng hoạt động kém hơn Bigram trên tập Validation và Test set.
- **Reason (Cơ sở lý thuyết)**:
  - Hiện tượng này được giải thích chặt chẽ thông qua **sự đánh đổi giữa độ chệch và phương sai (Bias–Variance Trade-off)**, **độ thưa dữ liệu (Data Sparsity)** và **hiện tượng quá khớp (Overfitting)**:
    1. **Về độ chệch (Bias)**: Mô hình Bigram áp đặt giả định Markov chặt hơn (chỉ nhớ 1 từ trước), bỏ qua nhiều ngữ cảnh xa nên có **độ chệch mô hình cao hơn (higher bias)** nhưng cấu trúc đơn giản. Trigram nới lỏng giả định Markov nên có **độ chệch thấp hơn (lower bias)**.
    2. **Về phương sai (Variance)**: Trigram có số lượng tham số cần ước lượng khổng lồ ($O(|\mathcal{V}|^3)$). Khi corpus nhỏ, số lượng vị trí token không đủ để cung cấp tần số thống kê đáng tin cậy cho đại đa số cụm 3 từ (phần lớn có tần số bằng 0 hoặc 1). Ước lượng tham số của Trigram trở nên rất nhạy cảm với mẫu dữ liệu ngẫu nhiên, dẫn đến **phương sai ước lượng rất cao (high variance)**. Ngược lại, Bigram có số tham số ít hơn nhiều ($O(|\mathcal{V}|^2)$), các cặp từ xuất hiện lặp lại thường xuyên hơn, do đó phương sai ước lượng thấp hơn và ổn định hơn.
    3. **Hậu quả trên dữ liệu mới (Validation/Test set)**: Khi dữ liệu huấn luyện hạn chế, thành phần phương sai bùng nổ của Trigram lấn át lợi thế về độ chệch thấp, gây ra hiện tượng quá khớp (overfitting). Trigram sẽ gặp rất nhiều ngữ cảnh chưa từng thấy (unseen contexts); khi áp dụng làm mịn (smoothing), xác suất bị pha loãng mạnh và khiến Perplexity trên validation/test set thường cao hơn (tệ hơn) so với Bigram.
- **Confidence (Độ tin cậy)**: **High**
