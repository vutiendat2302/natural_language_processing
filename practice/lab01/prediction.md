# Part C — Prediction Before Experiment

## 1. Prediction 1 — Vocabulary

### 1.1. Câu hỏi nghiên cứu
> *Nếu corpus có 30.000 documents (30K documents), vocabulary sẽ có khoảng bao nhiêu unique terms?*

### 1.2. Dự đoán cụ thể (Hypothesis)
* **Dự đoán không cắt ngưỡng tần suất (no pruning):**
  $$\hat{V} \approx 100.000 \text{ đến } 150.000 \text{ unique terms}$$

---

## 2. Prediction 2 — Sparsity of the TF-IDF Matrix

### 2.1. Câu hỏi nghiên cứu
> *TF-IDF matrix sẽ dense hay sparse? Tỷ lệ zero entries có thể lớn đến mức nào?*

### 2.2. Cơ sở lý thuyết & Lập luận định lượng
1. **Định nghĩa độ thưa (Sparsity):**
   Độ thưa $S$ của ma trận đặc trưng $X \in \mathbb{R}^{N \times V}$ được tính bằng tỷ lệ số phần tử mang giá trị 0 trên tổng số ô nhớ:
   $$S = 1 - \frac{\text{nnz}(X)}{N \times V}$$
   Trong đó:
   - $N = 30.000$ (số documents).
   - $V \approx 100.000$ (kích thước vocabulary dự đoán).
   - Tổng số phần tử lý thuyết: $N \times V \approx 30.000 \times 100.000 = 3 \times 10^9$ phần tử .
   - $\text{nnz}(X)$ là tổng số phần tử khác không.

2. **Dung lượng từ vựng kích hoạt trên một document:**
   - Dù từ vựng toàn cục $V$ rất lớn ($\sim 10^5$), một document cụ thể chỉ chứa từ vài chục đến vài trăm từ vựng khác nhau:
     $$\text{Unique terms per doc} \approx 100 - 250 \text{ terms}$$
   - Nghĩa là trong mỗi vector tài liệu biểu diễn bởi một hàng gồm $100.000$ chiều, chỉ có khoảng $100 - 250$ tọa độ nhận giá trị $> 0$ (tần số xuất hiện $\ge 1$). Hơn $99.750$ tọa độ còn lại mang giá trị chính xác là $0.0$.
   - Tỷ lệ phần tử khác không trên mỗi hàng:
     $$\text{Density per row} = \frac{\text{nnz}(d)}{V} \approx \frac{150}{100.000} \approx 0.0015 = 0.15\%$$

### 2.3. Dự đoán cụ thể
* Ma trận TF-IDF chắc chắn sẽ ở trạng thái **cực kỳ thưa (extremely sparse)**.
* Tỷ lệ phần tử 0 (Zero entries ratio) dự kiến đạt:
  $$\hat{S} \ge 98\% \quad (\text{dao động từ } 98\% \text{ đến } 99\%)$$

---

## 3. Prediction 3 — Search Ranking & Semantic Relevance 

### 3.1. Câu hỏi nghiên cứu
> *Với một query bất kỳ: Các documents đứng đầu kết quả tìm kiếm có nhất thiết là documents gần nghĩa nhất không?*

### 3.2. Cơ sở lý thuyết & Lập luận bản chất
1. **Bản chất của TF-IDF và Cosine Similarity:**
   - TF-IDF hoàn toàn là một biểu diễn dựa trên **thống kê bề mặt từ vựng rời rạc (Discrete Lexical Matching)**.
   - Vector của document và query được xây dựng dựa trên sự xuất hiện chính xác của các ký tự/chuỗi từ vựng (*exact token overlap*).
   - Tích vô hướng $\mathbf{q} \cdot \mathbf{d} = \sum_{t \in q \cap d} tfidf(t, q) \cdot tfidf(t, d)$ chỉ nhận giá trị dương khi và chỉ khi có ít nhất một token trong truy vấn trùng khớp y hệt với token trong tài liệu.

2. **Các hạn chế cốt lõi khiến top ranking KHÔNG đồng nghĩa với gần nghĩa nhất:**

   * **Vấn đề đồng nghĩa & đa dạng từ vựng (Synonymy / Vocabulary Mismatch):**
     - Con người diễn đạt cùng một ý niệm bằng nhiều từ khác nhau.
     - *Ví dụ:* Query là `"heart attack treatment"` (điều trị đau tim), nhưng văn bản y khoa hàn lâm chuẩn xác nhất lại sử dụng thuật ngữ `"myocardial infarction therapy"` (liệu pháp nhồi máu cơ tim).
     - Kết quả: Lexical overlap $= \emptyset \implies \text{Cosine Similarity} = 0.0$. Tài liệu cực kỳ phù hợp về ngữ nghĩa sẽ bị hệ thống TF-IDF loại bỏ hoàn toàn khỏi top kết quả.
   
   * **Vấn đề từ đa nghĩa và ngữ cảnh (Polysemy & Context Insensitivity):**
     - Một từ có nhiều nghĩa phụ thuộc ngữ cảnh.
     - *Ví dụ:* Query là `"apple release"` (tìm thông tin công nghệ Apple ra mắt sản phẩm), hệ thống có thể xếp lên đầu các văn bản nông nghiệp mô tả việc *"release fresh apple varieties into the market"* chỉ vì văn bản đó có tần suất từ `apple` và `release` cao. TF-IDF không có khả năng phân biệt ngữ cảnh phân bố.

   * **Thiên lệch do tần số lặp từ (Term Repetition Bias):**
     - Một văn bản ngắn nhưng cố tình lặp lại từ khóa truy vấn nhiều lần sẽ có giá trị Term Frequency (TF) cao, đẩy Cosine Similarity lên vị trí số 1, dù văn bản đó có thể hoàn toàn là spam hoặc có chất lượng thông tin rất kém.

### 3.3. Dự đoán cụ thể 
* **Kết luận:** Các documents đứng đầu kết quả tìm kiếm của TF-IDF **KHÔNG NHẤT THIẾT** là các documents gần nghĩa nhất với query của người dùng. Vì TF-IDF + cosine similarity chủ yếu dựa trên sự trùng khớp và trọng số của các term, chứ không thực sự hiểu semantic meaning.


