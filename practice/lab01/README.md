

## 1. Mục Tiêu & Bối Cảnh Bài Thực Hành 

### Các mục tiêu học tập cốt lõi (Learning Outcomes):
1. **Hiểu bản chất toán học:** Nắm vững cấu trúc không gian vector (Vector Space Model - VSM), công thức Term Frequency ($\text{TF}$), Document Frequency ($\text{DF}$), Inverse Document Frequency ($\text{IDF}$ có làm mượt và không làm mượt), và độ tương đồng Cosine kết hợp chuẩn hóa L2 norm.
2. **Kỹ năng tự lập trình (From-scratch Implementation):** Tự xây dựng thuật toán vector hóa TF-IDF và hàm đo Cosine Similarity thuần Python mà không phụ thuộc vào thư viện bên ngoài, sau đó đối chiếu độ chính xác với `scikit-learn`.
3. **Thực nghiệm quy mô lớn (30.000 văn bản):**
   * Đo lường kích thước từ vựng thực tế và độ thưa (Sparsity) của ma trận đặc trưng.
   * Thực hiện **Ablation Study** đánh giá ảnh hưởng của 3 chiến lược tiền xử lý:
     * **Pipeline A (Minimal):** Lowercase + Tokenization cơ bản.
     * **Pipeline B (Normalized):** Lowercase + Chuẩn hóa dấu câu (Punctuation) + Tokenization + Lọc từ dừng (Stopwords).
     * **Pipeline C (Extended):** Subword Tokenization bằng thuật toán WordPiece (`bert-base-uncased` tokenizer) giữ nguyên độ dài tài liệu (`truncation=False`).
4. **Xây dựng ứng dụng tìm kiếm (Information Retrieval):** Xây dựng Search Engine trên ma trận đặc trưng thưa, truy hồi Top-K tài liệu phù hợp cho các câu truy vấn thực tế.
5. **Đo lường định lượng nghiêm ngặt:** Đánh giá độ chính xác qua 3 chỉ số Information Retrieval kinh điển: **Precision@5**, **Recall@5**, và **Mean Reciprocal Rank (MRR)**.
6. **Mổ xẻ nguyên nhân thất bại (Error Analysis):** Phân tích hiện tượng phân mảnh từ vựng (*Lexical Mismatch*), sự áp đảo của các từ hiếm (*High-IDF Dominance*), và lý giải nguyên nhân dẫn đến bước nhảy vọt sang mô hình nhúng ngữ nghĩa (*Word Embeddings*) và *Transformers*.

---

## 2. Cấu Trúc Thư Mục & Ý Nghĩa Các Deliverables

```text
practice/lab01/
├── README.md                      # [File hiện tại] Hướng dẫn chi tiết, bối cảnh bài toán và kết quả
├── calculations.md                # [Part B] Bài tập tính toán tay giải tích chi tiết trên toy corpus
├── prediction.md                  # [Part C] 5 giả thuyết định lượng được đặt ra trước khi chạy thực nghiệm
├── implementation.py              # [Part E] Cài đặt thuật toán TF-IDF & Cosine from scratch + 7 Unit Tests
├── experiments.ipynb              # [Part D, F, G, H] Toàn bộ code chạy thực nghiệm trên tập 30K tài liệu
├── results.csv                    # [Part H] Bảng dữ liệu định lượng đánh giá 8 queries trên 3 pipelines
└── reflection.md                  # [Part I, J, 14, 15, 16] Báo cáo phân tích lỗi, Learning Check & Đúc kết
└── 23000111-VuTienDat-lab01.pdf    # File pdf chứa phần calculations, prediction, learning check
```

### Chi tiết nhiệm vụ của từng file:
* **`calculations.md` (10 điểm):** Lời giải phân tích toán học chi tiết từng bước cho bài toán 3 tài liệu đồ chơi ($D_1, D_2, D_3$). Trình bày ma trận Count Vector, bảng tần số $\text{TF}$, $\text{DF}$, $\text{IDF}$ (cả Smooth và Non-smooth), tính toán vector trọng số $\text{TF-IDF}$, chuẩn Euclid $L_2$ và giá trị $\text{Cosine Similarity}$.
* **`prediction.md` (10 điểm):** Lưu trữ 5 dự đoán độc lập của sinh viên trước khi đọc dữ liệu: kích thước từ vựng $\hat{V}$, độ thưa ma trận $\hat{S}$, tính chính xác của Top ranking, ảnh hưởng của việc lọc stopwords, và tỷ lệ từ vựng ngoài danh mục (OOV).
* **`implementation.py` (20 điểm):** Mã nguồn Python tự lập trình hoàn toàn các hàm:
  - `compute_tf(doc_tokens)`
  - `compute_idf(corpus_tokens, smooth=True)`
  - `compute_tfidf(corpus_tokens, idf_weights, smooth=True)`
  - `cosine_similarity(vec_a, vec_b)`
  - Tích hợp sẵn bộ 7 unit test tự động (bao gồm kiểm thử tính trực giao, kiểm thử khép kín pipeline và kiểm thử sai số với `scikit-learn`).
* **`experiments.ipynb` (30 điểm):** Notebook Jupyter thực thi hoàn chỉnh các thí nghiệm trên toàn bộ 30.000 tài liệu, chia tách thành các cell riêng biệt theo đúng cấu trúc:
  - Mục 7: Phân tích phân bố từ vựng và độ thưa ma trận.
  - Mục 9: So sánh đối đầu 3 Pipeline tiền xử lý A, B, C.
  - Mục 10: Xây dựng hệ thống tìm kiếm tài liệu (Document Search Engine).
  - Mục 11: Đánh giá định lượng $P@5, R@5, MRR$ trên 8 chủ đề truy vấn đa dạng.
* **`results.csv`:** Lưu kết quả định lượng chi tiết gồm 24 dòng dữ liệu thực tế ($8 \text{ queries} \times 3 \text{ pipelines}$).
* **`reflection.md` (15 điểm):** Báo cáo phân tích lỗi chuyên sâu cho 2 query tốt và 2 query kém; mổ xẻ nguyên nhân toán học của thất bại trên query y khoa; luận chứng chuyển dịch sang Word2Vec/Transformers; khai báo AI minh bạch; trả lời 7 câu hỏi Learning Check và đúc kết 6 bài học kinh nghiệm (~500 từ).

---

## 3. Hướng Dẫn Chuẩn Bị Dữ Liệu (Dataset Setup)

- Tải file trên gg classroom đổi tên thành: 30k.json 
---

## 4. Hướng Dẫn Cài Đặt Môi Trường (Python 3.12)

Đề án được cấu hình và kiểm thử hoạt động tối ưu nhất trên nền tảng **Python 3.12** (khuyên dùng Python $\ge 3.12.0$).

### Bước 1: Khởi tạo Virtual Environment với Python 3.12
Từ thư mục gốc của repository (`natural_language_processing/`):
```bash
# Kiểm tra phiên bản Python
python3.12 --version

# Tạo môi trường ảo mang tên .venv
python3.12 -m venv .venv

# Kích hoạt môi trường ảo
# Trên Linux/macOS:
source .venv/bin/activate
# Trên Windows:
# .venv\Scripts\activate
```

### Bước 2: Cài đặt các gói phụ thuộc
Nâng cấp `pip` và cài đặt các thư viện cần thiết từ `requirements.txt`:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Các thư viện chính được sử dụng trong bài:
* `scikit-learn` ($\ge 1.3.0$): Vector hóa TF-IDF chuẩn công nghiệp và tính ma trận tương đồng.
* `transformers` ($\ge 4.35.0$) & `tokenizers` ($\ge 0.14.0$): HuggingFace WordPiece Tokenizer (`bert-base-uncased`).
* `scipy` ($\ge 1.10.0$) & `numpy` ($\ge 1.24.0$): Tính toán ma trận thưa `csr_matrix` và tối ưu vector hóa.
* `pandas` ($\ge 2.0.0$): Xử lý bảng dữ liệu kết quả đánh giá.
* `matplotlib` & `seaborn`: Vẽ biểu đồ độ thưa và trực quan hóa kết quả tìm kiếm.

### Bước 3: Đăng ký Kernel cho Jupyter Notebook
Để giao diện Jupyter Notebook hoặc VS Code nhận diện đúng kernel Python 3.12:
```bash
python -m ipykernel install --user --name=nlp-lab01 --display-name "Python 3.12 (nlp-lab01)"
```

---

## 5. Hướng Dẫn Thực Thi Chi Tiết Từng Phần

### 5.1. Kiểm thử thuật toán tự cài đặt (`implementation.py`)
Mã nguồn `implementation.py` có thể chạy độc lập dưới dạng một module dòng lệnh. Nó sẽ tự động kích hoạt bộ 7 Unit Tests và in bảng đối chiếu toán học với `scikit-learn`:

```bash
cd practice/lab01
python implementation.py
```

**Kết quả mong đợi trên màn hình Terminal:**
```text
================================================================================
CHẠY BỘ KIỂM THỬ TỰ ĐỘNG (UNIT TESTS) CHO THUẬT TOÁN TF-IDF TỰ CÀI ĐẶT
================================================================================
[OK] Test 1: compute_tf()
[OK] Test 2: compute_idf()
[OK] Test 3: compute_tfidf()
[OK] Test 4: cosine_similarity()
[OK] Test 5: Cosine Similarity với 2 vector trực giao
[OK] Test 6: Pipeline xử lý văn bản thô khép kín
[OK] Test 7: Tính toán Cosine Similarity trực tiếp trên toy corpus D1/D2/D3

>>> TẤT CẢ 7 BÀI TEST ĐÃ VƯỢT QUA THÀNH CÔNG!
```

---

### 5.2. Chạy toàn bộ thực nghiệm lớn trên Notebook (`experiments.ipynb`)
Có thể thực thi notebook theo 2 phương thức:

#### Phương thức 1: Chạy tương tác qua giao diện đồ họa (Jupyter / VS Code)
1. Khởi chạy Jupyter Lab:
   ```bash
   jupyter lab
   ```
2. Mở tệp `experiments.ipynb` (hoặc `23000111_VuTienDat_BT1.ipynb`).
3. Chọn Kernel: **`Python 3.12 (nlp-lab01)`**.
4. Chạy tuần tự từng ô lệnh (Run All Cells). Notebook đã được thiết kế sẵn các khối try-catch an toàn bộ nhớ và phân đoạn batching hợp lý cho tokenizer.

#### Phương thức 2: Chạy tự động qua dòng lệnh (Headless Execution)
Nếu muốn chạy kiểm chứng ngầm toàn bộ thí nghiệm mà không cần mở trình duyệt:
```bash
jupyter nbconvert --to notebook --execute experiments.ipynb --output experiments_output.ipynb
```

---

## 6. Tổng Kết Các Kết Quả Thực Nghiệm Nổi Bật

### 6.1. Phân tích độ thưa và kích thước từ vựng (Corpus 30K)
* **Số lượng tài liệu:** $N = 30.000$ documents.
* **Tổng số từ vựng toàn cục (Vocabulary size):** $V = 193.540$ unique terms.
* **Số phần tử khác không (Non-zero entries):** $\text{nnz} = 4.985.822$ phần tử.
* **Tổng số phần tử lý thuyết:** $N \times V = 30.000 \times 193.540 = 5.806.200.000$ phần tử.
* **Độ thưa của ma trận đặc trưng:**
  $$S = 1 - \frac{4.985.822}{5.806.200.000} = 99.9141\%$$
  *Nhận xét:* Hơn $99.91\%$ ô nhớ trong ma trận chứa giá trị $0.0$. Việc lưu trữ dưới định dạng Compressed Sparse Row (`csr_matrix`) giúp giảm dung lượng RAM từ $\sim 46.4\text{ GB}$ (dạng dense) xuống chỉ còn $\sim 59.8\text{ MB}$.

---

### 6.2. So sánh đối đầu 3 Pipeline tiền xử lý (Preprocessing Ablation)

| Chỉ số đo lường | Pipeline A (Minimal) | Pipeline B (Normalized) | Pipeline C (Extended) |
| :--- | :---: | :---: | :---: |
| **Quy trình tiền xử lý** | Raw $\to$ Lowercase $\to$ Tokenize | Raw $\to$ Lowercase $\to$ Strip Punct $\to$ Stopwords | Raw $\to$ WordPiece Subwords (`bert-base-uncased`) |
| **Kích thước từ vựng ($V$)** | $193.540$ | $192.764$ | **$28.339$** (giảm $85.3\%$) |
| **Số token trung bình / doc** | $369.7$ tokens | $197.5$ tokens | $465.1$ subwords |
| **Độ thưa ma trận ($S$)** | $99.9141\%$ | $99.9351\%$ | **$99.3200\%$** |
| **Tỷ lệ từ vựng ngoài từ điển (OOV)** | $30.0\%$ | $30.0\%$ | **$0.0\%$** (triệt tiêu hoàn toàn) |
| **Thời gian vector hóa 30K docs** | $\approx 2.45\text{s}$ | $\approx 3.75\text{s}$ | $\approx 55.4\text{s}$ (xử lý qua HuggingFace) |

---

### 6.3. Đánh giá chất lượng tìm kiếm (Information Retrieval Evaluation)

Kết quả trích xuất từ bảng `results.csv` trên 8 câu truy vấn kiểm thử:

| Nhóm Query | Câu truy vấn thử nghiệm | Pipeline A ($P@5 / \text{MRR}$) | Pipeline B ($P@5 / \text{MRR}$) | Pipeline C ($P@5 / \text{MRR}$) | Nhận xét bản chất |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Good Queries** | `"climate change renewable energy"` | **1.0 / 1.0** | **1.0 / 1.0** | **1.0 / 1.0** | Từ vựng chuyên ngành hẹp, co-occurrence mạnh. |
| | `"credit card interest rate loan"` | **1.0 / 1.0** | **1.0 / 1.0** | **1.0 / 1.0** | Khớp từ vựng tài chính chuẩn xác. |
| | `"real estate property investment"` | **1.0 / 1.0** | **1.0 / 1.0** | **1.0 / 1.0** | Tần suất thuật ngữ thị trường bất động sản cao. |
| **Moderate Queries** | `"recipes cooking ingredients"` | 0.6 / 1.0 | 0.6 / 1.0 | 0.6 / 0.5 | Khớp tốt từ vựng ẩm thực, bị nhiễu bài viết chung. |
| | `"deep learning healthcare"` | 0.2 / 0.5 | 0.2 / 0.5 | 0.2 / 0.33 | Tài liệu kết hợp hai lĩnh vực tương đối hiếm. |
| | `"transformer language model"` | 0.2 / 1.0 | 0.2 / 1.0 | 0.2 / 1.0 | Từ `transformer` dễ lẫn sang biến áp điện lực. |
| **Poor Queries** | `"natural language processing"` | 0.2 / 0.25 | **0.0 / 0.0** | 0.2 / 0.33 | Bị phân mảnh nghĩa sang cài đặt ngôn ngữ điện thoại. |
| | `"medical image classification"` | **0.0 / 0.0** | **0.0 / 0.0** | **0.2 / 0.2** | **Thất bại hoàn toàn do Lexical Mismatch & High-IDF**. |

---

## 7. Kết Luận Khoa Học Từ Thực Nghiệm

1. **Hiệu ứng High-IDF Dominance:** Một từ hiếm duy nhất (như `classification`, $\text{IDF}=6.7038$) xuất hiện trong một tài liệu ngắn ngẫu nhiên có thể chiếm tỷ trọng vector L2 cực lớn, đẩy tài liệu sai chủ đề lên vị trí số 1 và làm sụt giảm độ chính xác của toàn hệ thống.
2. **Rào cản từ vựng rời rạc (Lexical Mismatch):** TF-IDF và mô hình Bag-of-Words bất lực trước các khái niệm đồng nghĩa (ví dụ: truy vấn tìm kiếm `"medical image"` không thể tìm ra tài liệu chứa `"chest X-ray"` hoặc `"computed tomography scan"` do không có token overlap).
3. **Động lực chuyển dịch kiến trúc:** Thất bại thực nghiệm trên chính là minh chứng thực tế rõ ràng nhất cho nhu cầu chuyển đổi sang **Biểu diễn ngữ nghĩa phân bố (Distributional Semantics)**:
   $$\text{TF-IDF (Thưa, rời rạc)} \implies \text{Word2Vec / GloVe (Dày đặc, hiểu ngữ cảnh cục bộ)} \implies \text{Transformers / BERT (Ngữ cảnh động hai chiều)}$$
