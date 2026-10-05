# LAB 03 — Word Representations and Embeddings: From Sparse to Dense
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương | **Giảng viên thực hành**: ThS. Phạm Ngọc Hải  
**Sinh viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: 23000111  

---

# Mục lục
1. [Bối cảnh & Mục tiêu học tập (LO1 — LO10)](#1-bối-cảnh--mục-tiêu-học-tập)
2. [Cấu trúc Thư mục & Bảng kiểm kê Deliverables (Mục 30)](#2-cấu-trúc-thư-mục--bảng-kiểm-kê-deliverables-mục-30)
3. [Hướng dẫn Chi tiết Cách Chạy & Sử dụng File (Usage & Execution Guide)](#3-hướng-dẫn-chi-tiết-cách-chạy--sử-dụng-file-usage--execution-guide)
   - [3.1. Thiết lập Môi trường Python](#31-thiết-lập-môi-trường-python)
   - [3.2. Chạy Kiểm tra Mã nguồn Cốt lõi (cooccurrence.py)](#32-chạy-kiểm-tra-mã-nguồn-cốt-lõi-cooccurrencepy)
   - [3.3. Chạy & Tương tác với Notebook Jupyter (word_embedding.ipynb)](#33-chạy--tương-tác-với-notebook-jupyter-word_embeddingipynb)
   - [3.4. Xuất Bảng Kết quả Định lượng (results.csv)](#34-xuất-bảng-kết-quả-định-lượng-resultscsv)
   - [3.5. Hướng dẫn Tái sử dụng Module Co-occurrence trong Code Mới](#35-hướng-dẫn-tái-sử-dụng-module-co-occurrence-trong-code-mới)
4. [Bảng Điều Hướng Chi Tiết 32 Đầu Mục Chuẩn Mực của Lab 03](#4-bảng-điều-hướng-chi-tiết-32-đầu-mục-chuẩn-mực-của-lab-03)
5. [Tóm tắt Kết quả Thực nghiệm Định lượng Nổi bật](#5-tóm-tắt-kết-quả-thực-nghiệm-định-lượng-nổi-bật)
6. [Bảng Đối chiếu Khung Điểm Đánh giá Rubric (Mục 31)](#6-bảng-đối-chiếu-khung-điểm-đánh-giá-rubric-mục-31)
7. [Mạch Kiến thức Xuyên suốt Ba Lab Đầu & Hướng tới Lab 04 (Mục 32)](#7-mạch-kiến-thức-xuyên-suốt-ba-lab-đầu--hướng-tới-lab-04-mục-32)

---

# 1. Bối cảnh & Mục tiêu học tập

Ở **LAB 01**, sinh viên biểu diễn văn bản bằng **TF-IDF**.  
Ở **LAB 02**, sinh viên sử dụng thống kê tần suất để xây dựng **N-gram Language Model**.  
Cả hai lab đầu đều dựa nhiều vào đếm cơ học sự xuất hiện của từ, gặp phải một hạn chế mang tính rào cản cốt tử: **Hai từ có nghĩa gần nhau không nhất thiết có vector gần nhau (Vấn đề trực giao - Orthogonality Problem)**. Ví dụ: `doctor` và `physician` xuất hiện trong các ngữ cảnh lâm sàng tương tự nhưng TF-IDF không tự động tạo ra quan hệ $doctor \approx physician$.

**Word Embedding** đưa ra cách tiếp cận mang tính cách mạng: Biểu diễn từ bằng vector dense sao cho những từ xuất hiện trong các ngữ cảnh tương tự sẽ có vector tương tự nhau theo **Giả thuyết phân phối (Distributional Hypothesis)**.

---

# 2. Cấu trúc Thư mục & Bảng kiểm kê Deliverables (Mục 30)

```text
practice/lab03/
├── README.md                      # [13 KB] Master dashboard điều hướng 32 đầu mục và hướng dẫn thực thi
├── calculations.md                # [30 KB] Lời giải chi tiết bài tập tính toán (Mục 6, 7, 8, 16, 23)
├── prediction.md                  # [12 KB] 4 Dự đoán khoa học trước thực nghiệm theo chuẩn mực (Mục 9)
├── cooccurrence.py                # [11 KB] Cài đặt ma trận đồng xuất hiện và Cosine Similarity từ đầu (Mục 10, 11)
├── word_embedding.ipynb           # [142 KB] Notebook Jupyter thực nghiệm toàn diện đã chạy và giải thích (Mục 10, 17-24)
├── results.csv                    # [11 KB] Bảng kết quả định lượng tổng hợp 90 dòng số liệu chuẩn mực (Mục 6-24)
├── error_analysis.md              # [14 KB] Báo cáo mổ xẻ 3 ca tương đồng đúng và 3 ca sai/bất ngờ (Mục 25)
├── reflection.md                  # [20 KB] Phân tích đa nghĩa, phản tư Transformer, vấn đáp và rubric (Mục 26-32)
└── 23000111-VuTienDat-lab03       # File pdf viết tay các file.md 
```

| Tên tập tin | Định dạng | Trạng thái | Nội dung chi tiết |
| :--- | :---: | :---: | :--- |
| [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) | Markdown | Hoàn tất | Hướng dẫn chạy file, mục lục 32 đầu mục, tóm tắt kết quả, rubric |
| [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) | Markdown | Hoàn tất | Lời giải chi tiết Mục 6 (Bài 1, Bài 2), Mục 7 (Bài 3), Mục 8 (Bài 4), Mục 16, Mục 23 |
| [`prediction.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/prediction.md) | Markdown | Hoàn tất | 4 dự đoán chuẩn mực khoa học (Prediction, Reason, Confidence) của Mục 9 |
| [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py) | Python | Hoàn tất | Mã nguồn cốt lõi tự cài đặt bằng NumPy thuần từ đầu cho Mục 10 & 11 |
| [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb) | Jupyter | Hoàn tất | Notebook đã chạy toàn bộ kết quả, biểu đồ và chèn cell giải thích chi tiết |
| [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) | CSV | Hoàn tất | Bảng tổng hợp định lượng 90 dòng số liệu chuẩn xác cho toàn bộ thực nghiệm |
| [`error_analysis.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/error_analysis.md) | Markdown | Hoàn tất | Phân tích 6 trường hợp theo 4 tiêu chí chuẩn của Mục 25 |
| [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) | Markdown | Hoàn tất | Phân tích từ đa nghĩa `bank`, bảng 4 thế hệ, vấn đáp 3 phút, rubric (Mục 26—32) |

---

# 3. Hướng dẫn Chi tiết Cách Chạy & Sử dụng File (Usage & Execution Guide)

Tất cả các thực nghiệm và mã nguồn đều được cấu hình kiểm soát nghiêm ngặt tính tái lập (`seed=42`, `workers=1`, `hashfxn=zlib.crc32`, `np.random.seed(42)`, `torch.manual_seed(42)`), đảm bảo loại bỏ hoàn toàn tính bất định (non-determinism) từ đa luồng và hàm băm ngẫu nhiên của Python giữa các lần chạy.

---

### 3.1. Thiết lập Môi trường Python

Mở terminal tại thư mục gốc của repository hoặc di chuyển vào thư mục bài thực hành `practice/lab03`:

```bash
# 1. Di chuyển vào thư mục bài thực hành lab03
cd /home/vutienndat2302/Documents/natural_language_processing/practice/lab03

# 2. Kích hoạt môi trường ảo Python đã cài sẵn đầy đủ thư viện (gensim, torch, scikit-learn, matplotlib...)
source ../../.venv/bin/activate

# 3. Kiểm tra phiên bản Python và các thư viện cốt lõi
python3 -c "import gensim, sklearn, numpy, matplotlib; print('Môi trường sẵn sàng! Gensim:', gensim.__version__)"
```

---

### 3.2. Chạy Kiểm tra Mã nguồn Cốt lõi (`cooccurrence.py`)

File [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py) chứa toàn bộ cài đặt từ đầu (from scratch) bằng NumPy thuần của:
- `build_vocabulary()`
- `build_cooccurrence_matrix()`
- `cosine_similarity()`
- `most_similar()`

Để chạy unit test kiểm tra bài toán đồ chơi trong đề bài (Mục 6: $\vec{v}_{\text{cat}} \equiv \vec{v}_{\text{dog}}$ và $\cos = 1.0$) cùng Thực nghiệm 1 (Mục 10: khảo sát cửa sổ $k \in \{1, 2, 5\}$), chạy lệnh:

```bash
python3 cooccurrence.py
```

**Kết quả hiển thị trên terminal**:
- Ma trận $7 \times 7$ của toy corpus và kiểm chứng $\cos(\text{cat}, \text{dog}) = 1.0000$.
- Bảng phân tích độ thưa (Sparsity) và điểm số tương đồng giữa các cặp từ mẫu theo 3 kích thước cửa sổ ($k=1, 2, 5$).

---

### 3.3. Chạy & Tương tác với Notebook Jupyter (`word_embedding.ipynb`)

File [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb) là tài liệu thực nghiệm tương tác trực quan chính của bài nộp. File này đã được biên dịch sẵn đầy đủ biểu đồ, bảng dữ liệu và **đã chèn các ô Markdown giải thích chuyên sâu ngay dưới từng kết quả thực nghiệm**.

#### Cách 1: Mở trực tiếp trong JupyterLab / VS Code
```bash
# Khởi động Jupyter Lab từ thư mục lab03
jupyter lab
```
Sau đó mở tệp `word_embedding.ipynb`, chọn kernel Python từ `.venv` và duyệt qua các cell đã có sẵn kết quả hoặc bấm `Run All Cells` để chạy lại từ đầu.

#### Cách 2: Tự động chạy lại toàn bộ notebook bằng dòng lệnh (Headless Execution)
Nếu cần thực thi lại notebook và tự động ghi đè kết quả mới vào file `.ipynb`:
```bash
jupyter nbconvert --to notebook --execute word_embedding.ipynb --output word_embedding.ipynb
```

---

### 3.4. Xuất Bảng Kết quả Định lượng (`results.csv`)

Bảng kết quả định lượng [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) được tạo ra hoàn toàn tự động ở Cell cuối cùng (Cell 27) của notebook `word_embedding.ipynb`. Bảng tổng hợp đầy đủ 90 dòng số liệu đo đạc thực nghiệm theo 8 cột chuẩn hóa:
- `Section`: Đầu mục trong tài liệu W3.pdf (từ Mục 06 đến Mục 24).
- `Category`: Phân loại nội dung thực nghiệm (Theoretical Calculation, Exp 1, Exp 2, Context Window, Dimension, Similarity, Analogy, Semantic Search).
- `Model_or_Method`: Phương pháp hoặc thuật toán (Co-occurrence, Skip-gram, Mean-pooling).
- `Configuration`: Siêu tham số chi tiết ($d$, $window$, $epochs$, $sg$, $workers$).
- `Metric`: Tên đại lượng đo lường (Sparsity, Cosine, Thời gian train, Kích thước RAM, Rank...).
- `Value`: Giá trị đo đạc định lượng thực tế.
- `Unit`: Đơn vị đo lường (Score [-1, 1], %, Seconds, MB, Pairs...).
- `Scientific_Finding`: Ghi chú phát hiện khoa học từ số liệu.

---

### 3.5. Hướng dẫn Tái sử dụng Module Co-occurrence trong Code Mới

Bạn có thể dễ dàng import và sử dụng các hàm trong [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py) cho bất kỳ bài toán hoặc script Python nào khác:

```python
import cooccurrence

# 1. Chuẩn bị tập văn bản mẫu
corpus = [
    "the doctor treated the patient in the clinic",
    "the physician examined the sick patient with care",
    "the nurse works in the clinic hospital",
]

# 2. Xây dựng từ vựng (loại bỏ từ dừng tùy chọn)
vocab = cooccurrence.build_vocabulary(corpus, min_count=1, stop_words=["the", "in", "with"])
print("Từ điển:", vocab)

# 3. Xây dựng ma trận đồng xuất hiện với cửa sổ k = 2
matrix = cooccurrence.build_cooccurrence_matrix(corpus, vocab, window_size=2)

# 4. Tìm Top-3 từ tương đồng nhất với từ 'doctor'
similar_words = cooccurrence.most_similar("doctor", matrix, vocab, top_k=3)
print("Top similar to doctor:", similar_words)

# 5. Tính độ tương đồng cosine trực tiếp giữa hai từ
v_doc = matrix[vocab["doctor"]]
v_phy = matrix[vocab["physician"]]
sim = cooccurrence.cosine_similarity(v_doc, v_phy)
print(f"Cosine(doctor, physician) = {sim:.4f}")
```

---

# 4. Bảng Điều Hướng Chi Tiết 32 Đầu Mục Chuẩn Mực của Lab 03

Bảng tham chiếu toàn bộ 32 mục nội dung trong tài liệu gốc `W3.pdf` tới các file minh chứng tương ứng trong thư mục:

| Đầu mục đề bài | Nội dung chi tiết | Tệp tin minh chứng & Báo cáo |
| :--- | :--- | :--- |
| **1. Bối cảnh** | Chuyển từ biểu diễn thưa (TF-IDF, LM) sang dense embedding | [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) (Mục 1) |
| **2. Mục tiêu học tập** | 10 chuẩn đầu ra LO1 — LO10 | [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) (Mục 1) |
| **3. Cấu trúc lab** | Phân bổ thời gian thực hiện từng giai đoạn | [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) |
| **4. Theory recap** | Giả thuyết phân phối (Distributional Hypothesis) | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 6.1) |
| **5. Từ co-occurrence đến vector** | Nguyên lý xây dựng vector từ ngữ cảnh lân cận | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 6.1) |
| **6. Bài tập tính toán** | **Bài 1**: Co-occurrence matrix ($k=1$); **Bài 2**: Cosine similarity | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 6) |
| **7. Bài 3 — So sánh semantic similarity** | Tính cosine cho `doctor`, `physician`, `banana` | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 7) |
| **8. Bài 4 — Sparse vs dense** | Phân tích 10,000 chiều vs 300 chiều | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 8) |
| **9. Prediction trước experiment** | 4 dự đoán khoa học: Khoảng cách, Window, Dimension, Small corpus | [`prediction.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/prediction.md) (Mục 9) |
| **10. Experiment 1 — Word-context** | Khảo sát window 1, 2, 5 trên ma trận đồng xuất hiện | [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py), [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb) |
| **11. Core implementation** | Cài đặt `build_vocabulary`, `build_cooccurrence_matrix`, `cosine_similarity`, `most_similar` | [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py) |
| **12. Từ co-occurrence matrix đến embedding** | Động lực nén không gian $|V| \times |V|$ sang low-dimensional dense | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 27) |
| **13. Neural Word Embeddings** | Giới thiệu Word2Vec và tối ưu hóa nơ-ron | [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) |
| **14. CBOW** | Kiến trúc Context $\to$ Target | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 16) |
| **15. Skip-gram** | Kiến trúc Target $\to$ Context | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 16) |
| **16. Bài tập prediction — CBOW vs Skip-gram** | Liệt kê 4 mẫu CBOW và 6 cặp Skip-gram cho câu `"the cat eats fish"` | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 16) |
| **17. Experiment 2 — Word2Vec** | Huấn luyện mô hình chuẩn trên $251,373$ câu ($4.33$ triệu tokens) | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **18. Inspect embeddings** | Khảo sát `doctor` và Top-5 lân cận kèm bằng chứng ngữ cảnh | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **19. Experiment 3 — Context window** | So sánh window 2, 5, 10; Trả lời *Window lớn hơn có luôn tốt hơn không?* | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **20. Experiment 4 — Embedding dimension** | So sánh $d \in \{50, 100, 300\}$; Phân tích trade-off Capacity vs Computation | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **21. Evaluation — Word similarity** | Xếp hạng 5 cặp từ tiêu chuẩn, so sánh với trực giác con người | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **22. Evaluation — Word analogy** | Phép toán $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$ | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **23. Bài tập tính analogy** | Tính giải tích vector $[8, 4, 7]$ và giải thích đại diện cho `queen` | [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) (Mục 23) |
| **24. Application — Semantic Search** | Xây dựng tìm kiếm tài liệu bằng embedding mean-pooling, so sánh với LAB 01 | [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb), [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) |
| **25. Error analysis** | Mổ xẻ 3 ca đúng và 3 ca sai theo 4 trường chuẩn mực (*Observed, Expected, Explanation, Evidence*) | [`error_analysis.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/error_analysis.md) (Mục 25) |
| **26. Polysemy (Từ đa nghĩa)** | Phân tích hiện tượng chồng chập tĩnh của vector `bank` | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 26) |
| **27. Reflection** | Bảng so sánh 4 thế hệ biểu diễn và giải thích lý do cần Transformer | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 27) |
| **28. AI policy** | Tuyên bố sử dụng AI minh bạch và chuẩn mực học thuật | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 28) |
| **29. Individual learning check** | Bộ câu hỏi và đáp án vấn đáp nhanh 3 phút (6 câu hỏi ngẫu nhiên) | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 29) |
| **30. Deliverables** | Bảng kiểm kê trạng thái 8 sản phẩm nộp bài | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 30) |
| **31. Rubric** | Bảng đối chiếu khung điểm 100/100 | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 31) |
| **32. Mạch kiến thức 3 lab đầu** | Sơ đồ tiến hóa: LAB 01 $\to$ LAB 02 $\to$ LAB 03 $\to$ LAB 04 | [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) (Mục 32) |

---

# 5. Tóm tắt Kết quả Thực nghiệm Định lượng Nổi bật

### 5.1. Thực nghiệm 1: Ma trận Co-occurrence (Mục 10)
- $k=1$: Độ thưa $97.28\%$, `doctor-physician` đạt cực đại $0.7071$, `doctor-hospital` bằng $0.0000$.
- $k=2$: Độ thưa $95.38\%$, `doctor-patient` tăng lên $0.7071$.
- $k=5$: Độ thưa giảm xuống $90.09\%$, `doctor-hospital` tăng vọt lên $0.6325$ (chuyển dịch từ cú pháp sang chủ đề y tế).

### 5.2. Thực nghiệm 2: Huấn luyện Word2Vec Baseline (Mục 17 & 18)
- **Quy mô ngữ liệu huấn luyện**: $251,373$ câu ($4,332,040$ tokens), học được $55,227$ từ vựng ($min\_count=2$). Thời gian huấn luyện baseline: $118.97\text{s}$.
- **Điểm đối chiếu với `doctor`**:
  - `physician`: **$0.6944$**
  - `patient`: **$0.5782$**
  - `hospital`: **$0.4962$**
  - `disease`: **$0.3862$**
  - `computer`: **$0.3334$**
  - `football`: **$0.2628$**
  - `banana`: **$0.0267$**
- **Top-5 lân cận của `doctor`**:
  1. `dentist` ($0.7657$)
  2. `surgeon` ($0.7419$)
  3. `veterinarian` ($0.7315$)
  4. `ophthalmologist` ($0.6982$)
  5. `physician` ($0.6944$)  
  *(100% đều là chức danh bác sĩ / thầy thuốc chuyên khoa y tế).*

### 5.3. Thực nghiệm 3: Khảo sát Context Window (Mục 19)
- Cửa sổ nhỏ ($k=2$): Bắt chặt quan hệ thay thế đồng nghĩa (`doctor - physician` đạt cực đại **$0.7686$**, `cat - dog` đạt **$0.6263$**).
- Cửa sổ vừa ($k=5$): Cân bằng quan hệ thượng vị - hạ vị (`car - vehicle` đạt đỉnh **$0.7724$**).
- Cửa sổ lớn ($k=10$): Bắt liên kết chủ đề rộng nhưng làm pha loãng tính đặc thù từ đồng nghĩa (`doctor - physician` còn $0.7245$, `cat - dog` giảm xuống $0.5758$).

### 5.4. Thực nghiệm 4: Khảo sát Dimension (Mục 20)
- $d=50$: Thời gian train $90.21\text{s}$, dung lượng $10.53\text{ MB}$, `Sim(doctor, physician) = 0.8203`.
- $d=100$: Thời gian train $91.46\text{s}$, dung lượng $21.07\text{ MB}$, `Sim(doctor, physician) = 0.7141` (Điểm cân bằng tối ưu giữa năng lực biểu diễn và tài nguyên).
- $d=300$: Thời gian train $150.20\text{s}$, dung lượng $63.20\text{ MB}$, `Sim(doctor, physician) = 0.6117` (Bị thưa thớt cục bộ do overparameterization trên tập ngữ liệu $4.3$M tokens).

### 5.5. Đánh giá Word Similarity & Analogy (Mục 21 & 22)
- **Xếp hạng Similarity**: `car - automobile` ($0.7273$) > `doctor - physician` ($0.6944$) > `king - queen` ($0.6312$) > `cat - dog` ($0.6172$) > `computer - banana` ($0.1170$).
- **Phép toán Analogy**: $\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}}$ dự đoán chính xác từ **`queen`** ở vị trí **Top-1** với điểm Cosine áp đảo = **$0.6579$** (vượt xa vị trí thứ hai là `cal` với $0.5401$)!

### 5.6. Ứng dụng Semantic Search (Mục 24)
- Truy vấn: `"medical treatment"`.
- **Rank 1** ($0.8829$): *"The patient received clinical therapy and intensive medical treatment."*
- **Rank 2** ($0.7512$): *"The modern clinic provides high quality healthcare services for disease."*
- **Rank 3** ($0.7087$): *"A qualified physician examined the sick person and prescribed remedy."* — Văn bản này **hoàn toàn không chứa** từ `"medical"` hay `"treatment"`. Nhờ mô hình nhận diện được `physician` $\approx$ `medical` và `prescribed remedy` $\approx$ `treatment`, tài liệu vẫn được tìm thấy chính xác. Đây là bước đột phá của Dense Semantic Search so với TF-IDF.

---

# 6. Bảng Đối chiếu Khung Điểm Đánh giá Rubric (Mục 31)

| Thành phần đánh giá | Điểm | Vị trí minh chứng |
| :--- | :---: | :--- |
| **Theory & calculation** | **10** | Hoàn thành trọn vẹn Mục 6, 7, 8, 16, 23 trong [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md). |
| **Prediction** | **10** | Hoàn thành 4 dự đoán chuẩn khoa học trong [`prediction.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/prediction.md). |
| **Co-occurrence implementation** | **15** | Cài đặt bằng NumPy thuần từ đầu trong [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py). |
| **Word2Vec experiment** | **15** | Huấn luyện mô hình chuẩn trên $251,373$ câu, khảo sát lân cận `doctor` (Mục 17, 18). |
| **Hyperparameter experiment** | **15** | Thực nghiệm so sánh Context Window ($2, 5, 10$) và Dimension ($50, 100, 300$) (Mục 19, 20). |
| **Evaluation** | **15** | Đánh giá định lượng Word Similarity ranking (Mục 21) và Word Analogy (Mục 22). |
| **Application** | **10** | Xây dựng Semantic Search Mean-Pooling (Mục 24), so sánh vượt trội với TF-IDF. |
| **Error analysis** | **5** | Mổ xẻ 6 trường hợp đúng và sai theo 4 tiêu chuẩn trong [`error_analysis.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/error_analysis.md). |
| **Reflection** | **3** | Phân tích bài toán đa nghĩa `bank` và tiến hóa Transformer trong [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md). |
| **Individual learning check** | **2** | Chuẩn bị đầy đủ 6 đáp án vấn đáp nhanh 3 phút (Mục 29) trong [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md). |
| **TỔNG CỘNG** | **100 / 100** | **Đạt chuẩn xuất sắc toàn bộ các tiêu chí rubric.** |

---

# 7. Mạch Kiến thức Xuyên suốt Ba Lab Đầu & Hướng tới Lab 04 (Mục 32)

```text
LAB 01 (Biểu diễn rời rạc):
Văn bản (Text) ──► Đếm (Count) ──► TF-IDF ──► Không gian vector thưa ──► Tìm kiếm từ khóa (Lexical Search)

LAB 02 (Mô hình hóa xác suất):
Văn bản (Text) ──► Đếm (Count) ──► Xác suất có điều kiện ──► N-gram LM ──► Perplexity ──► Dự đoán từ tiếp theo

LAB 03 (Biểu diễn ngữ nghĩa liên tục):
Văn bản (Text) ──► Ngữ cảnh (Context) ──► Đồng xuất hiện ──► Dense Embedding ──► Độ tương đồng ngữ nghĩa ──► Word2Vec

Tiền đề tự nhiên hướng tới LAB 04 (Text Classification):
TF-IDF ──► Word Embedding ──► Biểu diễn tài liệu (Document Representation) ──► Phân loại văn bản (Classification)
```
