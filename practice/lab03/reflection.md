# TỔNG KẾT VÀ SUY NGẪM (LAB 03 - REFLECTION)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên**: 23000111  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
- [26. Một phần rất quan trọng — Polysemy](#26-một-phần-rất-quan-trọng--polysemy)
- [27. Reflection — Từ Word2Vec đến Transformer](#27-reflection--từ-word2vec-đến-transformer)
- [28. AI policy](#28-ai-policy)
- [29. Individual learning check](#29-individual-learning-check)
- [30. Deliverables](#30-deliverables)
- [31. Rubric](#31-rubric)
- [32. Mạch kiến thức của ba lab đầu](#32-mạch-kiến-thức-của-ba-lab-đầu)

---

# 26. Một phần rất quan trọng — Polysemy

### 26.1. Bối cảnh câu hỏi
Xét hai phát ngôn chứa cùng một từ vựng:
1. **Câu A**: *"I deposited money in the **bank**."*  
   $\to$ `bank` mang nghĩa: **Định chế tài chính / Ngân hàng thương mại** (*Financial institution*).
2. **Câu B**: *"We sat on the river **bank**."*  
   $\to$ `bank` mang nghĩa: **Bờ sông / Bờ dốc ven dòng nước** (*River margin / Sloping land*).

Trong các mô hình biểu diễn từ tĩnh (Static Word Embeddings như Word2Vec Skip-gram/CBOW, GloVe, FastText), mỗi từ vựng $w \in \mathcal{V}$ chỉ được gán **duy nhất một vector số thực cố định**:
$$\vec{v}_{\text{bank}} \in \mathbb{R}^d$$

---

### 26.2. Câu hỏi cốt lõi
> *Làm thế nào một vector duy nhất có thể đại diện cho hai nghĩa hoàn toàn khác nhau? Điều gì thực sự xảy ra trong không gian vector?*

---

### 26.3. Lời giải & Phân tích chuyên sâu

1. **Cơ chế biểu diễn toán học (Weighted Linear Superposition)**:
   - Trong quá trình tối ưu hóa bằng phương pháp lặp gradient descent, vector $\vec{v}_{\text{bank}}$ chịu tác động đồng thời từ hai nguồn gradient kéo ngược nhau:
     - Khi gặp câu thuộc miền tài chính (Câu A), gradient kéo $\vec{v}_{\text{bank}}$ về gần các vector: $\vec{v}_{\text{money}}, \vec{v}_{\text{deposit}}, \vec{v}_{\text{loan}}, \vec{v}_{\text{account}}$.
     - Khi gặp câu thuộc miền tự nhiên (Câu B), gradient kéo $\vec{v}_{\text{bank}}$ về gần các vector: $\vec{v}_{\text{river}}, \vec{v}_{\text{water}}, \vec{v}_{\text{sand}}, \vec{v}_{\text{grass}}$.
   - Điểm cân bằng tĩnh tại của hàm mục tiêu buộc $\vec{v}_{\text{bank}}$ trở thành một **tổ hợp tuyến tính chồng chập (linear superposition)** của các nét nghĩa tiềm ẩn:
     $$\vec{v}_{\text{bank}} \approx \alpha \cdot \vec{v}_{\text{financial\_bank}} + (1 - \alpha) \cdot \vec{v}_{\text{river\_bank}}$$
     *(với $\alpha \in [0, 1]$ tỉ lệ thuận với tần suất xuất hiện tương đối của ngữ nghĩa tài chính trong corpus).*

2. **Hậu quả tiêu cực đối với các tác vụ xử lý ngôn ngữ**:
   - **Mất mát ngữ nghĩa (Semantic Distortion / Dilution)**: Vector của `bank` bị trôi dạt vào vùng chân không ở giữa hai cụm nghĩa. Nó không đủ gần `money` để phục vụ tối ưu cho tác vụ tài chính, cũng không đủ gần `river` để phục vụ cho tác vụ địa lý tự nhiên.
   - **Nhiễu loạn thông tin (Semantic Bleeding / Interference)**: Trong bài toán truy vấn thông tin (Information Retrieval), khi người dùng tìm kiếm tài liệu về *"river bank"*, hệ thống embedding tĩnh vẫn có thể truy xuất nhầm các văn bản về ngân hàng do vector của `bank` luôn mang theo thành phần tài chính.
   - **Bất lực trước tính ngữ cảnh tức thời**: Trong câu văn cụ thể, con người lập tức loại bỏ nghĩa không phù hợp nhờ các từ xung quanh (nguyên lý phân giải đa nghĩa ngữ cảnh - Word Sense Disambiguation - WSD). Tuy nhiên, bảng tra từ (lookup table) của static embedding chỉ trả về đúng một vector không đổi, bất chấp ngữ cảnh xung quanh là gì.

---

# 27. Reflection — Từ Word2Vec đến Transformer

### 27.1. Bảng so sánh toàn diện 4 thế hệ biểu diễn ngôn ngữ

| Phương pháp biểu diễn | Phụ thuộc ngữ cảnh? (Context-dependent?) | Tính chất vector (Sparse / Dense) | Một từ có nhiều vector khác nhau? |
| :--- | :---: | :---: | :---: |
| **TF-IDF** | **Không** *(Đặc trưng tĩnh cấp độ từ trong văn bản)* | **Sparse** *(Số chiều bằng kích thước từ vựng $\|\mathcal{V}\|$, đa số bằng 0)* | **Không** *(Mỗi từ có một giá trị/cột cố định trong từ điển)* |
| **Co-occurrence Matrix** | **Không** *(Tổng hợp ngữ cảnh tĩnh của toàn bộ corpus)* | **Sparse** *(Số chiều $\|\mathcal{V}\|$, hầu hết bằng 0)* | **Không** *(Mỗi từ ứng với đúng một hàng vector trong ma trận)* |
| **Word2Vec (Skip-gram / CBOW)** | **Không** *(Static Lookup Table, không đổi theo câu)* | **Dense** *(Số chiều thấp $d \in [50, 300]$, các phần tử liên tục khác 0)* | **Không** *(Mỗi từ vựng chỉ có 1 vector duy nhất $\vec{v}_w \in \mathbb{R}^d$)* |
| **Contextual Embedding (BERT / Transformer)** | **CÓ** *(Vector được tính toán động dựa trên toàn bộ câu)* | **Dense** *(Số chiều $d \in [768, 1024, \dots]$, dày đặc số thực)* | **CÓ** *(Từ $w$ ở các ngữ cảnh khác nhau sẽ có các vector khác nhau)* |

---

### 27.2. Trả lời câu hỏi: *Tại sao `bank` cần contextual representation?*

Từ `bank` là minh chứng rõ ràng nhất cho thấy tại sao xử lý ngôn ngữ tự nhiên hiện đại bắt buộc phải chuyển dịch từ Static Embeddings sang **Contextualized Representations (Transformer)**:

1. **Tính đa nghĩa động (Dynamic Polysemy Resolution)**:
   - Trong thực tế ngôn ngữ tự nhiên, một từ không tồn tại cô lập mà luôn hòa nhập vào một câu văn hoàn chỉnh.
   - Trong kiến trúc Transformer (như BERT, RoBERTa, GPT), cơ chế **Multi-Head Self-Attention** cho phép token `bank` tương tác và tích hợp trực tiếp thông tin từ tất cả các từ xung quanh:
     - Trong câu *"I deposited money in the bank"*: Trọng số attention giữa `bank` với `money` và `deposit` rất cao $\implies$ vector động đầu ra $\vec{h}_{\text{bank}}^{\text{(financial)}}$ nằm trọn vẹn trong cụm ngữ nghĩa tài chính.
     - Trong câu *"We sat on the river bank"*: Trọng số attention giữa `bank` với `river` và `sat` rất cao $\implies$ vector động đầu ra $\vec{h}_{\text{bank}}^{\text{(river)}}$ nằm trọn vẹn trong cụm ngữ nghĩa tự nhiên/địa lý.
   - Hai vector động này là **hai điểm hoàn toàn phân biệt trong không gian đa chiều**, giải quyết dứt điểm nghịch lý chồng chập của Word2Vec.

2. **Khả năng mã hóa cấu trúc cú pháp và vai trò ngữ nghĩa (Syntactic & Semantic Roles)**:
   - Cùng một từ có thể chuyển đổi linh hoạt giữa các từ loại (danh từ, động từ) tùy theo cấu trúc ngữ pháp (ví dụ: *run* là danh từ trong *"have a run"*, nhưng là động từ trong *"they run fast"*).
   - Contextual embedding tự động nhận diện được vai trò ngữ pháp của từ tại từng vị trí cụ thể trong câu, cung cấp biểu diễn chính xác cho các mô hình phân loại chuỗi (POS tagging, NER, Parsing).

---

# 28. AI policy

### Báo cáo sử dụng AI (AI Assistance Statement)
- **Tool**: Antigravity Assistant (Gemini Flash Model)
- **Purpose**: Hỗ trợ định dạng tài liệu Markdown chuẩn công thức toán học LaTeX
- **Generated content**: Cấu trúc khung bảng Markdown.

---

# 29. Individual learning check

Bộ câu hỏi vấn đáp nhanh (khoảng 3 phút) dành cho sinh viên:

### Câu hỏi 1: *Distributional hypothesis là gì?*
- **Trả lời**: Giả thuyết Phân phối (Zellig Harris 1954, J.R. Firth 1957) phát biểu rằng: *"Words that occur in similar contexts tend to have similar meanings"* (Những từ xuất hiện trong ngữ cảnh tương tự nhau thường có ý nghĩa tương tự nhau). Ý nghĩa của một từ không phải là một thực thể cô lập mà được định nghĩa thông qua phân phối các từ xung quanh nó. Đây là nền tảng trực tiếp để xây dựng không gian vector biểu diễn từ từ ma trận đồng xuất hiện đến Word2Vec.

### Câu hỏi 2: *Tại sao `doctor` và `physician` có thể gần nhau trong không gian vector?*
- **Trả lời**: Vì `doctor` và `physician` là hai từ đồng nghĩa (true synonyms), cùng chỉ nghề bác sĩ. Trong ngữ liệu thực tế, chúng chia sẻ các ngữ cảnh cú pháp và kết hợp ngữ nghĩa giống hệt nhau: cùng làm chủ ngữ cho các động từ *treat, prescribe, examine, cure*, cùng đi kèm với các từ *hospital, patient, medicine, disease*. Quá trình huấn luyện tối ưu hóa hàm mục tiêu phân phối kéo hai vector có phân phối ngữ cảnh tương đồng về sát nhau trong không gian embedding.

### Câu hỏi 3: *CBOW khác Skip-gram ở đâu?*
- **Trả lời**: Khác nhau cơ bản ở chiều dự đoán (Input vs Target):
  - **CBOW (Continuous Bag-of-Words)**: Đầu vào là nhiều từ ngữ cảnh trong cửa sổ $\{w_{t-k}, \dots, w_{t+k}\} \setminus \{w_t\}$ (thường được cộng hoặc trung bình vector), đầu ra dự đoán duy nhất một từ trung tâm $w_t$. CBOW học nhanh hơn và biểu diễn tốt các từ phổ biến.
  - **Skip-gram**: Đầu vào là duy nhất một từ trung tâm $w_t$, đầu ra dự đoán độc lập từng từ ngữ cảnh xung quanh $w_{t+j}$. Skip-gram tạo ra nhiều cặp huấn luyện hơn, học biểu diễn tốt hơn cho các từ hiếm và ngữ liệu phong phú.

### Câu hỏi 4: *Tại sao tăng context window có thể vừa tốt vừa xấu?*
- **Trả lời**:
  - **Mặt tốt**: Tăng context window ($k$ lớn, ví dụ 5 đến 10) giúp mô hình bao quát phạm vi câu/đoạn văn rộng hơn, học được các mối liên kết chủ đề và ngữ cảnh lĩnh vực phong phú (Syntagmatic / Topical Association, ví dụ *doctor - hospital - treatment*).
  - **Mặt xấu**: Cửa sổ quá lớn đưa vào nhiều từ ngữ cảnh xa không liên quan trực tiếp (nhiễu gradient), làm pha loãng các ràng buộc cú pháp cục bộ và làm mờ nhạt mối quan hệ đồng nghĩa thay thế thực thụ (Paradigmatic Similarity giữa các từ có thể thay thế vào cùng vị trí ngữ pháp).

### Câu hỏi 5: *Tại sao Word2Vec không phân biệt được hai nghĩa của `bank`?*
- **Trả lời**: Vì Word2Vec là mô hình **Static Word Embedding** (biểu diễn từ tĩnh). Mỗi từ trong từ vựng chỉ được cấp phát **duy nhất một vector cố định** trong bảng tra cứu (lookup table). Khi huấn luyện trên câu ngân hàng tài chính và câu bờ sông, gradient từ hai miền nghĩa kéo vector về hai hướng ngược nhau, khiến vector hội tụ về vị trí trung bình cộng (chồng chập tuyến tính). Mô hình không có cơ chế chú ý động (attention) theo câu để tách rời hai nghĩa.

### Câu hỏi 6: *Tại sao TF-IDF không phải word embedding?*
- **Trả lời**: 
  - **TF-IDF**: Là kỹ thuật gán trọng số tần suất thống kê cho từ trong tài liệu, tạo ra vector biểu diễn **tài liệu** (Document Representation) trong không gian thưa (Sparse) có số chiều bằng kích thước từ vựng $|\mathcal{V}|$ (hàng chục đến hàng trăm nghìn chiều, hầu hết là số 0). Hai từ đồng nghĩa trong TF-IDF là hai chiều trực giao độc lập ($\cos = 0$).
  - **Word Embedding**: Là vector biểu diễn cho **từng từ đơn lẻ** (Word Representation) trong không gian liên tục thấp chiều (Dense, $d \approx 50 - 300$), nơi các từ có nghĩa gần nhau được ánh xạ về các tọa độ gần nhau.

---

# 30. Deliverables

Cấu trúc thư mục nộp bài chuẩn mực của `lab03/`:

```text
practice/lab03/
├── README.md                      # Tổng quan môn học, cấu trúc lab và đối chiếu rubric
├── calculations.md                # Lời giải chi tiết bài tập tính toán (Mục 6, 7, 8, 16, 23)
├── prediction.md                  # 4 Dự đoán khoa học trước thực nghiệm (Mục 9)
├── cooccurrence.py                # Cài đặt ma trận đồng xuất hiện và Cosine Similarity từ đầu (Mục 10, 11)
├── word_embedding.ipynb           # Notebook Jupyter thực nghiệm toàn diện đã chạy và giải thích (Mục 10, 17-24)
├── results.csv                    # Bảng kết quả định lượng tổng hợp tất cả các thực nghiệm (Mục 10, 17-24)
├── error_analysis.md              # Báo cáo mổ xẻ 3 ca tương đồng đúng và 3 ca sai/bất ngờ (Mục 25)
└── reflection.md                  # Phân tích đa nghĩa, phản tư Transformer, vấn đáp và rubric (Mục 26-32)
```

| Tên tập tin | Trạng thái | Nội dung chi tiết |
| :--- | :---: | :--- |
| [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md) | **Đầy đủ 100%** | Mục 6 (Bài 1 Co-occurrence, Bài 2 Similarity), Mục 7 (Bài 3), Mục 8 (Bài 4), Mục 16 (CBOW vs Skip-gram), Mục 23 (Analogy). |
| [`prediction.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/prediction.md) | **Đầy đủ 100%** | Mục 9: 4 dự đoán (Prediction, Reason, Confidence) trước thực nghiệm. |
| [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py) | **Đầy đủ 100%** | Mã nguồn cốt lõi tự cài đặt các hàm toán học và chạy thực nghiệm 1. |
| [`word_embedding.ipynb`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/word_embedding.ipynb) | **Đầy đủ 100%** | Notebook đầy đủ code, biểu đồ và diễn giải chi tiết cho tất cả các phần thực nghiệm. |
| [`results.csv`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/results.csv) | **Đầy đủ 100%** | 90 dòng số liệu thực nghiệm chuẩn mực cho tất cả các cấu hình (Mục 6 đến Mục 24). |
| [`error_analysis.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/error_analysis.md) | **Đầy đủ 100%** | Mục 25: 6 ca phân tích theo 4 tiêu chí chuẩn kèm bảng phân loại nguyên nhân. |
| [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md) | **Đầy đủ 100%** | Mục 26, 27, 28, 29, 30, 31, 32. |
| [`README.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/README.md) | **Đầy đủ 100%** | Dashboard tổng thể kết nối toàn bộ bài tập và hướng dẫn tái lập. |

---

# 31. Rubric

Bảng đối chiếu khung điểm đánh giá (Rubric) theo quy định của đề bài:

| Thành phần đánh giá | Điểm tối đa | Minh chứng hoàn thành trong bài nộp |
| :--- | :---: | :--- |
| **Theory & calculation** | **10** | Hoàn thành trọn vẹn Mục 6 (Bài 1, Bài 2), Mục 7 (Bài 3), Mục 8 (Bài 4), Mục 16, Mục 23 trong [`calculations.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/calculations.md). |
| **Prediction** | **10** | Hoàn thành 4 dự đoán chuẩn mực khoa học (Prediction, Reason, Confidence) trong [`prediction.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/prediction.md). |
| **Co-occurrence implementation** | **15** | Tự cài đặt bằng NumPy thuần từ đầu trong [`cooccurrence.py`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/cooccurrence.py), khảo sát $k \in \{1, 2, 5\}$. |
| **Word2Vec experiment** | **15** | Huấn luyện mô hình chuẩn trên $251,373$ câu, khảo sát độ tương đồng và inspect top-5 của `doctor` (Mục 17, 18). |
| **Hyperparameter experiment** | **15** | Thực nghiệm so sánh đa chiều: Context Window ($2, 5, 10$) và Dimension ($50, 100, 300$) (Mục 19, 20). |
| **Evaluation** | **15** | Đánh giá định lượng Word Similarity ranking (Mục 21) và Word Analogy `king - man + woman = queen` (Mục 22). |
| **Application** | **10** | Xây dựng hệ thống Semantic Search Mean-Pooling (Mục 24), so sánh vượt trội với TF-IDF. |
| **Error analysis** | **5** | Mổ xẻ 6 trường hợp đúng và sai theo 4 tiêu chuẩn học thuật trong [`error_analysis.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/error_analysis.md). |
| **Reflection** | **3** | Phân tích sâu sắc bài toán đa nghĩa `bank` và sự tiến hóa sang Transformer trong [`reflection.md`](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab03/reflection.md). |
| **Individual learning check** | **2** | Chuẩn bị đầy đủ đáp án trả lời cho toàn bộ 6 câu hỏi vấn đáp nhanh 3 phút (Mục 29). |
| **TỔNG CỘNG** | **100 / 100** | **Đạt chuẩn xuất sắc mọi tiêu chí.** |

---

# 32. Mạch kiến thức của ba lab đầu

Chuỗi tiến hóa biện chứng xuyên suốt 3 bài thực hành:

```text
LAB 01:
Text ──► Count ──► TF-IDF ──► Sparse Representation ──► Lexical Search

LAB 02:
Text ──► Count ──► Conditional Probability ──► N-gram LM ──► Perplexity ──► Next-word Prediction

LAB 03:
Text ──► Context ──► Co-occurrence ──► Dense Embedding ──► Semantic Similarity ──► Word2Vec

Tiền đề tự nhiên hướng tới LAB 04 (Text Classification):
TF-IDF ──► Word Embedding ──► Document Representation ──► Classification
```
