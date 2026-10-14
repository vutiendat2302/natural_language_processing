# BÁO CÁO PHẢN TƯ VÀ TỔNG KẾT NỀN TẢNG (LAB 04 - REFLECTION)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [12. Phân tích Nguyên lý Phòng chống Rò rỉ Dữ liệu (Data Leakage)](#12-phân-tích-nguyên-lý-phòng-chống-rò-rỉ-dữ-liệu-data-leakage)
   - [12.1. Quy trình Chuẩn mực (Correct Machine Learning Pipeline)](#121-quy-trình-chuẩn-mực-correct-machine-learning-pipeline)
   - [12.2. Lỗi Rò rỉ Dữ liệu Phổ biến: Fit Preprocessing trước khi Phân chia](#122-lỗi-rò-rỉ-dữ-liệu-phổ-biến-fit-preprocessing-trước-khi-phân-chia)
   - [12.3. Giải thích Bản chất Toán học: Vì sao đây là Data Leakage?](#123-giải-thích-bản-chất-toán-học-vì-sao-đây-là-data-leakage)
2. [16. So sánh Bản chất: TF-IDF vs Average Word2Vec](#16-so-sánh-bản-chất-tf-idf-vs-average-word2vec)
   - [16.1. Tại sao hai representation tạo ra kết quả phân loại khác nhau?](#161-tại-sao-hai-representation-tạo-ra-kết-quả-phân-loại-khác-nhau)
   - [16.2. Bảng Đối chiếu Chuyên sâu Đặc tính Kỹ thuật](#162-bảng-đối-chiếu-chuyên-sâu-đặc-tính-kỹ-thuật)
3. [17. Giới hạn của Bag-of-Words: Bài toán Thứ tự Từ (Word Order Experiment)](#17-giới-hạn-của-bag-of-words-bài-toán-thứ-tự-từ-word-order-experiment)
   - [17.1. Thí nghiệm Đối chứng: Câu Khẳng định vs Câu Phủ định](#171-thí-nghiệm-đối-chứng-câu-khẳng-định-vs-câu-phủ-định)
   - [17.2. Phân tích: Liệu Average Word Embedding có hiểu được vai trò của "not"?](#172-phân-tích-liệu-average-word-embedding-có-hiểu-được-vai-trò-của-not)
4. [18. Mở rộng Không gian Đặc trưng: Thí nghiệm N-gram (N-gram Features Experiment)](#18-mở-rộng-không-gian-đặc-trưng-thí-nghiệm-n-gram-n-gram-features-experiment)
   - [18.1. So sánh Định lượng: Unigram vs Unigram + Bigram](#181-so-sánh-định-lượng-unigram-vs-unigram--bigram)
   - [18.2. Giải thích: Vì sao Bigram cải thiện phân loại nhưng bùng nổ không gian đặc trưng?](#182-giải-thích-vì-sao-bigram-cải-thiện-phân-loại-nhưng-bùng-nổ-không-gian-đặc-trưng)
5. [25. Vấn đáp Kiểm tra Năng lực Cá nhân (Individual Learning Check)](#25-vấn-đáp-kiểm-tra-năng-lực-cá-nhân-individual-learning-check)
   - [Câu 1: Tại sao Accuracy 95% chưa chắc đã tốt?](#câu-1-tại-sao-accuracy-95-chưa-chắc-đã-tốt)
   - [Câu 2: TF-IDF khác Average Word2Vec ở những điểm cốt lõi nào?](#câu-2-tf-idf-khác-average-word2vec-ở-những-điểm-cốt-lõi-nào)
   - [Câu 3: Tại sao cụm từ "not good" có thể gây lỗi nghiêm trọng cho classifier?](#câu-3-tại-sao-cụm-từ-not-good-có-thể-gây-lỗi-nghiêm-trọng-cho-classifier)
   - [Câu 4: Hiện tượng Data Leakage cụ thể xảy ra ở đâu nếu fit TF-IDF trước khi split?](#câu-4-hiện-tượng-data-leakage-cụ-thể-xảy-ra-ở-đâu-nếu-fit-tf-idf-trước-khi-split)
   - [Câu 5: Sự khác biệt bản chất giữa Precision và Recall là gì?](#câu-5-sự-khác-biệt-bản-chất-giữa-precision-và-recall-là-gì)
   - [Câu 6: Tại sao bổ sung Bigram lại giúp cải thiện hiệu năng Sentiment Analysis?](#câu-6-tại-sao-bổ-sung-bigram-lại-giúp-cải-thiện-hiệu-năng-sentiment-analysis)
6. [29. Mạch Nối Tri Thức: Từ LAB 01 tới LAB 05 (Curriculum Trajectory)](#29-mạch-nối-tri-thức-từ-lab-01-tới-lab-05-curriculum-trajectory)
   - [29.1. Tiến trình Tiến hóa của 4 Lab Nền tảng](#291-tiến-trình-tiến-hóa-của-4-lab-nền-tảng)
   - [29.2. Hai Chân lý Giới hạn Lớn được Vạch trần tại Lab 04](#292-hai-chân-lý-giới-hạn-lớn-được-vạch-trần-tại-lab-04)
   - [29.3. Bước đệm Tất yếu Sang LAB 05: Sequence Models & Deep NLP](#293-bước-đệm-tất-yếu-sang-lab-05-sequence-models--deep-nlp)

---

# 12. Phân tích Nguyên lý Phòng chống Rò rỉ Dữ liệu (Data Leakage)

## 12.1. Quy trình Chuẩn mực (Correct Machine Learning Pipeline)

Trong mọi hệ thống Học máy có giám sát (Supervised Machine Learning), nguyên tắc tối thượng là: **Tập kiểm tra (Test set / Validation set) phải đóng vai trò là dữ liệu hoàn toàn chưa từng biết tới (unseen data)**, mô phỏng chính xác môi trường vận hành thực tế khi mô hình được triển khai ngoài đời thực.

Quy trình chuẩn hóa bắt buộc phải tuân thủ nghiêm ngặt thứ tự sau:

```text
       Toàn bộ tập dữ liệu thô (Raw Dataset)
                        │
                        ▼
       ┌─────────────────────────────────┐
       │   Train / Val / Test Split      │  (Phân chia dữ liệu trước tiên)
       └─────────────────────────────────┘
           │             │            │
           ▼             ▼            ▼
      Train Set       Val Set      Test Set
           │             │            │
           ▼             │            │
   ┌───────────────┐     │            │
   │ Fit Preproc.  │     │            │  (Chỉ học từ vựng và IDF trên TRAIN)
   └───────────────┘     │            │
           │             │            │
           ├─────────────┼────────────┤
           ▼             ▼            ▼
     Transform      Transform    Transform  (Dùng chung từ điển TRAIN để chiếu vector)
           │             │            │
           ▼             ▼            ▼
     Train Matrix    Val Matrix   Test Matrix
           │
           ▼
   ┌───────────────┐
   │  Train Model  │
   └───────────────┘
```

Trong mã nguồn [classification.ipynb](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab04/classification.ipynb), quy trình này được hiện thực hóa bằng:
```python
# 1. Phân chia dữ liệu độc lập trước
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = split_data(...)

# 2. CHỈ fit trên tập TRAIN
tfidf = TfidfVectorizer(min_df=2)
X_train_tfidf = tfidf.fit_transform(train_texts)

# 3. CHỈ transform trên VAL và TEST
X_val_tfidf = tfidf.transform(val_texts)
X_test_tfidf = tfidf.transform(test_texts)
```

---

## 12.2. Lỗi Rò rỉ Dữ liệu Phổ biến: Fit Preprocessing trước khi Phân chia

Một sai lầm rất phổ biến của người mới bắt đầu làm Học máy là áp dụng pipeline sai trái:
```text
  Entire Dataset ──► Fit & Transform TF-IDF ──► Train/Test Split ──► Train Model
```
Ở pipeline này, lập trình viên đưa toàn bộ văn bản của tập Train lẫn Test vào phương thức `.fit_transform()` cùng một lúc, sau đó mới lấy ma trận kết quả chia thành tập huấn luyện và kiểm tra.

---

## 12.3. Giải thích Bản chất Toán học: Vì sao đây là Data Leakage?

Áp dụng tiền xử lý lên toàn bộ tập dữ liệu trước khi phân chia cấu thành hành vi **Rò rỉ dữ liệu (Data Leakage)** nghiêm trọng vì hai nguyên nhân toán học cụ thể:

1. **Rò rỉ Tập từ vựng (Vocabulary Leakage)**:
   - Khi gọi `fit()` trên toàn bộ dữ liệu, tập từ vựng $\mathcal{V}$ của bộ vector hóa sẽ chứa tất cả các từ đặc thù (domain terms, tên riêng, từ mới) chỉ xuất hiện trong tập Test.
   - Trong kịch bản thực tế, mô hình sẽ gặp các từ chưa từng thấy trong tập train (Out-of-Vocabulary - OOV) và bắt buộc phải bỏ qua hoặc xử lý dưới dạng token đặc biệt. Việc "nhìn thấy trước" từ vựng của tập Test đã triệt tiêu hoàn toàn bài toán OOV, tạo ra một lợi thế gian lận cho mô hình.
2. **Rò rỉ Trọng số Nghịch đảo Tần số Văn bản (IDF Leakage)**:
   - Trọng số IDF của một từ $t$ được định nghĩa:
     $$\text{IDF}(t) = \ln \left( \frac{1 + N}{1 + \text{DF}(t)} \right) + 1$$
   - Khi tính toán trên toàn bộ dataset, tổng số tài liệu $N = N_{\text{train}} + N_{\text{test}}$, và tần số tài liệu $\text{DF}(t) = \text{DF}_{\text{train}}(t) + \text{DF}_{\text{test}}(t)$.
   - Điều này đồng nghĩa với việc phân phối thống kê toàn cục của tập Test đã bị thẩm thấu trực tiếp vào trọng số của từng từ trong tập Train. Độ quan trọng của các từ trong tập Train đã bị bóp méo dựa trên thông tin tương lai của tập Test.
3. **Hậu quả thực tiễn**:
   - Khiến các chỉ số đo lường (Accuracy, F1) trên tập Test trở nên "lạc quan giả tạo" (artificially inflated).
   - Khi mô hình được đóng gói và đưa vào phục vụ người dùng thực tế trong môi trường sản xuất (Production), hiệu năng sẽ suy giảm thảm hại (performance drop), gây rủi ro kinh doanh lớn.

---

# 16. So sánh Bản chất: TF-IDF vs Average Word2Vec

## 16.1. Tại sao hai representation tạo ra kết quả phân loại khác nhau?

Mục tiêu cốt lõi của nghiên cứu không phải chỉ dừng lại ở việc xem mô hình nào có Accuracy cao hơn, mà phải lý giải: **Tại sao cùng một bộ phân loại Logistic Regression nhưng hai phương pháp biểu diễn lại dẫn tới kết quả khác biệt rõ rệt?**

Kết quả thực nghiệm trên tập Test của IMDb cho thấy:
- **TF-IDF + Logistic Regression**: $\text{Accuracy} = \mathbf{89.98\%}$, $F_1 = \mathbf{0.9009}$
- **Average Word2Vec + Logistic Regression**: $\text{Accuracy} = \mathbf{85.06\%}$, $F_1 = \mathbf{0.8521}$

Sự chênh lệch gần **5%** bắt nguồn từ sự đối lập sâu sắc trong bản chất biểu diễn của hai không gian đặc trưng.

---

## 16.2. Bảng Đối chiếu Chuyên sâu Đặc tính Kỹ thuật

| Tiêu chí so sánh | Biểu diễn TF-IDF (Term Frequency - Inverse Document Frequency) | Biểu diễn Average Word2Vec (Dense Semantic Centroid) |
| :--- | :--- | :--- |
| **Bản chất toán học** | **Rời rạc, cực kỳ thưa (Sparse, Lexical representation)** | **Liên tục, mật độ dày (Dense, Continuous representation)** |
| **Kích thước không gian** | Số chiều rất lớn: $d = |\mathcal{V}| \approx 54,511$ chiều (hoặc $> 600,000$ khi thêm Bigram) | Số chiều cố định, nhỏ gọn: $d = 100$ chiều tiềm ẩn |
| **Cơ chế biểu diễn** | Mỗi chiều ứng với sự hiện diện và trọng số thống kê của một từ vựng độc lập | Mỗi từ là một vector liên tục; văn bản là trọng tâm hình học (vector trung bình) |
| **Lưu giữ thông tin từ vựng (Term Information)** | **Giữ trọn vẹn 100% danh tính từng từ khóa**. Từ nào xuất hiện sẽ bật sáng chiều tương ứng | **Bị hòa tan hoàn toàn**. Phép cộng trung bình san phẳng dấu vết của từng từ riêng lẻ |
| **Khả năng bắt sắc thái phân cực mạnh** | **Cực kỳ xuất sắc**. Logistic Regression chỉ cần gán trọng số lớn ($w > 8$) cho các từ như `great`, `worst` là đủ để quyết định nhãn | **Yếu**. Vector của từ cảm xúc bị cộng dồn và chia đều với hàng trăm vector từ trung tính khác trong bài review |
| **Khả năng nắm bắt tương đồng ngữ nghĩa** | **Kém**. Hai từ đồng nghĩa hoàn toàn (`doctor` và `physician`) bị xem như hai chiều trực giao độc lập | **Rất tốt**. Các từ ngữ cảnh tương tự sẽ nằm gần nhau trong không gian Euclid |
| **Ảnh hưởng của độ dài văn bản** | Ít bị ảnh hưởng nhờ chuẩn hóa vector L2 ($\|\vec{x}\|_2 = 1$) | **Bị suy thoái nghiêm trọng khi văn bản dài**. Càng nhiều từ, vector trung bình càng trôi về tâm nhiễu |
| **Thời gian huấn luyện** | Rất nhanh: **13.86 giây** cho toàn bộ pipeline | Chậm hơn: **55.60 giây** (phải duyệt từng từ để tra bảng embedding và tính trung bình) |

---

# 17. Giới hạn của Bag-of-Words: Bài toán Thứ tự Từ (Word Order Experiment)

## 17.1. Thí nghiệm Đối chứng: Câu Khẳng định vs Câu Phủ định

Trong [classification.ipynb](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab04/classification.ipynb) (Mục 17), chúng ta thực hiện thí nghiệm kiểm thử trên hai câu văn có cấu trúc đối lập nhau về ngữ nghĩa:
- **Câu $S_1$ (Khen ngợi)**: `"this movie is good"` $\implies$ Nhãn con người: **Positive**
- **Câu $S_2$ (Phê bình phủ định)**: `"this movie is not good"` $\implies$ Nhãn con người: **Negative**

Kết quả đo đạc thực nghiệm từ hai mô hình:

| Mô hình | Xác suất $P(\text{positive} \mid S_1)$ | Xác suất $P(\text{positive} \mid S_2)$ | Nhận xét hành vi dự đoán |
| :--- | :---: | :---: | :--- |
| **Model A (TF-IDF + LR)** | **0.8593** (Positive) | **0.5668** (Vẫn bị đoán là Positive!) | Bị lừa bởi từ khóa `good` |
| **Model B (Avg Word2Vec + LR)** | **0.9882** (Positive) | **0.5073** (Hoàn toàn phân vân / Bấp bênh) | Bị lừa bởi vector của `good` |

Đặc biệt, khoảng cách góc Cosine Similarity giữa vector biểu diễn của $S_1$ và $S_2$ trong mô hình Average Word2Vec đạt tới:
$$\text{Cosine Similarity}(\vec{x}_{S_1}, \vec{x}_{S_2}) = \mathbf{0.9581}$$

---

## 17.2. Phân tích: Liệu Average Word Embedding có hiểu được vai trò của "not"?

### Câu trả lời:
**HOÀN TOÀN KHÔNG HIỂU.**

### Luận giải cơ chế thất bại:
1. **Phép cộng vector giao hoán (Commutativity of Vector Addition)**:
   Vector đại diện cho văn bản được tính theo công thức:
   $$\vec{x} = \frac{1}{n} \sum_{i=1}^n \vec{w}_i$$
   Trong đó:
   $$\vec{x}_{S_1} = \frac{1}{4} (\vec{w}_{\text{this}} + \vec{w}_{\text{movie}} + \vec{w}_{\text{is}} + \vec{w}_{\text{good}})$$
   $$\vec{x}_{S_2} = \frac{1}{5} (\vec{w}_{\text{this}} + \vec{w}_{\text{movie}} + \vec{w}_{\text{is}} + \vec{w}_{\text{not}} + \vec{w}_{\text{good}})$$
2. **Vai trò ngữ pháp của từ phủ định bị biến thành một "phần tử cộng thêm"**:
   - Trong ngôn ngữ tự nhiên, từ `"not"` là một **toán tử logic (Operator)** có chức năng đảo ngược toàn bộ cực tính ngữ nghĩa của tính từ đứng ngay sau nó: $\text{NOT}(\text{good}) \equiv \text{bad}$.
   - Nhưng trong Average Word2Vec, `"not"` chỉ đơn thuần là một vector từ tĩnh $\vec{w}_{\text{not}}$ được cộng thêm vào tổng số, rồi chia cho 5. Nó không hề có khả năng "nhân", "đảo dấu" hay "thay đổi cấu trúc" vector của từ $\vec{w}_{\text{good}}$.
3. **Giải thích con số Cosine Similarity = 0.9581**:
   Vì 4 trên 5 từ của $S_2$ hoàn toàn trùng khớp với $S_1$, vector tổng của chúng chia sẻ phần lớn không gian phương hướng. Kết quả là mô hình xem hai câu trái ngược nhau về cảm xúc như hai câu gần như giống hệt nhau về ngữ nghĩa.

Đây là minh chứng trực quan, sắc bén nhất về **giới hạn không thể vượt qua của Bag-of-Words và Static Averaged Embeddings**.

---

# 18. Mở rộng Không gian Đặc trưng: Thí nghiệm N-gram (N-gram Features Experiment)

## 18.1. So sánh Định lượng: Unigram vs Unigram + Bigram

Để khắc phục phần nào điểm yếu mất thứ tự từ của Unigram, chúng ta mở rộng dải trích xuất đặc trưng sang cụm 2 từ liền kề (Bigram: `ngram_range=(1, 2)`).

Số liệu thực nghiệm ghi nhận tại `results.csv` và `classification.ipynb`:

| Cấu hình đặc trưng | Kích thước từ vựng ($|\mathcal{V}|$) | Thời gian huấn luyện (giây) | Accuracy (Test) | Precision (Test) | Recall (Test) | F1-Score (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unigram (1-gram)** | 54,511 | **10.30 s** | 89.98% | 89.40% | 90.80% | **90.09%** |
| **Uni + Bigram (1-gram + 2-gram)** | **614,975** | **26.55 s** | **90.18%** | 89.22% | **91.48%** | **90.34%** |

---

## 18.2. Giải thích: Vì sao Bigram cải thiện phân loại nhưng bùng nổ không gian đặc trưng?

### 1. Vì sao Bigram giúp ích cho Text Classification?
- **Nắm bắt trực tiếp cụm phủ định cục bộ**:
  Khi sử dụng Bigram, chuỗi `"not good"` không còn bị tách làm hai nửa mà trở thành một đặc trưng độc lập: token `not_good`. Mô hình Logistic Regression có thể học trực tiếp một trọng số âm mạnh cho đặc trưng này ($w_{\text{not\_good}} < 0$).
  - Bằng chứng thực tế tại Mục 23: Khi đưa câu `"this movie is not good"` vào pipeline Uni+Bigram, xác suất dự báo rơi tự do từ $0.5668$ xuống còn **$0.112$**, giúp mô hình dự đoán chính xác nhãn Negative.
- **Nắm bắt các cụm danh từ và cụm gia tăng cường độ**:
  Các cụm như `"waste of"`, `"must see"`, `"very good"`, `"terribly bad"` được giữ nguyên vẹn sắc thái ngữ nghĩa liên kết.

### 2. Vì sao Không gian Đặc trưng lại bùng nổ khủng khiếp (Feature Space Explosion)?
- **Quy luật tổ hợp nhân tử**:
  Nếu một tập ngữ liệu có kích thước từ vựng unigram là $V$, số lượng cặp 2 từ liên tiếp trên lý thuyết có thể đạt tới $V^2$.
  Mặc dù trên thực tế nhiều cặp từ không xuất hiện, số lượng bigram phân biệt trong IMDb vẫn khiến từ vựng vọt từ **54,511** lên tới **614,975** (tăng gấp **11.28 lần**).
- **Hệ quả của sự bùng nổ**:
  - Không gian ma trận trở nên siêu thưa (ultra-sparse).
  - Chi phí bộ nhớ RAM tăng vọt.
  - Thời gian huấn luyện tăng gần gấp 3 lần ($10.30s \to 26.55s$).
  - Mức tăng hiệu năng chỉ ở mức khiêm tốn ($+0.20\%$ Accuracy, $+0.25\%$ F1). Hiện tượng này phản ánh sự suy giảm hiệu suất biên (diminishing marginal returns) khi mở rộng chiều đặc trưng tuyến tính.

---

# 25. Vấn đáp Kiểm tra Năng lực Cá nhân (Individual Learning Check)

Dưới đây là câu trả lời chuẩn mực, súc tích dành cho 6 câu hỏi ngẫu nhiên trong bài kiểm tra vấn đáp 3 phút:

---

### Câu 1: Tại sao Accuracy 95% chưa chắc đã tốt?
- **Trả lời**:  
  Bởi vì trên các tập dữ liệu bị **mất cân bằng lớp nghiêm trọng (Class Imbalance)**, một mô hình ngây thơ chỉ cần luôn luôn dự đoán nhãn của lớp đa số là đã có thể đạt Accuracy 95% mà không cần học bất kỳ đặc trưng nào (ví dụ: tập dữ liệu có 950 mẫu âm và 50 mẫu dương).
  Mô hình này hoàn toàn vô dụng trên thực tế vì tỷ lệ nhận diện lớp thiểu số (Recall và F1 của lớp Positive) bằng **0%**, khiến toàn bộ các ca mục tiêu quan trọng (như bệnh nhân ung thư, giao dịch gian lận ngân hàng) bị bỏ sót hoàn toàn.

---

### Câu 2: TF-IDF khác Average Word2Vec ở những điểm cốt lõi nào?
- **Trả lời**:  
  Khác nhau ở 3 điểm cốt lõi:
  1. **Tính thưa vs Tính dày**: TF-IDF biểu diễn văn bản trong không gian cực kỳ thưa (sparse) với hàng chục ngàn chiều; trong khi Average Word2Vec là không gian số chiều thấp, dày đặc (dense, thường 100-300 chiều).
  2. **Bản chất từ ngữ vs Ngữ nghĩa**: TF-IDF giữ chính xác danh tính từng từ vựng (lexical identity), rất nhạy với các từ chỉ cảm xúc mạnh; còn Word2Vec biểu diễn tương đồng ngữ nghĩa phân phối (distributional semantics).
  3. **Hành vi khi văn bản dài**: TF-IDF chuẩn hóa chiều dài tốt; còn Average Word2Vec bị hiện tượng pha loãng thông tin (dilution) do phép lấy trung bình cộng san phẳng mọi đỉnh tín hiệu cảm xúc.

---

### Câu 3: Tại sao cụm từ "not good" có thể gây lỗi nghiêm trọng cho classifier?
- **Trả lời**:  
  Bởi vì các mô hình Bag-of-Words (Unigram TF-IDF hoặc Average Word2Vec) hoàn toàn **bỏ qua thứ tự xuất hiện của từ (loss of word order)**.
  Khi tách từ, `"not"` và `"good"` bị xử lý độc lập. Từ `"good"` có trọng số dương rất lớn trong mô hình, trong khi từ `"not"` chỉ là một từ bổ trợ thông thường. Do đó, điểm số dương tính của `"good"` sẽ lấn át, khiến mô hình phân loại sai một câu chê tiêu cực thành câu khen tích cực.

---

### Câu 4: Hiện tượng Data Leakage cụ thể xảy ra ở đâu nếu fit TF-IDF trước khi split?
- **Trả lời**:  
  Hiện tượng rò rỉ xảy ra ở **hai vị trí toán học**:
  1. **Rò rỉ Từ vựng (Vocabulary Leakage)**: Bộ trích xuất biết trước tất cả các từ đặc thù chỉ có trong tập Test, loại bỏ một cách gian lận bài toán xử lý từ chưa biết (Out-Of-Vocabulary) ngoài thực tế.
  2. **Rò rỉ Thống kê Toàn cục (IDF Leakage)**: Công thức $\text{IDF} = \ln((1+N)/(1+\text{DF}))$ tính toán trên toàn bộ dữ liệu sẽ đưa thông tin về tần số xuất hiện của các từ trong tập Test vào trọng số của tập Train, làm sai lệch phân phối thực nghiệm.

---

### Câu 5: Sự khác biệt bản chất giữa Precision và Recall là gì?
- **Trả lời**:  
  - **Precision (Độ chuẩn xác)**: Đo lường trong số các mẫu mà mô hình *tuyên bố là Positive*, có bao nhiêu mẫu *thực sự là Positive* ($\frac{TP}{TP+FP}$). Phản ánh cái giá của việc báo động nhầm (False Positive).
  - **Recall (Độ nhạy / Độ bao phủ)**: Đo lường trong số tất cả các mẫu *thực sự là Positive*, mô hình *bắt được bao nhiêu mẫu* ($\frac{TP}{TP+FN}$). Phản ánh cái giá của việc bỏ sót mục tiêu (False Negative).
  - Hai chỉ số này mâu thuẫn nhau thông qua ngưỡng quyết định $\tau$: Tăng $\tau$ giúp tăng Precision nhưng giảm Recall, và ngược lại.

---

### Câu 6: Tại sao bổ sung Bigram lại giúp cải thiện hiệu năng Sentiment Analysis?
- **Trả lời**:  
  Bởi vì Bigram (cụm 2 từ liền kề) giúp mô hình **nắm bắt được cấu trúc phủ định cục bộ và ngữ cảnh ghép đôi**.
  Thay vì tách rời, mô hình có thêm các đặc trưng chuyên biệt như `"not good"`, `"never watch"`, `"very bad"`, `"must see"`. Nhờ đó, mô hình có thể gán trọng số phân loại riêng biệt cho cụm phủ định mà không bị nhầm lẫn với ý nghĩa của từng từ đơn lẻ.

---

# 29. Mạch Nối Tri Thức: Từ LAB 01 tới LAB 05 (Curriculum Trajectory)

## 29.1. Tiến trình Tiến hóa của 4 Lab Nền tảng

Nhìn lại toàn bộ hành trình từ đầu học kỳ, cấu trúc chương trình học của môn NLP được thiết kế theo một lộ trình tuyến tính, chặt chẽ và nhất quán:

```text
 ┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
 │    LAB 01     │     │    LAB 02     │     │    LAB 03     │     │    LAB 04     │
 │  Text to TFIDF│ ──► │  Text to Ngram│ ──► │Text to Embed  │ ──► │Text to Classif│
 │ Sparse Search │     │ Probability LM│     │ Semantic Sim. │     │ Supervised NLP│
 └───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
```

1. **LAB 01 (Biểu diễn Thưa)**: Học cách đếm tần số và đánh trọng số nghịch đảo tài liệu ($TF \times IDF$). Văn bản trở thành các vector thưa trong không gian trực giao để giải quyết bài toán tìm kiếm và truy hồi thông tin (Search & Information Retrieval).
2. **LAB 02 (Mô hình Ngôn ngữ Thống kê)**: Sử dụng chuỗi Markov và ước lượng hợp lý cực đại ($N$-gram LM) để mô hình hóa xác suất xuất hiện của chuỗi từ, đo lường độ trôi chảy bằng Perplexity và sinh văn bản tự động.
3. **LAB 03 (Biểu diễn Dày Phân phối)**: Nhận thức hạn chế của tính trực giao (orthogonality), chuyển sang không gian vector dày (Dense Embeddings - Word2Vec Co-occurrence) dựa trên Giả thuyết phân phối của Harris & Firth: *"Các từ cùng ngữ cảnh sẽ có vector gần nhau"*.
4. **LAB 04 (Ứng dụng Học máy Có giám sát)**: Mang toàn bộ các biểu diễn trên đặt vào bài toán phân loại có nhãn (Supervised Text Classification), kiểm thử toàn diện quy trình Machine Learning: Pipeline không rò rỉ, đánh giá đa chiều (Confusion Matrix, Precision/Recall/F1), phân tích lỗi thực tế và giải thích trọng số mô hình.

---

## 29.2. Hai Chân lý Giới hạn Lớn được Vạch trần tại Lab 04

Đóng góp khoa học quan trọng nhất của LAB 04 không phải là huấn luyện được một mô hình có Accuracy 90%, mà là giúp sinh viên nhận thức sâu sắc **hai ranh giới bất khả kháng** của các kỹ thuật biểu diễn cổ điển:

$$\Large \mathbf{TF\text{-}IDF \neq Semantic\ Understanding}$$
*TF-IDF chỉ là sự trùng khớp từ vựng bề mặt (Lexical Matching). Nó hoàn toàn không hiểu được từ đồng nghĩa, từ trái nghĩa hay sự biến đổi linh hoạt của ngôn ngữ.*

$$\Large \mathbf{Word2Vec \neq Contextual\ Understanding}$$
*Word2Vec tĩnh chỉ gán một vector bất biến cho mỗi từ và phép cộng trung bình vector làm mất hoàn toàn thứ tự từ lẫn phạm vi ngữ pháp của từ phủ định.*

Các ví dụ minh chứng kinh điển:
- **Hiện tượng phủ định đảo chiều**:
  `"I like this movie"` $\Longleftrightarrow$ `"I do not like this movie"`
  $\to$ Bag-of-words và Average Word2Vec bất lực trước vai trò của `"not"`.
- **Hiện tượng đa nghĩa theo ngữ cảnh (Polysemy)**:
  `"bank"` trong `"deposit money in the bank"` (ngân hàng) vs `"sit on the river bank"` (bờ sông).
  $\to$ Static embedding bị trôi dạt vào trung điểm vô nghĩa của hai trường nghĩa.

---

## 29.3. Bước đệm Tất yếu Sang LAB 05: Sequence Models & Deep NLP

Chính hai giới hạn căn bản trên đã giải thích tại sao ngành Xử lý Ngôn ngữ Tự nhiên bắt buộc phải chuyển mình:

```text
       LAB 04: Biểu diễn Tĩnh (Static / Bag-of-Words / Average)
                                  │
                                  ▼  (Nhu cầu mô hình hóa thứ tự và ngữ cảnh)
       LAB 05: Mô hình Chuỗi (Sequence Models: RNN / LSTM / GRU)
                                  │
                                  ▼  (Nhu cầu giải quyết phụ thuộc xa & song song hóa)
       LAB 06+: Transformer & Contextual Representations (Self-Attention / BERT)
```

**LAB 05 — Sequence Models** sẽ đưa sinh viên tiếp cận với **Mạng nơ-ron hồi quy (Recurrent Neural Networks - RNN, LSTM, GRU)**. Ở đó:
- Văn bản không còn bị nén thành một vector tĩnh thô thiển.
- Thay vào đó, chuỗi từ được duyệt tuần tự từng bước thời gian ($t = 1, \dots, T$). Trạng thái ẩn $h_t$ cập nhật liên tục thông qua các cổng kiểm soát thông tin (Gates), cho phép mô hình:
  1. Ghi nhớ thứ tự trước sau của các từ.
  2. Bắt trọn vẹn phạm vi tác động của từ phủ định `"not"`.
  3. Mô hình hóa phụ thuộc cú pháp phức tạp trong câu văn.

Đây là nền tảng tri thức sống còn trước khi tiến tới các kiến trúc đột phá hiện đại nhất như **Cơ chế chú ý (Self-Attention)** và **Mô hình Transformer (BERT, GPT)** trong nửa sau của chương trình môn học.
