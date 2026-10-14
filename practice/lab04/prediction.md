# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM (LAB 04 - PREDICTION)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [5. Dự đoán về Biểu diễn Văn bản: TF-IDF vs Average Word2Vec](#5-dự-đoán-về-biểu-diễn-văn-bản-tf-idf-vs-average-word2vec)
   - [5.1. Hai classifier giống nhau nhưng representation khác nhau có cho kết quả giống nhau không?](#51-hai-classifier-giống-nhau-nhưng-representation-khác-nhau-có-cho-kết-quả-giống-nhau-không)
   - [5.2. Lựa chọn Representation tối ưu cho Dataset IMDb Sentiment Analysis](#52-lựa-chọn-representation-tối-ưu-cho-dataset-imdb-sentiment-analysis)
2. [10. Năm Dự đoán Khoa học trước Experiment](#10-năm-dự-đoán-khoa-học-trước-experiment)
   - [10.1. Prediction 1 — Hiệu năng của TF-IDF + Logistic Regression](#101-prediction-1--hiệu-năng-của-tf-idf--logistic-regression)
   - [10.2. Prediction 2 — Liệu Average Word2Vec có chắc chắn tốt hơn TF-IDF?](#102-prediction-2--liệu-average-word2vec-có-chắc-chắn-tốt-hơn-tf-idf)
   - [10.3. Prediction 3 — Tăng Vocabulary Size có chắc chắn làm tăng Accuracy?](#103-prediction-3--tăng-vocabulary-size-có-chắc-chắn-làm-tăng-accuracy)
   - [10.4. Prediction 4 — Loại bỏ Stopwords có chắc chắn cải thiện Performance?](#104-prediction-4--loại-bỏ-stopwords-có-chắc-chắn-cải-thiện-performance)
   - [10.5. Prediction 5 — Accuracy và F1 có thể dẫn tới kết luận trái ngược nhau?](#105-prediction-5--accuracy-và-f1-có-thể-dẫn-tới-kết-luận-trái-ngược-nhau)
3. [Bảng Tổng hợp Dự đoán & Đối chiếu Kết quả Thực nghiệm Thực tế](#bảng-tổng-hợp-dự-đoán--đối-chiếu-kết-quả-thực-nghiệm-thực-tế)

---

# 5. Dự đoán về Biểu diễn Văn bản: TF-IDF vs Average Word2Vec

## 5.1. Hai classifier giống nhau nhưng representation khác nhau có cho kết quả giống nhau không?

### Trả lời:
**HOÀN TOÀN KHÔNG GIỐNG NHAU.**

### Cơ sở lý thuyết:
Mô hình phân loại (chẳng hạn Logistic Regression) hoạt động trên không gian đặc trưng đầu vào $\mathcal{X} \subseteq \mathbb{R}^d$. Khi thay đổi phương pháp trích xuất đặc trưng (Text Representation), bản chất hình học, số chiều và phân phối xác suất của không gian $\mathcal{X}$ bị biến đổi hoàn toàn:
- **TF-IDF**: Tạo ra không gian **thưa cực cao (high-dimensional, highly sparse)** với $d \approx 10^4 - 10^5$, trong đó mỗi chiều đại diện cho một token cụ thể (lexical feature). Phép phân cách tuyến tính tìm kiếm siêu phẳng dựa trên sự hiện diện hoặc vắng mặt của các từ chỉ sắc thái mang tính quyết định.
- **Average Word2Vec**: Tạo ra không gian **đặc, số chiều thấp (low-dimensional, dense)** với $d \approx 100 - 300$, trong đó mỗi văn bản được nén thành trọng tâm của các vector từ (semantic centroid). Không gian này biểu diễn tương đồng ngữ nghĩa tổng thể nhưng làm phẳng hoàn toàn các từ phân cực cá biệt.

Do siêu phẳng quyết định được tối ưu hóa trên hai không gian toán học có cấu trúc tô-pô khác biệt hoàn toàn, biên quyết định (decision boundary), độ nhạy, lỗi phân loại và toàn bộ các chỉ số hiệu năng (Accuracy, Precision, Recall, F1) chắc chắn sẽ phân kỳ đáng kể.

---

## 5.2. Lựa chọn Representation tối ưu cho Dataset IMDb Sentiment Analysis

### Câu hỏi nghiên cứu:
> *Đối với tập dữ liệu IMDb Large Movie Review Dataset (phân loại cảm xúc tích cực / tiêu cực), representation nào sẽ phù hợp hơn và cho hiệu năng vượt trội hơn?*

### Lập luận khoa học:
1. **Đặc tính văn bản của bài toán Sentiment**:
   Cảm xúc của một bài đánh giá phim thường được xác định bởi sự xuất hiện của các từ định tính phân cực mạnh (Sentiment Lexicons) như: *"masterpiece", "outstanding", "terrible", "garbage", "waste"*.
2. **Ưu thế của TF-IDF trong Sentiment Analysis**:
   - TF-IDF gán trọng số cao cho các từ hiếm và giàu thông tin phân loại, đồng thời chuẩn hóa độ dài văn bản (L2 normalization).
   - Mô hình tuyến tính dễ dàng gán trọng số cực lớn ($w_i \gg 0$ hoặc $w_i \ll 0$) trực tiếp cho các từ cảm xúc này mà không bị nhiễu bởi các từ trung tính khác trong bài viết.
3. **Điểm yếu chí mạng của Average Word2Vec trên văn bản dài**:
   - Độ dài trung bình của một bài review IMDb lên tới hơn 230 từ.
   - Khi tính trung bình cộng $\vec{x}_{\text{doc}} = \frac{1}{n} \sum_{i=1}^n \vec{v}_{w_i}$, vài từ cảm xúc mạnh sẽ bị "pha loãng" (diluted) bởi hàng trăm từ mô tả tình tiết cốt truyện, tên nhân vật, giới từ và động từ hành động trung tính.
   - Word2Vec tĩnh làm mất hoàn toàn thứ tự từ (word order) và cấu trúc phủ định (`"not good"` bị hòa trộn với `"good"`).

**Dự đoán**: **TF-IDF sẽ vượt trội rõ rệt so với Average Word2Vec** trên tập dữ liệu IMDb.

---

# 10. Năm Dự đoán Khoa học trước Experiment

Theo chuẩn mực khoa học thực nghiệm của học phần NLP, mỗi dự đoán phải được xác lập minh bạch với đầy đủ 3 thành phần: **Prediction**, **Reason** và **Confidence**.

---

## 10.1. Prediction 1 — Hiệu năng của TF-IDF + Logistic Regression

### Câu hỏi:
> *TF-IDF + Logistic Regression có khả năng hoạt động tốt không?*

- **Prediction**:
  Mô hình TF-IDF kết hợp với Logistic Regression sẽ **hoạt động rất tốt**, đạt độ chính xác (Accuracy) và F1-score ở mức cao, dự kiến dao động trong khoảng **88% — 91%** trên tập dữ liệu kiểm tra (Test set) của IMDb.
- **Reason**:
  - **Khả năng phân tách tuyến tính cao trong không gian số chiều lớn (Cover's Theorem)**: Khi biểu diễn văn bản bằng TF-IDF với kích thước từ vựng $V > 50,000$, dữ liệu được ánh xạ vào không gian số chiều rất cao. Theo định lý Cover, một tập dữ liệu phi tuyến trong không gian thấp sẽ có xác suất phân tách tuyến tính rất cao khi được chiếu lên không gian số chiều cao.
  - **Tối ưu hóa lồi toàn cục (Convex Optimization)**: Hàm mất mát Log-loss (nhị phân entropy chéo) kết hợp chuẩn hóa L2 của Logistic Regression là một bài toán tối ưu lồi, đảm bảo tìm được nghiệm tối ưu toàn cục ổn định, không bị mắc kẹt ở cực tiểu cục bộ như mạng nơ-ron sâu.
  - **Khớp hoàn hảo với túi từ phân cực**: Trong bài toán cảm xúc, chỉ cần một vài từ chìa khóa có trọng số lớn là đủ để đưa logit $z$ vượt xa ngưỡng quyết định.
- **Confidence**: **High (Rất cao)**

---

## 10.2. Prediction 2 — Liệu Average Word2Vec có chắc chắn tốt hơn TF-IDF?

### Câu hỏi:
> *Average Word2Vec có chắc chắn tốt hơn TF-IDF không?*

- **Prediction**:
  Average Word2Vec **CHẮC CHẮN KHÔNG TỐT HƠN** TF-IDF, thậm chí hiệu năng sẽ **thua kém đáng kể** (thấp hơn khoảng 3% — 6% về cả Accuracy và F1-score so với TF-IDF baseline).
- **Reason**:
  - **Hiệu ứng pha loãng tín hiệu (Semantic Dilution in Long Documents)**: Bài đánh giá IMDb là các đoạn văn dài ($mean \approx 231$ từ). Việc lấy trung bình số học không trọng số san phẳng tất cả các đỉnh tín hiệu cảm xúc.
  - **Mất mát cú pháp và phạm vi phủ định (Loss of Negation Scope)**: Word2Vec tĩnh biểu diễn từng từ độc lập. Phép cộng vector không thể phân biệt giữa `"this movie is good"` và `"this movie is not good"`, khiến vector biểu diễn của hai câu trái ngược ý nghĩa vẫn có độ tương đồng cosin tiến sát 1.0.
  - **Thiếu trọng số quan trọng theo tài liệu**: Không giống TF-IDF có cơ chế IDF phạt các từ phổ biến, Average Word2Vec gán trọng số 1 cho mọi từ trong câu, dẫn đến vector văn bản bị kéo về phía các từ phổ biến mang ít giá trị cảm xúc.
- **Confidence**: **High (Rất cao)**

---

## 10.3. Prediction 3 — Tăng Vocabulary Size có chắc chắn làm tăng Accuracy?

### Câu hỏi:
> *Nếu tăng vocabulary size thì accuracy chắc chắn tăng không?*

- **Prediction**:
  Tăng vocabulary size **KHÔNG CHẮC CHẮN LÀM TĂNG ACCURACY**. Khi kích thước từ vựng vượt qua một ngưỡng tối ưu nhất định, Accuracy sẽ bão hòa (plateau) hoặc thậm chí suy giảm nhẹ do hiện tượng quá khớp (overfitting) với nhiễu dữ liệu.
- **Reason**:
  - **Hiện tượng từ hiếm (Hapax Legomena & Long Tail Noise)**: Theo Định luật Zipf, phần lớn các từ bổ sung khi mở rộng từ vựng là các từ chỉ xuất hiện 1 hoặc 2 lần (typos, tên riêng rác, ký tự dị biệt). Những từ này không có tính khái quát hóa cho tập test mà chỉ khiến mô hình học vẹt tập train.
  - **Lời nguyền số chiều (Curse of Dimensionality)**: Khi số chiều $d$ tăng vọt mà số lượng mẫu $N$ cố định, mật độ dữ liệu trong không gian trở nên cực kỳ loãng, làm tăng nguy cơ mô hình bị phân mảnh biên quyết định.
  - **Chi phí tính toán bùng nổ**: Dung lượng bộ nhớ RAM và thời gian huấn luyện tăng tuyến tính theo số chiều đặc trưng mà không mang lại giá trị gia tăng tương xứng.
- **Confidence**: **High (Rất cao)**

---

## 10.4. Prediction 4 — Loại bỏ Stopwords có chắc chắn cải thiện Performance?

### Câu hỏi:
> *Nếu bỏ stopwords thì performance chắc chắn tăng không?*

- **Prediction**:
  Loại bỏ toàn bộ stopwords **KHÔNG CHẮC CHẮN LÀM TĂNG PERFORMANCE**, thậm chí trong bài toán Sentiment Classification, việc loại bỏ stopwords máy móc từ danh sách dựng sẵn (predefined stopword list) có thể **làm giảm độ chính xác**.
- **Reason**:
  - **Các stopword mang sắc thái phủ định sống còn**: Danh sách stopword chuẩn (như của NLTK, Scikit-learn) thường chứa các từ như: `"not"`, `"no"`, `"never"`, `"neither"`, `"nor"`, `"barely"`, `"hardly"`. Trong phân loại cảm xúc, đây chính là các toán tử đảo ngược cực tính (polarity inverters). Loại bỏ `"not"` sẽ biến `"not good"` thành `"good"`, khiến mô hình dự đoán sai hoàn toàn.
  - **Các hư từ bổ nghĩa mức độ (Intensifiers)**: Các từ như `"very"`, `"too"`, `"so"`, `"extremely"` thường bị coi là stopword nhưng lại đóng vai trò khuếch đại cường độ cảm xúc.
  - **Giải pháp tối ưu hơn**: Thay vì loại bỏ stopword thủ công, nên giữ lại toàn bộ và kết hợp mở rộng cụm từ n-gram (Bi-gram: `"not good"`, `"very bad"`) để giữ trọn vẹn ngữ nghĩa ngữ cảnh.
- **Confidence**: **High (Rất cao)**

---

## 10.5. Prediction 5 — Accuracy và F1 có thể dẫn tới kết luận trái ngược nhau?

### Câu hỏi:
> *Accuracy và F1 có thể cho kết luận khác nhau không?*

- **Prediction**:
  **HOÀN TOÀN CÓ THỂ**. Trong nhiều kịch bản thực nghiệm, đặc biệt khi dữ liệu có sự mất cân bằng giữa các lớp hoặc khi các mô hình có sự đánh đổi khác nhau giữa Precision và Recall, mô hình có Accuracy cao hơn lại có thể sở hữu F1-score thấp hơn.
- **Reason**:
  - **Bản chất toán học khác nhau**:
    - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$: Tính trọng số đồng đều trên toàn bộ mẫu, bị chi phối áp đảo bởi lớp đa số.
    - $F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$: Là trung bình điều hòa (harmonic mean) giữa Precision và Recall, phạt cực kỳ nặng nếu một trong hai chỉ số này nhận giá trị thấp.
  - **Kịch bản mất cân bằng nhãn (Class Imbalance)**: Nếu lớp Positive chỉ chiếm 5%, một mô hình dự đoán toàn bộ là Negative sẽ có Accuracy = 95% nhưng F1-score của lớp Positive bằng 0%. Một mô hình thứ hai dự đoán phát hiện được các ca Positive với Precision = 80%, Recall = 70% có thể có Accuracy = 93% (thấp hơn mô hình thứ nhất) nhưng F1-score = 74.6% (vượt trội hoàn toàn).
  - **Sự phân kỳ quyết định**: Khi lựa chọn mô hình triển khai thực tế, chỉ dựa vào Accuracy sẽ dẫn đến quyết định sai lầm nghiêm trọng.
- **Confidence**: **High (Rất cao)**

---

# Bảng Tổng hợp Dự đoán & Đối chiếu Kết quả Thực nghiệm Thực tế

Dưới đây là bảng đối chiếu trực tiếp giữa các giả thuyết khoa học đã thiết lập trước thực nghiệm và số liệu đo đạc thực tế từ quá trình chạy mã nguồn trên Test set của IMDb:

| Mã giả thuyết | Nội dung dự đoán | Mức độ tin cậy | Kết quả thực nghiệm đo được | Đánh giá kiểm định giả thuyết |
| :--- | :--- | :---: | :---: | :---: |
| **Prediction 1** | TF-IDF + Logistic Regression đạt Accuracy/F1 cao (88% — 91%) | **High** | $\text{Accuracy} = \mathbf{89.98\%}$, $F_1 = \mathbf{0.9009}$ | **XÁC NHẬN CHÍNH XÁC (MATCH)** |
| **Prediction 2** | Average Word2Vec kém hơn TF-IDF (thua 3% — 6%) | **High** | Avg Word2Vec: $\text{Acc} = \mathbf{85.06\%}$, $F_1 = \mathbf{0.8521}$ (kém TF-IDF ~4.9%) | **XÁC NHẬN CHÍNH XÁC (MATCH)** |
| **Prediction 3** | Tăng Vocab không đảm bảo tăng tuyến tính Accuracy | **High** | Mở rộng Uni+Bigram làm Vocab tăng 11.3 lần ($54k \to 615k$) nhưng Accuracy chỉ nhích nhẹ từ $89.98\% \to 90.18\%$ | **XÁC NHẬN CHÍNH XÁC (MATCH)** |
| **Prediction 4** | Giữ stopword và bổ sung Bigram giúp bắt được phủ định (`"not good"`) | **High** | Pipeline Uni+Bigram nhận diện chính xác `"this movie is not good"` $\to$ negative ($P=0.112$), vượt trội hoàn toàn Unigram thuần túy | **XÁC NHẬN CHÍNH XÁC (MATCH)** |
| **Prediction 5** | Accuracy và F1 phản ánh các khía cạnh khác nhau của mô hình | **High** | Model C (Linear SVM) có Accuracy ($89.96\%$) thấp hơn Model A ($89.98\%$) nhưng Precision ($89.55\%$) lại cao hơn ($89.40\%$) | **XÁC NHẬN CHÍNH XÁC (MATCH)** |
