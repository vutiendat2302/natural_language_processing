# BÁO CÁO PHÂN TÍCH LỖI VÀ ĐỘ GIẢI THÍCH MÔ HÌNH (LAB 04 - ERROR ANALYSIS)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
1. [20. Phân tích Ma trận Nhầm lẫn (Confusion Matrix Analysis)](#20-phân-tích-ma-trận-nhầm-lẫn-confusion-matrix-analysis)
   - [20.1. Thống kê Ma trận Nhầm lẫn trên Tập kiểm tra (Test Set)](#201-thống-kê-ma-trận-nhầm-lẫn-trên-tập-kiểm-tra-test-set)
   - [20.2. So sánh Ma trận Nhầm lẫn giữa 3 Mô hình](#202-so-sánh-ma-trận-nhầm-lẫn-giữa-3-mô-hình)
   - [20.3. Đánh giá Xu hướng Lỗi: Class nào bị nhầm nhiều nhất?](#203-đánh-giá-xu-hướng-lỗi-class-nào-bị-nhầm-nhiều-nhất)
2. [21. Mổ xẻ Chi tiết Mẫu Dữ liệu Thực tế (Detailed Error Case Studies)](#21-mổ-xẻ-chi-tiết-mẫu-dữ-liệu-thực-tế-detailed-error-case-studies)
   - [21.1. Ba trường hợp mô hình phân loại chính xác (Correctly Classified Examples)](#211-ba-trường-hợp-mô-hình-phân-loại-chính-xác-correctly-classified-examples)
   - [21.2. Ba trường hợp mô hình phân loại sai (Incorrectly Classified Examples)](#212-ba-trường-hợp-mô-hình-phân-loại-sai-incorrectly-classified-examples)
   - [21.3. Bảng phân loại nguồn gốc gây lỗi theo danh mục chuẩn](#213-bảng-phân-loại-nguồn-gốc-gây-lỗi-theo-danh-mục-chuẩn)
3. [22. Kiểm tra Trọng số Mô hình (Inspect Model Weights & Interpretability)](#22-kiểm-tra-trọng-số-mô-hình-inspect-model-weights--interpretability)
   - [22.1. Top 15 Đặc trưng Dương tính mạnh nhất (Positive Features)](#221-top-15-đặc-trưng-dương-tính-mạnh-nhất-positive-features)
   - [22.2. Top 15 Đặc trưng Âm tính mạnh nhất (Negative Features)](#222-top-15-đặc-trưng-âm-tính-mạnh-nhất-negative-features)
   - [22.3. Trả lời câu hỏi: Các đặc trưng này có thực sự có ý nghĩa đối với bài toán không?](#223-trả-lời-câu-hỏi-các-đặc-trưng-này-có-thực-sự-có-ý-nghĩa-đối-với-bài-toán-không)
   - [22.4. Khám phá "Hiện vật dữ liệu" (Data Artifacts / Spurious Correlation)](#224-khám-phá-hiện-vật-dữ-liệu-data-artifacts--spurious-correlation)

---

# 20. Phân tích Ma trận Nhầm lẫn (Confusion Matrix Analysis)

## 20.1. Thống kê Ma trận Nhầm lẫn trên Tập kiểm tra (Test Set)

Tập dữ liệu kiểm tra (Test set) sau khi tiền xử lý và loại bỏ các bài review trùng lặp gồm **4,959 mẫu**, với sự phân bố lớp gần như cân bằng hoàn hảo:
- Số mẫu nhãn Dương (Positive - nhãn 1): **2,489 mẫu** (50.19%)
- Số mẫu nhãn Âm (Negative - nhãn 0): **2,470 mẫu** (49.81%)

Đối với mô hình đường cơ sở **Model A (TF-IDF + Logistic Regression)**, ma trận nhầm lẫn thu được như sau:

| | **Predicted Negative (0)** | **Predicted Positive (1)** | **Tổng hàng (Actual)** |
| :--- | :---: | :---: | :---: |
| **Actual Negative (0)** | $TN = \mathbf{2,202}$ | $FP = \mathbf{268}$ | $N = 2,470$ |
| **Actual Positive (1)** | $FN = \mathbf{229}$ | $TP = \mathbf{2,260}$ | $P = 2,489$ |
| **Tổng cột (Predicted)** | $TN + FN = 2,431$ | $FP + TP = 2,528$ | **Tổng số = 4,959** |

Trực quan hóa ma trận nhầm lẫn của các mô hình được lưu trữ tại file ảnh: [confusion_matrix.png](file:///home/vutienndat2302/Documents/natural_language_processing/practice/lab04/confusion_matrix.png).

---

## 20.2. So sánh Ma trận Nhầm lẫn giữa 3 Mô hình

Dưới đây là bảng đối chiếu chi tiết số lượng $TP, FN, FP, TN$ đo được trên cùng tập test $N = 4,959$:

| Mô hình | Biểu diễn (Representation) | Bộ phân loại (Classifier) | True Positive ($TP$) | False Negative ($FN$) | False Positive ($FP$) | True Negative ($TN$) | Accuracy | F1-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | TF-IDF (Unigram) | Logistic Regression | **2,260** | 229 | 268 | 2,202 | **89.98%** | **90.09%** |
| **Model B** | Average Word2Vec ($d=100$) | Logistic Regression | 2,134 | 355 | 386 | 2,084 | 85.06% | 85.21% |
| **Model C** | TF-IDF (Unigram) | Linear SVM | 2,254 | 235 | **263** | **2,207** | 89.96% | 90.05% |

---

## 20.3. Đánh giá Xu hướng Lỗi: Class nào bị nhầm nhiều nhất?

1. **Class nào bị nhầm nhiều nhất?**
   - Trong Model A: Có **268 mẫu Negative bị phân loại nhầm thành Positive** (tương ứng False Positive Rate $\text{FPR} = \frac{268}{2470} = \mathbf{10.85\%}$).
   - Trong khi đó: Chỉ có **229 mẫu Positive bị phân loại nhầm thành Negative** (tương ứng False Negative Rate $\text{FNR} = \frac{229}{2489} = \mathbf{9.20\%}$).
   - $\implies$ **Lớp Negative bị mô hình dự đoán nhầm nhiều hơn lớp Positive** ($268 > 229$).
2. **Loại error nào chiếm ưu thế?**
   - Lỗi **False Positive ($FP = 268$)** chiếm ưu thế so với lỗi **False Negative ($FN = 229$)**.
   - Tỷ lệ lỗi $FP$ chiếm $53.92\%$ tổng số ca phân loại sai của mô hình ($268 / (268 + 229) = 268 / 497$).
   - Nguyên nhân cốt lõi: Nhiều bài đánh giá tiêu cực (chê bai thậm tệ) trong IMDb vẫn sử dụng các cấu trúc văn phong chứa nhiều từ tích cực, ví dụ: các lời khen mang tính an ủi (*"the actors were good, but the script was terrible"*), hoặc các câu châm biếm, mỉa mai (*"what a masterpiece of garbage"*). Vì TF-IDF Unigram chỉ cộng dồn trọng số từ độc lập, các từ mang sắc thái khen ngợi đã vô tình đẩy logit của bài chê lên mức dương.
3. **Metric nào phản ánh vấn đề đó?**
   - Do $FP > FN$, mẫu số của Precision ($TP + FP$) bị nới rộng nhiều hơn mẫu số của Recall ($TP + FN$).
   - Kết quả: **Precision ($\mathbf{0.8940}$) thấp hơn Recall ($\mathbf{0.9080}$)**.
   - Chỉ số **Precision** chính là thước đo phản ánh rõ nét sự xuất hiện của các ca False Positive này.

---

# 21. Mổ xẻ Chi tiết Mẫu Dữ liệu Thực tế (Detailed Error Case Studies)

Dưới đây là phân tích chuyên sâu các trường hợp điển hình được trích xuất trực tiếp từ tập dữ liệu kiểm tra của `classification.ipynb`.

## 21.1. Ba trường hợp mô hình phân loại chính xác (Correctly Classified Examples)

### Trường hợp 1 (Correct Positive): Review khen ngợi xuất sắc với nhiều từ phân cực mạnh
- **Đoạn văn bản (Text)**:
  > *"this outstanding film has about the best acting that you'll ever see, and that alone makes this a must-see. the entire cast is excellent, but then again, it had to be in order to keep up with dustin hoffman and meryl streep. it didn't take me long to get hooked on this film, and aside from a courtroom scene that is merely good, this is top-notch entertainment. this is a rare film that actually..."*
- **Nhãn thực tế (True Label)**: `positive` (1)
- **Dự đoán của mô hình (Predicted Label)**: `positive` (1)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.992$** (99.2%)
- **Phân tích cơ chế (Analysis)**:
  Văn bản tập trung một mật độ dày đặc các từ khóa nằm trong Top trọng số dương tính cao nhất của mô hình: `"outstanding"` ($+3.51$), `"best"` ($+5.64$), `"excellent"` ($+7.31$), `"good"` ($+2.82$). Tổ hợp tuyến tính tạo ra logit dương rất lớn ($z > 4.8$), kéo xác suất qua hàm Sigmoid chạm ngưỡng bão hòa gần như tuyệt đối ($0.992$).

### Trường hợp 2 (Correct Positive): Review khen ngợi nhẹ nhàng, nhất quán
- **Đoạn văn bản (Text)**:
  > *"i really like traci lords. she may not be the greatest actress in the world but she's rather good in this. she play the dowdy, conservative, reporter to a 't'. it's a great little thriller which keeps you guessing for a good while. jeff fahey is also good as traci's boss. i think given a decent break traci could be a top actress. she's certainly no worse than some of today's leading ladies."*
- **Nhãn thực tế (True Label)**: `positive` (1)
- **Dự đoán của mô hình (Predicted Label)**: `positive` (1)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.860$** (86.0%)
- **Phân tích cơ chế (Analysis)**:
  Văn bản chứa các từ chỉ cảm xúc tích cực ổn định như `"like"` ($+2.31$), `"great"` ($+8.45$), lặp lại từ `"good"` 3 lần. Mặc dù có các cụm phủ định như `"not be the greatest"`, nhưng sự lấn lướt của các tính từ tích cực khác giúp mô hình gán nhãn chính xác.

### Trường hợp 3 (Correct Negative): Review phê bình với các từ khóa chê bai rõ ràng
- **Đoạn văn bản (Text)**:
  > *"this is an admirable attempt from first time filmmaker ham tran, offering little-glanced perspective dealing with vietnam war victims struggling for liberation, but plays out as a glorified history special. with clunky, self-consciously informative dialog and sub-par acting, even a relatively impressive budget with attention to detail will not spring to life this sagging, albeit historically w..."*
- **Nhãn thực tế (True Label)**: `negative` (0)
- **Dự đoán của mô hình (Predicted Label)**: `negative` (0)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.446$** $\implies P(y=0 \mid x) = \mathbf{0.554}$
- **Phân tích cơ chế (Analysis)**:
  Mặc dù bài viết có nhắc đến một số từ khen mang tính động viên (*"admirable", "impressive"*), các tính từ chỉ trích chuyên môn (*"clunky", "sub-par", "sagging"*) đã kéo điểm số tổng thể xuống dưới ngưỡng 0.5. Mô hình nhận diện chính xác nhãn Negative.

---

## 21.2. Ba trường hợp mô hình phân loại sai (Incorrectly Classified Examples)

### Trường hợp 4 (False Positive - Mã lỗi: `lexical ambiguity` & `domain-specific term`): Lời chê về khâu kỹ thuật bị lẫn với lời khen
- **Đoạn văn bản (Text)**:
  > *"the core message is strong, the cast has given it their best shot, the packaging is excellent, but the screenplay is seriously over-dramatized and every cliche in the book on women's suffering in india has been over-used to the max."*
- **Nhãn thực tế (True Label)**: `negative` (0)
- **Dự đoán của mô hình (Predicted Label)**: `positive` (1)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.879$** (87.9% tin vào Positive)
- **Phân loại lỗi**: `lexical ambiguity` / `concession structure` (cấu trúc nhượng bộ).
- **Phân tích chuyên sâu (Root Cause Analysis)**:
  - **Mâu thuẫn ngữ nghĩa**: Người đánh giá sử dụng kỹ thuật viết "nhượng bộ - phản biện" (*concession*): Trước tiên khen ngợi mặt hình thức (*"message is strong"*, *"best shot"*, *"packaging is excellent"*) để làm bàn đạp cho lời chê cốt lõi ở vế sau (*"screenplay is seriously over-dramatized"*, *"cliche"*, *"over-used"*).
  - **Cơ chế gây lỗi của TF-IDF**: Mô hình Bag-of-Words tính tổng điểm độc lập của từng từ. Từ `"excellent"` đóng góp $+7.31$, `"best"` đóng góp $+5.64$, `"strong"` đóng góp $+1.8$. Trong khi đó, các từ chê bai như `"over-dramatized"`, `"cliche"`, `"over-used"` có trọng số âm khiêm tốn hoặc là từ ít gặp. Kết quả là điểm số dương tính áp đảo hoàn toàn, khiến mô hình đưa ra dự đoán sai lệch nghiêm trọng với độ tự tin lên tới $87.9\%$.

### Trường hợp 5 (False Negative - Mã lỗi: `data noise` & `lexical ambiguity`): Review khen ngợi nhưng sử dụng từ vựng tiêu cực
- **Đoạn văn bản (Text)**:
  > *"this silly movie is really fun for the younger audiences. its heros are a couple of dud detectives whose sophomoric attitudes lead them down some very silly roads. chasing the big murder case, you will see these detectives go to every length to solve the crime. no nudity, but lots of sexual implication, slapstick silliness...everything adolescents go for. low budget, but very entertaining. def..."*
- **Nhãn thực tế (True Label)**: `positive` (1)
- **Dự đoán của mô hình (Predicted Label)**: `negative` (0)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.409$** (Dưới ngưỡng 0.5)
- **Phân loại lỗi**: `domain-specific term` / `lexical ambiguity` / `data noise`.
- **Phân tích chuyên sâu (Root Cause Analysis)**:
  - **Bản chất ngữ nghĩa**: Đây là bài đánh giá tích cực dành cho một bộ phim hài bựa, nhảm dành cho lứa tuổi thanh thiếu niên (*"really fun", "very entertaining"*).
  - **Cơ chế gây lỗi**: Tác giả sử dụng một loạt các từ vốn thường mang ngữ nghĩa xấu trong các thể loại phim khác: `"silly"` (xuất hiện 2 lần), `"dud"` (vô dụng), `"crime"`, `"murder"`, `"low budget"`. Trọng số âm của các từ `"silly"`, `"dud"`, `"crime"` đã triệt tiêu hoàn toàn điểm số của `"fun"` và `"entertaining"`, kéo logit về phía âm và khiến mô hình đoán sai.

### Trường hợp 6 (False Positive - Mã lỗi: `sarcasm`): Thách thức kinh điển của châm biếm, mỉa mai
- **Đoạn văn bản (Text)**:
  > *"Great, another two hours of my life I will never get back."*  
  *(Mẫu thực nghiệm bổ sung tại Mục 23)*
- **Nhãn thực tế (True Label)**: `negative` (0) (Tác giả cực kỳ thất vọng và tiếc thời gian)
- **Dự đoán của mô hình (Predicted Label)**: `positive` (1)
- **Xác suất dự báo $P(y=1 \mid x)$**: **$0.657$** (65.7% Positive)
- **Phân loại lỗi**: `sarcasm` (châm biếm, mỉa mai).
- **Phân tích chuyên sâu (Root Cause Analysis)**:
  - **Bản chất ngữ nghĩa**: Con người ngay lập tức hiểu rằng từ *"Great"* đứng đầu câu là sự mỉa mai chua chát khi kết hợp với mệnh đề *"another two hours of my life I will never get back"*.
  - **Cơ chế gây lỗi**: Từ `"great"` là đặc trưng dương tính mạnh nhất toàn bộ mô hình ($w = +8.450$). Cả câu chỉ có một vài từ, và cụm biểu đạt sự thất vọng *"never get back"* gồm các từ đơn lẻ không mang trọng số âm đủ mạnh để vượt qua trọng số khổng lồ của `"great"`. Đây là giới hạn không thể khắc phục của các mô hình tuyến tính Bag-of-Words khi thiếu vắng cơ chế hiểu ngữ cảnh sâu (contextual understanding).

---

## 21.3. Bảng phân loại nguồn gốc gây lỗi theo danh mục chuẩn

| Danh mục lỗi (Error Category) | Cơ chế phát sinh trong thực tế phân loại văn bản | Mẫu văn bản minh họa tiêu biểu | Giải pháp kỹ thuật khắc phục |
| :--- | :--- | :--- | :--- |
| **`negation` (Phủ định)** | Từ phủ định (`not`, `never`) đi kèm từ tích cực nhưng bị tách rời thành các unigram độc lập | `"this movie is not good"` $\to$ dự đoán Positive vì từ `good` có trọng số quá lớn | Bổ sung **Bi-gram features** (`not good`) hoặc dùng **Sequence Models (LSTM/Transformer)** |
| **`sarcasm` (Châm biếm / Mỉa mai)** | Dùng từ ngữ tích cực ở bề mặt biểu kiến để diễn đạt ý chê bai sâu cay | `"Great, another two hours of my life I will never get back"` | Cần cơ chế **Self-Attention** để liên kết từ `great` với ngữ cảnh tiếc nuối phía sau |
| **`lexical ambiguity` (Mơ hồ từ vựng)** | Từ ngữ mang tính chất đa nghĩa hoặc phụ thuộc vào thể loại phim (hài hước vs chính kịch) | `"silly"`, `"dud"`, `"ridiculous"` là khen trong phim hài nhưng là chê trong phim tài liệu | Nhúng vector ngữ cảnh động (**Contextual Embeddings - BERT**) |
| **`domain-specific term` (Thuật ngữ chuyên ngành)** | Các từ ngữ mô tả điện ảnh chuyên sâu (*"cliche", "microphone boom"*) mang trọng số lệch | Bài review chê micro xuất hiện trong khung hình $\to$ mô hình không hiểu đây là lỗi kỹ thuật | Mở rộng tập ngữ liệu huấn luyện chuyên ngành điện ảnh |
| **`short text` (Văn bản quá ngắn)** | Văn bản quá ngắn chỉ có 3 - 5 từ, thiếu hụt ngữ cảnh trầm trọng | Review chỉ có `"not bad"` hoặc `"could be better"` | Bổ sung cơ chế tiền xử lý hoặc gán ngưỡng tự tin cao hơn trước khi ra quyết định |
| **`data noise` (Nhiễu dữ liệu)** | Lỗi chính tả, thẻ HTML sót lại, hoặc nhãn ban đầu của người dùng bị gán nhầm | Bài review chấm điểm 1/10 nhưng ghi nhãn positive trên hệ thống gốc | Làm sạch dữ liệu tự động và kiểm tra tính nhất quán nhãn (Label Noise Cleaning) |
| **`class imbalance` (Mất cân bằng lớp)** | Một lớp chiếm tỷ lệ áp đảo làm dịch chuyển siêu phẳng quyết định | Mô hình có xu hướng đoán toàn bộ về lớp đa số để tối ưu hàm mất mát | Cân bằng trọng số lớp (`class_weight='balanced'`), dùng SMOTE hoặc Focal Loss |

---

# 22. Kiểm tra Trọng số Mô hình (Inspect Model Weights & Interpretability)

Mô hình Logistic Regression tuyến tính có hàm quyết định:
$$z = \vec{w}^T \vec{x} + b = \sum_{j=1}^V w_j x_j + b$$
Nhờ tính chất tuyến tính, mỗi trọng số $w_j$ cho biết trực tiếp mức độ và chiều hướng tác động của đặc trưng từ vựng $j$ lên log-odds của lớp Positive:
- $w_j \gg 0$: Khi từ $j$ xuất hiện, nó làm tăng mạnh xác suất mẫu thuộc lớp Positive.
- $w_j \ll 0$: Khi từ $j$ xuất hiện, nó làm tăng mạnh xác suất mẫu thuộc lớp Negative.

## 22.1. Top 15 Đặc trưng Dương tính mạnh nhất (Positive Features)

Dưới đây là 15 đặc trưng từ vựng có hệ số hồi quy dương lớn nhất được trích xuất từ `model_a.coef_[0]`:

| Thứ hạng | Đặc trưng (Feature) | Trọng số ($w_j$) | Nhận xét ngôn ngữ học |
| :---: | :--- | :---: | :--- |
| **1** | `great` | **+8.450** | Tính từ khen ngợi phổ quát, tần suất rất cao trong các bài khen |
| **2** | `excellent` | **+7.314** | Tính từ chỉ chất lượng vượt trội |
| **3** | `perfect` | **+5.674** | Biểu đạt sự hoàn hảo không tì vết |
| **4** | `best` | **+5.635** | Cấp so sánh cao nhất, chỉ các tác phẩm đỉnh cao |
| **5** | `wonderful` | **+5.235** | Cảm xúc thán phục, hài lòng |
| **6** | `amazing` | **+5.156** | Biểu đạt sự kinh ngạc trước chất lượng tác phẩm |
| **7** | `loved` | **+4.680** | Động từ biểu thị tình cảm yêu thích sâu sắc |
| **8** | `today` | **+4.465** | Hiện vật dữ liệu (Data artifact) — Thường gặp trong cụm so sánh thời nay |
| **9** | `brilliant` | **+4.444** | Khen ngợi tài năng đạo diễn, kịch bản hoặc diễn xuất |
| **10** | `enjoyed` | **+4.418** | Động từ chỉ sự tận hưởng bộ phim |
| **11** | `favorite` | **+4.232** | Tác phẩm yêu thích hàng đầu của người xem |
| **12** | `fun` | **+4.153** | Mang tính giải trí cao, vui vẻ |
| **13** | `love` | **+4.077** | Từ chỉ cảm xúc yêu thích |
| **14** | `well` | **+4.053** | Phó từ đi kèm các cụm khen (*well-done, well-acted, well-written*) |
| **15** | `superb` | **+4.017** | Tính từ khen ngợi xuất sắc |

---

## 22.2. Top 15 Đặc trưng Âm tính mạnh nhất (Negative Features)

Dưới đây là 15 đặc trưng từ vựng có hệ số hồi quy âm sâu nhất:

| Thứ hạng | Đặc trưng (Feature) | Trọng số ($w_j$) | Nhận xét ngôn ngữ học |
| :---: | :--- | :---: | :--- |
| **1** | `worst` | **-10.997** | Từ chê bai nặng nề nhất, xuất hiện áp đảo trong các bài đánh giá 1 sao |
| **2** | `bad` | **-8.695** | Tính từ tiêu cực cơ bản, tần suất cực lớn |
| **3** | `awful` | **-8.265** | Chỉ sự kinh khủng, dở tệ |
| **4** | `waste` | **-7.575** | Xuất hiện dày đặc trong cụm *"waste of time", "waste of money"* |
| **5** | `boring` | **-7.442** | Sự tẻ nhạt, buồn ngủ, thiếu hấp dẫn |
| **6** | `poor` | **-6.702** | Chất lượng kém (*"poor acting", "poor direction"*) |
| **7** | `terrible` | **-6.420** | Chỉ sự tồi tệ không thể chấp nhận |
| **8** | `nothing` | **-5.636** | Mang tính phủ định triệt để (*"nothing works", "nothing good"*) |
| **9** | `worse` | **-5.267** | Cấp so sánh hơn của tệ (*"gets worse and worse"*) |
| **10** | `horrible` | **-5.164** | Cảm giác kinh hãi trước một tác phẩm tồi |
| **11** | `dull` | **-5.107** | Nhàm chán, buồn tẻ |
| **12** | `poorly` | **-5.007** | Phó từ chỉ sự yếu kém (*"poorly executed", "poorly written"*) |
| **13** | `fails` | **-4.682** | Động từ chỉ sự thất bại của tác phẩm |
| **14** | `stupid` | **-4.603** | Chỉ trích kịch bản hoặc nhân vật ngớ ngẩn |
| **15** | `annoying` | **-4.528** | Gây khó chịu, bực bội cho người xem |

---

## 22.3. Trả lời câu hỏi: Các đặc trưng này có thực sự có ý nghĩa đối với bài toán không?

### Kết luận:
**CÓ, ĐA PHẦN CÁC ĐẶC TRƯNG NÀY CÓ Ý NGHĨA TRỰC QUAN HOÀN TOÀN TỰ NHIÊN VÀ PHÙ HỢP TUYỆT ĐỐI VỚI BÀI TOÁN PHÂN TÍCH CẢM XÚC (High Semantic Plausibility).**

### Luận điểm phân tích:
1. **Tính trực tiếp và minh bạch (Model Transparency & Interpretability)**:
   - Các từ có trọng số tuyệt đối lớn nhất chính là những từ mà con người sử dụng phổ biến nhất khi khen hoặc chê một bộ phim:
     - Khen: `great`, `excellent`, `amazing`, `perfect`, `superb`.
     - Chê: `worst`, `bad`, `awful`, `waste`, `boring`, `terrible`.
   - Điều này chứng minh mô hình Logistic Regression không học ngẫu nhiên mà đã hội tụ về đúng các biểu thức cảm xúc (Sentiment Lexicons) cốt lõi của ngôn ngữ tự nhiên.
2. **Cường độ tương quan với giá trị trọng số**:
   - Từ mang tính cực đoan nhất (`worst`) có độ lớn trọng số lên tới $-10.997$, cao hơn từ tiêu cực chung chung (`bad` mang $-8.695$).
   - Sự phân bậc trọng số này hoàn toàn tương thích với trực giác đo lường cường độ cảm xúc (sentiment intensity) của con người.

---

## 22.4. Khám phá "Hiện vật dữ liệu" (Data Artifacts / Spurious Correlation)

Bên cạnh các tính từ cảm xúc hiển nhiên, việc phân tích trọng số còn làm lộ diện các hiện tượng thú vị:
- **Từ `today` bất ngờ lọt vào Top 8 đặc trưng tích cực ($w = +4.465$)**:
  - Tại sao một danh từ/phó từ chỉ thời gian lại mang trọng số dương cao như vậy?
  - **Bản chất ngữ liệu (Corpus Artifact)**: Trong các bài review khen ngợi trên IMDb, người viết thường sử dụng các mẫu câu so sánh bộ phim với chuẩn mực điện ảnh đương đại, ví dụ: *"one of the best films made **today**"*, *"standing out in the cinema of **today**"*, hoặc *"rarely seen in movies **today**"*.
  - **Tương quan giả (Spurious Correlation)**: Từ `today` vô tình đồng xuất hiện với tỷ lệ cao trong các bài viết tích cực, khiến mô hình học thành một quy tắc suy luận bề mặt. Nếu triển khai mô hình sang một miền dữ liệu khác (ví dụ: tin tức thời sự), đặc trưng `today` sẽ trở thành nguồn gây nhiễu nguy hiểm.
- **Giá trị của việc kiểm tra trọng số (Model Weights Inspection)**:
  Đây là bước thiết yếu trong quy trình kiểm định mô hình có trách nhiệm (Responsible & Interpretable AI), giúp kỹ sư phát hiện thiên kiến ngầm (bias), phòng ngừa rủi ro mô hình "học vẹt" hiện vật dữ liệu trước khi đưa vào môi trường sản xuất thực tế.
