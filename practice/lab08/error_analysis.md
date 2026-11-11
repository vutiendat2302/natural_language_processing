# PHÂN TÍCH LỖI, ĐỘ TỰ TIN VÀ NGHIÊN CỨU ABLATION (LAB 08 - ERROR ANALYSIS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 08 — Pretrained Transformers and BERT Fine-Tuning  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 19. Application — Sentiment Classification

Ứng dụng mô hình **Pretrained Transformer (BERT-style Fine-Tuned)** vào bài toán phân loại cảm xúc nhị phân trên tập dữ liệu **IMDb Movie Reviews**:

```text
Review Text
  ↓
Pretrained Transformer (WordPiece Tokenizer + 6-layer Encoder + [CLS] Linear Head)
  ↓
Positive (1) / Negative (0)
```

### 19.1. Bảng kết quả định lượng trên tập Test ($N = 2,500$ mẫu):

| Thước đo (Metric) | Công thức toán học | Kết quả Pretrained Fine-Tuned | Đối chiếu Transformer From Scratch (LAB 07) | Mức cải thiện ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: |
| **Accuracy** | $\frac{TP + TN}{N} = \frac{1171 + 1166}{2500}$ | **$93.48\%$** ($0.9348$) | $88.60\%$ ($0.8860$) | **$+4.88\%$** |
| **Precision** | $\frac{TP}{TP + FP} = \frac{1171}{1171 + 84}$ | **$93.31\%$** ($0.9331$) | $88.88\%$ ($0.8888$) | **$+4.43\%$** |
| **Recall** | $\frac{TP}{TP + FN} = \frac{1171}{1171 + 79}$ | **$93.68\%$** ($0.9368$) | $88.24\%$ ($0.8824$) | **$+5.44\%$** |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | **$93.49\%$** ($0.9349$) | $88.56\%$ ($0.8856$) | **$+4.93\%$** |

### 19.2. Phân tích Confusion Matrix (`confusion_matrix.png`):

| Thực tế \ Dự đoán | Predicted Negative (`0`) | Predicted Positive (`1`) | Tổng mẫu thực tế |
| :--- | :---: | :---: | :---: |
| **True Negative (`0`)** | **$TN = 1,166$** ($93.28\%$) | $FP = 84$ ($6.72\%$) | $1,250$ |
| **True Positive (`1`)** | $FN = 79$ ($6.32\%$) | **$TP = 1,171$** ($93.68\%$) | $1,250$ |
| **Tổng mẫu dự đoán** | $1,245$ | $1,255$ | **$2,500$** |

* **Đánh giá đa chiều (vượt ra ngoài con số Accuracy):**
  * Tổng số lỗi trên $2,500$ mẫu kiểm thử đã giảm từ $285$ lỗi (ở mô hình From Scratch LAB 07: $138\text{ FP} + 147\text{ FN}$) xuống chỉ còn **$163$ lỗi** ($84\text{ FP} + 79\text{ FN}$), tương ứng mức **giảm $42.8\%$ tổng số ca dự đoán sai**.
  * Sai số giữa $FP = 84$ và $FN = 79$ là rất nhỏ (chỉ lệch $5$ mẫu), chứng tỏ mô hình có cả **Precision ($93.31\%$)** lẫn **Recall ($93.68\%$)** cân bằng, không hề xảy ra hiện tượng thiên lệch lớp (class bias).

---

# 20. Error Analysis

Theo đúng yêu cầu của **Mục 20**, ta trích xuất và phân tích sâu **8 ca dự đoán sai tiêu biểu** từ $163$ lỗi trên tập Test, bao gồm:
* **3 False Positives (`FP1`, `FP2`, `FP3`)**: Nhãn thật là `Negative (0)`, mô hình dự đoán nhầm thành `Positive (1)`.
* **3 False Negatives (`FN1`, `FN2`, `FN3`)**: Nhãn thật là `Positive (1)`, mô hình dự đoán nhầm thành `Negative (0)`.
* **2 High-Confidence Errors (`HC1`, `HC2`)**: Các mẫu mô hình dự đoán **rất tự tin ($c > 0.96$) nhưng hoàn toàn sai**.

Tất cả các lỗi đều được phân loại nghiêm ngặt theo đúng 9 nhóm nguyên nhân ngôn ngữ học:
`negation`, `sarcasm`, `long-range dependency`, `ambiguous sentiment`, `domain-specific expression`, `rare word`, `tokenization`, `context`, `label ambiguity`.

---

## 20.1. Bảng tổng hợp Error Analysis (Chuẩn định dạng Mục 20)

| ID | Text | True | Pred | Confidence | Error type | Hypothesis |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **FP1** | *"With an Oscar-winning cast, stunning trailers, and a $150M budget, you would think this film couldn't possibly fail—yet it fails on every conceivable level."* | Neg (`0`) | Pos (`1`) | $0.824$ | `long-range dependency` + `context` | Mật độ tính từ khen ngợi ở mệnh đề giả định đầu câu (`Oscar-winning`, `stunning`) áp đảo vector `[CLS]`, trong khi sự đảo chiều ở cuối câu phụ thuộc vào cấu trúc giả định *"you would think... yet"* quá xa. |
| **FP2** | *"The first hour builds a fascinating mystery with brilliant cinematography, gorgeous costumes, and haunting music... [140 words later] ...rendering the entire experience a hollow, pointless waste of time."* | Neg (`0`) | Pos (`1`) | $0.886$ | `context` (Truncation tại $T=128$) | Bài review dài $210$ từ mở đầu bằng đoạn khen kỹ thuật quay phim nhưng chê kịch bản ở đoạn kết. Cửa sổ cắt `max_length=128` đã loại bỏ hoàn toàn câu kết luận tiêu cực, khiến mô hình chỉ đọc được phần khen đầu bài. |
| **FP3** | *"Critics call it a bold, avant-garde triumph of modern cinema, though to me it felt like watching paint dry in a pretentious art gallery."* | Neg (`0`) | Pos (`1`) | $0.742$ | `ambiguous sentiment` + `domain-specific expression` | Mô hình gán trọng số cao cho cụm khen trích dẫn (`bold, avant-garde triumph`), nhưng không hiểu nghĩa ẩn dụ của thành ngữ chê bai *"watching paint dry"* (nhàm chán tột độ) vì không có tính từ tiêu cực trực diện. |
| **FN1** | *"It is neither conventional nor easy to stomach; the bleak, suffocating atmosphere and relentless tragedy make it an unforgettable triumph of realist cinema."* | Pos (`1`) | Neg (`0`) | $0.815$ | `negation` + `domain-specific expression` | Trong thể loại phim tâm lý bi kịch/kinh dị, các từ mang sắc thái u ám (`bleak`, `suffocating`, `tragedy`) được dùng để khen sự chân thực của bộ phim, nhưng mô hình lại hiểu theo nghĩa tiêu cực thông thường. |
| **FN2** | *"I expected a complete disaster after reading the terrible reviews, and for the first ten minutes I almost walked out, but I have never been so glad to be proven wrong."* | Pos (`1`) | Neg (`0`) | $0.791$ | `negation` + `long-range dependency` | Mệnh đề đầu chứa nhiều từ tiêu cực mạnh (`disaster`, `terrible`, `walked out`), còn mệnh đề khen ở cuối dùng phủ định bậc hai gián tiếp (`never been so glad to be proven wrong`), làm mô hình không đảo cực hết được tín hiệu âm. |
| **FN3** | *"A delightfully unpretentious, bittersweet little indie gem that refuses to pander to blockbuster clichés, even if the pacing meanders and the ending feels open."* | Pos (`1`) | Neg (`0`) | $0.684$ | `tokenization` + `rare word` + `label ambiguity` | WordPiece tách `unpretentious` thành `['un', '##pretentious']` và `meanders` thành `['me', '##ander', '##s']`; tiền tố phủ định `un` kết hợp với `refuses`, `clichés`, `bittersweet` tạo ra tổ hợp đặc trưng nghiêng nhẹ về lớp Negative. |
| **HC1** | *"Oh, brilliant! Another masterclass in how to turn a hundred-million-dollar budget into two hours of coma-inducing perfection. Truly a triumph of insomnia cure!"* | Neg (`0`) | Pos (`1`) | **$0.978$** | `sarcasm` | **High-Confidence Error 1:** Câu mỉa mai châm biếm (`sarcasm`) sử dụng dày đặc các từ khen cấp độ cao nhất (`brilliant`, `masterclass`, `perfection`, `triumph`) để chế giễu bộ phim gây buồn ngủ (`coma-inducing`, `insomnia cure`). |
| **HC2** | *"If you love wooden acting, laughable CGI, and a script written by a sleepy toddler, then this absolute masterpiece is definitely the best movie of the year for you!"* | Neg (`0`) | Pos (`1`) | **$0.964$** | `sarcasm` + `context` | **High-Confidence Error 2:** Cấu trúc điều kiện giả định mỉa mai (*"If you love [những thứ tệ hại], then this [lời khen cực đại]"*); mệnh đề chính ở cuối chứa `"absolute masterpiece"` và `"best movie of the year"` đẩy xác suất Softmax lên $96.4\%$. |

---

## 20.2. Phân tích cơ chế chi tiết từng nhóm lỗi

### 1. Nhóm False Positives (`FP1`, `FP2`, `FP3`)
* **Tại `FP1` (`long-range dependency` + `context`):** Mô hình Pretrained đã xử lý tốt các câu tương phản ngắn (như *"bad trailer, but great film"*), nhưng khi mệnh đề nhượng bộ được đặt trong cấu trúc giả định kéo dài (*"you would think this film couldn't possibly fail — yet it fails"*), từ `"fail"` xuất hiện hai lần (một lần đi kèm phủ định `"couldn't"`, một lần khẳng định `"yet it fails"`), làm các Attention Heads bị nhiễu pha khi tổng hợp về `[CLS]`.
* **Tại `FP2` (`context` do giới hạn độ dài chuỗi):** Đây là minh chứng trực tiếp liên kết với **Experiment 4 (Mục 17)**. Ở cấu hình `max_length = 128`, bộ Tokenizer cắt bỏ mọi token từ vị trí $129$ trở đi. Do toàn bộ lời chê nằm ở đoạn kết bài review, thông tin quyết định nhãn thực chất chưa bao giờ đi vào mạng Transformer. Khi thử nghiệm nâng lên `max_length = 256`, mẫu `FP2` này được dự đoán đúng thành `Negative` với xác suất $0.912$.
* **Tại `FP3` (`ambiguous sentiment` + `domain-specific expression`):** Người viết trích dẫn lời khen của giới phê bình (`"bold, avant-garde triumph"`) trước khi bác bỏ bằng thành ngữ `"felt like watching paint dry"`. Vì cả `"watching"`, `"paint"`, `"dry"` đều là các từ trung tính trong không gian nhúng tiền huấn luyện, mô hình không nhận diện được hàm ý tiêu cực của cụm thành ngữ này.

### 2. Nhóm False Negatives (`FN1`, `FN2`, `FN3`)
* **Tại `FN1` (`negation` + `domain-specific expression`):** Đây là lỗi đặc thù của miền điện ảnh nghệ thuật/bi kịch. Các tính từ `"bleak"` (ảm đạm), `"suffocating"` (ngột ngạt), `"relentless tragedy"` (bi kịch dồn dập) mang nghĩa tiêu cực trong hầu hết các ngữ cảnh thông thường và các bài review phim giải trí, nhưng lại là lời khen đắt giá cho một bộ phim chính kịch hiện thực (`"unforgettable triumph"`).
* **Tại `FN2` (`negation` + `long-range dependency`):** Để hiểu câu *"I have never been so glad to be proven wrong"* là tích cực, mô hình phải thực hiện chuỗi suy luận logic 3 bước: (1) Kỳ vọng ban đầu là `"disaster"` (Tiêu cực); (2) Kỳ vọng đó bị `"proven wrong"` (Đảo cực thành Tích cực); (3) `"never been so glad"` khẳng định niềm vui về sự đảo cực đó. Đây là suy luận ngữ dụng phức tạp vượt quá khả năng tổng hợp của một vector `[CLS]` đơn lẻ.
* **Tại `FN3` (`tokenization` + `rare word` + `label ambiguity`):** Từ `"unpretentious"` (không màu mè, chân thật — nghĩa tích cực) khi qua WordPiece bị tách thành `['un', '##pretentious']`, trong đó tiền tố `'un'` và gốc `'##pretentious'` (kiêu căng) đều gắn với ngữ cảnh tiêu cực. Kết hợp với lời nhận xét pha trộn (*"even if the pacing meanders"*), mô hình nghiêng nhẹ về lớp Negative với độ tự tin thấp ($c = 0.684$).

### 3. Nhóm High-Confidence Errors (`HC1`, `HC2` — Dự đoán rất tự tin nhưng sai)
* **Cơ chế gây lỗi tự tin cao ở `HC1` ($c = 0.978$) và `HC2` ($c = 0.964$):**
  * Cả hai mẫu đều thuộc hiện tượng **châm biếm / mỉa mai (`sarcasm`)**. Trong `HC1`, người viết dùng tới 4 từ vựng có trọng số tích cực cực lớn trong bộ phân loại (`"brilliant"`, `"masterclass"`, `"perfection"`, `"triumph"`).
  * Để nhận ra đây là lời chê, cần phải hiểu tri thức thế giới rằng một bộ phim gây *"coma-inducing"* (hôn mê) hoặc dùng làm *"insomnia cure"* (thuốc chữa mất ngủ) là một bộ phim cực kỳ buồn tẻ. Tuy nhiên, các Attention Heads của BERT bị hút mạnh vào các từ cảm thán tích cực bề mặt, cộng dồn tạo ra logit lớp Positive rất lớn ($z_1 - z_0 > 3.5$), khiến hàm Softmax bão hòa ở mức $96.4\% - 97.8\%$.

---

# 21. Confidence Analysis

Thực hiện trích xuất phân phối xác suất và độ tự tin trên toàn bộ $2,500$ mẫu của tập Test IMDb:
```python
probabilities = softmax(logits)          # Shape: (2500, 2)
confidence = probabilities.max(axis=-1)  # Shape: (2500,)
```

### 21.1. Bảng thống kê định lượng theo 3 mức Confidence (`confidence_analysis.png`):

| Phân tầng Confidence | Khoảng giá trị $c$ | Số mẫu ($N$) | Tỷ trọng (%) | Số mẫu ĐÚNG | Số mẫu SAI | Accuracy (%) | Error Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **High confidence** | $0.85 \le c \le 1.00$ | $1,960$ | $78.4\%$ | $1,898$ | **$62$** | $96.84\%$ | **$3.16\%$** |
| **Medium confidence** | $0.65 \le c < 0.85$ | $365$ | $14.6\%$ | $321$ | $44$ | $87.95\%$ | $12.05\%$ |
| **Low confidence** | $0.50 \le c < 0.65$ | $175$ | $7.0\%$ | $118$ | $57$ | $67.43\%$ | $32.57\%$ |
| **Tổng cộng** | $0.50 \le c \le 1.00$ | **$2,500$** | **$100.0\%$** | **$2,337$** | **$163$** | **$93.48\%$** | **$6.52\%$** |

### 21.2. Phân rã chi tiết 163 mẫu dự đoán SAI theo 5 khoảng xác suất:

| Khoảng Confidence | Phân loại | Số mẫu ĐÚNG | Số mẫu SAI | Tỷ lệ mẫu sai trên tổng 163 lỗi | Đặc điểm ngôn ngữ chủ đạo của các lỗi trong khoảng |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$[0.50, 0.65)$** | Low | $118$ | $57$ | $35.0\%$ | Cảm xúc pha trộn khen-chê cân bằng (`ambiguous sentiment`), tách subword từ hiếm |
| **$[0.65, 0.75)$** | Med-Low | $134$ | $25$ | $15.3\%$ | Thành ngữ ẩn dụ (`"watching paint dry"`), câu dài nhiều mệnh đề phụ |
| **$[0.75, 0.85)$** | Med-High | $187$ | $19$ | $11.7\%$ | Phủ định tầm xa (`long-range dependency`), từ vựng đa nghĩa theo thể loại phim |
| **$[0.85, 0.95)$** | High | $612$ | **$34$** | **$20.9\%$** | Lỗi cắt cụt chuỗi (`truncation` tại $T=128$) làm mất hoàn toàn đoạn kết đảo chiều |
| **$[0.95, 1.00]$** | Very High | $1,286$ | **$28$** | **$17.2\%$** | Châm biếm / mỉa mai (`sarcasm` như `HC1`, `HC2`) hoặc nhiễu nhãn (`label ambiguity`) |

### 21.3. Trả lời câu hỏi trọng tâm Mục 21:
> **Confidence cao có đồng nghĩa với prediction đúng không?**

* **Kết luận khẳng định:** **Hoàn toàn KHÔNG ($\text{confidence} \neq \text{correctness}$).**
* **Giải thích khoa học:**
  1. Mặc dù độ tự tin có tương quan dương với độ chính xác trung bình (Accuracy tăng từ $67.43\%$ ở nhóm Low lên $96.84\%$ ở nhóm High), nhưng **tương quan thống kê không đồng nghĩa với bảo đảm đúng tuyệt đối trên từng mẫu**.
  2. Có tới **$62 / 163$ lỗi ($38.0\%$ tổng số lỗi của mô hình)** rơi vào vùng `High confidence` ($c \ge 0.85$), trong đó có **$28$ lỗi cực kỳ tự tin ($c \ge 0.95$)**.
  3. Nguyên nhân toán học: Mạng nơ-ron sâu sử dụng hàm mất mát Cross-Entropy và tầng kích hoạt tuyến tính cuối cùng có xu hướng tạo ra các giá trị logit có biên độ lớn khi bắt gặp các đặc trưng bề mặt quen thuộc (như từ `"masterpiece"`, `"brilliant"`), dẫn đến hiện tượng **quá tự tin (overconfidence / miscalibration)** ngay cả khi ngữ cảnh tổng thể là châm biếm hoặc câu văn đã bị cắt mất phần kết.

---

# 22. Ablation / Analysis

Để đánh giá định lượng đóng góp của từng thành phần kiến trúc và dữ liệu, ta thực hiện đầy đủ cả **4 phân tích Ablation có kiểm soát (A, B, C, D)**:

### Ablation A — Frozen vs Fine-tuned (Mục 15)
* **Thiết lập:** Giữ nguyên kiến trúc Pretrained BERT, tập dữ liệu IMDb ($100\%$), $T=128$; chỉ thay đổi trạng thái `requires_grad` của Encoder.
* **Kết quả:** Mở khóa Encoder để fine-tune giúp tăng Accuracy từ **$86.40\% \to 93.48\%$ ($+7.08\%$)** và F1 từ **$0.8636 \to 0.9349$ ($+0.0713$)**, đổi lại thời gian huấn luyện tăng từ $38.4\text{s}$ lên $142.5\text{s}$.
* **Ý nghĩa:** Chứng minh rằng việc điều chỉnh lại các tầng Self-Attention bên trong Encoder để thích nghi với tác vụ đích đóng vai trò quyết định để vượt ngưỡng $90\%$ Accuracy.

### Ablation B — 10% vs 100% Training Data (Mục 16)
* **Thiết lập:** Giữ nguyên mô hình và tập Test cố định; giảm dữ liệu huấn luyện từ $10,000$ mẫu ($100\%$) xuống $1,000$ mẫu ($10\%$).
* **Kết quả:** Pretrained BERT chỉ giảm **$-4.28\%$ Accuracy** ($93.48\% \to 89.20\%$), trong khi Transformer From Scratch sụt giảm tới **$-20.20\%$ Accuracy** ($88.60\% \to 68.40\%$).
* **Ý nghĩa:** Chứng minh khả năng chuyển giao tri thức (Transfer Learning) của Pretraining giúp tiết kiệm tới $90\%$ chi phí gán nhãn dữ liệu mà vẫn vượt qua mô hình huấn luyện từ đầu.

### Ablation C — Sequence Length $T \in \{64, 128, 256\}$ (Mục 17)
* **Thiết lập:** Giữ nguyên mô hình Pretrained Fine-tuned và $100\%$ dữ liệu; thay đổi ngưỡng cắt độ dài chuỗi $T$.
* **Kết quả:**
  * $T: 64 \to 128 \implies$ Accuracy tăng mạnh **$+3.64\%$** ($89.84\% \to 93.48\%$), thời gian tăng $1.92\times$ ($74.2\text{s} \to 142.5\text{s}$).
  * $T: 128 \to 256 \implies$ Accuracy chỉ tăng nhẹ **$+0.64\%$** ($93.48\% \to 94.12\%$), nhưng thời gian tăng $2.24\times$ ($142.5\text{s} \to 318.6\text{s}$) và bộ nhớ tăng từ $2.68\text{ GB} \to 5.84\text{ GB}$.
* **Ý nghĩa:** Minh họa rõ nét sự đánh đổi giữa hiện tượng mất thông tin do cắt cụt (`truncation` khi $T$ quá nhỏ) và rào cản chi phí bậc hai $\mathcal{O}(T^2)$ (khi $T$ quá lớn).

### Ablation D — Pretrained vs From Scratch (Mục 14)
* **Thiết lập:** Cùng tập dữ liệu IMDb ($100\%$), cùng $T=128$, cùng tập Test $2,500$ mẫu.
* **Kết quả:** Pretrained Transformer đạt **$93.48\%$ Accuracy (F1 $0.9349$)** so với **$88.60\%$ Accuracy (F1 $0.8856$)** của Transformer From Scratch.
* **Ý nghĩa:** Khẳng định giá trị cốt lõi của biểu diễn ngôn ngữ tiền huấn luyện MLM và Subword Tokenization so với việc học từ trọng số ngẫu nhiên.

---

# 28. Error Analysis — Câu hỏi bắt buộc

Trả lời đầy đủ 6 câu hỏi bắt buộc (**Q1 – Q6**) theo đúng chuỗi suy luận khoa học 3 bước:
$$\text{Observation (Quan sát hiện tượng)} \longrightarrow \text{Evidence (Bằng chứng số liệu)} \longrightarrow \text{Interpretation (Diễn giải cơ chế)}$$
*(Tuyệt đối không suy diễn cảm tính kiểu "accuracy = 93% nghĩa là model hiểu ngôn ngữ tốt").*

---

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
* **Observation:** Khi thu hẹp quy mô dữ liệu huấn luyện xuống mức khan hiếm ($10\%$ và $25\%$), mô hình Pretrained Transformer duy trì được độ chính xác ổn định và vượt xa mô hình Transformer huấn luyện từ đầu (From Scratch).
* **Evidence:** Theo số liệu **Experiment 3 (Mục 16)** tại `results.csv` và `data_efficiency.png`:
  * Tại mức $10\%$ dữ liệu ($1,000$ mẫu), Pretrained BERT đạt **Accuracy $89.20\%$, F1 $0.8915$** (chỉ giảm $4.28\%$ so với mức $93.48\%$ ở $100\%$ dữ liệu).
  * Ngược lại, tại mức $10\%$ dữ liệu, Transformer From Scratch rơi tự do xuống **Accuracy $68.40\%$, F1 $0.6792$** (sụt giảm tới $20.20\%$ so với mức $88.60\%$ ở $100\%$ dữ liệu).
  * Khoảng cách hiệu năng giữa Pretrained và From Scratch tăng vọt từ $+4.88\%$ (tại $100\%$ data) lên tới **$+20.80\%$** (tại $10\%$ data).
* **Interpretation:** **Pretrained model hưởng lợi nhiều hơn gấp nhiều lần khi dữ liệu huấn luyện ít.** Nguyên nhân là vì quá trình học biểu diễn từ vựng và cấu trúc cú pháp đã được hoàn tất trong giai đoạn Pretraining trên kho văn bản lớn; tập dữ liệu nhỏ ở hạ nguồn chỉ cần dùng để định vị siêu phẳng phân loại của lớp đầu ra, giúp mô hình không bị rơi vào tình trạng thiếu hụt từ vựng (OOV) và Overfitting nặng như mô hình khởi tạo ngẫu nhiên.

---

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
* **Observation:** Trên cùng bài toán phân loại cảm xúc IMDb với cùng phép chia dữ liệu và cùng độ dài chuỗi $T=128$, mô hình Pretrained Fine-tuned đạt kết quả vượt trội trên cả 4 chỉ số Accuracy, Precision, Recall và F1.
* **Evidence:** Theo **Experiment 1 (Mục 14)**:
  * Pretrained BERT đạt **Accuracy $93.48\%$, Precision $0.9331$, Recall $0.9368$, F1 $0.9349$** (chỉ mắc $163$ lỗi trên $2,500$ mẫu).
  * Transformer From Scratch chỉ đạt **Accuracy $88.60\%$, Precision $0.8888$, Recall $0.8824$, F1 $0.8856$** (mắc $285$ lỗi).
* **Interpretation:** Kết quả này **không có nghĩa là mô hình "hiểu ngôn ngữ như con người"**, mà chứng minh một nguyên lý khoa học cụ thể: **Các đặc trưng ngữ cảnh hai chiều thu được từ mục tiêu tự giám sát Masked Language Modeling (MLM) kết hợp với tách từ Subword chứa đựng tri thức cấu trúc ngôn ngữ có tính chuyển giao cao (transferable inductive bias)**, giúp bộ phân loại hạ nguồn phân biệt được các mẫu câu phức tạp (như phủ định gián tiếp hoặc từ vựng ít xuất hiện trong tập train) mà một mô hình học từ con số 0 chỉ trên dữ liệu bài toán đích không thể tự khái quát hóa được.

---

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
* **Observation:** Xét kịch bản thực nghiệm trong đó việc đóng băng toàn bộ Pretrained Encoder (`Frozen Mode`) cho ra Accuracy/F1 gần sát với việc cập nhật toàn bộ mạng (`Fine-tuning Mode`).
* **Evidence:** Ví dụ trên các tác vụ phân loại chủ đề văn bản đơn giản (như phân biệt tin Thể thao vs Kinh tế) hoặc khi khoảng cách giữa Frozen và Fine-tuned chỉ lệch $< 1\% - 2\%$.
* **Interpretation:** Điều đó chứng tỏ rằng **không gian biểu diễn vector `[CLS]` sinh ra sẵn bởi mục tiêu tiền huấn luyện MLM đã đủ để phân tách tuyến tính (linearly separable) các lớp của bài toán đích**. Nói cách khác, các đặc trưng cốt lõi cần cho bài toán đích (như từ khóa chủ đề) đã được mã hóa nổi bật ngay trong biểu diễn tổng quát của Pretrained Encoder mà không đòi hỏi phải tái cấu trúc lại trọng số của các tầng Self-Attention.

---

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
* **Observation:** Trong thực nghiệm phân loại cảm xúc IMDb của chúng ta, mở khóa Pretrained Encoder để fine-tune mang lại bước nhảy vọt lớn so với giữ nguyên Encoder đóng băng.
* **Evidence:** Theo **Experiment 2 (Mục 15)**:
  * `Mode A (Frozen Encoder)` chỉ đạt **Accuracy $86.40\%$, F1 $0.8636$**.
  * `Mode B (Full Fine-tuning)` tăng vọt lên **Accuracy $93.48\%$, F1 $0.9349$** (cải thiện **+$7.08\%$ Accuracy** và **+$0.0713$ F1**, giảm hơn một nửa tỷ lệ lỗi từ $13.60\%$ xuống $6.52\%$).
* **Interpretation:** Khoảng cách $+7.08\%$ này gợi ý rằng **biểu diễn ngữ cảnh mặc định từ bài toán MLM là biểu diễn đa dụng (task-agnostic) chứ chưa được căn chỉnh tối ưu cho cực tính cảm xúc (sentiment polarity)**. Trong một bài review phim dài, vector `[CLS]` của Frozen Encoder trộn lẫn cả thông tin tóm tắt cốt truyện, tên nhân vật lẫn nhận xét khen chê. Quá trình Fine-tuning là bắt buộc để đạo hàm từ hàm mất mát phân loại điều chỉnh lại các ma trận $W^Q, W^K, W^V$ bên trong các tầng Attention, giúp `[CLS]` học cách tập trung có chọn lọc vào các tính từ cảm xúc và mệnh đề đảo chiều (sau *"but"*, *"however"*).

---

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
* **Observation:** Mô hình Pretrained BERT sau khi đạt độ chính xác rất cao trên tập Test cùng miền (Movie Reviews) lại bị sụt giảm hiệu năng nghiêm trọng khi đem sang đánh giá trực tiếp trên các tập văn bản thuộc miền khác (Electronics Reviews và Financial/Scientific News).
* **Evidence:** Theo **Experiment 5 (Mục 18)**:
  * $\text{Performance}_{\text{in-domain}}$ (Domain A — IMDb Movies): **Accuracy $93.48\%$, F1 $0.9349$**.
  * $\text{Performance}_{\text{out-of-domain}}$ (Domain B1 — Electronics Product Reviews): **Accuracy $83.12\%$, F1 $0.8278$** (giảm **$-10.36\%$ Accuracy**, **$-0.1071$ F1**).
  * $\text{Performance}_{\text{out-of-domain}}$ (Domain B2 — Financial/Scientific News): **Accuracy $74.60\%$, F1 $0.7348$** (giảm **$-18.88\%$ Accuracy**, **$-0.2001$ F1**).
* **Interpretation:** Sự suy giảm mạnh này bắt nguồn từ 3 cơ chế của hiện tượng **Domain Shift**:
  1. **Dịch chuyển phân phối từ vựng (Covariate / Lexical Shift):** Nhiều thuật ngữ quyết định nhãn ở miền mới (như *"thermal throttling"*, *"coil whine"* trong đồ điện tử hay *"margin contraction"*, *"default risk"* trong tài chính) hoàn toàn vắng mặt trong tập huấn luyện phim ảnh IMDb.
  2. **Đảo chiều cực tính theo ngữ cảnh miền (Domain-Dependent Polarity Shift):** Các từ như *"unpredictable"*, *"dramatic"*, *"chilling"* mang nhãn `Positive` trong bình luận phim trinh thám/kinh dị, nhưng lại mang nhãn `Negative` khi mô tả sự cố thiết bị hoặc biến động thị trường tài chính.
  3. **Khác biệt về văn phong (Style & Register Shift):** Quá trình fine-tune trên IMDb đã làm mô hình phụ thuộc vào các tính từ cảm thán chủ quan, nên mất hiệu lực trước văn phong báo cáo số liệu khách quan của bản tin tài chính/khoa học.

---

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
* **Observation:** Khi nâng ngưỡng độ dài chuỗi tối đa từ $T = 128$ lên $T = 256$, các chỉ số Accuracy và F1 gần như bão hòa trong khi thời gian huấn luyện và dung lượng bộ nhớ tăng lên hơn gấp đôi.
* **Evidence:** Theo **Experiment 4 (Mục 17)**:
  * Khi nâng từ $T = 64 \to 128$: Accuracy tăng mạnh **$+3.64\%$** ($89.84\% \to 93.48\%$), F1 tăng $+0.0370$.
  * Nhưng khi nâng từ $T = 128 \to 256$: Accuracy chỉ nhích thêm **$+0.64\%$** ($93.48\% \to 94.12\%$, F1 tăng $+0.0064$), trong khi **Training time tăng gấp $2.24\times$** ($142.5\text{s} \to 318.6\text{s}$) và **Memory tăng gấp $2.18\times$** ($2.68\text{ GB} \to 5.84\text{ GB}$) do kích thước ma trận Self-Attention mỗi head tăng gấp $4$ lần ($128^2 = 16,384 \to 256^2 = 65,536$).
* **Interpretation:** **Tuyệt đối không nên tiếp tục tăng sequence length trong trường hợp này.** Về mặt toán học, chi phí của tầng Self-Attention tăng theo hàm bậc hai $\mathcal{O}(T^2)$ trong khi lợi ích thông tin tuân theo quy luật hiệu suất cận biên giảm dần (phần lớn thông tin cảm xúc cốt lõi đã nằm trong $128$ tokens đầu tiên). Tiếp tục nâng $T$ lên $512$ sẽ làm chi phí tính toán tăng thêm gấp $4$ lần nữa mà hầu như chỉ xử lý thêm các token `[PAD]` hoặc các câu mô tả cốt truyện dư thừa.
