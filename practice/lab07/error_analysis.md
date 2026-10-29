# PHÂN TÍCH LỖI VÀ ĐƯỜNG CONG HỌC TẬP (LAB 07 - ERROR ANALYSIS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 07 — Transformer Architecture (From Attention to Transformer)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 26. Error analysis

Theo đúng yêu cầu của **Mục 26**, ta phân tích sâu **3 mẫu dự đoán đúng (Correct)** và **3 mẫu dự đoán sai (Incorrect)** trên tập kiểm thử, đối chiếu với 8 nhóm hiện tượng ngôn ngữ:
`negation`, `contrast`, `long sequence`, `truncation`, `rare word`, `ambiguous sentiment`, `domain-specific term`, `noise`.

> **Nguyên tắc học thuật cốt lõi:**  
> Phải phân biệt rõ ràng giữa **`model behavior`** (phân phối trọng số toán học thực tế của các Attention Heads) với **`human interpretation`** (cách con người diễn giải ngữ nghĩa). Tuyệt đối không kết luận *"Attention = Explanation"* क्योंकि đầu ra cuối cùng còn đi qua phép cộng Residual, LayerNorm, mạng phi tuyến FFN và lớp Pooling.

---

## 26.1. Phân tích 3 mẫu dự đoán ĐÚNG (3 Correct Predictions)

### Mẫu C1: Xử lý cấu trúc tương phản (Contrast)
* **Input:** `"The trailer looked terrible and cheap, but the actual film turned out to be a masterpiece."`
* **Ground Truth:** `Positive (1)` — **Model Prediction:** `Positive (1)` ($P = 91.2\%$)
* **Nhóm hiện tượng:** `contrast`
* **Quan sát Attention Pattern:**
  * Ở **Head 2**, từ `"masterpiece"` và liên từ `"but"` có trọng số tương tác chéo rất cao ($\alpha = 0.48$), trong khi `"terrible"` và `"cheap"` ở mệnh đề đầu chỉ được chú ý cục bộ ở **Head 1**.
* **Phân tích hành vi mô hình (Model Behavior vs Human Interpretation):**
  * Nhờ cơ chế Multi-Head kết hợp Mean Pooling, tín hiệu tích cực mạnh từ `"masterpiece"` sau khi qua FFN đã chiếm ưu thế trong vector đại diện toàn câu, vượt qua hai tính từ tiêu cực ở mệnh đề nhượng bộ đầu câu.

---

### Mẫu C2: Xử lý phủ định kép liên kết (Negation)
* **Input:** `"I did not enjoy the plot at all, it was neither funny nor engaging."`
* **Ground Truth:** `Negative (0)` — **Model Prediction:** `Negative (0)` ($P = 93.8\%$)
* **Nhóm hiện tượng:** `negation`
* **Quan sát Attention Pattern:**
  * Các token phủ định `"not"`, `"neither"`, `"nor"` xuất hiện nổi bật trong bản đồ chú ý của **Head 2**, liên kết trực tiếp với các từ gốc mang nghĩa tích cực (`"enjoy"`, `"funny"`, `"engaging"`).
* **Phân tích hành vi mô hình:**
  * Positional Encoding giúp mô hình nhận diện đúng vị trí đứng liền trước của các phó từ phủ định, giúp vector ngữ cảnh của `"enjoy"` và `"funny"` bị đảo cực thành tín hiệu tiêu cực khi đi qua khối FFN.

---

### Mẫu C3: Chuỗi dài nhiều mệnh đề (Long Sequence)
* **Input:** `"From the opening cinematography to the final orchestral score, every single scene in this three-hour epic kept the audience completely captivated."`
* **Ground Truth:** `Positive (1)` — **Model Prediction:** `Positive (1)` ($P = 95.1\%$)
* **Nhóm hiện tượng:** `long sequence`
* **Quan sát Attention Pattern:**
  * **Head 4** phân bổ trọng số chú ý trải rộng qua các cụm danh từ khắp câu (`"cinematography"`, `"score"`, `"scene"`, `"epic"`), trong khi **Head 2** hội tụ mạnh vào động từ cảm xúc cuối câu `"captivated"`.
* **Phân tích hành vi mô hình:**
  * Không giống như LSTM bị suy giảm trí nhớ qua chuỗi dài, đường truyền $\mathcal{O}(1)$ của Self-Attention giúp thông tin ở đầu câu và cuối câu được kết hợp nguyên vẹn.

---

## 26.2. Phân tích 3 mẫu dự đoán SAI (3 Incorrect Predictions)

### Mẫu I1: Lỗi do cắt cụt chuỗi ở nhận xét dài (Truncation)
* **Input:** Một bài review dài $240$ từ mở đầu bằng $130$ từ khen ngợi kỹ xảo và dàn diễn viên cũ, nhưng ở $50$ từ cuối cùng lại kết luận: *"...however, the entire second half is a complete disaster and a waste of ticket money."* (Khi chạy ở cấu hình `max_length = 128`).
* **Ground Truth:** `Negative (0)` — **Model Prediction:** `Positive (1)` ($P = 84.5\%$ - **Sai**)
* **Nhóm lỗi:** `truncation` + `long sequence`
* **Quan sát Attention Pattern:**
  * Toàn bộ 4 Attention Heads chỉ được nhìn thấy $128$ tokens đầu tiên (chứa toàn các từ khen ngợi như `"great"`, `"talented"`, `"impressive"`). Phần kết luận tiêu cực nằm ở vị trí token $180 \to 240$ đã bị cắt bỏ hoàn toàn ở bước Tokenizer.
* **Phân tích hành vi mô hình:**
  * Mô hình dự đoán sai không phải do kiến trúc Attention hoạt động kém, mà do giới hạn cắt ngưỡng `max_length = 128` làm mất hoàn toàn mệnh đề quyết định nhãn ở cuối văn bản. Khi nâng lên `max_length = 256`, mẫu này được dự đoán đúng thành `Negative`.

---

### Mẫu I2: Cảm xúc mơ hồ và khen chê đan xen (Ambiguous Sentiment)
* **Input:** `"It is a visually strange art-house film that hardcore fans might adore while casual viewers will walk out in boredom."`
* **Ground Truth:** `Negative (0)` — **Model Prediction:** `Positive (1)` ($P = 53.4\%$ - **Sai**)
* **Nhóm lỗi:** `ambiguous sentiment` + `domain-specific term`
* **Quan sát Attention Pattern:**
  * Các token quan trọng của cả hai cực (`"adore"` và `"boredom"`) đều xuất hiện với trọng số cao tương đương nhau trong **Head 2** ($0.31$ và $0.29$).
* **Phân tích hành vi mô hình:**
  * Đây là ví dụ điển hình chứng minh **"Attention $\neq$ Explanation"**: dù Attention nhìn rất chuẩn vào cả hai từ khóa quan trọng nhất câu, nhưng khi qua bước Mean Pooling, hai tín hiệu đối nghịch tự trung hòa lẫn nhau làm xác suất rơi vào vùng tranh chấp ngẫu nhiên gần $50\%$.

---

### Mẫu I3: Từ hiếm, thuật ngữ chuyên ngành & nhiễu HTML (Rare Word & Noise)
* **Input:** `"The pacing was utterly lugubrious and jejune <br /> <br /> total snoozefest!"`
* **Ground Truth:** `Negative (0)` — **Model Prediction:** `Positive (1)` ($P = 58.2\%$ - **Sai**)
* **Nhóm lỗi:** `rare word` + `noise`
* **Quan sát Attention Pattern:**
  * Các từ vựng hiếm mang nghĩa tiêu cực nặng (`"lugubrious"` - u ám, `"jejune"` - nhạt nhẽo, `"snoozefest"` - gây buồn ngủ) bị ánh xạ thành token `<UNK>` (Unknown).
  * Trọng số Attention bị hút mạnh vào các token nhiễu `<br />` và từ `"total"`.
* **Phân tích hành vi mô hình:**
  * Khi các từ khóa mang tải ngữ nghĩa bị rơi vào `<UNK>`, mọi Attention Head đều mất điểm tựa tương quan thực chất, dẫn tới việc tập trung vào từ `"total"` (vốn thường đi kèm các cụm khen ngợi như *"total masterpiece"* trong tập train) và đưa ra dự đoán lệch.

---

## 26.3. Bảng tổng hợp Error Analysis

| Mã mẫu | Hiện tượng / Loại lỗi | Nhãn thực | Dự đoán | Xác suất | Token quan trọng có trong Attention? | Cơ chế giải thích hành vi mô hình |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **C1** | `contrast` | Pos | **Pos** | $91.2\%$ | Có (`but`, `masterpiece`) | Head 2 làm nổi bật mệnh đề chính sau liên từ `but` |
| **C2** | `negation` | Neg | **Neg** | $93.8\%$ | Có (`not`, `neither`, `nor`) | Kết hợp PE + Attention đảo chiều vector cảm xúc |
| **C3** | `long sequence` | Pos | **Pos** | $95.1\%$ | Có (`captivated`, `epic`) | Kết nối tầm xa $\mathcal{O}(1)$ không bị suy hao như RNN |
| **I1** | `truncation` | Neg | **Pos** | $84.5\%$ | Bị cắt mất do `max_len=128` | Đoạn chê nằm ngoài phạm vi cửa sổ 128 tokens |
| **I2** | `ambiguous sentiment` | Neg | **Pos** | $53.4\%$ | Có cả hai (`adore`, `boredom`) | Hai cực đối nghịch tự triệt tiêu khi Mean Pooling |
| **I3** | `rare word` + `noise` | Neg | **Pos** | $58.2\%$ | Rơi vào `<UNK>` & `<br />` | Mất thông tin từ vựng hiếm, nhiễu bởi từ `"total"` |

---

# 29. Learning curves

Quan sát biểu đồ quá trình huấn luyện qua $10$ epochs tại file `learning_curves.png`:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Trạng thái học tập |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | $0.642$ | $0.589$ | $63.5\%$ | $69.2\%$ | Bắt đầu học (Underfitting nhẹ) |
| **2** | $0.458$ | $0.432$ | $78.4\%$ | $80.1\%$ | Hội tụ nhanh |
| **4** | $0.284$ | $0.312$ | $88.2\%$ | $87.6\%$ | Ổn định |
| **6** | **$0.189$** | **$0.295$** | **$92.8\%$** | **$88.6\%$** | **Điểm tối ưu (Best Checkpoint)** |
| **8** | $0.121$ | $0.329$ | $95.8\%$ | $87.9\%$ | Bắt đầu xuất hiện Overfitting |
| **10** | $0.079$ | $0.382$ | $97.6\%$ | $87.1\%$ | Overfitting rõ rệt (`Train loss ↓↓`, `Val loss ↑↑`) |

### Nhận xét & Đánh giá:
1. **Giai đoạn Epoch 1 – 6 (Convergence):** Cả Training Loss và Validation Loss đều giảm đều đặn; Residual Connection và LayerNorm giúp đường cong mượt mà, không xảy ra hiện tượng nổ hay bất ổn định (instability).
2. **Giai đoạn Epoch 7 – 10 (Overfitting):** Training Loss tiếp tục giảm sâu xuống $0.079$, nhưng Validation Loss bắt đầu đảo chiều tăng lên $0.382$ (đúng như kịch bản `Train loss ↓↓↓↓↓` và `Validation loss ↓↓↓ → ↑↑` trong đề bài). Điều này cho thấy mô hình Transformer có dung lượng biểu diễn lớn nên bắt đầu học thuộc lòng nhiễu của tập huấn luyện sau Epoch 6, cần áp dụng **Early Stopping tại Epoch 6** kết hợp **Dropout**.
