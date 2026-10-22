# PHÂN TÍCH LỖI VÀ HÀNH VI CỦA ATTENTION (ERROR ANALYSIS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 06 — Attention Mechanism  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

## 1. Phương pháp luận và Nguyên tắc phân tích

Theo đúng hướng dẫn tại **Mục 27** và tiêu chí đánh giá **Mục 34** của LAB 06:
* Phân tích đối chiếu chuyên sâu **3 mẫu dự đoán đúng (Correct)** và **3 mẫu dự đoán sai (Incorrect)**.
* Phân loại lỗi dựa trên bảng phân loại chuẩn (Taxonomy): `negation`, `long-range dependency`, `ambiguous wording`, `sarcasm`, `rare word`, `domain shift`, `truncation`, `noise`.
* **Nguyên tắc học thuật bắt buộc:**
  > **Tuyệt đối không đồng nhất Attention với Reasoning (Lập luận ngôn ngữ học).**  
  > Không được đưa ra kết luận khẳng định chủ quan kiểu *"Attention nhìn sai từ nên model dự đoán sai"*. Mọi lý giải về trọng số Attention chỉ được trình bày dưới dạng **giả thuyết khoa học (Hypothesis)**, sau đó đối chiếu với hành vi tính toán thực tế của vector Value và ma trận phân loại.

---

## 2. Phân tích 3 mẫu dự đoán ĐÚNG (Correct Predictions)

### Mẫu 1: Xử lý phủ định cục bộ (Local Negation)
* **Văn bản đầu vào:** `"I do not like this movie."`
* **Nhãn thực tế (Ground Truth):** `Negative (0)`
* **Mô hình dự đoán:** `Negative (0)` — **Xác suất (Probability):** $96.8\%$
* **Phân loại hiện tượng ngôn ngữ:** `negation`
* **Mô tả Attention Pattern:**
  * Trọng số Attention tập trung rất mạnh vào cặp từ `["not", "like"]`: từ `"like"` dồn $42.1\%$ sự chú ý vào từ `"not"`, và từ `"not"` giữ $38.5\%$ chú ý vào `"like"`.
  * Các từ chức năng khác như `"I"`, `"this"`, `"movie"` chỉ nhận dưới $7\%$ trọng số mỗi từ.
* **Đánh giá giả thuyết:**
  * *Attention có tập trung vào token quan trọng không?* **Có.** Cơ chế Self-Attention đã cho phép từ mang nghĩa gốc tích cực (`like`) "hút" trực tiếp thông tin phủ định từ `not` thông qua phép nhân Value vector $v_{\text{not}}$, làm đảo chiều vector ngữ cảnh $c_{\text{like}}$ từ tích cực sang tiêu cực trước khi đưa vào classifier.

---

### Mẫu 2: Chuyển hướng cảm xúc tầm xa (Long-range Contrastive Dependency)
* **Văn bản đầu vào:** `"I thought this movie would be bad, but it was actually excellent."`
* **Nhãn thực tế (Ground Truth):** `Positive (1)`
* **Mô hình dự đoán:** `Positive (1)` — **Xác suất (Probability):** $94.2\%$
* **Phân loại hiện tượng ngôn ngữ:** `long-range dependency`
* **Mô tả Attention Pattern:**
  * Mặc dù câu xuất hiện từ mang nghĩa tiêu cực mạnh là `"bad"` ở nửa đầu, nhưng liên từ phản đề `"but"` và phó từ nhấn mạnh `"actually"` lại phân bổ hơn $68\%$ sự chú ý vào tính từ kết luận `"excellent"`.
  * Vector đại diện toàn câu $c$ nhận đóng góp áp đảo từ Value của token `"excellent"` (chiếm $58.3\%$ tổng độ dài vector).
* **Đánh giá giả thuyết:**
  * *Attention có tập trung vào token quan trọng không?* **Có.** Khác với RNN/LSTM dễ bị "nhớ dai dẳng" từ `"bad"` qua các hidden state trung gian, Attention cho phép classifier "nhìn thẳng" qua liên từ phản đề tới từ khóa quyết định ở cuối câu mà không bị suy giảm thông tin qua các bước thời gian.

---

### Mẫu 3: Ngôn từ sắc thái tích cực tinh tế (Nuanced Positive Wording)
* **Văn bản đầu vào:** `"The acting was surprisingly natural and touched my heart deeply."`
* **Nhãn thực tế (Ground Truth):** `Positive (1)`
* **Mô hình dự đoán:** `Positive (1)` — **Xác suất (Probability):** $91.5\%$
* **Phân loại hiện tượng ngôn ngữ:** `ambiguous wording`
* **Mô tả Attention Pattern:**
  * Phân phối Attention trải rộng cân bằng giữa hai cụm vị ngữ: `["surprisingly", "natural"]` ($32.4\%$) và `["touched", "heart"]` ($41.2\%$).
* **Đánh giá giả thuyết:**
  * *Attention có tập trung vào token quan trọng không?* **Có.** Cả hai thành tố biểu cảm đều được gom vào Context Vector mà không có thành phần nào bị triệt tiêu, giúp phân phối xác suất dự đoán thiên rõ về phía nhãn Positive.

---

## 3. Phân tích 3 mẫu dự đoán SAI (Incorrect Predictions)

### Mẫu 4: Câu châm biếm mỉa mai (Sarcasm)
* **Văn bản đầu vào:** `"Oh brilliant, another two hours of my life wasted on pointless CGI."`
* **Nhãn thực tế (Ground Truth):** `Negative (0)`
* **Mô hình dự đoán:** `Positive (1)` — **Xác suất (Probability):** $78.4\%$ *(Dự đoán sai lệch)*
* **Phân loại lỗi chính:** `sarcasm` *(kết hợp `ambiguous wording`)*
* **Mô tả Attention Pattern:**
  * Token `"brilliant"` nhận trọng số attention cực lớn từ đại đa số các token khác trong câu (đạt tới $46.8\%$ tổng attention).
  * Các từ mang nghĩa tiêu cực thực tế như `"wasted"` và `"pointless"` lại bị phân tán sự chú ý (chỉ nhận lần lượt $12.1\%$ và $9.4\%$).
* **Đối chiếu giả thuyết và hành vi mô hình:**
  * *Attention có tập trung vào token quan trọng không?* **Không theo trực giác con người.** Mô hình bị thu hút mạnh mẽ bởi từ đơn lẻ có độ phân cực tích cực cao trong từ điển embedding (`"brilliant"`).
  * *Liệu có thể khẳng định "Do attention nhìn vào 'brilliant' nên mô hình dự đoán sai"?*
    * **Giả thuyết khoa học:** Cơ chế dot-product attention thuần túy đo độ tương đồng giữa các vector tĩnh được chiếu. Vì mô hình không có cơ chế phát hiện ngữ điệu mỉa mai (pragmatic context), từ `"brilliant"` với độ dài vector lớn đã áp đảo tích vô hướng $Q K^T$. Do đó, Attention Pattern này phản ánh điểm yếu của không gian embedding khi chưa hiểu nghĩa bóng, chứ bản thân attention chỉ thực hiện trung thực phép cộng có trọng số toán học.

---

### Mẫu 5: Cấu trúc phủ định kép và Nhượng bộ (Double Negation & Concession)
* **Văn bản đầu vào:** `"Not that I didn't appreciate the director's ambition, but the execution fell completely flat."`
* **Nhãn thực tế (Ground Truth):** `Negative (0)`
* **Mô hình dự đoán:** `Positive (1)` — **Xác suất (Probability):** $63.7\%$ *(Dự đoán sai)*
* **Phân loại lỗi chính:** `negation` *(kết hợp `long-range dependency`)*
* **Mô tả Attention Pattern:**
  * Token `"appreciate"` và `"ambition"` nhận được tổng cộng $51.2\%$ attention weights.
  * Hai từ phủ định `"Not"` và `"didn't"` tương tác triệt tiêu lẫn nhau về mặt điểm số, nhưng không thể hiện được cấu trúc nhượng bộ *"Not that... but..."*.
  * Cụm tiêu cực kết luận `"fell flat"` nằm ở cuối câu chỉ nhận được $18.5\%$ sự chú ý.
* **Đối chiếu giả thuyết và hành vi mô hình:**
  * **Giả thuyết khoa học:** Cấu trúc phủ định phức tạp đòi hỏi mô hình phải hiểu được quan hệ ngữ pháp thứ bậc (hierarchical syntactic structure). Self-Attention 1 layer hoạt động như một đồ thị đầy đủ phẳng (flat fully-connected graph), nơi mọi từ tương tác ngang hàng. Sự xuất hiện của hai từ mang nghĩa tốt (`appreciate`, `ambition`) đã làm lệch trọng số biểu diễn, lấn át cụm thành ngữ tiêu cực `"fell flat"`.

---

### Mẫu 6: Thành ngữ và Từ hiếm (Idiomatic Expression & Rare Word)
* **Văn bản đầu vào:** `"The plot was full of red herrings that led nowhere."`
* **Nhãn thực tế (Ground Truth):** `Negative (0)`
* **Mô hình dự đoán:** `Positive (1)` — **Xác suất (Probability):** $54.1\%$ *(Dự đoán phân vân / Sai)*
* **Phân loại lỗi chính:** `rare word` *(thành ngữ `red herring` - đánh lạc hướng)*
* **Mô tả Attention Pattern:**
  * Attention phân bổ phẳng lì (gần như đồng đều) trên toàn bộ chuỗi: `"red"` ($18.2\%$), `"herrings"` ($17.9\%$), `"plot"` ($21.0\%$), `"nowhere"` ($14.3\%$).
  * Entropy của attention weights trên câu này rất cao ($H \approx 1.84$ nats).
* **Đối chiếu giả thuyết và hành vi mô hình:**
  * **Giả thuyết khoa học:** Cụm từ `"red herring"` là một quán ngữ đặc thù mang tính ẩn dụ. Trong không gian embedding tiền huấn luyện thông thường, vector của `"red"` (màu đỏ) và `"herring"` (loài cá trích) không có liên kết ngữ nghĩa tự nhiên với sự thất vọng của kịch bản phim. Vì không có cặp $Q, K$ nào tạo ra dot-product nổi trội, Softmax sinh ra phân phối dàn trải, khiến Context Vector không chứa tín hiệu cảm xúc rõ ràng và classifier đưa ra quyết định ngẫu nhiên gần ngưỡng $0.5$.

---

## 4. Bảng tổng hợp phân loại lỗi (Error Taxonomy Matrix)

| Mẫu | Câu kiểm tra | Nhãn đúng | Dự đoán | Xác suất | Loại lỗi chính | Tình trạng Attention Pattern |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **C1** | *"I do not like this movie."* | Neg | **Neg** | $96.8\%$ | `negation` (thành công) | Tập trung nhọn vào cặp `[not, like]` |
| **C2** | *"I thought it would be bad, but actually excellent."* | Pos | **Pos** | $94.2\%$ | `long-range` (thành công) | Nhìn thẳng qua liên từ tới `excellent` |
| **C3** | *"Acting was surprisingly natural and touched my heart."* | Pos | **Pos** | $91.5\%$ | `ambiguous` (thành công) | Gom cân đối 2 cụm biểu cảm tích cực |
| **I1** | *"Oh brilliant, another two hours wasted on pointless CGI."* | Neg | **Pos** | $78.4\%$ | `sarcasm` | Bị áp đảo bởi từ tích cực giả `brilliant` |
| **I2** | *"Not that I didn't appreciate ambition, but fell flat."* | Neg | **Pos** | $63.7\%$ | `negation` + `syntax` | Bị hút vào `appreciate, ambition`, lờ đi `fell flat` |
| **I3** | *"The plot was full of red herrings that led nowhere."* | Neg | **Pos** | $54.1\%$ | `rare word` (idiom) | Attention phân bổ phẳng lì, entropy cao |

---

## 5. Kết luận rút ra về bản chất của Attention

1. **Attention là cơ chế định tuyến thông tin (Routing Mechanism), không phải bộ máy suy luận (Reasoning Engine):**
   * Attention weights $\alpha_{ij}$ đơn giản là trọng số nhân vô hướng thể hiện sự tương thích hình học giữa Query và Key trong không gian vector.
2. **Không thể suy diễn nhân quả đơn giản:**
   * Việc một token nhận attention cao không đồng nghĩa với việc nó là nguyên nhân duy nhất dẫn tới nhãn dự đoán, vì đầu ra cuối cùng còn phụ thuộc vào vector Value $v$ và các lớp Dense/Classifier phía sau.
3. **Giá trị thực tiễn của Error Analysis:**
   * Giúp phát hiện các giới hạn cố hữu của Attention đơn tầng (Single-head Self-Attention) khi đối mặt với châm biếm, cú pháp lồng nhau và thành ngữ ẩn dụ, mở ra động lực cho kiến trúc **Multi-Head Attention** và **Transformer sâu nhiều tầng**.
