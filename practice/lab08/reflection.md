# BÁO CÁO TỔNG KẾT, VẤN ĐÁP CÁ NHÂN VÀ MẠCH NỐI HỌC PHẦN (LAB 08 - REFLECTION)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 08 — Pretrained Transformers and BERT Fine-Tuning  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 27. Individual Learning Check

Trả lời ngắn gọn, rõ ràng và bằng chính lời văn tự đúc kết cho toàn bộ **12 câu hỏi kiểm tra kiến thức cá nhân (Mục 27)**:

### 1. Pretraining khác fine-tuning ở đâu?
* **Trả lời:** **Pretraining** là giai đoạn huấn luyện mô hình từ đầu trên kho văn bản khổng lồ không cần nhãn (như Wikipedia, BookCorpus) bằng các bài toán tự giám sát (như dự đoán từ bị che MLM) nhằm học biểu diễn ngôn ngữ tổng quát ($\theta_{\text{pretrained}}$). Ngược lại, **Fine-tuning** lấy chính bộ trọng số $\theta_{\text{pretrained}}$ đó làm điểm xuất phát, gắn thêm một lớp đầu ra chuyên biệt (`task-specific head`) và tiếp tục huấn luyện có giám sát trên tập dữ liệu có nhãn của bài toán cụ thể ($D_{\text{task}}$) với tốc độ học nhỏ để thu được $\theta_{\text{task}}$.

### 2. MLM là gì?
* **Trả lời:** **MLM (Masked Language Modeling)** là mục tiêu tiền huấn luyện tự giám sát cốt lõi của BERT: một tỷ lệ nhỏ các token trong câu (thường là $15\%$) bị thay thế ngẫu nhiên bằng ký hiệu `[MASK]`, và mô hình phải dựa vào toàn bộ ngữ cảnh ở cả bên trái lẫn bên phải để dự đoán lại đúng token gốc tại các vị trí bị che thông qua hàm mất mát Cross-Entropy.

### 3. Tại sao cần tokenizer?
* **Trả lời:** Mô hình Transformer chỉ có thể tính toán trên các tensor số học chứ không thể đọc trực tiếp chuỗi ký tự văn bản thô. **Tokenizer** thực hiện việc chuẩn hóa văn bản, cắt câu thành chuỗi các đơn vị từ/mảnh từ (`tokens`) theo quy tắc nhất quán với lúc pretrain, rồi ánh xạ từng token thành chỉ số nguyên (`input_ids`) để mạng tra cứu được đúng vector trong bảng `Embedding`.

### 4. Subword token giải quyết vấn đề gì?
* **Trả lời:** **Subword tokenization** (như WordPiece, BPE) giải quyết triệt để bài toán đánh đổi giữa kích thước từ điển và hiện tượng từ lạ ngoài từ điển (**Out-of-Vocabulary — OOV**). Thay vì biến mọi từ hiếm, từ ghép hoặc từ biến đổi hình thái thành token vô nghĩa `[UNK]` (như word-level tokenizer), subword giữ nguyên các từ phổ biến và tách các từ hiếm thành các mảnh có nghĩa (như `"unbelievable"` $\to$ `["un", "##believe", "##able"]`), giúp mô hình vẫn hiểu được cấu trúc gốc từ và hậu tố.

### 5. `[CLS]` thường được sử dụng như thế nào?
* **Trả lời:** Token đặc biệt **`[CLS]`** luôn được chèn ở vị trí đầu tiên ($t = 0$) của mọi chuỗi đầu vào trong các mô hình họ BERT. Do không mang nghĩa từ vựng riêng của một từ cụ thể nào, sau khi đi qua nhiều tầng Self-Attention hai chiều, vector ẩn cuối cùng của `[CLS]` ($h_{\text{[CLS]}} \in \mathbb{R}^{d_{\text{model}}}$) đóng vai trò là vector tổng hợp đại diện cho toàn bộ câu và được truyền thẳng vào lớp `Linear` (Classification Head) để phân loại văn bản.

### 6. Attention mask dùng để làm gì?
* **Trả lời:** Khi gom nhiều câu có độ dài ngắn dài khác nhau vào cùng một batch, các câu ngắn phải chèn thêm các token đệm `[PAD]`. **`attention_mask`** (gồm `1` tại vị trí token thật và `0` tại vị trí `[PAD]`) được dùng để gán giá trị $-\infty$ (hoặc $-10^9$) vào điểm tương quan Attention của các cột `[PAD]` trước khi qua hàm Softmax, đảm bảo trọng số chú ý vào `[PAD]` bằng $0$ tuyệt đối và ngăn không cho vùng đệm làm sai lệch biểu diễn của câu.

### 7. Frozen encoder nghĩa là gì?
* **Trả lời:** **Frozen encoder** (chế độ Feature Extraction) có nghĩa là khóa cố định toàn bộ các tham số bên trong khối Pretrained Transformer Encoder (`requires_grad = False`), không cập nhật chúng trong lúc lan truyền ngược. Mô hình chỉ sử dụng Encoder như một bộ trích xuất đặc trưng tĩnh và chỉ huấn luyện duy nhất các trọng số của tầng `Classification Head` ở đầu ra.

### 8. Fine-tuning thay đổi những tham số nào?
* **Trả lời:** Trong chế độ **Fine-tuning toàn phần (Full Fine-tuning)**, quá trình tối ưu hóa cập nhật đồng thời **cả các tham số của tầng Classification Head mới khởi tạo lẫn toàn bộ các tham số bên trong Pretrained Transformer Encoder** (bao gồm bảng Embedding, các ma trận chiếu $W^Q, W^K, W^V, W^O$ của Multi-Head Attention, các tầng Feed-Forward Network và LayerNorm).

### 9. Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?
* **Trả lời:** Khi tập dữ liệu bài toán đích nhỏ (ví dụ chỉ có $1,000$ mẫu), một mô hình học từ đầu sẽ không đủ dữ liệu để học ngữ nghĩa từ vựng và cấu trúc câu nên rất dễ bị Overfitting. Ngược lại, **Pretrained model** đã tích lũy sẵn tri thức ngôn ngữ từ hàng tỷ từ ở giai đoạn Pretraining; tập dữ liệu nhỏ lúc này chỉ cần dùng để tinh chỉnh nhẹ ranh giới quyết định của bộ phân loại trên một không gian đặc trưng đã có cấu trúc rất tốt.

### 10. Pretrained model có đảm bảo không bị domain shift không?
* **Trả lời:** **Không.** Dù được tiền huấn luyện trên kho văn bản lớn, khi mô hình được fine-tune trên một miền nguồn (như bình luận phim IMDb) rồi đem áp dụng sang một miền đích khác biệt về từ vựng chuyên ngành, quy ước cực tính hoặc văn phong (như đánh giá thiết bị kỹ thuật hoặc tin tức tài chính/y sinh), hiệu năng vẫn bị suy giảm mạnh (như giảm $-10.36\%$ đến $-18.88\%$ trong Experiment 5).

### 11. Vì sao tăng sequence length làm chi phí attention tăng?
* **Trả lời:** Bởi vì trong cơ chế Self-Attention, mỗi một token trong tổng số $T$ tokens phải tính tích vô hướng Query-Key với tất cả $T$ tokens trong chuỗi, tạo ra ma trận Attention kích thước $T \times T$. Do đó, độ phức tạp tính toán và dung lượng bộ nhớ tăng theo **hàm bậc hai $\mathcal{O}(T^2)$** (khi nhân đôi độ dài chuỗi $T \to 2T$, kích thước ma trận Attention tăng gấp $4$ lần).

### 12. Confidence cao có đảm bảo prediction đúng không?
* **Trả lời:** **Không ($\text{confidence} \neq \text{correctness}$).** Giá trị `confidence = max(softmax(logits))` chỉ đo mức độ nghiêng của logit đầu ra về một lớp dựa trên các đặc trưng mà mô hình nhìn thấy. Khi gặp câu châm biếm (`sarcasm`) chứa nhiều từ khen bề mặt hoặc câu dài bị cắt mất đoạn chê ở cuối (`truncation`), mô hình vẫn có thể đưa ra dự đoán sai với độ tự tin trên $96\%$.

---

# 29. Reflection

> **Chủ đề bài viết (300–500 từ):**  
> *Sau LAB 07, chúng ta đã biết cách xây dựng Transformer. Sau LAB 08, tại sao trong thực tế người ta thường không bắt đầu bằng việc train Transformer từ đầu?*  
> *(Đề cập đầy đủ 6 khía cạnh bắt buộc: `1. data`, `2. pretrained representation`, `3. computational cost`, `4. downstream task`, `5. fine-tuning`, `6. limitation/domain shift`).*

### Bài luận Reflection (435 từ):

Qua hành trình từ **LAB 07** đến **LAB 08**, sự chuyển dịch từ việc tự xây dựng và huấn luyện một kiến trúc Transformer từ đầu (*training from scratch*) sang việc kế thừa và tinh chỉnh một mô hình ngôn ngữ tiền huấn luyện (*Pretrained Transformer + Fine-tuning*) đã lý giải rõ ràng vì sao các hệ thống NLP hiện đại hầu như không bao giờ khởi tạo Transformer từ trọng số ngẫu nhiên cho các bài toán ứng dụng.

Lý do đầu tiên nằm ở bài toán **dữ liệu (`1. data`)**. Kiến trúc Transformer sở hữu dung lượng biểu diễn khổng lồ nhưng lại thiếu các giả định quy nạp tuần tự cứng như RNN, khiến nó cực kỳ "đói" dữ liệu. Trong thực nghiệm của chúng ta, khi lượng dữ liệu huấn luyện giảm xuống chỉ còn $10\%$ ($1,000$ mẫu), mô hình Transformer huấn luyện từ đầu sụt giảm hiệu năng tới $20.20\%$ (xuống còn $68.40\%$ Accuracy) do hiện tượng Overfitting và thiếu hụt độ phủ từ vựng. Trong thực tế doanh nghiệp và nghiên cứu, việc thu thập và gán nhãn thủ công hàng triệu mẫu cho mỗi bài toán mới là bất khả thi.

Đây chính là lúc **biểu diễn tiền huấn luyện (`2. pretrained representation`)** phát huy giá trị cốt lõi. Thông qua các mục tiêu tự giám sát như Masked Language Modeling (MLM) trên kho văn bản không nhãn quy mô hàng tỷ từ, các mô hình như BERT đã mã hóa sẵn cấu trúc cú pháp, quan hệ từ đồng nghĩa, phạm vi phủ định và cơ chế tách từ Subword vào trong các tầng Self-Attention. Thay vì bắt mô hình vừa phải học tiếng Anh từ con số 0 vừa phải học cách phân loại, ta tách bạch hoàn toàn hai giai đoạn này.

Về mặt **chi phí tính toán (`3. computational cost`)**, quá trình Pretraining một mô hình Transformer lớn đòi hỏi hàng trăm giờ xử lý trên các cụm GPU/TPU chuyên dụng và tiêu tốn nguồn năng lượng khổng lồ. Bằng cách tái sử dụng các checkpoint đã được cộng đồng tiền huấn luyện sẵn, chúng ta tiết kiệm được toàn bộ chi phí xây dựng nền tảng đó và chỉ phải trả một chi phí tính toán rất nhỏ (vài phút cho $2-3$ epochs) ở giai đoạn hạ nguồn.

Đối với từng **tác vụ hạ nguồn cụ thể (`4. downstream task`)** như phân loại cảm xúc IMDb, thực nghiệm chứng minh việc chỉ đóng băng Encoder để trích xuất đặc trưng tĩnh đạt $86.40\%$ Accuracy, nhưng khi áp dụng **tinh chỉnh toàn phần (`5. fine-tuning`)**, độ chính xác tăng vọt lên $93.48\%$ (F1 đạt $0.9349$, vượt xa mức $88.60\%$ của mô hình huấn luyện từ đầu). Fine-tuning cho phép đạo hàm từ lớp Classification Head định hướng lại các Attention Heads bên trong Encoder để nhạy bén hơn với các tín hiệu đặc thù của bài toán đích.

Tuy nhiên, Pretrained Transformer không phải là "viên đạn bạc" không có **giới hạn và sự dịch chuyển miền (`6. limitation/domain shift`)**. Khi đem mô hình fine-tune trên bình luận phim sang kiểm thử trên miền đánh giá đồ điện tử và tin tức tài chính, Accuracy sụt giảm lần lượt $-10.36\%$ và $-18.88\%$ do sự thay đổi từ vựng chuyên ngành và đảo chiều cực tính ngữ nghĩa. Bên cạnh đó, rào cản chi phí bậc hai $\mathcal{O}(T^2)$ theo độ dài chuỗi và hiện tượng dự đoán sai với độ tự tin cao trước các câu châm biếm (`sarcasm`) đòi hỏi người kỹ sư NLP phải luôn kết hợp Fine-tuning với việc thích nghi miền (domain adaptation), kiểm soát độ dài chuỗi và phân tích lỗi nghiêm ngặt.

---

# 30. Bridge to LAB 09

**LAB 08** khép lại chuỗi bài toán phân loại ở cấp độ toàn chuỗi (**Sequence Classification**):

```text
Pretrained Transformer
  ↓
Fine-tuning
  ↓
Text Classification (1 nhãn duy nhất cho toàn bộ văn bản)
```

Tuy nhiên, việc gán **một nhãn duy nhất cho cả câu** không thể trả lời được câu hỏi chi tiết hơn ở cấp độ từng từ:
> *Nếu chúng ta muốn biết **từ nào** trong câu là tên người (`PERSON`), tổ chức (`ORGANIZATION`), địa điểm (`LOCATION`) hoặc thực thể y sinh, thì sao?*

### Ví dụ đối chiếu trực quan:
Cho câu đầu vào:
```text
Barack Obama visited Paris.
```
* **Ở LAB 08 (Sequence Classification):**
  ```text
  h_[CLS] ──► Linear Head ──► "political / news text" (1 nhãn cho cả câu)
  ```
* **Ở LAB 09 (Token Classification / Named Entity Recognition):**
  Thay vì chỉ lấy một vector $h_{\text{[CLS]}}$ tại vị trí $0$, ta tận dụng **toàn bộ chuỗi các vector biểu diễn ngữ cảnh** $H = [h_1, h_2, \dots, h_T] \in \mathbb{R}^{T \times d_{\text{model}}}$ ở tầng cuối của Pretrained Transformer để phân loại cho từng token:
  ```text
  Barack Obama  ──►  PERSON    (B-PER, I-PER)
  visited       ──►  O         (Outside)
  Paris         ──►  LOCATION  (B-LOC)
  ```

### Sơ đồ chuyển tiếp sang LAB 09:
```text
Sequence Classification (LAB 08 — [CLS] representation -> 1 label per text)
  ↓
Token Classification (LAB 09 — Contextual token representations h_t -> 1 label per token)
  ↓
Named Entity Recognition (NER — BIO / BIOES Tagging)
  ↓
Information Extraction (Trích xuất thông tin có cấu trúc từ văn bản)
```
Như vậy, ở **LAB 09**, chúng ta sẽ tiếp tục khai thác sức mạnh của **Pretrained Transformer**, nhưng mở rộng từ việc đọc vector `[CLS]` sang việc khai thác biểu diễn ngữ cảnh tại **mọi vị trí token** kèm bài toán căn chỉnh nhãn trên các mảnh Subword (`subword label alignment`).
