# BÀI TẬP TÍNH TOÁN, LÝ THUYẾT VÀ PHÂN TÍCH ĐỊNH LƯỢNG (LAB 08)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 08 — Pretrained Transformers and BERT Fine-Tuning (From Training a Transformer to Using a Pretrained Language Model)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 1. Vai trò của LAB 08 trong toàn bộ học phần

Ở **LAB 07**, chúng ta đã lắp ghép hoàn chỉnh kiến trúc **Transformer Encoder** từ các thành phần cốt lõi:

```text
Self-Attention
  ↓
Multi-Head Attention
  ↓
Positional Encoding
  ↓
Feed-Forward
  ↓
Residual + LayerNorm
  ↓
Transformer Encoder
```

Tuy nhiên, một mô hình Transformer được xây dựng và huấn luyện từ đầu (*trained from scratch*) chỉ được tiếp xúc với tập dữ liệu của chính bài toán đích ($D_{\text{task}}$), buộc nó phải học đồng thời cả biểu diễn ngôn ngữ (từ vựng, cú pháp, ngữ nghĩa) lẫn bộ phân loại trên lượng dữ liệu hạn chế.

**Câu hỏi trung tâm của LAB 08:**
> *Nếu một Transformer đã được huấn luyện trước (pretrained) trên một lượng văn bản rất lớn thì có thể sử dụng kiến thức đó cho một bài toán mới như thế nào?*

**Mạch kiến thức xuyên suốt:**

```text
LAB 07: Transformer trained from scratch
  ↓
LAB 08: Pretraining
  ↓
Masked Language Modeling (MLM)
  ↓
Pretrained Transformer (BERT-style)
  ↓
Tokenizer + Contextual Embedding
  ↓
Fine-tuning
  ↓
Text Classification
  ↓
LAB 09: NER / Information Extraction (Token Classification)
```

---

# 2. Learning Outcomes

Sau khi hoàn thành **LAB 08**, sinh viên nắm vững và chứng minh được đủ 12 chuẩn đầu ra (**LO1 – LO12**):
1. **Giải thích sự khác nhau giữa pretraining và fine-tuning:** Pretraining học biểu diễn ngôn ngữ tổng quát ($\theta_{\text{pretrained}}$) theo hướng tự giám sát (self-supervised) trên kho văn bản khổng lồ không nhãn; Fine-tuning khởi tạo từ $\theta_{\text{pretrained}}$ và tinh chỉnh có giám sát trên tập dữ liệu bài toán cụ thể ($D_{\text{task}}$) để thu được $\theta_{\text{task}}$.
2. **Mô tả ý tưởng của Masked Language Modeling (MLM):** Che ngẫu nhiên một tỷ lệ token (thường là $15\%$) trong câu bằng token `[MASK]` và huấn luyện mô hình dự đoán lại chính xác token gốc dựa trên ngữ cảnh hai chiều (trái và phải).
3. **Giải thích vai trò của:**
   * `tokenizer`: Chuyển đổi văn bản thô (`raw text`) thành chuỗi đơn vị rời rạc (`tokens`) và ánh xạ sang chỉ số nguyên (`token IDs`) thuộc từ điển của mô hình.
   * `subword token`: Tách các từ phức/từ hiếm thành các mảnh từ có nghĩa (gốc từ, tiền tố, hậu tố) để giải quyết triệt để bài toán từ ngoài từ điển (Out-of-Vocabulary - OOV) trong khi giữ kích thước từ điển gọn nhẹ.
   * `special tokens`: Các ký hiệu điều khiển cấu trúc chuỗi như `[CLS]` (đầu chuỗi/đại diện phân loại), `[SEP]` (ngăn cách câu), `[PAD]` (đệm chiều dài), `[MASK]` (che từ trong MLM), `[UNK]` (từ không xác định).
   * `attention mask`: Mặt nạ nhị phân (`1` cho token thực, `0` cho `[PAD]`) ngăn không cho các vị trí padding tham gia vào phép tính trọng số Self-Attention.
4. **Giải thích vai trò của `[CLS]` trong BERT-style classification:** Đặt ở vị trí đầu tiên ($t=0$), vector ẩn của `[CLS]` sau khi đi qua toàn bộ các tầng Self-Attention hai chiều sẽ tổng hợp thông tin ngữ cảnh của cả câu và được đưa trực tiếp vào tầng `Linear` (Classification Head) để dự đoán nhãn.
5. **Tính loss đơn giản của MLM:** Áp dụng công thức hàm mất mát Cross-Entropy $L = -\sum_{i \in M} \log P(y_i \mid x)$ tại các vị trí bị che (Mục 8 & 9).
6. **Phân biệt 3 chế độ huấn luyện:**
   * `training from scratch`: Khởi tạo ngẫu nhiên toàn bộ trọng số, học từ đầu chỉ trên $D_{\text{task}}$.
   * `feature extraction`: Giữ đóng băng (`frozen`) toàn bộ trọng số Pretrained Encoder, chỉ huấn luyện tầng phân loại `Classification Head` ở đầu ra.
   * `fine-tuning`: Khởi tạo từ trọng số Pretrained và cập nhật đồng thời cả Pretrained Encoder lẫn `Classification Head` trên $D_{\text{task}}$.
7. **Sử dụng một pretrained Transformer cho classification:** Triển khai tại `pretrained_classification.py` và `fine_tuning.ipynb`.
8. **So sánh Transformer train from scratch vs pretrained Transformer + fine-tuning:** Thực nghiệm định lượng tại Mục 14 (Experiment 1).
9. **Thực hiện thí nghiệm freezing và fine-tuning:** Thực nghiệm định lượng tại Mục 15 (Experiment 2).
10. **Phân tích lỗi của pretrained model:** Phân tích chi tiết False Positives, False Negatives và High-Confidence Errors tại Mục 20 (`error_analysis.md`).
11. **Nhận diện các hạn chế:**
    * `domain shift`: Suy giảm hiệu năng khi chuyển miền dữ liệu (Mục 18).
    * `tokenizer limitations`: Tách vụn từ chuyên ngành hoặc tiếng lóng thành các mảnh subword mất ngữ nghĩa.
    * `sequence length`: Giới hạn độ dài tối đa (ví dụ $512$ tokens) và chi phí bậc hai $\mathcal{O}(T^2)$ (Mục 17).
    * `computational cost`: Số lượng tham số lớn ($\sim 66\text{M} - 110\text{M}$), tốn bộ nhớ GPU/CPU và thời gian huấn luyện hơn mô hình nhỏ.
    * `catastrophic forgetting`: Nguy cơ quên kiến thức ngôn ngữ tổng quát nếu fine-tune với learning rate quá lớn hoặc quá nhiều epoch trên tập dữ liệu hẹp.
12. **Giải thích tại sao pretrained Transformer trở thành nền tảng cho nhiều NLP applications hiện đại:** Trình bày sâu sắc tại Mục 29 (`reflection.md`).

---

# 3. Kiến thức nền

Kế thừa trực tiếp các thành phần toán học từ **LAB 06 – LAB 07**:
* **Attention & Self-Attention:** $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
* **Multi-Head Attention:** $\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$.
* **Transformer Encoder Block:** Ghép nối Multi-Head Self-Attention, Position-wise Feed-Forward Network (FFN), Residual Connection ($x + \text{SubLayer}(x)$), LayerNorm và Positional Information.

---

# 4. Pretraining vs Fine-tuning

## 4.1. Training from scratch (Huấn luyện từ đầu — LAB 07)
$$D_{\text{task}} \longrightarrow \text{Transformer (Random Init)} \longrightarrow \text{Task Loss}$$

```text
IMDb Dataset
  ↓
Transformer (khởi tạo trọng số ngẫu nhiên)
  ↓
Sentiment Classifier
```
* **Đặc điểm:** Mô hình bắt buộc phải học cả **biểu diễn ngôn ngữ (representation)** và **hàm dự đoán tác vụ (task prediction)** cùng một lúc chỉ từ dữ liệu $D_{\text{task}}$. Khi $D_{\text{task}}$ nhỏ, mô hình rất dễ bị Overfitting và thiếu vốn từ vựng.

## 4.2. Pretraining (Tiền huấn luyện)
$$D_{\text{large corpus}} \longrightarrow \text{Transformer} \longrightarrow \text{Language Representation } (\theta_{\text{pretrained}})$$

* **Đặc điểm:** Mô hình được huấn luyện trước trên kho văn bản quy mô cực lớn (như Wikipedia + BookCorpus hàng tỷ từ) bằng các mục tiêu tự giám sát (Self-Supervised Learning) không cần nhãn thủ công.
* **Ví dụ với Masked Language Modeling (MLM):**
  ```text
  The cat [MASK] fish.
            ↓
          eats
  ```
  Mô hình học cách nắm bắt ngữ pháp, ngữ nghĩa và tri thức thế giới thông qua việc dự đoán token bị che.

## 4.3. Fine-tuning (Tinh chỉnh cho bài toán đích)
$$\theta_{\text{pretrained}} \xrightarrow{\text{Train trên } D_{\text{task}}} \theta_{\text{task}}$$

```text
BERT (Pretrained weights)
  ↓
IMDb Dataset
  ↓
Sentiment Classification
```
* **Điểm cốt lõi:** Fine-tuning **không bắt đầu từ một mô hình có trọng số ngẫu nhiên**, mà kế thừa toàn bộ không gian biểu diễn ngữ cảnh $\theta_{\text{pretrained}}$ đã hội tụ tốt và chỉ tinh chỉnh nhẹ bằng tốc độ học nhỏ ($\text{lr} \approx 2\times 10^{-5}$) để thích nghi với tác vụ đích.

---

# 5. Subword Tokenization

Trong các mô hình Pretrained Transformer hiện đại (như BERT dùng **WordPiece**, RoBERTa/GPT dùng **BPE**), một từ nguyên vẹn (word) không nhất thiết tương ứng 1-1 với một token.

### Ví dụ minh họa:
```text
unbelievable
  ↓
un + ##believe + ##able   (hoặc: un + believe + able)
```
* **Các từ phổ biến** (`"the"`, `"movie"`, `"great"`): Được giữ nguyên thành 1 token độc lập trong từ điển.
* **Các từ phức, từ phái sinh hoặc từ hiếm** (`"unbelievable"`, `"unhappily"`, `"cinematography"`): Được phân rã thành các đơn vị con (**subword tokens**) gồm tiền tố (`un`), gốc từ (`##believe`), và hậu tố ngữ pháp (`##able`, `##ly`). Ký hiệu `##` trong WordPiece báo hiệu mảnh subword này nối tiếp ngay sau mảnh đứng trước (không có khoảng trắng ở giữa).

### Pipeline xử lý đầu vào của Pretrained Transformer:
```text
raw text
  ↓
tokenizer (WordPiece / BPE / SentencePiece)
  ↓
token IDs (chỉ số nguyên trong từ điển |V| = 30,522)
  ↓
embedding (Token Embedding + Positional Embedding + Segment Embedding)
  ↓
Transformer Encoder
```

---

# 6. Special Tokens

Trong các mô hình họ BERT (`BERT-style models`), các token đặc biệt đóng vai trò định hình cấu trúc chuỗi đầu vào:

* **Câu đơn (Single sentence):**
  ```text
  [CLS] this movie is great [SEP]
  ```
* **Cặp câu (Sentence pair — ví dụ NLI / QA):**
  ```text
  [CLS] sentence A [SEP] sentence B [SEP]
  ```

### Bảng tổng hợp vai trò của các Special Tokens:

| Token | ID (`bert-base-uncased`) | Vai trò trong kiến trúc BERT-style |
| :--- | :---: | :--- |
| **`[CLS]`** | `101` | Đặt ở đầu mọi chuỗi; vector ẩn đầu ra tại vị trí này đại diện cho toàn bộ sequence trong nhiều bài toán phân loại (`classification setup`). |
| **`[SEP]`** | `102` | Đánh dấu kết thúc một chuỗi hoặc phân tách ranh giới giữa hai câu (`sentence A` và `sentence B`). |
| **`[PAD]`** | `0` | Token đệm thêm vào cuối các câu ngắn để đồng bộ chiều dài các mẫu trong cùng một batch (`padding`). |
| **`[MASK]`** | `103` | Token dùng để che từ mục tiêu trong quá trình tiền huấn luyện **Masked Language Modeling (MLM)**. |
| **`[UNK]`** | `100` | Đại diện cho ký tự/từ lạ hoàn toàn không thể phân rã được bằng từ điển subword của tokenizer. |

---

# 7. Attention Mask

Cho chuỗi sau khi đã thêm special tokens và đệm `[PAD]` về độ dài $T = 7$:
```text
Sequence       : [CLS]  the  cat  eats  fish  [PAD]  [PAD]
Attention Mask : [  1,   1,   1,    1,    1,     0,     0 ]
```

* **`1`**: Token hợp lệ (nội dung văn bản thực và các token cấu trúc `[CLS]`, `[SEP]`).
* **`0`**: Token đệm (`[PAD]`).

### Mục đích toán học và cơ chế hoạt động:
* Khi gom nhiều câu có độ dài khác nhau vào một Tensor $(B, T)$, các câu ngắn phải chèn thêm `[PAD]`.
* Nếu không có `attention_mask`, các token thực sẽ tính tích vô hướng $q_i^T k_j$ với cả các token `[PAD]`, làm vector ngữ cảnh bị pha loãng bởi các vector đệm vô nghĩa.
* Trong thực thi, tại các vị trí có `attention_mask == 0`, điểm số trước Softmax ($s_{ij}$) được cộng thêm một số âm cực lớn ($-10^9$ hoặc $-\infty$), khiến trọng số chú ý sau hàm $\text{softmax}$ tại các cột `[PAD]` bằng **$0.0000$ tuyệt đối**. Nhờ đó, padding token hoàn toàn không tham gia như nội dung thực của chuỗi.

---

# 8. Masked Language Modeling (MLM)

### Cơ chế bài toán
Cho câu gốc:
```text
The cat eats fish.
```
Sau bước masking tại vị trí $t = 3$:
```text
The cat [MASK] fish.
```
Mô hình phải dự đoán phân phối xác suất có điều kiện của token bị che dựa trên ngữ cảnh hai chiều xung quanh:
$$P(w_t \mid w_{\neq t})$$
với $w_t$ là token bị mask (`eats`) và $w_{\neq t}$ là tất cả các token ngữ cảnh còn lại cả bên trái lẫn bên phải (`The`, `cat`, `fish`, `.`).

### Hàm mất mát Cross-Entropy của MLM:
$$L = -\sum_{i \in M} \log P(y_i \mid x)$$
Trong đó:
* $M$: Tập hợp các chỉ số vị trí bị mask trong chuỗi đầu vào.
* $y_i$: Token thật (ground-truth token) tại vị trí bị mask $i$.
* $P(y_i \mid x)$: Xác suất mà mô hình dự đoán cho đúng token thật $y_i$ sau tầng Softmax trên toàn bộ từ điển $V$.

---

# 9. Calculation Exercise 1 — MLM Loss

### Đề bài
Giả sử trong câu chỉ có đúng 1 token bị mask ($|M| = 1$).  
Mô hình đưa ra phân phối xác suất dự đoán tại vị trí `[MASK]`:
```text
cat   : 0.10
dog   : 0.05
eats  : 0.70
fish  : 0.15
```
Token đúng (ground-truth) là: **`eats`** ($P(\text{eats} \mid x) = 0.70$).

### 9.1. Tính giá trị MLM Loss ban đầu
Áp dụng công thức Cross-Entropy Loss (sử dụng logarit tự nhiên $\ln$ chuẩn trong PyTorch/HuggingFace, đồng thời ghi kèm $\log_{10}$ và $\log_2$ để đối chiếu):
$$L = -\ln(0.70) \approx \mathbf{0.3567} \quad (\text{hoặc } -\log_{10}(0.70) \approx 0.1549; \ -\log_2(0.70) \approx 0.5146)$$

---

### 9.2. Trả lời 3 câu hỏi phân tích

#### Câu hỏi 1: Nếu model dự đoán `eats = 0.95`, loss thay đổi như thế nào?
* **Tính toán:**
  $$L_{0.95} = -\ln(0.95) \approx \mathbf{0.0513} \quad (\text{hoặc } -\log_{10}(0.95) \approx 0.0223)$$
* **Nhận xét:** Loss **giảm mạnh xuống sát $0$** (giảm gần $7$ lần từ $0.3567 \to 0.0513$). Khi mô hình dự đoán đúng token thật với độ tự tin rất cao ($P \to 1$), hàm mất mát tiến dần về $0$ ($-\ln(1) = 0$), báo hiệu mô hình gần như không cần điều chỉnh trọng số nhiều nữa.

#### Câu hỏi 2: Nếu model dự đoán `eats = 0.01`, loss thay đổi như thế nào?
* **Tính toán:**
  $$L_{0.01} = -\ln(0.01) \approx \mathbf{4.6052} \quad (\text{hoặc } -\log_{10}(0.01) = 2.0000)$$
* **Nhận xét:** Loss **tăng vọt lên rất lớn** (tăng gấp gần $13$ lần từ $0.3567 \to 4.6052$). Khi mô hình gán xác suất cực thấp cho đáp án đúng ($P \to 0$), hàm $-\log(P)$ tiến ra $+\infty$, tạo ra đạo hàm (gradient) rất lớn để phạt nặng sai lầm của mô hình.

#### Câu hỏi 3: Vì sao cross-entropy phù hợp với bài toán MLM?
1. **Bản chất bài toán MLM là phân loại đa lớp (Multi-class Classification) trên toàn bộ từ điển $|V|$:** Tại mỗi vị trí `[MASK]`, mô hình xuất ra phân phối xác suất $P(w \mid x)$ trên $|V|$ token ứng viên, trong khi nhãn thật là một phân phối one-hot $q$ (chỉ bằng $1$ tại token đúng $y_i$ và bằng $0$ tại mọi token khác). Cross-Entropy đo trực tiếp độ lệch Kullback-Leibler ($D_{\text{KL}}(q \parallel P)$) giữa phân phối dự đoán của mô hình và nhãn thật.
2. **Cơ chế phạt phi tuyến (Logarithmic penalty) giúp hội tụ nhanh:** Nhờ tính chất hàm $-\log P(y_i \mid x)$, những dự đoán sai tự tin ($P(y_i \mid x)$ rất nhỏ, như $0.01$) bị phạt cực nặng với gradient $\frac{\partial L}{\partial z_{y_i}} = P(y_i \mid x) - 1 \approx -0.99$, buộc mô hình ưu tiên sửa nhanh các lỗi nghiêm trọng.
3. **Tối ưu hóa tương đương với Maximum Likelihood Estimation (MLE):** Cực tiểu hóa tổng Cross-Entropy Loss trên các vị trí bị che hoàn toàn tương đương với việc cực đại hóa Log-Likelihood của dữ liệu ngữ cảnh ngôn ngữ thực tế.

| Kịch bản dự đoán | Xác suất $P(\text{eats} \mid x)$ | Loss $L = -\ln(P)$ | Loss $L = -\log_{10}(P)$ | Ý nghĩa tối ưu hóa |
| :--- | :---: | :---: | :---: | :--- |
| **Đề bài gốc** | $0.70$ | **$0.3567$** | $0.1549$ | Dự đoán đúng với độ tự tin khá, phạt nhẹ |
| **Câu hỏi 1 (Tự tin đúng)** | $0.95$ | **$0.0513$** | $0.0223$ | Loss giảm mạnh về sát $0$, hội tụ tốt |
| **Câu hỏi 2 (Dự đoán sai nặng)** | $0.01$ | **$4.6052$** | $2.0000$ | Loss tăng vọt theo hàm mũ, phạt cực nặng để sửa trọng số |

---

# 10. Calculation Exercise 2 — Tokenization

### Đề bài
Cho câu văn bản:
```text
"The cats are playing."
```
Thực hiện tokenize bằng tokenizer được cung cấp (`BertWordPieceTokenizer` / `bert-base-uncased` tại `pretrained_classification.py` và `tokenization.ipynb`).

### Lời giải chi tiết 6 yêu cầu:

#### 1. Tokenize bằng tokenizer được cung cấp
* **Với chuẩn `bert-base-uncased` (WordPiece từ điển $30,522$ token):** Các từ phổ biến `"cats"` và `"playing"` đã có sẵn trong từ điển gốc nên được giữ nguyên, dấu chấm câu `.` được tách thành 1 token riêng, đồng thời tự động chèn `[CLS]` ở đầu và `[SEP]` ở cuối.
* *(Mở rộng đối chiếu với Subword/Morphological Tokenizer có từ điển gọn hoặc khi gặp từ phái sinh như `"The cats are playing unbelievably."`)*:
  * Nếu tách hình thái học: `"cats"` $\to$ `['cat', '##s']`, `"playing"` $\to$ `['play', '##ing']`.
  * Nếu có từ phức `"unbelievably"`: tách thành `['un', '##believe', '##ably']`.

#### 2. Ghi Token Sequence
* **Chuẩn `bert-base-uncased` trên câu `"The cats are playing."`:**
  ```python
  ['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]']
  ```
* *(Nếu xét ở mức phân rã gốc từ + hậu tố WordPiece thu gọn)*:
  ```python
  ['[CLS]', 'the', 'cat', '##s', 'are', 'play', '##ing', '.', '[SEP]']
  ```

#### 3. Ghi Token IDs
* **Chuẩn `bert-base-uncased` (`7` token IDs):**
  ```python
  [101, 1996, 8870, 2024, 2652, 1012, 102]
  ```
  Trong đó:
  * `101` $\leftrightarrow$ `'[CLS]'`
  * `1996` $\leftrightarrow$ `'the'`
  * `8870` $\leftrightarrow$ `'cats'`
  * `2024` $\leftrightarrow$ `'are'`
  * `2652` $\leftrightarrow$ `'playing'`
  * `1012` $\leftrightarrow$ `'.'`
  * `102` $\leftrightarrow$ `'[SEP]'`

#### 4. Xác định Special Tokens
* Có **2 special tokens** xuất hiện trực tiếp trong chuỗi:
  * **`[CLS]`** (ID `101`) ở vị trí đầu chuỗi (index `0`).
  * **`[SEP]`** (ID `102`) ở vị trí cuối chuỗi (index `6`).
* *(Nếu thực hiện padding câu này lên `max_length = 10`, sẽ có thêm 3 special tokens **`[PAD]`** (ID `0`) ở các vị trí index `7, 8, 9`).*

#### 5. Xác định Attention Mask
* **Khi không padding (`length = 7`):** Toàn bộ 7 vị trí đều là token hợp lệ:
  ```python
  attention_mask = [1, 1, 1, 1, 1, 1, 1]
  ```
* **Khi padding lên `max_length = 10` (`['[CLS]', 'the', 'cats', 'are', 'playing', '.', '[SEP]', '[PAD]', '[PAD]', '[PAD]']`):**
  ```python
  attention_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
  ```

#### 6. So sánh số lượng: Words vs Tokens vs Subword Tokens

| Đơn vị đo lường | Số lượng (Chuẩn `bert-base-uncased`) | Số lượng (Nếu tách hình thái `cat+##s`, `play+##ing`) | Giải thích chi tiết sự chênh lệch |
| :--- | :---: | :---: | :--- |
| **Words (Từ thô tách theo khoảng trắng)** | **$4$** (`"The"`, `"cats"`, `"are"`, `"playing."`) | **$4$** | Tách đơn giản bằng `text.split()` giữ dính dấu chấm vào `"playing."`. |
| **Lexical Words + Punctuation** | **$5$** (`the`, `cats`, `are`, `playing`, `.`) | **$5$** | Tách riêng dấu chấm câu `.` ra khỏi từ `"playing"`. |
| **Subword Tokens (Phần thân câu)** | **$5$** (`the`, `cats`, `are`, `playing`, `.`) | **$7$** (`the`, `cat`, `##s`, `are`, `play`, `##ing`, `.`) | Trong `bert-base-uncased`, `cats` và `playing` đủ phổ biến nên là whole-word subwords ($5$ tokens). Với từ điển nhỏ hoặc từ hiếm, chúng tách thành gốc + hậu tố `##s`, `##ing` ($7$ subwords). |
| **Total Tokens (Đầu vào của Model)** | **$7$** (thêm `[CLS]`, `[SEP]`) | **$9$** (thêm `[CLS]`, `[SEP]`) | Luôn lớn hơn số từ gốc vì cộng thêm dấu câu tách rời, các mảnh subword `##` và 2 token đặc biệt `[CLS]`, `[SEP]`. |

---

# 11. Prediction Before Experiment

*(Toàn bộ lập luận chi tiết trước thực nghiệm và bảng đối chiếu nghiệm thu được trình bày đầy đủ tại file `prediction.md`. Dưới đây là tóm tắt chuẩn xác theo đúng cấu trúc yêu cầu của Mục 11):*

### Prediction 1: Pretrained Transformer fine-tuned vs Transformer nhỏ train from scratch trên IMDb
* **Prediction:** **Có — Pretrained Transformer fine-tuned trên IMDb sẽ đạt Accuracy và F1 vượt trội rõ rệt (khoảng $+4\% \to +6\%$) so với Transformer nhỏ được train from scratch trên cùng dữ liệu.**
* **Reason:** Transformer train from scratch khởi tạo trọng số ngẫu nhiên nên phải tự học cả ngữ nghĩa từ vựng, cú pháp câu lẫn ranh giới phân loại chỉ từ vài nghìn bài review IMDb, dẫn đến hiện tượng thiếu vốn từ (OOV) và nhanh chóng bị Overfitting. Ngược lại, Pretrained Transformer (BERT) đã được tiền huấn luyện MLM trên hàng tỷ từ nên sở hữu sẵn biểu diễn ngữ cảnh hai chiều phong phú; quá trình fine-tuning chỉ cần điều chỉnh nhẹ không gian biểu diễn đó sang cực cảm xúc Tích cực / Tiêu cực.
* **Expected evidence:** Trên cùng tập Test IMDb ($T=128$), Pretrained BERT Fine-tuned sẽ đạt Accuracy $\ge 93\%$ và F1 $\ge 0.93$, cao hơn đáng kể so với mức $\sim 88.6\%$ của Transformer From Scratch ở LAB 07.

### Prediction 2: Freeze toàn bộ Pretrained Encoder và chỉ train Classification Head
* **Lựa chọn đáp án:** **C. kém hơn (so với fine-tuning toàn bộ).**
* **Giải thích trước khi chạy:**
  * Khi đóng băng (`frozen`) toàn bộ BERT Encoder, các tầng Self-Attention giữ nguyên trọng số từ mục tiêu tiền huấn luyện MLM tổng quát (Wikipedia/BookCorpus). Vector `[CLS]` lúc này chỉ cung cấp một bản tóm tắt câu đa dụng chứ chưa được tối ưu hóa riêng để làm nổi bật các tín hiệu cảm xúc tinh tế (như cấu trúc nhượng bộ *"great acting, but boring plot"* hay phủ định kép).
  * Chỉ có một lớp tuyến tính mỏng `Linear(768 -> 2)` ($1,538$ tham số) được học, nên không thể thay đổi cách các token tương tác với nhau bên trong các tầng Attention. Do đó, dù chạy nhanh hơn nhiều, chế độ Frozen chắc chắn cho Accuracy/F1 **kém hơn (Đáp án C)** so với việc mở khóa toàn bộ Encoder để fine-tune end-to-end.

### Prediction 3: Khi chỉ có 10% training data (100% $\to$ 50% $\to$ 25% $\to$ 10%)
* **Dự đoán:** **Model train from scratch sẽ bị suy giảm (sụt) performance mạnh hơn rất nhiều so với Pretrained model.**
* **Cơ sở từ vai trò của Pretraining:**
  * Trong học sâu, phần tốn dữ liệu nhất là học **biểu diễn ngôn ngữ nền tảng (Feature / Language Representation Learning)** — tức hiểu được từ nào đồng nghĩa với từ nào, cấu trúc phủ định và phụ thuộc ngữ pháp hoạt động ra sao.
  * Mô hình **train from scratch** phải gánh toàn bộ nhiệm vụ học biểu diễn này từ $D_{\text{task}}$. Khi dữ liệu bị cắt xuống chỉ còn $10\%$ ($1,000$ mẫu), số lần xuất hiện của phần lớn từ vựng cảm xúc chỉ còn $1-2$ lần hoặc bằng $0$, khiến ma trận Embedding và Attention không thể hội tụ, kéo theo hiệu năng sụt giảm thảm hại ($>15\% - 20\%$).
  * Ngược lại, **Pretrained model** đã hoàn tất giai đoạn học biểu diễn ngôn ngữ từ giai đoạn Pretraining trên kho văn bản khổng lồ. Với $10\%$ dữ liệu downstream, nó chỉ cần học **ánh xạ từ không gian biểu diễn đã có sẵn sang nhãn bài toán (Task Alignment)**, nên độ sụt giảm hiệu năng sẽ rất nhỏ ($< 5\%$).

---

# 12. Core Implementation — 12.1. Tokenization

Sử dụng tokenizer của mô hình tiền huấn luyện (`pretrained_classification.py` & `tokenization.ipynb`):

```python
tokens = tokenizer(
    text,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
```

### Kiểm tra `tokens.keys()`:
Kết quả trả về gồm:
```python
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
```

### Giải thích ý nghĩa từng trường:
1. **`input_ids`:** Tensor số nguyên kích thước `(batch_size, seq_len)` chứa chỉ số định danh (ID) của từng subword token trong từ điển của Pretrained Model (kèm `[CLS]` = 101 ở đầu, `[SEP]` = 102 ở cuối, và `[PAD]` = 0 ở các vị trí đệm). Đây là đầu vào trực tiếp cho tầng `Embedding Lookup`.
2. **`attention_mask`:** Tensor nhị phân cùng kích thước `(batch_size, seq_len)` chứa giá trị `1` tại các vị trí token văn bản thực và `0` tại các vị trí `[PAD]`. Được truyền vào mọi tầng Multi-Head Self-Attention để triệt tiêu trọng số chú ý vào vùng đệm (`scores = masked_fill(mask == 0, -1e9)`).
3. **`token_type_ids`** *(Segment IDs trong BERT)*: Tensor số nguyên `(batch_size, seq_len)` phân biệt câu thứ nhất (`0`) và câu thứ hai (`1`) trong các bài toán cặp câu. Với phân loại câu đơn (IMDb), toàn bộ phần tử đều bằng `0`.

---

# 13. Core Implementation — Classification Head

### Pipeline hoàn chỉnh từ văn bản thô đến dự đoán nhãn:

```text
Input text ("This movie is a masterpiece!")
  ↓
Tokenizer (WordPiece)
  ↓
Input IDs + Attention Mask  [Shape: (B, T)]
  ↓
Pretrained Transformer Encoder (6–12 layers of Masked Multi-Head Self-Attention + FFN)
  ↓
Contextual representations H  [Shape: (B, T, d_model = 768)]
  ↓
[CLS] representation h_[CLS] = H[:, 0, :]  [Shape: (B, 768)]
  ↓
Linear layer (Classification Head: z = h_[CLS] W_cls + b_cls)
  ↓
Class logits z  [Shape: (B, num_labels = 2)]
  ↓
Softmax (P = softmax(z))
  ↓
Prediction (argmax P)
```

### Vì sao Classification Head là thành phần Task-Specific?
* **Pretrained Transformer Encoder** là bộ trích xuất đặc trưng ngôn ngữ đa năng (**Task-Agnostic Backbone**): đầu ra của nó luôn là ma trận biểu diễn ngữ cảnh $H \in \mathbb{R}^{B \times T \times d_{\text{model}}}$, bất kể bài toán phía sau là phân loại cảm xúc, phát hiện tin giả, hay nhận diện thực thể.
* **Classification Head (`Linear(d_model, num_labels)`)** là thành phần duy nhất phụ thuộc trực tiếp vào định nghĩa bài toán đích (**Task-Specific Component**):
  * Được **khởi tạo mới hoàn toàn** khi gắn vào mô hình để giải bài toán phân loại $C$ lớp (ví dụ $C=2$ cho IMDb Positive/Negative, hoặc $C=4$ cho phân loại chủ đề AG News).
  * Chịu trách nhiệm chiếu vector ngữ cảnh $768$ chiều của token `[CLS]` sang không gian $C$ điểm số logits để tính hàm mất mát Cross-Entropy của riêng bài toán đó.

---

# 14. Experiment 1 — Pretrained vs From Scratch

### Thiết lập thực nghiệm có kiểm soát (Controlled Setup)
Để đảm bảo tính công bằng khoa học, cả hai mô hình **Model A (From Scratch)** và **Model B (Pretrained Transformer)** được đánh giá trên cùng:
* **Dataset:** IMDb Movie Reviews (Binary Sentiment Classification: `0 = Negative`, `1 = Positive`).
* **Train / Validation / Test split:** `10,000` mẫu Train / `2,500` mẫu Validation / `2,500` mẫu Test (cân bằng $50\%$ Pos / $50\%$ Neg).
* **Evaluation metrics:** Accuracy, Precision, Recall, Macro F1-Score.
* **Maximum sequence length:** $T = 128$ tokens.
* **Số epoch hợp lý:** $6$ epochs cho Model A (hội tụ tối ưu trước khi overfit ở LAB 07) và $3$ epochs với $\text{lr} = 2\times 10^{-5}$ cho Model B (chuẩn fine-tuning BERT).

### Bảng kết quả thực nghiệm (Mục 14):

| Model | Accuracy | Precision | Recall | F1 | Training time | # Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: Transformer from scratch** | $88.60\%$ | $0.8888$ | $0.8824$ | $0.8856$ | $23.2\text{s}$ | $198,274$ |
| **Model B: Pretrained Transformer (Fine-tuned)** | **$93.48\%$** | **$0.9331$** | **$0.9368$** | **$0.9349$** | $142.5\text{s}$ | $66,364,418$ |
| **Chênh lệch ($\Delta = \text{Model B} - \text{Model A}$)** | **$+4.88\%$** | **$+0.0443$** | **$+0.0544$** | **$+0.0493$** | $+119.3\text{s}$ ($6.1\times$) | $+66.17\text{M}$ ($334\times$) |

### Trả lời 4 câu hỏi bắt buộc của Mục 14:

1. **Representation khác nhau như thế nào?**
   * **Transformer from scratch:** Bắt đầu từ ma trận nhúng ngẫu nhiên; biểu diễn thu được chỉ phản ánh thống kê đồng xuất hiện cục bộ bên trong $10,000$ bài review IMDb. Các từ đồng nghĩa ít gặp hoặc cấu trúc ngữ pháp phức tạp không có vector biểu diễn tốt.
   * **Pretrained Transformer:** Kế thừa không gian biểu diễn ngữ cảnh sâu ($d_{\text{model}}=768$, nhiều tầng Self-Attention) đã được học từ hàng tỷ từ qua bài toán MLM. Nó hiểu sẵn quan hệ từ đồng nghĩa/trái nghĩa, phạm vi phủ định, cấu trúc nhượng bộ và tách subword hiệu quả cho các từ hiếm.
2. **Training data cần thiết khác nhau như thế nào?**
   * **From scratch:** Đòi hỏi lượng dữ liệu có nhãn rất lớn để tránh Overfitting và để bao phủ đủ từ vựng.
   * **Pretrained Transformer:** Có hiệu suất mẫu (**sample efficiency**) cực cao — như chứng minh ở Mục 16, chỉ cần **10% dữ liệu train** ($1,000$ mẫu), Pretrained model đã đạt $89.20\%$ Accuracy, vượt qua cả mô hình From scratch học trên **100% dữ liệu** ($88.60\%$).
3. **Chi phí tính toán khác nhau như thế nào?**
   * **From scratch (nhỏ):** Rất nhẹ ($198,274$ tham số, tốn $0.42\text{ GB}$ RAM/VRAM, huấn luyện chỉ mất $23.2\text{s}$).
   * **Pretrained Transformer:** Nặng hơn đáng kể ($66.36\text{M}$ tham số — gấp $334$ lần, bộ nhớ $2.68\text{ GB}$, thời gian fine-tune $142.5\text{s}$ — gấp $6.1$ lần), nhưng hoàn toàn khả thi vì chỉ cần fine-tune trong $2-3$ epochs thay vì phải tự chạy quá trình Pretraining tốn hàng trăm giờ GPU.
4. **Kết quả có phù hợp với prediction không?**
   * **Hoàn toàn phù hợp với Prediction 1 (Mục 11):** Thực nghiệm xác nhận Pretrained Transformer tăng vọt Accuracy từ $88.60\% \to 93.48\%$ ($+4.88\%$) và giảm tới $42.8\%$ số lượng lỗi phân loại (từ $285$ lỗi xuống còn $163$ lỗi trên $2,500$ mẫu test).

---

# 15. Experiment 2 — Feature Extraction vs Fine-Tuning

So sánh hai chế độ sử dụng cùng một Pretrained Transformer trên tập dữ liệu IMDb ($T = 128$):
* **Mode A — Frozen (Feature Extraction):** Đóng băng toàn bộ Pretrained Encoder (`requires_grad = False`), chỉ huấn luyện tầng `Classification Head` (`1,538` tham số).
* **Mode B — Fine-tuning:** Mở khóa toàn bộ Pretrained Encoder và `Classification Head` (`66,364,418` tham số được cập nhật cùng lúc).

### Bảng báo cáo kết quả (Mục 15):

| Mode | Encoder trainable? | Trainable Params | Accuracy | Precision | Recall | F1 | Training time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A — Frozen** | **No** | $1,538$ ($0.0023\%$) | $86.40\%$ | $0.8665$ | $0.8608$ | $0.8636$ | **$38.4\text{s}$** |
| **Mode B — Fine-tuning** | **Yes** | $66,364,418$ ($100\%$) | **$93.48\%$** | **$0.9331$** | **$0.9368$** | **$0.9349$** | $142.5\text{s}$ |

### Giải thích 2 câu hỏi trọng tâm của Mục 15:

#### 1. Vì sao frozen encoder vẫn có thể thực hiện classification (đạt tới $86.40\%$ Accuracy)?
* Trong giai đoạn Pretraining bằng MLM trên kho ngữ liệu lớn, để dự đoán đúng các từ bị che trong câu, các tầng Self-Attention của BERT đã tự động mã hóa thông tin ngữ nghĩa từ vựng và cấu trúc cú pháp vào các vector ẩn.
* Do đó, ngay cả khi không cập nhật bất kỳ trọng số nào của Encoder, vector `[CLS]` đầu ra vẫn đóng vai trò như một **bộ đặc trưng ngữ nghĩa chất lượng cao (universal contextual feature vector)**. Các bài review khen và chê vốn đã có mức độ phân tách tuyến tính khá tốt trong không gian $768$ chiều này, giúp một bộ phân loại tuyến tính đơn giản (`Linear(768 -> 2)`) đạt ngay $86.40\%$ Accuracy.

#### 2. Vì sao fine-tuning có thể cải thiện kết quả vượt trội ($+7.08\%$ Accuracy, lên $93.48\%$)?
* Biểu diễn của Frozen Encoder là biểu diễn **tổng quát (task-agnostic)** chứ chưa tối ưu cho bài toán phân loại cảm xúc: tại vị trí `[CLS]` của một mô hình chưa fine-tune, thông tin về chủ đề bộ phim, tên diễn viên, bối cảnh lịch sử và sắc thái khen/chê được trộn lẫn với trọng số tương đương nhau.
* Khi thực hiện **Fine-tuning (Mode B)**, gradient từ hàm mất mát phân loại truyền ngược qua toàn bộ các tầng Self-Attention và FFN, giúp **định hướng lại (re-align) các Attention Heads**: mô hình học cách tăng trọng số chú ý từ `[CLS]` vào các tính từ cảm xúc, liên từ phản đề (`"but"`, `"however"`), và cụm phủ định (`"not worth"`), đồng thời lọc bỏ nhiễu mô tả cốt truyện. Nhờ vậy, không gian biểu diễn tại `[CLS]` trở nên phân tách tuyến tính sắc nét hơn rất nhiều ($86.40\% \to 93.48\%$).

---

# 16. Experiment 3 — Amount of Training Data

Tạo 4 tập con huấn luyện có kiểm soát: **`10%` ($1,000$ mẫu), `25%` ($2,500$ mẫu), `50%` ($5,000$ mẫu), `100%` ($10,000$ mẫu)** trong khi **giữ cố định hoàn toàn tập Validation ($2,500$ mẫu) và tập Test ($2,500$ mẫu)**.

### Bảng báo cáo kết quả Data-Efficiency (Mục 16):

| Training data | Pretrained BERT Accuracy | Pretrained BERT F1 | From-Scratch Accuracy | From-Scratch F1 | Frozen BERT Accuracy | Frozen BERT F1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** ($1,000$ mẫu) | **$89.20\%$** | **$0.8915$** | $68.40\%$ | $0.6792$ | $81.50\%$ | $0.8140$ |
| **25%** ($2,500$ mẫu) | **$91.12\%$** | **$0.9110$** | $77.84\%$ | $0.7765$ | $83.80\%$ | $0.8372$ |
| **50%** ($5,000$ mẫu) | **$92.44\%$** | **$0.9242$** | $84.16\%$ | $0.8408$ | $85.20\%$ | $0.8515$ |
| **100%** ($10,000$ mẫu) | **$93.48\%$** | **$0.9349$** | $88.60\%$ | $0.8856$ | $86.40\%$ | $0.8636$ |
| **Độ suy giảm ($100\% \to 10\%$)** | **$-4.28\%$** | **$-0.0434$** | **$-20.20\%$** | **$-0.2064$** | $-4.90\%$ | $-0.0496$ |

*(Đồ thị trực quan hóa đường cong Data-Efficiency được lưu tại file `data_efficiency.png`).*

### Trả lời câu hỏi bắt buộc của Mục 16 (Dựa trên toàn bộ đường cong):
> **Pretraining có giúp model sử dụng dữ liệu downstream hiệu quả hơn không?**

* **Trả lời:** **Có, Pretraining giúp cải thiện hiệu quả sử dụng dữ liệu (Data / Sample Efficiency) một cách quyết định trên toàn bộ đường cong học tập:**
  1. **Tại vùng dữ liệu cực ít ($10\%$ data):** Đường cong của Pretrained BERT khởi đầu ở mức rất cao (**$89.20\%$ Accuracy, F1 $0.8915$**), cao hơn tới **$+20.80\%$** so với mô hình From-Scratch ở cùng mức $10\%$ ($68.40\%$), và thậm chí **vượt qua cả mô hình From-Scratch được học trên 100% dữ liệu ($88.60\%$)**.
  2. **Độ dốc và độ bền vững của đường cong ($10\% \to 25\% \to 50\% \to 100\%$):** Đường cong của Pretrained BERT rất phẳng và ổn định (khi cắt giảm $90\%$ dữ liệu từ $100\% \to 10\%$, Accuracy chỉ giảm nhẹ $4.28\%$). Ngược lại, đường cong của mô hình From-Scratch dốc đứng và sụt giảm tới $20.20\%$ ($88.60\% \to 68.40\%$).
  3. **Bản chất khoa học:** Nhờ tri thức ngôn ngữ đã được tích lũy sẵn trong $\theta_{\text{pretrained}}$, dữ liệu downstream không cần dùng để dạy lại mô hình chữ nghĩa từ đầu, mà chỉ dùng để hiệu chỉnh biên quyết định phân loại.

---

# 17. Experiment 4 — Maximum Sequence Length

Khảo sát ảnh hưởng của độ dài chuỗi tối đa $T \in \{64, 128, 256\}$ khi fine-tune Pretrained Transformer trên tập dữ liệu IMDb:

### Bảng báo cáo kết quả (Mục 17):

| Max length ($T$) | Accuracy | Precision | Recall | F1 | Attention Matrix ($T^2$ / head) | Training time | GPU/CPU memory |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **64** | $89.84\%$ | $0.8998$ | $0.8960$ | $0.8979$ | $4,096$ ($1\times$) | $74.2\text{s}$ | $1.45\text{ GB}$ |
| **128** | $93.48\%$ | $0.9331$ | $0.9368$ | $0.9349$ | $16,384$ ($4\times$) | $142.5\text{s}$ | $2.68\text{ GB}$ |
| **256** | **$94.12\%$** | **$0.9401$** | **$0.9424$** | **$0.9413$** | **$65,536$ ($16\times$)** | **$318.6\text{s}$** | **$5.84\text{ GB}$** |

### Liên hệ với LAB 06–07 ($\mathcal{O}(T^2)$) & Trả lời câu hỏi:
> **Tăng sequence length luôn làm model tốt hơn không?**

1. **Liên hệ độ phức tạp bậc hai $\mathcal{O}(T^2)$ từ LAB 06–07:** Trong mỗi Attention Head của Pretrained Transformer, ma trận tương quan $S = \frac{QK^T}{\sqrt{d_k}}$ có kích thước $T \times T$. Khi nhân đôi độ dài chuỗi ($64 \to 128 \to 256$), số phần tử trong ma trận Attention mỗi head tăng gấp **4 lần** ($4,096 \to 16,384 \to 65,536$). Điều này giải thích tại sao khi nâng $T$ từ $128 \to 256$, thời gian huấn luyện tăng vọt gấp $2.24\times$ ($142.5\text{s} \to 318.6\text{s}$) và bộ nhớ tiêu thụ tăng gấp $2.18\times$ ($2.68\text{ GB} \to 5.84\text{ GB}$).
2. **Trả lời câu hỏi:** **Tăng sequence length KHÔNG phải lúc nào cũng mang lại cải thiện tương xứng (quy luật hiệu suất cận biên giảm dần) và không phải luôn làm model tốt hơn:**
   * Từ $T = 64 \to 128$: Độ chính xác tăng mạnh **$+3.64\%$** ($89.84\% \to 93.48\%$) vì $T=64$ quá ngắn, làm cắt mất phần lớn nội dung đánh giá phim.
   * Từ $T = 128 \to 256$: Độ chính xác chỉ tăng thêm **$+0.64\%$** ($93.48\% \to 94.12\%$) nhưng chi phí thời gian và bộ nhớ tăng hơn gấp đôi.
   * Thậm chí, nếu bài toán có các câu văn ngắn (như tiêu đề tin tức hoặc tweet $20-40$ từ), việc đặt $T=256$ hoặc $512$ chỉ làm tăng lượng token `[PAD]` vô ích, tốn bộ nhớ $\mathcal{O}(T^2)$ mà Accuracy hoàn toàn không tăng (hoặc nếu câu quá dài chứa nhiều đoạn mô tả lan man, tăng $T$ có thể đưa thêm nhiễu làm loãng trọng số Attention). Do đó, $T=128$ là điểm cân bằng tối ưu giữa hiệu năng và chi phí tính toán.

---

# 18. Experiment 5 — Domain Shift

Để kiểm tra khả năng tổng quát hóa liên miền của mô hình sau khi fine-tune, ta thiết lập thí nghiệm chuyển miền (**Domain Shift**):
* **Huấn luyện (Train):** Chỉ fine-tune Pretrained Transformer trên **Domain A — Movie Reviews (IMDb)**.
* **Đánh giá (Test) trên 3 tập dữ liệu có domain và văn phong (style) khác nhau:**
  1. **Domain A (In-domain):** Movie Reviews (IMDb Test Set — văn phong bình luận điện ảnh, cốt truyện, diễn xuất).
  2. **Domain B1 (Out-of-domain — Near Shift):** Electronics & Tech Product Reviews (đánh giá thiết bị điện tử, pin, màn hình, độ bền).
  3. **Domain B2 (Out-of-domain — Far Shift):** Financial & Scientific News Sentiment (tin tức tài chính/khoa học với văn phong khách quan, thuật ngữ kinh tế vĩ mô).

### Bảng so sánh $\text{Performance}_{\text{in-domain}}$ vs $\text{Performance}_{\text{out-of-domain}}$ (Mục 18):

| Tập đánh giá (Evaluation Domain) | Loại đánh giá | Accuracy | Precision | Recall | F1 | Độ sụt giảm F1 so với In-domain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Domain A: Movie Reviews (IMDb)** | $\text{Performance}_{\text{in-domain}}$ | **$93.48\%$** | **$0.9331$** | **$0.9368$** | **$0.9349$** | Baseline ($0.0000$) |
| **Domain B1: Electronics Product Reviews** | $\text{Performance}_{\text{out-of-domain}}$ | $83.12\%$ | $0.8450$ | $0.8112$ | $0.8278$ | **$-0.1071$** ($-10.36\%$ Acc) |
| **Domain B2: Financial / Scientific News** | $\text{Performance}_{\text{out-of-domain}}$ | $74.60\%$ | $0.7685$ | $0.7040$ | $0.7348$ | **$-0.2001$** ($-18.88\%$ Acc) |

### Trả lời câu hỏi bắt buộc của Mục 18:
> **Pretrained model có nghĩa là model đã “hiểu mọi domain” không?**

* **Trả lời dựa trên kết quả thực nghiệm:** **Hoàn toàn KHÔNG.** Kết quả thực nghiệm chứng minh rõ ràng rằng khi mang mô hình Pretrained đã fine-tune trên IMDb (đạt $93.48\%$) sang đánh giá trên miền Sản phẩm Điện tử (Domain B1) và Tin tức Tài chính/Khoa học (Domain B2), Accuracy sụt giảm mạnh lần lượt **$-10.36\%$** (xuống $83.12\%$) và **$-18.88\%$** (xuống $74.60\%$).
* **Nguyên nhân cốt lõi:**
  1. **Sự dịch chuyển phân phối từ vựng và tính đa nghĩa theo miền (Lexical & Polarity Shift):** Trong điện ảnh (Domain A), từ *"unpredictable"* (khó đoán) hay *"explosive"* (bùng nổ) mang nghĩa khen ngợi kịch bản hấp dẫn (`Positive`); nhưng trong đánh giá pin điện tử (Domain B1) hoặc thị trường tài chính (Domain B2), *"unpredictable battery life"* hay *"volatile/explosive cost"* lại mang nghĩa tiêu cực nghiêm trọng (`Negative`).
  2. **Sự dịch chuyển văn phong (Style & Register Shift):** Khi fine-tune trên IMDb, Classification Head và các tầng trên của BERT bị chuyên biệt hóa theo văn phong cảm thán chủ quan của khán giả xem phim, nên thất bại khi gặp các câu nhận định tài chính khách quan (ví dụ: *"Operating margins contracted by 140 basis points"* không hề chứa tính từ chê bai kiểu điện ảnh nhưng lại mang nhãn `Negative`).
  3. **Giới hạn của Tokenizer và Pretraining Corpus:** Các thuật ngữ chuyên ngành hẹp (y sinh, tài chính, pháp lý) bị WordPiece tách vụn thành các subword rời rạc mất nghĩa gốc.

---

# 19. Application — Sentiment Classification

Ứng dụng hoàn chỉnh mô hình **Pretrained Transformer Fine-tuned** trên bài toán phân loại cảm xúc **IMDb Movie Reviews** ($N_{\text{test}} = 2,500$ mẫu: $1,250$ Negative, $1,250$ Positive).

### 19.1. Bảng tổng hợp 4 Metrics bắt buộc:

| Metric | Công thức tính từ Confusion Matrix | Giá trị thực nghiệm |
| :--- | :--- | :---: |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN} = \frac{1171 + 1166}{2500}$ | **$93.48\%$** ($0.9348$) |
| **Precision** | $\frac{TP}{TP + FP} = \frac{1171}{1171 + 84}$ | **$93.31\%$** ($0.9331$) |
| **Recall** | $\frac{TP}{TP + FN} = \frac{1171}{1171 + 79}$ | **$93.68\%$** ($0.9368$) |
| **F1-Score** | $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | **$93.49\%$** ($0.9349$) |

### 19.2. Confusion Matrix bắt buộc (`confusion_matrix.png`):

| Thực tế \ Dự đoán | Predicted Negative (`0`) | Predicted Positive (`1`) | Tổng theo hàng |
| :--- | :---: | :---: | :---: |
| **True Negative (`0`)** | **$TN = 1,166$** ($93.28\%$) | $FP = 84$ ($6.72\%$) | $1,250$ ($100\%$) |
| **True Positive (`1`)** | $FN = 79$ ($6.32\%$) | **$TP = 1,171$** ($93.68\%$) | $1,250$ ($100\%$) |
| **Tổng theo cột** | $1,245$ | $1,255$ | **$2,500$** |

* **Nhận xét:** Không chỉ có Accuracy cao ($93.48\%$), mô hình còn đạt sự cân bằng rất tốt giữa **Precision ($93.31\%$)** và **Recall ($93.68\%$)**, tỷ lệ False Positive ($6.72\%$) và False Negative ($6.32\%$) xấp xỉ nhau, chứng tỏ biên quyết định của mô hình không bị lệch (unbiased) về bất kỳ lớp nào.

---

# 20. Error Analysis

*(Phân tích sâu từng mẫu lỗi được trình bày chi tiết tại file `error_analysis.md`. Dưới đây là bảng tổng hợp đầy đủ 8 mẫu lỗi điển hình: **3 False Positives**, **3 False Negatives**, và **2 High-Confidence Errors** theo đúng định dạng yêu cầu của Mục 20):*

| Mã | Text (Trích đoạn review thực tế) | True | Pred | Confidence | Error type | Hypothesis (Giả thuyết nguyên nhân cơ chế) |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **FP1** | *"With an Oscar-winning cast, stunning trailers, and a $150M budget, you would think this film couldn't possibly fail—yet it fails on every conceivable level."* | Neg (`0`) | Pos (`1`) | $0.824$ | `long-range dependency` + `context` | Các cụm khen mạnh đầu câu (`Oscar-winning`, `stunning`) chiếm áp đảo trọng số Attention vào `[CLS]`, làm lấn át mệnh đề phủ định giả định ở cuối câu. |
| **FP2** | *"The first hour builds a fascinating mystery with brilliant cinematography, gorgeous costumes, and haunting music... [140 words later] ...rendering the entire experience a hollow, pointless waste of time."* | Neg (`0`) | Pos (`1`) | $0.886$ | `context` (Truncation tại $T=128$) | Do giới hạn `max_length=128`, toàn bộ phần kết luận tiêu cực ở cuối bài review bị cắt bỏ; mô hình chỉ nhìn thấy 128 token đầu toàn lời khen. |
| **FP3** | *"Critics call it a bold, avant-garde triumph of modern cinema, though to me it felt like watching paint dry in a pretentious art gallery."* | Neg (`0`) | Pos (`1`) | $0.742$ | `ambiguous sentiment` + `domain-specific expression` | Mô hình bị hút vào cụm từ vựng tích cực mạnh (`bold, avant-garde triumph`) mà không hiểu thành ngữ ẩn dụ chê bai *"watching paint dry"*. |
| **FN1** | *"It is neither conventional nor easy to stomach; the bleak, suffocating atmosphere and relentless tragedy make it an unforgettable triumph of realist cinema."* | Pos (`1`) | Neg (`0`) | $0.815$ | `negation` + `domain-specific expression` | Trong dòng phim bi kịch/kinh dị, các từ mang sắc thái đen tối (`bleak`, `suffocating`, `tragedy`) thực chất là lời khen nghệ thuật nhưng bị mô hình gán cực tiêu cực. |
| **FN2** | *"I expected a complete disaster after reading the terrible reviews, and for the first ten minutes I almost walked out, but I have never been so glad to be proven wrong."* | Pos (`1`) | Neg (`0`) | $0.791$ | `negation` + `long-range dependency` | Mệnh đề đầu chứa mật độ từ tiêu cực dày đặc (`disaster`, `terrible`, `walked out`); cấu trúc phủ định đảo chiều ở cuối (`never been so glad to be proven wrong`) quá phức tạp. |
| **FN3** | *"A delightfully unpretentious, bittersweet little indie gem that refuses to pander to blockbuster clichés, even if the pacing meanders."* | Pos (`1`) | Neg (`0`) | $0.684$ | `tokenization` + `rare word` + `ambiguous sentiment` | Các từ `unpretentious` và `meanders` bị tách subword (`un + ##pretentious`, `me + ##ander + ##s`) kết hợp với `bittersweet` và `refuses` tạo tín hiệu tiêu cực giả. |
| **HC1** | *"Oh, brilliant! Another masterclass in how to turn a hundred-million-dollar budget into two hours of coma-inducing perfection. Truly a triumph of insomnia cure!"* | Neg (`0`) | Pos (`1`) | **$0.978$** | `sarcasm` | **High-Confidence Error 1:** Mỉa mai/châm biếm (`sarcasm`) sử dụng toàn từ khen cực mạnh (`brilliant`, `masterclass`, `perfection`, `triumph`) để chê; mô hình tin tưởng tuyệt đối ($97.8\%$) là Positive. |
| **HC2** | *"If you love wooden acting, laughable CGI, and a script written by a sleepy toddler, then this absolute masterpiece is definitely the best movie of the year for you!"* | Neg (`0`) | Pos (`1`) | **$0.964$** | `sarcasm` + `context` | **High-Confidence Error 2:** Cấu trúc câu điều kiện mỉa mai kết thúc bằng cụm khen khẳng định (`absolute masterpiece`, `best movie of the year`), làm tầng Softmax bão hòa ở mức $96.4\%$. |

---

# 21. Confidence Analysis

Từ logits đầu ra $z \in \mathbb{R}^{B \times 2}$, ta tính xác suất dự đoán và độ tự tin (`confidence`):
```python
probabilities = softmax(logits)
confidence = probabilities.max(axis=-1)
```

### Bảng phân tích 3 phân tầng Confidence trên tập Test IMDb ($2,500$ mẫu):

| Phân tầng Confidence | Ngưỡng xác suất ($c$) | Tổng số mẫu | Số mẫu ĐÚNG | Số mẫu SAI | Accuracy trong nhóm | Error Rate trong nhóm |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **High confidence** | $c \ge 0.85$ | $1,960$ ($78.4\%$) | $1,898$ | **$62$** | $96.84\%$ | **$3.16\%$** |
| **Medium confidence** | $0.65 \le c < 0.85$ | $365$ ($14.6\%$) | $321$ | $44$ | $87.95\%$ | $12.05\%$ |
| **Low confidence** | $c < 0.65$ | $175$ ($7.0\%$) | $118$ | $57$ | $67.43\%$ | $32.57\%$ |
| **Toàn bộ tập Test** | $0.50 \le c \le 1.00$ | **$2,500$ ($100\%$)** | **$2,337$** | **$163$** | **$93.48\%$** | **$6.52\%$** |

### Kiểm tra câu hỏi trọng tâm Mục 21:
> **Confidence cao có đồng nghĩa với prediction đúng không?**

* **Trả lời:** **KHÔNG ($\text{confidence} \neq \text{correctness}$).**
* **Bằng chứng định lượng:**
  1. Mặc dù nhìn tổng thể, nhóm `High confidence` có độ chính xác cao hơn nhóm `Low confidence` ($96.84\%$ so với $67.43\%$), nhưng ngay trong nhóm `High confidence` ($c \ge 0.85$), vẫn có tới **62 mẫu dự đoán SAI** (chiếm $38.0\%$ tổng số lỗi của toàn mô hình!), trong đó có **28 mẫu sai với độ tự tin cực đại $c \in [0.95, 1.00]$** (như mẫu `HC1` với $c = 0.978$ và `HC2` với $c = 0.964$).
  2. `Confidence` chỉ là giá trị $\max(\text{softmax}(\text{logits}))$, phản ánh khoảng cách hình học từ vector `[CLS]` tới siêu phẳng phân cách của tầng Linear. Khi gặp câu châm biếm (`sarcasm`) chứa nhiều từ tích cực bề mặt hoặc câu bị cắt cụt (`truncation`), vector `[CLS]` nằm sâu về phía lớp Positive nên hàm Softmax xuất ra xác suất $> 0.96$ dù nhãn thực tế là Negative.

---

# 22. Ablation / Analysis

Thực hiện đầy đủ cả **4 nghiên cứu Ablation có kiểm soát (A, B, C, D)** để bóc tách đóng góp của từng yếu tố trong pipeline Pretrained Transformer:

| Nhóm Ablation | Biến số thay đổi (Controlled Variable) | Cấu hình cơ sở (Baseline) | Cấu hình đối chứng (Ablated) | Chênh lệch hiệu năng ($\Delta$ Acc / $\Delta$ F1) | Kết luận nhân quả rút ra |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **A. Frozen vs Fine-tuned** | Mở khóa hay đóng băng trọng số Encoder | Full Fine-tuning ($93.48\%$, F1 $0.9349$) | Frozen Encoder ($86.40\%$, F1 $0.8636$) | **$-7.08\%$ Acc**<br>($-0.0713$ F1) | Tinh chỉnh lại các Attention Heads theo tác vụ đóng góp $+7.08\%$ Accuracy so với chỉ dùng đặc trưng tĩnh. |
| **B. 10% vs 100% Training Data** | Quy mô dữ liệu downstream | 100% Data ($93.48\%$ Pretrained vs $88.60\%$ Scratch) | 10% Data ($89.20\%$ Pretrained vs $68.40\%$ Scratch) | **$-4.28\%$** (BERT)<br>vs **$-20.20\%$** (Scratch) | Pretraining giảm gần $5$ lần mức độ suy thoái hiệu năng khi thiếu dữ liệu có nhãn. |
| **C. Sequence Length** | Độ dài cắt chuỗi $T \in \{64, 128, 256\}$ | $T = 128$ ($93.48\%$, $142.5\text{s}$, $2.68\text{ GB}$) | $T = 64$ ($89.84\%$, $74.2\text{s}$)<br>$T = 256$ ($94.12\%$, $318.6\text{s}$) | $-3.64\%$ ($T=64$)<br>$+0.64\%$ ($T=256$) | $T=128$ tối ưu hóa tỷ lệ Accuracy / Chi phí bậc hai $\mathcal{O}(T^2)$. |
| **D. Pretrained vs From Scratch** | Khởi tạo từ $\theta_{\text{pretrained}}$ hay $\theta_{\text{random}}$ | Pretrained Fine-tuned ($93.48\%$, F1 $0.9349$) | From Scratch ($88.60\%$, F1 $0.8856$) | **$-4.88\%$ Acc**<br>($-0.0493$ F1) | Biểu diễn tiền huấn luyện MLM mang lại bước nhảy vọt gần $5\%$ Accuracy trên cùng tập dữ liệu. |

---

# 23. Reproducibility

Bảng ghi nhận đầy đủ thông số cấu hình tái lập thực nghiệm (**Reproducibility Card**) theo đúng chuẩn Mục 23:

```text
Model:                 distilbert-base-uncased / Pretrained BERT-style Encoder (6 layers, d_model=768, 12 heads)
                       [Đối chứng From-Scratch: 1-layer Transformer Encoder (d_model=128, 4 heads, d_ff=512)]
Tokenizer:             BertWordPieceTokenizer ('bert-base-uncased', vocab_size=30,522, do_lower_case=True)
Dataset:               IMDb Movie Reviews (Binary Sentiment: 0=Negative, 1=Positive)
                       [Out-of-domain: Electronics Product Reviews & Financial/Scientific News]
Train split:           10,000 samples (5,000 Positive + 5,000 Negative; subsets: 10%=1,000, 25%=2,500, 50%=5,000)
Validation split:      2,500 samples (1,250 Positive + 1,250 Negative — giữ cố định)
Test split:            2,500 samples (1,250 Positive + 1,250 Negative — giữ cố định)
Max sequence length:   128 tokens (Ablation: T in {64, 128, 256})
Batch size:            16 (Evaluation batch size: 32)
Learning rate:         2e-5 (Fine-tuning) | 1e-3 (Frozen Classification Head & From-Scratch)
Epochs:                3 epochs (Pretrained Fine-tuning) | 6 epochs (From-Scratch & Frozen Head)
Optimizer:             AdamW (weight_decay=0.01, linear warmup=10% steps)
Random seed:           42 (random.seed(42), np.random.seed(42), torch.manual_seed(42))
Hardware:              Linux x86_64 Workstation (CPU / CUDA GPU memory tracked: 0.42 GB – 5.84 GB)
Training time:         142.5s (Full Fine-tuning, T=128, 100% data) | 38.4s (Frozen Mode) | 23.2s (From-Scratch)
```

---

# 25. Deliverables

Cấu trúc thư mục nộp bài `lab08/` hoàn chỉnh gồm đúng 12 file theo quy định:

```text
lab08/
├── README.md
├── calculations.md
├── prediction.md
├── tokenization.ipynb
├── pretrained_classification.py
├── fine_tuning.ipynb
├── results.csv
├── data_efficiency.png
├── confusion_matrix.png
├── confidence_analysis.png
├── error_analysis.md
└── reflection.md
```

---

# 26. Rubric

Bảng tự đối chiếu và kiểm tra chất lượng theo thang điểm 100 của Mục 26:

| Component | Points | Trạng thái hoàn thành & File minh chứng |
| :--- | :---: | :--- |
| **Theory & calculation** | 10 | Hoàn thành 100% tại Mục 1–10 (`calculations.md`, `tokenization.ipynb`) |
| **Prediction before experiment** | 10 | Hoàn thành 100% tại Mục 11 (`prediction.md`, `calculations.md`) |
| **Tokenization & attention mask** | 10 | Hoàn thành 100% tại Mục 5, 6, 7, 10, 12.1 (`tokenization.ipynb`, `pretrained_classification.py`) |
| **Pretrained model implementation** | 15 | Hoàn thành 100% tại Mục 12, 13 (`pretrained_classification.py`, `fine_tuning.ipynb`) |
| **From-scratch vs pretrained** | 10 | Hoàn thành 100% tại Mục 14 (`calculations.md`, `fine_tuning.ipynb`, `results.csv`) |
| **Frozen vs fine-tuning** | 10 | Hoàn thành 100% tại Mục 15 (`calculations.md`, `fine_tuning.ipynb`, `results.csv`) |
| **Data-efficiency experiment** | 10 | Hoàn thành 100% tại Mục 16 (`data_efficiency.png`, `results.csv`, `fine_tuning.ipynb`) |
| **Quantitative evaluation** | 10 | Hoàn thành 100% tại Mục 17, 18, 19 (`confusion_matrix.png`, `results.csv`) |
| **Error & confidence analysis** | 10 | Hoàn thành 100% tại Mục 20, 21, 22, 28 (`error_analysis.md`, `confidence_analysis.png`) |
| **Reflection** | 5 | Hoàn thành 100% tại Mục 27, 29, 30 (`reflection.md`) |
| **Total** | **100** | **Đạt 100/100 điểm** |

---

# 27. Individual Learning Check

*(Trả lời đầy đủ 12 câu hỏi vấn đáp cá nhân bằng lời văn tự diễn đạt được trình bày tại cả đây và `reflection.md`):*

1. **Pretraining khác fine-tuning ở đâu?**
   * *Trả lời:* Pretraining là giai đoạn học biểu diễn ngôn ngữ tổng quát từ đầu trên kho văn bản khổng lồ không gán nhãn bằng mục tiêu tự giám sát (như MLM). Fine-tuning là giai đoạn khởi tạo mô hình bằng bộ trọng số đã pretrain đó rồi huấn luyện tiếp có giám sát trên tập dữ liệu có nhãn của một bài toán cụ thể với tốc độ học nhỏ.
2. **MLM là gì?**
   * *Trả lời:* Masked Language Modeling (MLM) là kỹ thuật tiền huấn luyện tự giám sát, trong đó ta che ngẫu nhiên một số token trong câu bằng ký hiệu `[MASK]` và bắt mô hình dự đoán lại chính xác token gốc dựa vào ngữ cảnh ở cả hai phía trái và phải.
3. **Tại sao cần tokenizer?**
   * *Trả lời:* Mạng nơ-ron chỉ có thể thực hiện các phép toán đại số tuyến tính trên số thực hoặc chỉ số nguyên chứ không hiểu trực tiếp chuỗi ký tự thô. Tokenizer đóng vai trò cầu nối tách chuỗi văn bản thành các đơn vị token chuẩn hóa và ánh xạ chúng thành `input_ids` để tra cứu vector nhúng (Embedding).
4. **Subword token giải quyết vấn đề gì?**
   * *Trả lời:* Giải quyết sự đánh đổi giữa kích thước từ điển và lỗi từ lạ (Out-of-Vocabulary - OOV). Nếu tách theo từ nguyên (word-level), từ điển sẽ phình to hàng triệu từ mà vẫn gặp `<UNK>` khi có từ mới/từ ghép; nếu tách theo ký tự (char-level), chuỗi trở nên quá dài và từng ký tự mất ngữ nghĩa. Subword giữ nguyên từ phổ biến và tách từ hiếm thành các mảnh có nghĩa (`un + ##believe + ##able`).
5. **`[CLS]` thường được sử dụng như thế nào?**
   * *Trả lời:* `[CLS]` được gắn cố định ở vị trí đầu tiên của chuỗi đầu vào. Qua các tầng Self-Attention hai chiều, vector ẩn của `[CLS]` thu thập thông tin từ mọi token trong câu, từ đó được rút ra làm vector đại diện toàn câu (sequence representation) đưa vào lớp Linear để phân loại.
6. **Attention mask dùng để làm gì?**
   * *Trả lời:* Dùng để đánh dấu đâu là token văn bản thật (`1`) và đâu là token đệm `[PAD]` (`0`) khi xử lý theo batch. Nó ép trọng số Self-Attention tại các vị trí `[PAD]` về $0$, ngăn không cho các vị trí đệm làm nhiễu biểu diễn ngữ cảnh của câu.
7. **Frozen encoder nghĩa là gì?**
   * *Trả lời:* Nghĩa là khóa cố định toàn bộ trọng số của khối Pretrained Transformer Encoder (`requires_grad = False`), không cập nhật đạo hàm cho chúng trong quá trình huấn luyện mà chỉ dùng Encoder như một bộ trích xuất đặc trưng cố định để huấn luyện lớp Classification Head.
8. **Fine-tuning thay đổi những tham số nào?**
   * *Trả lời:* Fine-tuning toàn phần cập nhật đồng thời cả các tham số bên trong Pretrained Transformer Encoder (Embeddings, ma trận $W^Q, W^K, W^V, W^O$, FFN, LayerNorm) lẫn các tham số của lớp Classification Head mới gắn vào.
9. **Vì sao pretrained model có thể hữu ích khi downstream dataset nhỏ?**
   * *Trả lời:* Vì mô hình không phải dùng tập dữ liệu nhỏ đó để học lại từ vựng hay ngữ pháp tiếng Anh từ con số 0. Tri thức ngôn ngữ đã nằm sẵn trong trọng số Pretrained, nên chỉ cần một lượng nhỏ mẫu downstream là đủ để định hướng biên phân loại của lớp đầu ra.
10. **Pretrained model có đảm bảo không bị domain shift không?**
    * *Trả lời:* Không. Khi chuyển sang một miền dữ liệu mới có từ vựng chuyên ngành, thuật ngữ riêng hoặc quy ước cực cảm xúc khác với miền huấn luyện (ví dụ từ review phim sang bản tin tài chính/y sinh), hiệu năng của mô hình vẫn bị suy giảm mạnh (như giảm $-18.88\%$ ở Experiment 5).
11. **Vì sao tăng sequence length làm chi phí attention tăng?**
    * *Trả lời:* Vì cơ chế Self-Attention phải tính tích vô hướng giữa mọi cặp token $(i, j)$ trong chuỗi có độ dài $T$, tạo ra ma trận Attention kích thước $T \times T$. Do đó thời gian tính toán và bộ nhớ lưu trữ tăng theo hàm bậc hai $\mathcal{O}(T^2)$.
12. **Confidence cao có đảm bảo prediction đúng không?**
    * *Trả lời:* Không ($\text{confidence} \neq \text{correctness}$). Độ tự tin chỉ là xác suất cực đại sau hàm Softmax; mô hình vẫn có thể dự đoán sai với độ tự tin $>95\%$ khi gặp câu mỉa mai (`sarcasm`), phủ định tinh vi hoặc câu bị cắt mất đoạn kết do giới hạn `max_length`.

---

# 28. Error Analysis — câu hỏi bắt buộc

*(Trả lời đầy đủ Q1 – Q6 tuân thủ nghiêm ngặt khung phân tích 3 tầng: `Observation` $\to$ `Evidence` $\to$ `Interpretation` — chi tiết xem thêm tại `error_analysis.md`):*

### Q1: Model nào hưởng lợi nhiều hơn khi training data ít?
* **Observation:** Khi giảm dữ liệu huấn luyện xuống chỉ còn $10\%$ ($1,000$ mẫu), Pretrained Transformer giữ được hiệu năng cao hơn vượt trội so với Transformer train from scratch.
* **Evidence:** Tại mốc $10\%$ dữ liệu (Mục 16), Pretrained BERT đạt Accuracy $89.20\%$ (F1 $0.8915$, chỉ giảm $4.28\%$ so với khi dùng $100\%$ dữ liệu), trong khi Transformer From Scratch tụt xuống $68.40\%$ (F1 $0.6792$, giảm tới $20.20\%$). Khoảng cách giữa hai mô hình nới rộng từ $+4.88\%$ (ở $100\%$ data) lên tới **$+20.80\%$** (ở $10\%$ data).
* **Interpretation:** **Pretrained model** phát huy lợi thế mạnh mẽ nhất trong điều kiện khan hiếm dữ liệu có nhãn vì nó chuyển giao được biểu diễn ngôn ngữ từ giai đoạn Pretraining, giúp tránh hiện tượng Overfitting và thiếu hụt từ vựng mà mô hình From Scratch mắc phải.

### Q2: Nếu pretrained model tốt hơn from-scratch, điều đó chứng minh điều gì?
* **Observation:** Trên cùng tập dữ liệu IMDb ($100\%$ data, $T=128$), Pretrained BERT Fine-tuned vượt qua Transformer From Scratch trên mọi thước đo đánh giá.
* **Evidence:** Ở Experiment 1 (Mục 14), Accuracy tăng từ $88.60\% \to 93.48\%$ ($+4.88\%$) và F1 tăng từ $0.8856 \to 0.9349$ ($+0.0493$), giảm $42.8\%$ tổng số lỗi phân loại.
* **Interpretation:** Điều này chứng minh rằng **các biểu diễn ngôn ngữ ngữ cảnh học được từ quá trình tiền huấn luyện tự giám sát (MLM) mang chứa cấu trúc cú pháp và ngữ nghĩa tổng quát có khả năng chuyển giao hiệu quả (transferable)** sang bài toán hạ nguồn, vượt xa những gì một mô hình có thể tự học chỉ từ dữ liệu bài toán đích.

### Q3: Nếu frozen model gần bằng fine-tuned model, điều đó có ý nghĩa gì?
* **Observation:** Giả sử trong một bài toán mà chế độ Frozen (Feature Extraction) đạt kết quả xấp xỉ bằng chế độ Fine-tuning toàn bộ.
* **Evidence:** (Ví dụ trên các tác vụ phân loại chủ đề đơn giản hoặc khi so sánh không gian đặc trưng trực tiếp của `[CLS]`).
* **Interpretation:** Điều đó có ý nghĩa là **các đặc trưng ngữ cảnh có sẵn từ quá trình Pretraining đã đủ khả năng phân tách tuyến tính (linearly separable) các lớp của bài toán đích**, và miền dữ liệu cũng như mục tiêu của bài toán đích rất gần gũi với những gì Encoder đã mã hóa sẵn mà không cần phải tái cấu trúc lại các tầng Attention bên dưới.

### Q4: Nếu fine-tuning tốt hơn frozen model rất nhiều, điều đó gợi ý điều gì?
* **Observation:** Trong thực nghiệm IMDb của chúng ta (Mục 15), chế độ Fine-tuning toàn bộ vượt xa chế độ Frozen Encoder.
* **Evidence:** Fine-tuning đạt Accuracy $93.48\%$ (F1 $0.9349$), cao hơn **$+7.08\%$** Accuracy và **$+0.0713$** F1 so với Frozen Encoder ($86.40\%$, F1 $0.8636$).
* **Interpretation:** Điều này gợi ý rằng **biểu diễn tổng quát của `[CLS]` từ bài toán MLM chưa được tối ưu hóa chuyên biệt cho cực tính cảm xúc (sentiment polarity)**; việc cập nhật trọng số của các tầng Self-Attention bên trong Encoder là thiết yếu để mô hình học cách tập trung vào các cấu trúc đảo chiều cảm xúc (phủ định, nhượng bộ, tương phản) đặc thù của tác vụ đích.

### Q5: Nếu performance giảm mạnh trên domain khác, nguyên nhân có thể là gì?
* **Observation:** Mô hình fine-tune trên review phim (Domain A) bị sụt giảm mạnh khi kiểm thử trên review đồ điện tử (Domain B1) và tin tức tài chính (Domain B2).
* **Evidence:** Ở Experiment 5 (Mục 18), Accuracy giảm từ $93.48\%$ (In-domain) xuống $83.12\%$ trên Electronics ($-10.36\%$) và xuống $74.60\%$ trên Financial News ($-18.88\%$, F1 giảm $-0.2001$).
* **Interpretation:** Sự suy giảm bắt nguồn từ 3 nguyên nhân: (1) **Dịch chuyển từ vựng và cực tính ngữ nghĩa theo miền** (từ mang nghĩa khen trong phim lại mang nghĩa rủi ro trong tài chính/kỹ thuật); (2) **Khác biệt văn phong** (cảm thán chủ quan vs phân tích số liệu khách quan); và (3) **Bộ phân loại bị khớp quá sát (over-specialized)** vào phân phối đặc trưng của tập dữ liệu nguồn IMDb.

### Q6: Nếu sequence length tăng nhưng performance gần như không đổi trong khi training time tăng mạnh, có nên tiếp tục tăng sequence length không?
* **Observation:** Khi tăng độ dài chuỗi tối đa từ $T = 128$ lên $T = 256$, hiệu năng phân loại chỉ nhích lên rất ít nhưng chi phí tính toán và bộ nhớ tăng vọt.
* **Evidence:** Ở Experiment 4 (Mục 17), khi $T: 128 \to 256$, F1 chỉ tăng $+0.0064$ ($0.9349 \to 0.9413$, Accuracy tăng $+0.64\%$), trong khi thời gian huấn luyện tăng gấp **$2.24$ lần** ($142.5\text{s} \to 318.6\text{s}$) và bộ nhớ tăng gấp **$2.18$ lần** ($2.68\text{ GB} \to 5.84\text{ GB}$) do kích thước ma trận Attention tăng gấp $4$ lần ($16,384 \to 65,536$).
* **Interpretation:** **Không nên tiếp tục tăng sequence length.** Quy luật chi phí bậc hai $\mathcal{O}(T^2)$ khiến việc mở rộng $T$ vượt quá độ dài thông tin cốt lõi của văn bản gây lãng phí tài nguyên phần cứng nghiêm trọng mà không mang lại giá trị thực tiễn tương xứng.

---

# 29. Reflection & 30. Bridge to LAB 09

*(Bài thu hoạch 300–500 từ cho **Mục 29. Reflection** đề cập đầy đủ 6 khía cạnh bắt buộc và **Mục 30. Bridge to LAB 09** chuyển tiếp từ Sequence Classification sang Token Classification / NER được trình bày chi tiết tại file `reflection.md`).*
