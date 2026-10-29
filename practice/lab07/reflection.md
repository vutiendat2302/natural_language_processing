# BÁO CÁO TỔNG KẾT, SO SÁNH TOÀN DIỆN VÀ VẤN ĐÁP (LAB 07 - REFLECTION)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 07 — Transformer Architecture (From Attention to Transformer)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 27. So sánh toàn bộ LAB 04 → LAB 07

## 27.1. Bảng đối chiếu tổng hợp các kiến trúc biểu diễn văn bản

| Property | TF-IDF (LAB 01/04) | Avg Word2Vec (LAB 03/04) | LSTM (LAB 05) | Attention (LAB 06) | Transformer (LAB 07) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sparse** | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ |
| **Dense** | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Word order** | Limited (N-gram) | $\times$ (Bag-of-Words) | $\checkmark$ (Tự nhiên) | $\checkmark^*$ | $\checkmark^*$ (Qua PE) |
| **Context-dependent** | $\times$ | $\times$ (Static) | $\checkmark$ | $\checkmark$ | $\checkmark$ (Sâu & Đa đầu) |
| **Direct token interaction** | $\times$ | $\times$ | Limited | $\checkmark$ | $\checkmark$ (Multi-Head) |
| **Sequential computation** | $\times$ | $\times$ | $\checkmark$ (Bắt buộc) | $\times$ | $\times$ |
| **Parallelizable** | $\checkmark$ | $\checkmark$ | Limited | $\checkmark$ | $\checkmark$ (Tối ưu GPU) |
| **Long-range interaction** | $\times$ | $\times$ | Difficult | Improved | **Improved ($\mathcal{O}(1)$ path)** |
| **Attention mechanism** | $\times$ | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |
| **Positional information** | $\times$ | $\times$ | Implicit | Required | **Required (PE)** |
| **Cost with sequence length** | $\sim \text{linear}$ | $\sim \text{linear}$ | $\sim \text{linear } \mathcal{O}(T)$ | $\mathcal{O}(T^2)$ | **$\mathcal{O}(T^2)$** |

*(\* Attention và Transformer bắt buộc cần Positional Information để biểu diễn thứ tự token).*

---

## 27.2. Chuỗi tiến hóa giải quyết các giới hạn từ LAB 01 tới LAB 07

> **Câu hỏi tổng hợp:** *Từ LAB 01 tới LAB 07, mỗi bước đã giải quyết limitation nào của bước trước?*

```text
TF-IDF (LAB 01 & LAB 04)
  ↓
  ├── Limitation: Biểu diễn thưa (sparse), số chiều khổng lồ bằng kích thước từ điển |V|,
  │               các từ đồng nghĩa hoàn toàn trực giao (không hiểu độ tương đồng ngữ nghĩa).
  ▼
Word2Vec / Avg Embedding (LAB 03 & LAB 04)
  ↓
  ├── Giải quyết: Nén thành vector đặc (dense) chiều thấp, nắm bắt được độ tương đồng ngữ nghĩa.
  ├── Limitation: Biểu diễn tĩnh (static - mỗi từ chỉ có 1 vector duy nhất bất kể ngữ cảnh) và
  │               phép lấy trung bình (Average Pooling) làm mất hoàn toàn thứ tự từ (word order).
  ▼
RNN / LSTM / GRU (LAB 05)
  ↓
  ├── Giải quyết: Đưa thứ tự chuỗi và trạng thái ẩn tích lũy theo thời gian vào mô hình.
  ├── Limitation: Tính toán tuần tự từng bước (không thể song song hóa), bị nút thắt cổ chai
  │               khi nén cả câu vào h_T và suy hao thông tin tầm xa (vanishing gradient).
  ▼
Attention Mechanism (LAB 06)
  ↓
  ├── Giải quyết: Cho phép tương tác trực tiếp token-to-token với đường truyền O(1), xóa bỏ
  │               nút thắt cổ chai h_T và mở khóa khả năng tính toán ma trận song song.
  ├── Limitation: Chỉ có 1 góc nhìn đơn lẻ (single-head), thiếu phép biến đổi phi tuyến sâu
  │               cho từng vị trí, mất thông tin thứ tự tự nhiên và khó xếp chồng nhiều tầng.
  ▼
Transformer Architecture (LAB 07)
  ↓
  ├── Giải quyết: Kết hợp Positional Encoding (thứ tự), Multi-Head Attention (đa góc nhìn),
  │               FFN (biến đổi phi tuyến) và Residual + LayerNorm (huấn luyện mạng sâu ổn định).
  └── Remaining limitation:
      1. Chi phí tính toán và bộ nhớ tăng bậc hai O(T^2) theo độ dài chuỗi T.
      2. Đòi hỏi lượng dữ liệu lớn (dễ overfit trên tập nhỏ nếu train từ đầu) -> Cần Pretraining (BERT/GPT ở LAB 08).
```

---

# 31. Individual learning check

Trả lời đầy đủ và ngắn gọn 9 câu hỏi vấn đáp ngẫu nhiên từ Giảng viên:

### Câu 1: Tại sao Transformer cần positional encoding?
* **Trả lời:** Vì phép toán Self-Attention ($QK^T$) và FFN hoàn toàn đối xứng với hoán vị vị trí (permutation equivariant). Nếu không cộng thêm Positional Encoding, mô hình coi câu văn như một tập hợp từ không thứ tự, không thể phân biệt *"dog bites man"* với *"man bites dog"*.

### Câu 2: Multi-head attention khác một attention head ở đâu?
* **Trả lời:** Một head chỉ chiếu dữ liệu vào 1 không gian duy nhất và tạo ra 1 phân phối chú ý. Multi-head attention chiếu $Q, K, V$ vào $h$ không gian con độc lập chạy song song, cho phép mô hình nắm bắt đồng thời nhiều kiểu quan hệ khác nhau (cú pháp, phủ định, thực thể) trước khi nối lại (`Concat`) và trộn qua $W^O$.

### Câu 3: Nếu $d_{\text{model}} = 128$ và có $8$ heads thì $d_k$ bằng bao nhiêu?
* **Trả lời:**
  $$d_k = \frac{d_{\text{model}}}{n_{\text{heads}}} = \frac{128}{8} = \mathbf{16}$$

### Câu 4: Tại sao attention matrix có kích thước $T \times T$?
* **Trả lời:** Vì trong Self-Attention trên chuỗi có $T$ tokens, mỗi một token trong $T$ tokens đóng vai trò Query phải tính tích vô hướng với tất cả $T$ tokens đóng vai trò Key. Phép nhân ma trận $(T \times d_k) \times (d_k \times T)$ tạo ra ma trận $T \times T$.

### Câu 5: Residual connection có vai trò gì?
* **Trả lời:** Công thức $y = x + F(x)$ đóng 2 vai trò: (1) Tạo đường truyền trực tiếp cho đạo hàm ($\frac{\partial y}{\partial x} = \mathbf{I} + \frac{\partial F}{\partial x}$) giúp chống triệt tiêu gradient khi xếp chồng nhiều tầng; (2) Bảo toàn thông tin gốc $x$ của token để không bị biến dạng qua các lớp.

### Câu 6: Feed-Forward Network làm gì trong Transformer block?
* **Trả lời:** Sau khi Self-Attention trộn thông tin giữa các token (tương tác liên vị trí), FFN thực hiện phép biến đổi **phi tuyến độc lập trên từng vị trí** ($\text{ReLU}(xW_1 + b_1)W_2 + b_2$) để trích xuất và làm giàu đặc trưng ngữ nghĩa cho từng token.

### Câu 7: Tại sao Transformer có thể xử lý các token song song hơn RNN?
* **Trả lời:** RNN phải đợi trạng thái ẩn $h_{t-1}$ mới tính được $h_t$ (phụ thuộc tuần tự theo thời gian). Transformer loại bỏ hoàn toàn vòng lặp đệ quy đó và thay bằng phép nhân ma trận toàn cục $QK^T$, cho phép GPU tính toán tất cả $T$ vị trí cùng một lúc.

### Câu 8: Nếu sequence dài gấp đôi, attention matrix thay đổi như thế nào?
* **Trả lời:** Khi chiều dài chuỗi tăng gấp đôi ($T \to 2T$), kích thước ma trận Attention tăng từ $T \times T$ lên $(2T) \times (2T)$, tức **số phần tử và bộ nhớ tăng gấp $4$ lần** ($(2T)^2 = 4T^2$).

### Câu 9: Vì sao không thể kết luận rằng attention weight cao đồng nghĩa token đó là "lý do" model đưa ra prediction?
* **Trả lời:** Vì Attention weights chỉ phản ánh mức độ tương thích tích vô hướng giữa Query và Key tại tầng đó. Trước khi ra dự đoán cuối cùng, thông tin còn được nhân với Value $V$, cộng trộn với vector gốc qua Residual Connection, biến đổi phi tuyến qua FFN và lấy trung bình ở lớp Pooling. Do đó Attention pattern chỉ mô tả hành vi định tuyến thông tin, không đồng nhất với giải thích nhân quả (explanation).

---

# 35. Mạch nối LAB 01 → LAB 07

Hành trình phát triển xuyên suốt học phần từ đầu kỳ tới hiện tại:

```text
LAB 01: Text → TF-IDF → Sparse representation → Search
LAB 02: Text → N-gram → Probability → Language Model → Perplexity
LAB 03: Text → Context → Word Embedding → Semantic Similarity
LAB 04: Text → Representation → Classifier → Evaluation → Error Analysis
LAB 05: Sequence → RNN / LSTM / GRU → Hidden state → Sequence classification
LAB 06: Sequence → Query / Key / Value → Attention → Context representation
LAB 07: Sequence → Self-Attention → Multi-Head + Positional Encoding + FFN + Residual/Norm → Transformer Block
```

**Bước chuẩn bị tiếp theo (Hướng tới LAB 08 — Pretrained Transformer / BERT):**
Khi đã tự tay lắp ghép thành công khối **Transformer Encoder Block** ở LAB 07, câu hỏi tiếp theo là: *Thay vì huấn luyện khối Transformer này từ con số 0 trên tập dữ liệu nhỏ (dễ bị Overfitting sau Epoch 6 như thực nghiệm), làm thế nào để tận dụng mô hình Transformer đã được tiền huấn luyện (Pretrained) trên hàng tỷ câu văn bản như **BERT**?* Đó chính là trọng tâm của **LAB 08**.
