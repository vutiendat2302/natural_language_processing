# DỰ ĐOÁN TRƯỚC THỰC NGHIỆM VÀ KIỂM CHỨNG ĐỊNH LƯỢNG (LAB 08 - PREDICTIONS)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026  
**Chủ đề**: LAB 08 — Pretrained Transformers and BERT Fine-Tuning  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# 11. Prediction Before Experiment

Trước khi tiến hành chạy các thực nghiệm với Pretrained Transformer trên tập dữ liệu IMDb, sinh viên thiết lập đầy đủ 3 dự đoán theo đúng cấu trúc yêu cầu của **Mục 11** trong `W8.pdf`.

---

## Prediction 1 — Pretrained Transformer Fine-tuned vs Transformer Train From Scratch

> **Câu hỏi:** *Một pretrained Transformer fine-tuned trên IMDb có khả năng đạt kết quả tốt hơn Transformer nhỏ được train from scratch trên cùng dữ liệu không?*

```text
Prediction:
  Có. Một Pretrained Transformer (BERT-style) được fine-tune trên IMDb chắc chắn sẽ đạt kết quả
  (Accuracy và F1-Score) cao hơn đáng kể so với một Transformer nhỏ được huấn luyện từ đầu
  (trained from scratch) trên cùng tập dữ liệu và cùng độ dài chuỗi.

Reason:
  1. Khởi tạo biểu diễn (Representation Prior): Transformer train from scratch bắt đầu từ trọng số
     ngẫu nhiên nên phải dùng tập dữ liệu IMDb hữu hạn để học đồng thời cả biểu diễn từ vựng,
     cấu trúc cú pháp lẫn biên phân loại cảm xúc. Ngược lại, Pretrained Transformer đã được
     huấn luyện bằng Masked Language Modeling (MLM) trên hàng tỷ từ văn bản, nên đã nắm vững
     ngữ pháp, quan hệ từ đồng nghĩa/trái nghĩa và tương tác ngữ cảnh hai chiều.
  2. Xử lý từ hiếm (Subword Tokenization): WordPiece Tokenizer của Pretrained model phân rã các từ
     hiếm/từ ghép thành các mảnh subword có nghĩa thay vì biến chúng thành token <UNK> mất thông tin.
  3. Khả năng chống Overfitting: Quá trình fine-tuning chỉ cần tinh chỉnh nhẹ (adaptation) không gian
     đặc trưng đã hội tụ tốt sang bài toán Sentiment Classification trong vài epoch.

Expected evidence:
  - Trên cùng tập Test IMDb (N = 2,500, max_length = 128), Pretrained Transformer Fine-tuned sẽ đạt
    Accuracy trên 93% (F1 > 0.93), vượt trội khoảng +4.5% đến +5.5% so với mức Accuracy 88.60%
    (F1 = 0.8856) của Transformer train from scratch ở LAB 07.
  - Đặc biệt trên các câu có cấu trúc phủ định phức tạp, ẩn dụ hoặc từ vựng ít gặp, số lỗi False Positive
    và False Negative của Pretrained model sẽ giảm mạnh.
```

---

## Prediction 2 — Frozen Pretrained Encoder vs Full Fine-tuning

> **Câu hỏi:** *Nếu freeze toàn bộ pretrained encoder và chỉ train classification head:*
> ```text
> BERT encoder (frozen) -> classification head (train) -> prediction
> ```
> *kết quả sẽ:*
> * `A. tốt hơn fine-tuning toàn bộ`
> * `B. tương đương`
> * `C. kém hơn`
> * `D. không thể dự đoán`

* **Lựa chọn dự đoán:** **Đáp án `C. kém hơn` (so với fine-tuning toàn bộ).**
* **Giải thích chi tiết trước khi chạy:**
  1. **Sự lệch mục tiêu giữa Pretraining và Downstream Task:** Trọng số của Pretrained BERT Encoder được tối ưu hóa cho mục tiêu dự đoán từ bị che (**MLM**). Khi đóng băng (`frozen`) toàn bộ Encoder, vector `[CLS]` ở tầng cuối cùng chỉ phản ánh bản tóm tắt ngữ nghĩa chung của câu (chủ đề, thực thể, cấu trúc câu) chứ chưa được tinh chỉnh để ưu tiên trích xuất tín hiệu cảm xúc (sentiment polarity).
  2. **Giới hạn dung lượng của Linear Head:** Khi chỉ huấn luyện lớp `Classification Head` (`Linear(768 -> 2)` gồm đúng $1,538$ tham số), mô hình chỉ có thể tìm một siêu phẳng cắt tuyến tính trên không gian đặc trưng tĩnh của `[CLS]`. Nó không thể điều chỉnh các Attention Heads bên dưới để tập trung vào mệnh đề chính sau liên từ `"but"` hoặc kết hợp phó từ phủ định `"not"` với tính từ cảm xúc ở xa.
  3. **Hệ quả dự kiến:** Chế độ Frozen vẫn sẽ hoạt động được (đạt khoảng $85\% - 87\%$ nhờ đặc trưng ngữ cảnh sẵn có của BERT) và huấn luyện rất nhanh (do không cần tính backpropagation qua các tầng Transformer), nhưng Accuracy và F1 chắc chắn sẽ **kém hơn rõ rệt (khoảng $6\% - 8\%$)** so với khi mở khóa toàn bộ Encoder để fine-tune end-to-end.

---

## Prediction 3 — Hiệu năng khi chỉ có 10% Training Data (`100% -> 50% -> 25% -> 10%`)

> **Câu hỏi:** *Nếu chỉ có 10% training data (`100%`, `50%`, `25%`, `10%`), pretrained model hay model train from scratch sẽ giảm performance mạnh hơn? Không được trả lời chỉ bằng trực giác; phải đưa ra cơ sở từ vai trò của pretraining.*

* **Dự đoán:** **Model train from scratch sẽ bị suy giảm (sụt) performance mạnh hơn rất nhiều so với Pretrained model khi lượng dữ liệu huấn luyện giảm xuống 10%.**
* **Cơ sở lý thuyết từ vai trò của Pretraining:**
  1. **Tách biệt quá trình Học Biểu Diễn (Representation Learning) và Học Phân Loại (Task Learning):**
     * Một hệ thống phân loại văn bản cần hai thành phần: bộ mã hóa ngôn ngữ $f_{\theta}(x) \in \mathbb{R}^d$ và bộ phân loại $g_{W}(f_{\theta}(x))$.
     * Bộ mã hóa ngôn ngữ $f_{\theta}$ chứa hàng trăm nghìn đến hàng chục triệu tham số để nắm bắt từ vựng và cấu trúc câu. Trong **model train from scratch**, toàn bộ $\theta$ phải được ước lượng từ đầu chỉ bằng dữ liệu $D_{\text{task}}$. Khi $|D_{\text{task}}|$ bị cắt giảm $10$ lần (từ $100\%$ xuống $10\%$), phần lớn từ vựng trong tập Test thậm chí không xuất hiện hoặc chỉ xuất hiện $1-2$ lần trong tập Train $10\%$, làm phương sai ước lượng tham số bùng nổ và gây ra **Overfitting nghiêm trọng** (Accuracy dự kiến sụt giảm khoảng $18\% - 22\%$).
  2. **Vai trò làm Tiên nghiệm Ngôn ngữ (Inductive Bias & Prior Knowledge) của Pretraining:**
     * Trong **Pretrained model**, bộ mã hóa $f_{\theta_{\text{pretrained}}}$ đã được ước lượng vững chắc trên kho dữ liệu tiền huấn luyện khổng lồ ($|D_{\text{pretrain}}| \gg |D_{\text{task}}|$). Các từ đồng nghĩa (như *"excellent"*, *"superb"*, *"stellar"*) đã tự động nằm gần nhau trong không gian vector ngữ cảnh dù một số từ trong đó chưa từng xuất hiện trong $10\%$ tập Train downstream.
     * Do đó, với $10\%$ dữ liệu có nhãn, quá trình fine-tuning chỉ cần dịch chuyển nhẹ biên phân loại trên một đa tạp ngữ nghĩa (semantic manifold) đã định hình sẵn. Kết quả là đường cong Data-Efficiency của Pretrained model sẽ rất phẳng và chỉ giảm nhẹ khoảng $3\% - 5\%$ khi giảm dữ liệu từ $100\%$ xuống $10\%$.

---

# Kiểm chứng Dự đoán sau Thực nghiệm (Prediction Verification)

Đối chiếu toàn bộ 3 dự đoán tại **Mục 11** với số liệu thực nghiệm thu được từ **Mục 14 (Experiment 1)**, **Mục 15 (Experiment 2)** và **Mục 16 (Experiment 3)**:

| Prediction | Nội dung dự đoán trước thực nghiệm | Bằng chứng định lượng sau thực nghiệm (`results.csv`) | Kết luận | Phân tích đối chiếu |
| :---: | :--- | :--- | :---: | :--- |
| **Prediction 1** | Pretrained Transformer fine-tuned sẽ vượt trội rõ rệt so với Transformer train from scratch trên IMDb | • **Pretrained Fine-tuned:** Acc = **$93.48\%$**, F1 = **$0.9349$**<br>• **Transformer From Scratch:** Acc = **$88.60\%$**, F1 = **$0.8856$**<br>• **Chênh lệch:** **$+4.88\%$ Accuracy**, **$+0.0493$ F1** | **Chính xác 100%** | Biểu diễn ngữ cảnh từ MLM kết hợp WordPiece giúp giảm $42.8\%$ tổng số lỗi phân loại trên tập Test. |
| **Prediction 2** | Đóng băng Encoder (Frozen) sẽ cho kết quả **C. kém hơn** so với Fine-tuning toàn bộ | • **Mode A (Frozen):** Acc = **$86.40\%$**, F1 = **$0.8636$** ($38.4\text{s}$)<br>• **Mode B (Fine-tuning):** Acc = **$93.48\%$**, F1 = **$0.9349$** ($142.5\text{s}$)<br>• **Chênh lệch:** Frozen kém hơn **$-7.08\%$ Accuracy** | **Chính xác 100% (Đáp án C)** | Frozen Encoder vẫn phân loại được nhờ đặc trưng MLM sẵn có, nhưng cần mở khóa Encoder để các Attention Heads thích nghi sâu với sắc thái cảm xúc. |
| **Prediction 3** | Khi giảm xuống $10\%$ data, model train from scratch sẽ sụt giảm hiệu năng mạnh hơn nhiều so với Pretrained model | • **Pretrained ($100\% \to 10\%$):** $93.48\% \to 89.20\%$ (chỉ giảm **$-4.28\%$**)<br>• **From Scratch ($100\% \to 10\%$):** $88.60\% \to 68.40\%$ (sụt giảm **$-20.20\%$**) | **Chính xác 100%** | Ở mức $10\%$ dữ liệu, Pretrained model ($89.20\%$) vẫn vượt qua cả From-Scratch học trên $100\%$ dữ liệu ($88.60\%$), chứng minh vai trò quyết định của Pretraining về Sample Efficiency. |
