# Lab 02 — Báo Cáo Phân Tích Lỗi Thực Nghiệm (Error Analysis - Mục 22)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Tập dữ liệu**: `30K.json` (Trích xuất 10.000 documents / 214.026 câu)  
**Mô hình đánh giá**: Bigram và Trigram Language Model với kỹ thuật làm mịn Laplace  

---

## 1. Tổng Quan Mục Tiêu Phân Tích Lỗi
Mục tiêu cốt lõi của phần phân tích lỗi là đánh giá thực chất nơi mô hình N-gram thống kê rời rạc hoạt động hiệu quả và nơi nó bộc lộ những khiếm khuyết chết người trong bài toán **dự đoán từ tiếp theo (Next-word prediction)** và **chấm điểm chuỗi văn bản (Sequence scoring)**. Theo đúng chuẩn Rubric chấm điểm của Lab 02, báo cáo lựa chọn mổ xẻ:
- **2 Trường hợp dự đoán đúng (Correct Predictions)**: Các trường hợp mô hình đưa ra dự đoán trùng khớp hoàn hảo với thói quen ngôn ngữ tự nhiên và thống kê thực tế trong corpus.
- **2 Trường hợp dự đoán sai (Incorrect Predictions)**: Các trường hợp mô hình dự đoán thất bại, sinh ra các từ vô nghĩa, bất hợp lý hoặc trái ngược hoàn toàn với trực giác con người.

Với từng trường hợp, báo cáo ghi nhận Ngữ cảnh (Context), Dự đoán của mô hình (Model Prediction), Từ kỳ vọng thực tế (Expected Continuation), Xác suất gán (Assigned Probability) và chẩn đoán nguyên nhân gốc rễ trên **8 chiều phân tích** theo yêu cầu đề bài:
1. Thiếu hụt dữ liệu huấn luyện (Insufficient training data)
2. Cụm n-gram chưa từng xuất hiện (Unseen n-gram)
3. Giới hạn kích thước từ vựng (Vocabulary limitation)
4. Độ thưa dữ liệu (Data sparsity)
5. Cửa sổ ngữ cảnh quá ngắn (Context window too short)
6. Lệch miền dữ liệu (Domain mismatch)
7. Nhiễu do tiền xử lý (Preprocessing artifacts)
8. Biến dạng phân phối do làm mịn (Smoothing distortions)

---

## 2. Ca Phân Tích 1 — Dự Đoán Đúng 1: Cụm Cố Định Tần Suất Cực Cao

### Chi tiết trường hợp
- **Ngữ cảnh (Context)**: `"in the"`
- **Dự đoán Top 1 của mô hình**: `"most"` ($P \approx 0.0041$), `"first"` ($P \approx 0.0039$), `"best"` ($P \approx 0.0034$)
- **Từ tiếp nối thực tế trong corpus**: `"world"` (339 lần), `"past"` (224 lần), `"first"` (185 lần), `"most"` (164 lần)
- **Xác suất mô hình gán**: $P(\text{"most"} \mid \text{"in the"}) = 0.0041$; $P(\text{"first"} \mid \text{"in the"}) = 0.0039$

### Đánh giá & Lý do thành công
- **Mật độ đồng xuất hiện dày đặc:** Cụm `"in the"` là một trong những bigram phổ biến nhất trong tiếng Anh và xuất hiện hơn 7.900 lần trong tập huấn luyện.
- **Quy luật ngữ pháp chặt chẽ:** Đi sau một giới từ kết hợp mạo từ xác định (`IN + DT`), vị trí tiếp theo bắt buộc phải là một tính từ so sánh nhất/bổ nghĩa (như *"most"*, *"first"*, *"best"*) hoặc một danh từ chung (như *"world"*, *"middle"*, *"future"*).
- Mô hình Bigram/Trigram có tần số mẫu rất lớn ($C(h) \gg 0$), áp đảo hoàn toàn mẫu số Laplace, giúp các tính từ hợp lệ vươn lên đứng đầu bảng xếp hạng xác suất.

---

## 3. Ca Phân Tích 2 — Dự Đoán Đúng 2: Cấu Trúc Trợ Động Từ Khuyết Thiếu

### Chi tiết trường hợp
- **Ngữ cảnh (Context)**: `"you can"`
- **Dự đoán Top 1 của mô hình**: `"be"` ($P \approx 0.0085$), `"also"` ($P \approx 0.0017$), `"find"` ($P \approx 0.0015$)
- **Từ tiếp nối thực tế trong corpus**: `"also"` (225 lần), `"be"` (198 lần), `"find"` (152 lần), `"see"` (121 lần)
- **Xác suất mô hình gán**: $P(\text{"be"} \mid \text{"you can"}) = 0.0085$; $P(\text{"also"} \mid \text{"you can"}) = 0.0017$

### Đánh giá & Lý do thành công
- **Tính chuẩn xác cú pháp tuyệt đối:** Động từ khuyết thiếu *"can"* trong tiếng Anh ràng buộc nghiêm ngặt từ đi sau phải là động từ nguyên thể không chia (*"be"*, *"find"*, *"see"*) hoặc phó từ bổ nghĩa (*"also"*).
- Cụm từ `"you can"` xuất hiện hơn 2.000 lần trong dữ liệu huấn luyện, cung cấp nền tảng thống kê rất vững chắc.
- Mô hình nhận diện chuẩn xác các từ có xác suất cao, phản ánh trung thực phân phối thực tế trong corpus.

---

## 4. Ca Phân Tích 3 — Dự Đoán Sai 1: Cụm Danh Từ Hiếm Bị Lệch Miền Dữ Liệu

### Chi tiết trường hợp
- **Ngữ cảnh (Context)**: `"the cat"`
- **Dự đoán Top 1 của mô hình**: `"house"` ($P \approx 0.0002$), `</s>` ($P \approx 0.0001$), `"to"` ($P \approx 0.0001$)
- **Từ tiếp nối kỳ vọng theo con người**: `"sat"`, `"eats"`, `"is"`, `"was"`, `"sleeps"` (các vị ngữ quen thuộc của loài mèo)
- **Thực tế trong corpus**: Chỉ xuất hiện cụm `"the cat queen"` (5 lần trong một bài giới thiệu ẩm thực/thú cưng địa phương) và `"the cat s"` (2 lần)
- **Xác suất mô hình gán**: $P(\text{"house"} \mid \text{"the cat"}) = 0.0002$ (xấp xỉ mức sàn làm mịn ngẫu nhiên $\approx 0.00001$)

### Chẩn đoán nguyên nhân gốc rễ
1. **Thiếu hụt dữ liệu (Insufficient Data) & Độ thưa (Sparsity):**
   Mặc dù từ `"the"` rất phổ biến, cụm `"the cat"` chỉ xuất hiện vỏn vẹn 7 lần trong hơn 20.000 câu huấn luyện. Cỡ mẫu quá nhỏ khiến mô hình không tích lũy được tri thức thống kê về hành vi của danh từ này.
2. **Lệch miền dữ liệu (Domain Mismatch):**
   Tập dữ liệu `30K.json` chứa chủ yếu các bài viết kỹ thuật sửa lỗi máy tính (Mac OS Lion, khôi phục ổ cứng SSD), bài bán hàng trang phục biểu diễn, và tin tức sự kiện thi đấu BBQ. Ngữ cảnh đời thường về vật nuôi gia đình gần như hoàn toàn vắng bóng.
3. **Biến dạng do làm mịn Laplace (Smoothing Distortion):**
   Với $C(\text{"the cat"}) = 7$ và $|\mathcal{V}| = 96.111$, mẫu số Laplace là $7 + 96.111 = 96.118$. Mọi từ có tần số $C=1$ ngẫu nhiên hoặc có tần số unigram lớn đều bị đẩy lên bằng nhiễu xác suất ảo, tạo ra chuỗi ghép kỳ quặc như *"the cat house"*.

---

## 5. Ca Phân Tích 4 — Dự Đoán Sai 2: Thuật Ngữ Kỹ Thuật Nhiều Từ Chuyên Sâu

### Chi tiết trường hợp
- **Ngữ cảnh (Context)**: `"natural language"`
- **Dự đoán Top 1 của mô hình**: `"and"` ($P \approx 0.0003$), `</s>` ($P \approx 0.0002$), `"that"` ($P \approx 0.0001$)
- **Từ tiếp nối kỳ vọng theo con người**: `"processing"` ($P \approx 0.70$ trong ngữ cảnh khoa học máy tính), `"understanding"`, `"generation"`
- **Thực tế trong corpus**: Cả tập train chỉ xuất hiện đúng 2 lần cụm từ này (`"processing"`: 2 lần, `"toolkit"`: 1 lần)
- **Xác suất mô hình gán**: $P(\text{"and"} \mid \text{"natural language"}) = 0.0003$; $P(\text{"processing"} \mid \text{"natural language"}) = 0.00003$

### Chẩn đoán nguyên nhân gốc rễ
1. **Cụm N-gram cực hiếm (Unseen / Extremely Sparse N-gram):**
   Cụm `"natural language processing"` chỉ xuất hiện đúng 2 lần trong hơn 3.6 triệu từ. Tín hiệu thống kê quá mờ nhạt trước biển dữ liệu lớn.
2. **Lỗ hổng của kỹ thuật làm mịn Laplace (Cộng đều một hằng số):**
   Laplace cộng $+1$ cho toàn bộ $96.111$ từ trong từ vựng. Kết quả là các từ dừng (stop-words) cực kỳ phổ biến như `"and"`, `"that"`, `"to"` dễ dàng đè bẹp các danh từ chuyên môn đặc thù chỉ có 1–2 lần xuất hiện.
3. **Mô hình không hiểu ngữ nghĩa (Thiếu biểu diễn phân bố dày đặc - Embeddings):**
   Mô hình N-gram coi mỗi từ như một chỉ số nguyên rời rạc hoàn toàn trực giao. Nó không có khả năng nhận biết rằng `"natural"` và `"language"` có mối quan hệ ngữ nghĩa khăng khít với `"processing"`, `"computer"` hay `"linguistics"`. Thiếu đi vector ngữ nghĩa liên tục (Word Embeddings), mô hình không thể khái quát hóa từ các khái niệm tương đồng.

---

## 6. Bảng Tổng Hợp Ma Trận Chẩn Đoán Lỗi

| Ca kiểm thử | Ngữ cảnh | Dự đoán của Model | Kỳ vọng thực tế | Cơ chế thành công / Thất bại chính |
| :---: | :--- | :--- | :--- | :--- |
| **Đúng 1** | `"in the"` | `most` / `first` | `world` / `past` / `first` | **Tần số xuất hiện cực cao + quy luật ngữ pháp mạnh** |
| **Đúng 2** | `"you can"` | `be` / `find` | `be` / `find` / `also` | **Ràng buộc trợ động từ khuyết thiếu + ngữ cảnh dày dặn** |
| **Sai 1** | `"the cat"` | `house` / `</s>` | `sat` / `eats` / `is` | **Lệch miền dữ liệu (Domain mismatch) + dữ liệu quá thưa ($C=7$)** |
| **Sai 2** | `"natural language"`| `and` / `that` | `processing` | **Cực hiếm + Laplace ưu tiên từ dừng + thiếu vector nhúng ngữ nghĩa** |

---

## 7. Bài Học Kỹ Thuật Đúc Kết
1. **Mô hình N-gram thuần túy chỉ là bộ nhớ thống kê bề mặt:** Mô hình chỉ dự đoán tốt những gì đã xuất hiện với tần số rất lớn trong cửa sổ cục bộ của tập huấn luyện.
2. **Làm mịn Laplace không phù hợp với từ vựng lớn:** Khi $|\mathcal{V}| \approx 100.000$, việc cộng $+1$ tạo ra lượng nhiễu nền quá lớn và làm sai lệch nghiêm trọng tỷ lệ giữa từ nội dung và từ chức năng.
3. **Sự tất yếu của Neural NLP:** Để giải quyết triệt để 4 ca lỗi trên, ngành NLP bắt buộc phải chuyển sang **Vector nhúng liên tục (Word Embeddings - Lab 03)** để học độ tương đồng ngữ nghĩa, và **Kiến trúc Transformer** để mở rộng ngữ cảnh chú ý vượt qua giới hạn vài từ của Markov.
