# PHÂN TÍCH LỖI VÀ GIỚI HẠN CỦA WORD EMBEDDINGS (LAB 03 - ERROR ANALYSIS)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên**: 23000111  
**Giảng viên lý thuyết**: TS. Lê Hồng Phương  
**Giảng viên thực hành**: ThS. Phạm Ngọc Hải  

---

# Mục lục
- [25. Error analysis](#25-error-analysis)
  - [25.1. Ba trường hợp similarity đúng (Expected Similarities)](#251-ba-trường-hợp-similarity-đúng-expected-similarities)
  - [25.2. Ba trường hợp similarity sai hoặc bất ngờ (Unexpected Similarities)](#252-ba-trường-hợp-similarity-sai-hoặc-bất-ngờ-unexpected-similarities)
  - [25.3. Bảng tổng kết phân loại nguyên nhân gây lỗi (Root Causes Summary)](#253-bảng-tổng-kết-phân-loại-nguyên-nhân-gây-lỗi-root-causes-summary)

---

# 25. Error analysis

Trong mô hình không gian vector và biểu diễn phân phối (Distributional Representations như Co-occurrence Matrix hay Word2Vec Skip-gram / CBOW), chất lượng của các vector từ được đo lường thông qua khả năng phản ánh trực giác ngữ nghĩa của con người. Tuy nhiên, do bản chất của thuật toán là **tối ưu hóa hàm mục tiêu thống kê dựa trên ngữ cảnh xuất hiện cục bộ (distributional statistics)** thay vì thực sự "hiểu" ngôn ngữ, mô hình thường xuyên bộc lộ các lỗi hệ thống hoặc đưa ra các kết quả phản trực giác.

Mục này tiến hành mổ xẻ chi tiết:
- **3 trường hợp dự đoán đúng / phù hợp trực giác (Successful/Expected Cases)**
- **3 trường hợp lỗi hoặc sai lệch bất ngờ (Failure/Surprising Cases)**

Mỗi trường hợp được phân tích toàn diện theo 4 tiêu chí bắt buộc:
1. **Observed (Quan sát thực nghiệm)**: Điểm số hoặc thứ hạng tương đồng thực tế từ mô hình (được trích xuất trực tiếp từ kết quả thực thi notebook thực tế).
2. **Expected (Kỳ vọng ngôn ngữ học)**: Mối quan hệ ngữ nghĩa thực tế theo trực giác con người.
3. **Possible explanation (Cơ chế sinh lỗi / Giải thích nguyên nhân)**: Phân tích cơ chế toán học và đặc tính phân phối sinh ra hiện tượng.
4. **Evidence from corpus (Bằng chứng ngữ liệu)**: Các mẫu câu và cấu trúc kết hợp *(kèm nhãn ví dụ ngữ cảnh minh họa)*.

---

## 25.1. Ba trường hợp similarity đúng (Expected Similarities)

### Trường hợp 1: Cặp từ đồng nghĩa `doctor` — `physician`
- **Observed (Quan sát thực nghiệm)**:
  - Điểm Cosine Similarity: **$0.6944$** (Skip-gram Baseline, $d=100$, $window=5$).
  - Trong thực nghiệm cửa sổ ngữ cảnh (Exp 3), điểm số đạt **$0.7686$** ($window=2$) và **$0.7141$** ($window=5$).
  - Nằm trong Top lân cận cao nhất của `doctor` và xếp thứ 2 trong bảng đánh giá tương đồng tổng quát (Mục 21).
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - Đây là cặp từ đồng nghĩa chân thực (true synonyms / paradigmatic equivalence). Kỳ vọng điểm tương đồng phải đạt mức rất cao và đứng đầu danh sách gợi ý.
- **Possible explanation (Giải thích nguyên nhân)**:
  - Cả hai từ đều đóng vai trò cú pháp là danh từ chỉ người làm nghề y tế, thực hiện các hành động chuyên môn giống hệt nhau (*examine, diagnose, prescribe, treat, advise*), và cùng đi kèm với các tân ngữ/ngữ cảnh tương tự (*patient, hospital, medicine, illness*).
  - Thuật toán Skip-gram tối đa hóa xác suất dự đoán ngữ cảnh $P(c \mid w)$, vì phân phối $P(c \mid doctor) \approx P(c \mid physician)$ nên hai vector bị kéo hội tụ về cùng một cụm trong không gian liên tục.
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ minh họa)*: *"The doctor treated the sick patient at the clinic."*
  - *(Ví dụ minh họa)*: *"A qualified physician treated the elderly patient with modern therapies."*
  - Ngữ cảnh chia sẻ các token cốt lõi: `treated`, `patient`, `clinic/hospital`.

---

### Trường hợp 2: Cặp quan hệ Thượng vị — Hạ vị `car` — `vehicle`
- **Observed (Quan sát thực nghiệm)**:
  - Điểm Cosine Similarity: **$0.7724$** (ở $window=5$, $d=100$) và **$0.7316$** (ở $window=2$).
  - Đồng thời, cặp đồng nghĩa `car - automobile` đạt điểm số dẫn đầu toàn bảng đánh giá Mục 21 với **$0.7273$**.
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - `vehicle` là từ thượng vị (hypernym) của `car` (hyponym). Trong giao tiếp thực tế, người ta thường dùng hai từ này thay thế cho nhau nhằm tránh lặp từ (anaphoric reference / synonymy in context).
- **Possible explanation (Giải thích nguyên nhân)**:
  - Cả `car` và `vehicle` đều xuất hiện cùng các động từ chỉ hành động cơ giới hóa (*drive, park, buy, repair, crash, sell, license*) và các tính từ mô tả đặc tính phương tiện (*electric, hybrid, fast, expensive, safe*).
  - Sự trùng lặp sâu sắc về không gian ngữ cảnh chức năng này giúp vector của chúng đạt khoảng cách rất ngắn.
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ minh họa)*: *"He parked his car in front of the garage."*
  - *(Ví dụ minh họa)*: *"Police inspected the damaged vehicle after the accident on the highway."*
  - Cả hai từ cùng làm chủ ngữ/tân ngữ cho hệ động từ giao thông đường bộ.

---

### Trường hợp 3: Cặp thực thể phân biệt ranh giới `computer` — `banana`
- **Observed (Quan sát thực nghiệm)**:
  - Điểm Cosine Similarity: **$0.1170$** (xấp xỉ mức trực giao, xếp cuối cùng trong bảng Mục 21).
  - Đối với từ `doctor`, cặp `doctor - banana` có độ tương đồng chỉ **$0.0267$** (xấp xỉ bằng $0$).
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - `computer` (công nghệ thông tin) và `banana` (nông sản / thực phẩm) thuộc hai trường từ vựng tách biệt tuyệt đối (disjoint semantic domains), không có quan hệ ngữ nghĩa hay liên tưởng.
- **Possible explanation (Giải thích nguyên nhân)**:
  - Tập từ ngữ cảnh của `computer` gồm: *software, hardware, cpu, screen, internet, algorithm, data*.
  - Tập từ ngữ cảnh của `banana` gồm: *fruit, yellow, peel, eat, monkey, potassium, sweet, tree*.
  - Tập giao ngữ cảnh gần như rỗng (ngoại trừ các từ dừng phổ thông). Do đó, tích vô hướng trong quá trình huấn luyện Negative Sampling bị đẩy về $0$, giữ hai vector gần như vuông góc trong không gian $d$ chiều.
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ minh họa)*: Không tồn tại văn cảnh chung nào mà `computer` và `banana` chia sẻ các từ nội dung lân cận ngoài các hư từ ngữ pháp.

---

## 25.2. Ba trường hợp similarity sai hoặc bất ngờ (Unexpected Similarities)

### Trường hợp 1: Nhầm lẫn giữa Tương đồng Chức năng và Liên kết Chủ đề (`doctor` — `hospital`)
- **Observed (Quan sát thực nghiệm)**:
  - Điểm Cosine Similarity: **$0.4962$** (ở baseline) và **$0.5120$ – $0.5936$** (trong khảo sát cửa sổ $window=2 \to 10$).
  - Trong không gian vector, `hospital` có điểm tương đồng với `doctor` rất cao, bám sát các từ chỉ người/đồng nghiệp.
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - `doctor` là danh từ chỉ người (animate agent / profession), trong khi `hospital` là danh từ chỉ địa điểm/tổ chức (inanimate location / institution).
  - Về mặt bản chất quan hệ ngôn ngữ học: Đây là quan hệ **liên kết chủ đề (Syntagmatic / Topical Association)** chứ **không phải quan hệ tương đồng ngữ nghĩa (Paradigmatic / Semantic Similarity)**. Một bác sĩ không phải là một bệnh viện.
- **Possible explanation (Cơ chế sinh lỗi)**:
  - **Ảnh hưởng của Context Window**: Khi mở rộng cửa sổ ngữ cảnh, mô hình bao quát phạm vi rộng cả câu. Trong các văn bản y tế, `doctor` và `hospital` hầu như luôn xuất hiện đồng thời trong cùng một đoạn văn.
  - Mô hình tĩnh Word2Vec không phân biệt được vai trò ngữ pháp (chủ ngữ vs trạng ngữ chỉ nơi chốn). Hàm mất mát chỉ tối ưu hóa việc "hai từ này hay xuất hiện gần nhau", dẫn đến việc kéo vector của thực thể người và địa điểm lại sát nhau, làm xóa nhòa ranh giới giữa *tính đồng nghĩa* và *tính liên tưởng chủ đề*.
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ minh họa)*: *"The doctor works long night shifts at the emergency hospital."*
  - Cả hai từ cùng xuất hiện trong phạm vi 3–5 token liên tiếp với tần số rất cao.

---

### Trường hợp 2: Nghịch lý từ trái nghĩa hoặc quan hệ đối kháng (`doctor` — `disease`)
- **Observed (Quan sát thực nghiệm)**:
  - Điểm Cosine Similarity: **$0.3862$**.
  - `disease` (bệnh tật) có độ tương đồng khá cao với `doctor` (bác sĩ), vượt xa các từ dị biệt bên ngoài miền y tế như `computer` ($0.3334$), `football` ($0.2628$) hay `banana` ($0.0267$).
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - `doctor` (chủ thể chữa bệnh) và `disease` (đối tượng bệnh lý cần tiêu diệt) là hai khái niệm đối lập về mặt mục tiêu và bản chất thực thể. Kỳ vọng chúng phải có sự phân biệt rõ ràng thay vì được mô hình đánh giá là có liên hệ chặt chẽ.
- **Possible explanation (Cơ chế sinh lỗi)**:
  - **Co-occurrence of Antonyms / Opposites (Hiện tượng đồng xuất hiện của các cặp đối kháng)**: Đây là một trong những điểm yếu kinh điển nhất của Giả thuyết Phân phối (Distributional Hypothesis).
  - Các cặp từ đối kháng nhau (như *doctor - disease*, *hot - cold*, *increase - decrease*, *good - bad*) luôn xuất hiện trong các cấu trúc ngữ pháp và ngữ cảnh đàm thoại giống hệt nhau:
    - *"The doctor studies the disease."*
    - *"Treating disease is the main duty of a doctor."*
  - Do có cùng tập hợp từ ngữ cảnh bao quanh (*treatment, medicine, severe, symptoms, cure*), vector của chúng bị tối ưu hóa để nằm gần nhau. Word2Vec hoàn toàn bất lực trong việc nhận diện quan hệ logic phủ định hoặc đối kháng ngữ nghĩa.
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ minh họa)*: *"Specialist doctors diagnose rare infectious diseases in clinical trials."*

---

### Trường hợp 3: Sự suy sụp biểu diễn do Hiện tượng Đa nghĩa (`bank` collapse - Phân tích ngữ nghĩa kinh điển)
- **Observed (Quan sát thực nghiệm - Ví dụ lý thuyết minh họa)**:
  - *(Ví dụ minh họa lý thuyết kinh điển)*: Khi khảo sát từ đa nghĩa `bank`, danh sách lân cận thường bị pha trộn hỗn loạn giữa các từ tài chính (*money, credit, finance*) và các từ địa lý/tự nhiên (*river, water, lake, stream*).
  - Điểm tương đồng giữa `bank` với `river` và `finance` bị kéo tụt đáng kể so với khi các từ này đứng trong miền đơn nghĩa chuyên biệt.
- **Expected (Kỳ vọng ngôn ngữ học)**:
  - Trong tiếng Anh, `bank` là từ đa nghĩa rõ rệt (Polysemy / Homonymy):
    - Nghĩa 1: Ngân hàng tài chính (*financial institution*).
    - Nghĩa 2: Bờ sông / bờ đê (*river edge / slope*).
  - Con người có thể phân biệt dứt khoát hai nghĩa này dựa trên ngữ cảnh tức thời của câu nói.
- **Possible explanation (Cơ chế sinh lỗi)**:
  - **Giới hạn cố hữu của Static Word Embeddings (Một từ chỉ có duy nhất một vector cố định)**:
    $$\vec{v}_{\text{bank}} = \alpha \vec{v}_{\text{finance}} + (1 - \alpha) \vec{v}_{\text{river}}$$
  - Trong quá trình huấn luyện, mỗi khi gặp câu về ngân hàng, gradient kéo vector $\vec{v}_{\text{bank}}$ về phía cực tài chính. Khi gặp câu về bờ sông, gradient lại kéo nó về phía cực thủy văn.
  - Kết quả cuối cùng là vector của `bank` bị treo lơ lửng ở vị trí trung bình trọng số giữa hai cụm nghĩa, không thực sự đại diện trọn vẹn cho bất kỳ nghĩa nào. Đây chính là động lực lịch sử thúc đẩy sự ra đời của **Contextualized Word Representations** (ELMo, BERT, Transformer).
- **Evidence from corpus (Bằng chứng ngữ liệu - Ví dụ minh họa)**:
  - *(Ví dụ ngữ cảnh 1)*: *"I deposited cash into my personal bank account."*
  - *(Ví dụ ngữ cảnh 2)*: *"They walked along the grassy river bank enjoying the cool breeze."*

---

## 25.3. Bảng tổng kết phân loại nguyên nhân gây lỗi (Root Causes Summary)

| STT | Nguyên nhân gốc rễ (Root Cause) | Biểu hiện thực nghiệm | Giải pháp khắc phục |
| :---: | :--- | :--- | :--- |
| **1** | **Context Window quá rộng** | Nhầm lẫn giữa từ đồng nghĩa (paradigmatic) và từ cùng chủ đề (syntagmatic) | Thu hẹp cửa sổ ($k = 1$ hoặc $k = 2$) khi cần ưu tiên quan hệ cú pháp và từ đồng nghĩa |
| **2** | **Hiện tượng đồng xuất hiện của từ đối lập (Antonyms)** | Các cặp từ trái nghĩa (*good-bad, doctor-disease*) có cosine similarity quá cao | Bổ sung tri thức từ vựng ngoài (WordNet), retrofitting hoặc dùng hàm mất mát có ràng buộc đối lập |
| **3** | **Từ đa nghĩa (Polysemy / Homonymy)** | Vector bị kéo về vị trí trung bình cộng, gây nhiễu cho cả hai trường nghĩa | Chuyển đổi sang mô hình ngôn ngữ ngữ cảnh (Contextual Embeddings: BERT, RoBERTa) |
| **4** | **Thiên lệch ngữ liệu (Domain Bias)** | Từ ngữ bị méo mó ý nghĩa nếu corpus tập trung quá mức vào một chủ đề | Huấn luyện trên ngữ liệu đa miền (multi-domain) cân bằng và đủ lớn |
| **5** | **Tần số xuất hiện thấp (Data Sparsity / Rare Words)** | Vector của các từ hiếm không hội tụ, bị chi phối bởi khởi tạo ngẫu nhiên | Sử dụng mô hình biểu diễn cấp độ ký tự / subword (FastText, BPE) |
