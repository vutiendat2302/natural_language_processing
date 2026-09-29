# Lab 02 — Báo Cáo Phản Tư & Cầu Nối Sang Mô Hình Ngôn Ngữ Neural (Mục 24 & 26)

**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Học viên**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  
**Chủ đề**: Phản tư bản chất của Mô hình N-gram, Độ thưa dữ liệu, Thước đo Perplexity và Bước chuyển dịch sang LLM  

---

## Phần 1: Trả Lời Chi Tiết 7 Câu Hỏi Phản Tư (Mục 24)

### Câu 1: Nếu tăng $n$, mô hình nhận thêm thông tin gì?
> *Khi chuyển từ Unigram sang Bigram, Trigram và các bậc cao hơn, mô hình tiếp thu thêm loại tri thức nào?*

**Trả lời chi tiết:**
- Khi tăng $n$ từ $1 \to 2 \to 3 \dots$, mô hình mở rộng **cửa sổ ngữ cảnh điều kiện hóa** từ $0$ từ phía trước (Unigram) lên $1$ từ (Bigram), $2$ từ (Trigram), và tổng quát là $(n-1)$ từ liền trước:
  $$P(w_t \mid w_{t-n+1}, \dots, w_{t-1})$$
- **Các dạng thông tin mô hình nhận thêm:**
  1. **Sự phụ thuộc cú pháp cục bộ (Short-range Syntactic Dependencies):** Mô hình học được các ràng buộc ngữ pháp cơ bản như sự hòa hợp chủ vị (ví dụ: `"he eats"` thay vì `"he eat"`), quy tắc mạo từ (ví dụ: `"an apple"` thay vì `"a apple"`), và sự kết hợp giới từ (ví dụ: `"depend on"`).
  2. **Cụm từ cố định và thuật ngữ (Collocations & Idiomatic Phrasing):** Mô hình nắm bắt được các cụm từ ghép mà nghĩa tổng thể không thể phân tách thành từng từ đơn lẻ, như `"machine learning"`, `"artificial intelligence"`, `"New York"`.
  3. **Giảm độ bất định (Entropy Reduction):** Dưới góc độ lý thuyết thông tin, bổ sung thêm biến điều kiện không bao giờ làm tăng entropy:
     $$H(W_t \mid W_{t-2}, W_{t-1}) \le H(W_t \mid W_{t-1}) \le H(W_t)$$
     Phân phối xác suất của từ tiếp theo trở nên sắc nét hơn, giúp mô hình tự tin hơn trong các dự đoán có ngữ cảnh cụ thể.

---

### Câu 2: Tại sao tăng $n$ lại làm sparsity tăng?
> *Tại sao việc tăng bậc $n$ tất yếu dẫn đến hiện tượng dữ liệu cực thưa (data sparsity)?*

**Trả lời chi tiết:**
- **Sự bùng nổ tổ hợp của không gian trạng thái (Curse of Dimensionality):**
  Với một tập từ vựng $\mathcal{V}$ có kích thước $V = |\mathcal{V}|$, số lượng cấu hình n-gram khả dĩ trên lý thuyết tăng theo cấp số nhân $V^n$:
  - Unigram: $V \approx 10^5$ tham số.
  - Bigram: $V^2 \approx 10^{10}$ tham số.
  - Trigram: $V^3 \approx 10^{15}$ tham số.
- **Giới hạn hữu hạn của kho ngữ liệu thực tế:**
  Dù tập huấn luyện có chứa hàng triệu hay hàng tỷ từ, nó cũng chỉ quan sát được một tỷ lệ cực kỳ nhỏ trong không gian tổ hợp khổng lồ này. Thực nghiệm trên file `30K.json` đã chỉ ra bằng chứng định lượng rõ rệt:
  - Unigram: $96.111$ loại từ $\to$ **$44.87\%$** là *hapax legomena* (chỉ xuất hiện đúng 1 lần).
  - Bigram: $1.159.102$ cụm $\to$ **$73.12\%$** là *hapax legomena*.
  - Trigram: $2.361.677$ cụm $\to$ **$88.14\%$** là *hapax legomena*!
- Càng tăng $n$, cụm từ càng mang tính đặc thù cao, tần suất lặp lại càng giảm sâu, và hầu như mọi ô trong ma trận chuyển tiếp đều mang giá trị $0$. Khi kiểm thử trên văn bản mới, xác suất gặp phải một cụm n-gram chưa từng thấy (unseen) có thể lên tới $90\% - 99\%$.

---

### Câu 3: Tại sao smoothing cần thiết?
> *Tại sao kỹ thuật làm mịn (smoothing) là bắt buộc đối với mô hình ngôn ngữ thống kê?*

**Trả lời chi tiết:**
- **Tránh thảm họa xác suất bằng 0 (Zero-Frequency Problem):**
  Theo ước lượng MLE truyền thống, bất kỳ cụm từ nào chưa từng xuất hiện trong tập huấn luyện sẽ bị gán xác suất bằng $0$. Do xác suất của một câu là tích của chuỗi các xác suất thành phần:
  $$P(W) = \prod_{t=1}^T P(w_t \mid context_t)$$
  Chỉ cần một cụm n-gram có xác suất bằng $0$, toàn bộ xác suất của câu lập tức bị kéo sụp về $0.0$. Hậu quả là log-probability trở thành $-\infty$ và Perplexity bùng nổ lên $+\infty$, khiến toàn bộ hệ thống tê liệt.
- **Phân biệt "Chưa thấy trong mẫu" khác với "Không thể xảy ra":**
  Một kho ngữ liệu dù lớn đến đâu cũng không thể bao quát hết khả năng sinh ngôn ngữ vô hạn của con người. Câu *"I study AI"* hoàn toàn hợp lệ dù tập train chỉ có *"I study NLP"*. Smoothing đóng vai trò "tái phân bổ" một phần nhỏ ngân sách xác suất từ các từ xuất hiện nhiều sang các trường hợp chưa từng thấy, tạo ra một sàn xác suất dương an toàn cho mọi câu văn mới.

---

### Câu 4: Perplexity đo điều gì?
> *Perplexity đo lường điều gì dưới góc nhìn toán học và trực giác ngôn ngữ?*

**Trả lời chi tiết:**
- **Định nghĩa toán học:** Perplexity là nghịch đảo trung bình nhân của các xác suất thành phần, tương đương với hàm mũ của mất mát cross-entropy trên tập kiểm thử:
  $$PP(W) = \exp\left(-\frac{1}{N} \sum_{i=1}^N \ln P(w_i \mid context_i)\right) = \left(\prod_{i=1}^N P(w_i \mid context_i)\right)^{-\frac{1}{N}}$$
- **Ý nghĩa trực quan (Số nhánh phân vân hiệu dụng - Effective Branching Factor):**
  Perplexity đo mức độ "bối rối", "ngạc nhiên" của mô hình khi đọc văn bản thực tế của con người:
  - Nếu một mô hình có Perplexity bằng $K$, điều đó có nghĩa là tại mỗi bước dự đoán từ tiếp theo, mô hình đang phân vân tương đương với việc **chọn ngẫu nhiên đều giữa $K$ từ có khả năng như nhau**.
  - Perplexity càng thấp chứng tỏ mô hình gán xác suất càng cao cho câu văn tự nhiên, độ ngạc nhiên càng nhỏ, và mô hình nắm bắt phân phối ngôn ngữ càng chuẩn xác.

---

### Câu 5: Một model có perplexity thấp hơn có luôn tạo ra văn bản tốt hơn đối với con người không? Giải thích.
> *Liệu một mô hình có perplexity thấp hơn trên test set có luôn sinh ra văn bản chất lượng cao hơn trong mắt con người không?*

**Trả lời chi tiết:**
**KHÔNG LUÔN LUÔN.**
1. **Perplexity đo khả năng chấm điểm, không đo chiến lược sinh văn bản (Generation/Decoding):**
   Perplexity chỉ đo xác suất của văn bản mẫu chuẩn do con người viết sẵn. Trong khi đó, việc sinh văn bản phụ thuộc vào giải thuật giải mã (Greedy Search, Beam Search, Sampling, Top-p/Top-k). Một mô hình có perplexity rất thấp vẫn có thể bị mắc kẹt trong vòng lặp vô tận (như *"the cat sat on the mat on the mat on the mat..."*) hoặc sinh ra văn bản đơn điệu, sáo rỗng.
2. **Thiên lệch từ chức năng (Function Word Bias):**
   Một mô hình có thể đạt perplexity rất đẹp nhờ đoán đúng các hư từ cực kỳ phổ biến (mạo từ *the, a*, giới từ *in, on*, liên từ *and*), nhưng lại đoán sai hoàn toàn các từ khóa nội dung quan trọng (danh từ, động từ chính), khiến câu văn trở nên vô nghĩa đối với người đọc.
3. **Phụ thuộc vào kích thước từ vựng và tiền xử lý:**
   Perplexity cực kỳ nhạy cảm với cách phân chia từ vựng. Một mô hình có từ vựng tí hon ($|\mathcal{V}|=500$) có thể đạt perplexity rất thấp chỉ vì không gian lựa chọn quá hẹp, nhưng nó hoàn toàn vô dụng để diễn đạt các khái niệm phong phú của con người.

---

### Câu 6: N-gram language model thất bại ở đâu khi so với cách con người hiểu ngôn ngữ?
> *Mô hình N-gram cổ điển thất bại ở những điểm cốt tử nào khi so sánh với cơ chế tư duy ngôn ngữ của con người?*

**Trả lời chi tiết:**
1. **Hoàn toàn mù tịt về ngữ nghĩa phân bố (Biểu diễn rời rạc, trực giao):**
   Mô hình N-gram coi mỗi từ như một chỉ số nguyên độc lập, không có sự liên hệ hình học. Đối với N-gram, khoảng cách giữa `"cat"` và `"dog"` cũng xa lạ hệt như giữa `"cat"` và `"refrigerator"`. Nếu mô hình quan sát thấy *"the dog barked"*, nó hoàn toàn không có khả năng liên tưởng sang *"the puppy barked"*.
2. **Bất lực trước cấu trúc ngữ pháp phân cấp (Hierarchical Structure):**
   Ngôn ngữ tự nhiên có cấu trúc cây cú pháp lồng nhau, không phải là một chuỗi Markov phẳng. Xét câu có quan hệ chủ-vị cách xa nhau:
   > *"The **books** that I borrowed from the university library last week **were** fascinating."*
   Một mô hình Trigram khi đứng trước từ `"were"` chỉ nhìn thấy 2 từ trước là `"last week"` (dạng số ít), nên nó sẽ dự đoán sai thành `"was"`, bỏ lỡ hoàn toàn chủ ngữ số nhiều `"books"` cách đó 10 từ.
3. **Không có tri thức thế giới thực (Grounded World Knowledge):**
   Con người hiểu ngôn ngữ nhờ liên hệ với thực tại vật lý, tính logic, và nhân quả. Mô hình N-gram đơn thuần chỉ là một bảng đếm tần số bề mặt không mang tri thức ngữ nghĩa.

---

### Câu 7: Nếu context dài 100 từ, trigram có sử dụng được thông tin của 97 từ đầu không?
> *Nếu đoạn ngữ cảnh dài 100 từ, liệu mô hình Trigram có thể tận dụng tri thức của 97 từ đầu tiên không?*

**Trả lời chi tiết:**
- **HOÀN TOÀN KHÔNG THỂ (Giới hạn tầm nhìn Markov - Markov Horizon Blindness).**
- Theo định nghĩa toán học của giả định Markov bậc hai, xác suất của từ thứ 100 chỉ điều kiện hóa trên 2 từ đứng liền kề nó:
  $$P(w_{100} \mid w_1, w_2, \dots, w_{99}) \equiv P(w_{100} \mid w_{98}, w_{99})$$
- Toàn bộ 97 từ đầu tiên ($w_1, \dots, w_{97}$) bị vứt bỏ hoàn toàn và bị coi là độc lập có điều kiện với $w_{100}$.
- **Cầu nối trực tiếp sang Deep Learning và Transformers:**
  - Điểm nghẽn nghiêm trọng này đã định hình lộ trình phát triển lịch sử của toàn ngành NLP:
    1. **Mạng nơ-ron hồi quy (RNN / LSTM / GRU):** Đưa vào trạng thái ẩn $h_t = f(h_{t-1}, x_t)$ nhằm nén và duy trì thông tin ngữ cảnh dài hạn qua thời gian.
    2. **Biểu diễn phân bố dày đặc (Word Embeddings - Lab 03):** Ánh xạ từ vựng vào không gian vector liên tục (Word2Vec, GloVe, FastText), giải quyết triệt để tính trực giao của N-gram.
    3. **Cơ chế chú ý (Self-Attention) & Transformer (LLM hiện đại):** Xóa bỏ hoàn toàn giới hạn tuần tự, cho phép token thứ 100 kết nối trực tiếp với toàn bộ 99 từ phía trước thông qua tích vô hướng ma trận chú ý ($\text{Attention}(Q, K, V)$).

---

## Phần 2: Chuẩn Bị Vấn Đáp Nhanh 3 Phút (Mục 26)

Dưới đây là 5 câu trả lời ngắn gọn, chuẩn xác, ghi điểm tối đa khi giảng viên hoặc trợ giảng (TA) chọn ngẫu nhiên:

### Câu hỏi 1: Vì sao bigram có zero probability?
- **Trả lời**: Vì kho ngữ liệu huấn luyện luôn hữu hạn. Khi kiểm tra một cặp từ $(w_{t-1}, w_t)$ chưa từng xuất hiện cùng nhau trong tập train, tần số $C(w_{t-1}, w_t) = 0$. Ước lượng MLE sẽ tính ra $0 / C(w_{t-1}) = 0.0$. Vì xác suất câu là tích các xác suất thành phần, một thừa số 0 sẽ triệt tiêu toàn bộ xác suất của câu về 0.

### Câu hỏi 2: Tại sao phải dùng log probability?
- **Trả lời**: Khi tính toán cho các câu dài hàng chục hoặc hàng trăm từ, việc nhân liên tiếp các xác suất nhỏ $p_i \in (0, 1)$ sẽ làm giá trị tích suy giảm cực nhanh, gây ra lỗi tràn số dưới (Numerical Underflow) trong bộ nhớ máy tính. Chuyển sang không gian Log ($\ln P(S) = \sum \ln p_i$) sẽ biến phép nhân thành phép cộng các số âm, bảo toàn độ chính xác số học tuyệt đối.

### Câu hỏi 3: Perplexity thấp nghĩa là gì?
- **Trả lời**: Perplexity đo độ "bối rối" của mô hình, tương đương với số lượng từ ứng viên mà mô hình đang phân vân lựa chọn tại mỗi bước (effective branching factor). Perplexity càng thấp chứng tỏ mô hình càng gán xác suất cao cho văn bản thực tế, mức độ bất ngờ càng nhỏ, và mô hình hóa ngôn ngữ càng chính xác.

### Câu hỏi 4: Tại sao trigram không nhất thiết tốt hơn bigram trên test set?
- **Trả lời**: Do hiện tượng quá khớp (Overfitting) và dữ liệu cực thưa (Data Sparsity). Không gian trạng thái của Trigram là $O(V^3)$. Trên tập train nhỏ, các cụm 3 từ rất hiếm khi lặp lại, dẫn đến ước lượng thống kê có phương sai rất lớn. Khi sang test set, Trigram gặp vô số cụm từ chưa từng thấy, buộc smoothing phải can thiệp thô bạo làm loãng xác suất, khiến test perplexity tăng vọt; trong khi Bigram có ước lượng ổn định và khái quát hóa tốt hơn.

### Câu hỏi 5: Nếu `"cat eats"` chưa xuất hiện trong training corpus thì model xử lý thế nào?
- **Trả lời**: 
  - Nếu dùng **MLE**: Mô hình gán $P(\text{eats} \mid \text{cat}) = 0.0$, kéo xác suất của toàn bộ câu chứa cụm này về 0.
  - Nếu dùng **Laplace Smoothing**: Mô hình bổ sung pseudo-count $k=1$ vào tử số và $V$ vào mẫu số:
    $$P_{Laplace}(\text{eats} \mid \text{cat}) = \frac{0 + 1}{C(\text{cat}) + V} = \frac{1}{C(\text{cat}) + V} > 0$$
    Giúp câu vẫn nhận được một xác suất dương nhỏ và tránh việc Perplexity bùng nổ vô hạn.
