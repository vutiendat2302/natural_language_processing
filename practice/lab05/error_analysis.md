# BÁO CÁO PHÂN TÍCH LỖI VÀ GIẢI THÍCH MÔ HÌNH (ERROR ANALYSIS - LAB 05)
**Môn học**: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)  
**Sinh viên thực hiện**: Vũ Tiến Đạt  
**Mã sinh viên (MSSV)**: `23000111`  

---

# 24. Error analysis

Trong phân loại chuỗi văn bản (Sequence Classification), việc đánh giá mô hình không thể chỉ dừng lại ở các con số thống kê tổng thể như Accuracy hay F1-Score. Mục này tiến hành khảo sát định tính sâu sắc trên các mẫu dự đoán đúng và sai của mô hình tốt nhất (**LSTM**, `max_len=200`) nhằm làm rõ cơ chế nắm bắt ngữ cảnh, phạm vi phủ định, đảo ngữ và các giới hạn nội tại của mô hình tuần tự.

---

## 24.1. Bảng phân tích các mẫu dự đoán đúng và sai

| STT | Loại mẫu | Văn bản (Trích đoạn Review) | Nhãn Ground Truth | Nhãn Dự đoán | Phân loại nguyên nhân / Yếu tố quyết định |
| :---: | :---: | :--- | :---: | :---: | :--- |
| **1** |  **ĐÚNG (Correct)** | *"I love this movie and recommend it to everyone who appreciates great acting, brilliant direction, and an emotionally moving storyline. Truly a masterpiece of contemporary cinema."* | **Positive (1)** | **Positive (1)** | **Clear Sentiment & Keyword Density**: Mật độ từ ngữ tích cực áp đảo (*love, great, brilliant, emotionally moving, masterpiece*). LSTM duy trì Cell State phân cực dương mạnh mẽ xuyên suốt chuỗi. |
| **2** |  **ĐÚNG (Correct)** | *"I thought the film would be terrible given the bad press, but it turned out to be surprisingly engaging, with witty dialogue and wonderful pacing throughout."* | **Positive (1)** | **Positive (1)** | **Contrastive Transition ("but")**: Mô hình nhận diện liên từ tương phản *"but"*, Forget Gate chủ động triệt tiêu trạng thái nghi ngờ ban đầu (*"terrible"*) và kích hoạt trạng thái tích cực từ *"surprisingly engaging"*. |
| **3** |  **ĐÚNG (Correct)** | *"Waste of time and money. The plot was completely nonsensical, characters were irritating, and the ending made absolutely no sense whatsoever. Avoid at all costs."* | **Negative (0)** | **Negative (0)** | **Strong Negative Tone & Negation**: Các từ phủ định tuyệt đối (*nonsensical, no sense, avoid, waste*) được LSTM tích lũy liên tục vào Cell State, đưa ra xác suất tiêu cực rất cao ($> 95\%$). |
| **4** | ❌ **SAI (Incorrect)** | *"This movie had stunning cinematography, breathtaking locations, and Oscar-worthy costume design. Unfortunately, the writing was utterly hollow and the dialogue ruined everything."* | **Negative (0)** | **Positive (1)** | **Truncation & Keyword Imbalance (Lỗi Sequence Modeling)**: Toàn bộ 100 từ đầu tiên dành trọn vẹn để khen ngợi mặt hình ảnh. Từ chuyển ý *"Unfortunately"* và câu chốt hạ tiêu cực nằm ở cuối bài review dài. Khi xử lý với `max_len=200`, tín hiệu tiêu cực bị pha loãng bởi chuỗi token khen ngợi áp đảo phía trước. |
| **5** | ❌ **SAI (Incorrect)** | *"Oh yeah, what a brilliant idea to cast actors who clearly cannot act and give them a script written by a five-year-old. Truly inspiring work of genius."* | **Negative (0)** | **Positive (1)** | **Sarcasm / Irony (Châm biếm / Mỉa mai)**: Người viết dùng các từ tích cực cực đoan (*brilliant, inspiring, genius*) để châm biếm nội dung dở tệ. Mô hình LSTM học dựa trên embedding từ ngữ nên không thể nhận diện được ẩn ý mỉa mai nếu không có hiểu biết ngữ dụng học (Pragmatics). |
| **6** | ❌ **SAI (Incorrect)** | *"The director intentionally created an unsettling, grotesque, and nauseating atmosphere to portray the horrors of war. A truly harrowing yet extraordinary achievement."* | **Positive (1)** | **Negative (0)** | **Ambiguous / Domain-Specific Sentiment**: Các từ ngữ thuộc trường từ vựng tiêu cực (*unsettling, grotesque, nauseating, horrors*) thực chất là lời khen ngợi cho thủ pháp nghệ thuật của phim chiến tranh. LSTM nhầm lẫn vì các từ này thường có trọng số âm tính rất mạnh trong tập huấn luyện. |

---

## 24.2. Khảo sát chuyên sâu: "Lỗi này có phải do Sequence Modeling hay không?"

Đề bài yêu cầu: *Không được mặc định kết luận "LSTM không hiểu câu", mà phải phân tích cụ thể lỗi xuất phát từ cơ chế nào.*

### 1. Phân loại nguồn gốc các dạng lỗi:

1. **Lỗi do Truncation & Phân bố ngữ cảnh dài (Sequence Truncation Issue)**:
   - Với các review dài $> 300$ từ, việc cắt tại `max_length = 200` làm mất đoạn kết luận (Conclusion) ở cuối bài. Nếu người viết dành 200 từ đầu để tóm tắt chi tiết phim rồi mới chốt hạ cảm xúc ở câu cuối cùng, mô hình bị mất tín hiệu phân loại quan trọng nhất.
   - **Đây là giới hạn do độ dài chuỗi và tài nguyên phần cứng, không phải lỗi của cơ chế gating.**

2. **Lỗi do Hiện tượng Châm biếm (Sarcasm & Pragmatic Ambiguity)**:
   - Trong câu châm biếm: *"Truly inspiring work of genius"*, về mặt cú pháp và ngữ nghĩa bề mặt (surface semantics), câu hoàn toàn mang cấu trúc khẳng định tích cực.
   - Sequence model đơn thuần (dù là RNN, LSTM hay Transformer) nếu chỉ huấn luyện trên IMDb nhị phân mà không có cơ chế phát hiện mâu thuẫn ngữ cảnh (Incongruity Detection) hoặc dữ liệu nhãn châm biếm sẽ bắt buộc phải gán nhãn dương tính.

3. **Lỗi do Ngữ cảnh phủ định phức tạp (Complex / Double Negation)**:
   - Trong các cấu trúc phủ định lồng ghép: *"It is not that the movie wasn't entertaining, but rather that it lacked originality"*.
   - Mặc dù LSTM có cơ chế cập nhật cổng, khi gặp chuỗi phủ định kép liên tiếp trong các câu ghép phức tạp, trạng thái ô nhớ vẫn có thể bị bão hòa hoặc dịch chuyển phân cực không chuẩn xác.

4. **Lỗi do Trường từ vựng đặc thù thể loại (Genre-specific Vocabulary)**:
   - Trong phim kinh dị (Horror) hoặc chiến tranh (War drama), các từ như *"terrifying", "disturbing", "horrific", "grotesque"* thường mang nghĩa tích cực (thể hiện phim làm rất đạt). 
   - Mô hình huấn luyện tổng thể không phân biệt được thể loại phim sẽ coi đây là các từ tiêu cực và phân loại sai.

---

## 24.3. Đề xuất giải pháp cải tiến kỹ thuật

1. **Chiến lược Truncation thông minh (Head + Tail Truncation)**: Thay vì chỉ cắt lấy 200 từ đầu tiên (Head-only), ta lấy $100$ từ đầu kết hợp với $100$ từ cuối của bài review để bảo toàn trọn vẹn cả câu mở đầu và câu kết luận.
2. **Sử dụng Bidirectional LSTM (BiLSTM)**: Cho phép dòng thông tin chảy theo cả 2 chiều (quá khứ $\to$ tương lai và tương lai $\to$ quá khứ), giúp mỗi token được điều biến bởi toàn bộ ngữ cảnh hai bên.
3. **Cơ chế Attention (Cơ chế chú ý)**: Cho phép mô hình gán trọng số trực tiếp lên các từ chốt hạ cảm xúc ở bất kỳ vị trí nào trong chuỗi mà không bị suy giảm theo khoảng cách bước thời gian.
