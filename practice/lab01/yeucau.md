  ### 2. Cấu trúc thư mục & Danh sách deliverables cần nộp (Mục 17)       
    ```                                                                      
  Toàn bộ bài làm của Lab 01 nằm trong thư mục practice/lab01/ với cây thư
  mục tiêu chuẩn:                                                                                                                      
    lab01/                                                                
    ├── README.md                      # Hướng dẫn chạy mã nguồn, môi     
  trường và tổng kết kết quả                                              
           
    ├── implementation.py              # Tự cài đặt cốt lõi TF-IDF &      
  Cosine Similarity (Part E)                                              
    ├── experiments.ipynb              # (hoặc 23000111_VuTienDat_BT1.    
  ipynb) Toàn bộ thí nghiệm trên corpus 30K (Part D, F, G, H)             
    ├── results.csv                    # Kết quả retrieval và đánh giá    
  định lượng (P@5, R@5, MRR)                                              
    └── reflection.md                  # Báo cáo phân tích lỗi, trả lời   
  Learning Check, Reflection & Khai báo AI                                                                                                
  │ Note                                           
  │ Theo quy chuẩn đặt tên file trong môn học (AGENTS.md): file notebook  
  │ bài tập cá nhân là 23000111_VuTienDat_BT1.ipynb (hoặc tạo             
  │ alias/symbolic link tương ứng nếu hệ thống nộp tự động yêu cầu đúng   
  tên                                                                     
  │ experiments.ipynb).                                                   
  ──────                        
  ```                                          
  ### 3. Chi tiết nội dung từng thành phần cần hoàn thiện                 
                                                                          
  #### 📄 calculations.md — Bài tập tính toán thủ công (Part B — 10 điểm) 
                                                                          
  Thực hiện tính toán từng bước chi tiết trên corpus đồ chơi gồm 3 câu:   
  D₁ = "cat eats fish", D₂ = "dog eats fish", D₃ = "cat likes fish".      
                                                                          
  • Ex 1 — Count Vector: Tính vector đếm với từ vựng 5 từ: ["cat", "dog", 
  "eats", "fish", "likes"].                                               
  • Ex 2 — TF: Tính ma trận Term Frequency theo các dạng chuẩn hóa khác   
  nhau.                                                                   
  • Ex 3 — IDF: Tính Document Frequency và Inverse Document Frequency theo
  công thức chuẩn và smoothed IDF.                                        
  • Ex 4 — TF-IDF: Nhân TF × IDF và chuẩn hóa L₂.                         
  • Ex 5 — Cosine Similarity: Tính độ tương đồng giữa các tài liệu.       
  • Ex 6 — Qualitative Prediction: Dự đoán mức độ tương đồng và xếp hạng  
  giữa các câu y khoa ("medical image classification" vs "medical image   
  analysis", "natural language processing").                              
                                                                          
  #### 📄 prediction.md — Dự đoán trước thực nghiệm (Part C — 10 điểm)    
                                                                          
  Ghi lại các dự đoán độc lập (không xem code/kết quả trước) về:          
                                                                          
  • Vocabulary size: Kích thước từ vựng dự kiến trên tập dữ liệu 30.000   
  tài liệu.                                                               
  • Matrix Sparsity: Tỷ lệ thưa của ma trận TF-IDF (S = 1 - nnz(X)/(N ×   
  V)).                                                                    
  • Search ranking: Hành vi xếp hạng khi truy vấn có/không có lexical     
  overlap.                                                                
                                                                          
  #### 🐍 implementation.py — Tự cài đặt thuật toán cốt lõi (Part E — 20  
  điểm)                                                                   
                                                                          
  Triển khai từ đầu (scratch), không sử dụng trực tiếp lớp TfidfVectorizer
  của scikit-learn cho phần lõi:                                          
                                                                          
  • Cài đặt đủ 6 hàm:                                                     
      1. build_vocabulary(corpus)                                         
      2. compute_counts(corpus, vocab)                                    
      3. compute_tf(counts)                                               
      4. compute_idf(corpus, vocab, smooth=...)                           
      5. compute_tfidf(tf, idf, norm=...)                                 
      6. cosine_similarity(vec1, vec2)                                    
  • Unit tests: Viết kiểm thử assert trên corpus 3 câu mẫu, đối chiếu sai 
  số < 10⁻⁹ so với kết quả giải tay trong calculations.md.                
  • So sánh với thư viện: Đối chiếu kết quả với scikit-learn và giải thích
  lý do chênh lệch (nếu có) về quy ước làm mịn (smoothing) hoặc chuẩn hóa 
  L₂.                                                                     
                                                                          
  #### 📓 experiments.ipynb (hoặc 23000111_VuTienDat_BT1.ipynb) — Thực    
  nghiệm & Đánh giá (45 điểm)                                             
                                                                          
  Chứa mã nguồn và kết quả chạy thực nghiệm trên tập 30K văn bản:         
                                                                          
  1. Experiment 1: Sparse Representation (Part D — Đã tích hợp):          
      • Đo đạc kích thước ma trận X ∈ ℝ^{N×V}, độ thưa S.                 
      • Trích xuất Top 20 từ theo Document Frequency, Top 20 từ có IDF cao
      nhất, Top 20 từ có TF-IDF cao nhất trong 1 văn bản đại diện.        
  2. Experiment 2: Preprocessing Ablation (Part F — 15 điểm):             
      • So sánh 3 pipeline:                                               
          • Pipeline A (Minimal): Lowercase → Tokenization.               
          • Pipeline B (Normalized): Lowercase → Punctuation removal →    
          Tokenization → Stopwords removal.                               
          • Pipeline C (Extended): Normalization → Subword Tokenization   
          (BPE/WordPiece).                                                
      • Đo lường bảng chỉ số: Vocab size, Average tokens/doc, Matrix      
      sparsity, OOV rate, Search performance.                             
  3. Application: Document Search Engine (Part G — 15 điểm):              
      • Xây dựng engine tìm kiếm tài liệu từ truy vấn người dùng dựa trên 
      TF-IDF và Cosine Similarity.                                        
      • Hiển thị bảng kết quả: Rank | Document ID | Similarity | Document 
      Preview (Top-5 tài liệu).                                           
  4. Quantitative Evaluation (Part H — 15 điểm) & results.csv:            
      • Thiết kế tập đánh giá gồm 5–10 queries mẫu có gán nhãn tài liệu   
      liên quan (ground truth).                                           
      • Đo 3 metric cốt lõi:                                              
          • Precision@5 (P@5)                                             
          • Recall@5 (R@5)                                                
          • Mean Reciprocal Rank (MRR)                                    
      • Xuất toàn bộ bảng kết quả đánh giá ra file results.csv.           
                                                                          
                                                                          
  #### 📄 reflection.md — Phân tích lỗi, Đánh giá & Khai báo AI (Part I,  
  J, 15, 16 — 15 điểm)                                                    
                                                                          
  File văn bản (tối đa ~500 từ cho phần reflection chính) bao gồm các mục:
                                                                          
  1. Error Analysis (Part I — 10 điểm):                                   
      • Phân tích 2 queries kết quả tốt và 2 queries kết quả kém.         
      • Mổ xẻ 1 Failure Case quan trọng nhất do rào cản ngữ nghĩa từ vựng 
      (ví dụ: query "heart attack treatment" không tìm thấy văn bản chứa  
      "myocardial infarction therapy" do không có lexical overlap).       
  2. From Failure to Next Representation (Part J):                        
      • Đưa ra giả thuyết giải pháp biểu diễn ngữ nghĩa vượt qua rào cản  
      từ vựng rời rạc (chuyển đổi từ TF-IDF → Word Embeddings →           
      Transformers).                                                      
  3. Learning Check (Mục 15):                                             
      • Trả lời ngắn gọn và chính xác 7 câu hỏi lý thuyết/bản chất cuối   
      lab.                                                                
  4. Reflection (Mục 16 — 5 điểm):                                        
      • Trả lời 6 câu hỏi đúc kết: Prediction nào sai? Kết quả nào bất    
      ngờ? Experiment nào bằng chứng rõ nhất? Nếu làm lại sẽ thay đổi gì? 
  5. AI Usage Policy (Mục 14 — Bắt buộc):                                 
      • Báo cáo minh bạch khối AI contribution (nêu rõ AI hỗ trợ phần nào:
      gợi ý tối ưu thuật toán, debug syntax,... và phần nào là sinh viên  
      tự tính toán, tự phân tích).                                        
                                                                          
  ──────                                                                  
  ### 4. Thang điểm đánh giá (Assessment Rubric — Mục 18)                 
                                                                          
   Thành phần đánh giá                                     │ Điểm tối đa
  ─────────────────────────────────────────────────────────┼──────────────
   Theory recap & calculation (calculations.md)            │      10
   Prediction (prediction.md)                              │      10
   Core implementation (implementation.py)                 │      20
   Preprocessing experiment (Ablation study trên Notebook) │      15
   Search application (Search Engine trên Notebook)        │      15
   Quantitative evaluation (P@5, R@5, MRR & results.csv)   │      15
   Error analysis (Phân tích lỗi trên reflection.md)       │      10
   Reflection / Learning Check (reflection.md)             │      5
   TỔNG CỘNG                                               │     100
                                                                          
  │ Nguyên tắc chấm điểm của Giảng viên:                                  
  │ Điểm số không dựa vào việc viết nhiều dòng code hay dùng thư viện phức
  │ tạp, mà chấm chủ yếu vào:                                             
                                                                          
    Understanding + Experimental Reasoning + Evidence + Interpretation    
                                                                          
  │ (Mức độ thấu hiểu bản chất + Lập luận thực nghiệm + Dẫn chứng số liệu +
  │ Phân tích kết quả).                                                   