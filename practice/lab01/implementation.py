from typing import List, Union, Optional
import numpy as np


# ==============================================================================
# 8.2 CÁC HÀM CẦN XÂY DỰNG
# ==============================================================================

def build_vocabulary(corpus: List[str]) -> List[str]:
    """
    Xây dựng từ điển (vocabulary) từ tập văn bản, được sắp xếp theo thứ tự bảng chữ cái.

    Args:
        corpus: Danh sách các văn bản thô (chuỗi ký tự).

    Returns:
        Danh sách các từ duy nhất (unique tokens) xuất hiện trong toàn bộ corpus, đã sắp xếp.
    """
    unique_terms = set()
    for doc in corpus:
        # Chuẩn hóa chữ thường và tách từ theo khoảng trắng
        tokens = doc.lower().split()
        unique_terms.update(tokens)
    return sorted(list(unique_terms))

def compute_counts(corpus: List[str], vocab: List[str]) -> np.ndarray:
    """
    Xây dựng ma trận đếm Bag-of-Words cho tập văn bản dựa trên từ điển cho trước.

    Args:
        corpus: Danh sách các chuỗi văn bản.
        vocab: Danh sách từ vựng (xác định thứ tự các cột).

    Returns:
        Mảng numpy 2D kích thước (N, V), trong đó phần tử (i, j) là số lần xuất hiện
        của từ j trong văn bản i.
    """
    term_to_idx = {term: idx for idx, term in enumerate(vocab)}
    num_docs = len(corpus)
    vocab_size = len(vocab)
    counts = np.zeros((num_docs, vocab_size), dtype=np.int64)

    for doc_idx, doc in enumerate(corpus):
        tokens = doc.lower().split()
        for token in tokens:
            if token in term_to_idx:
                counts[doc_idx, term_to_idx[token]] += 1

    return counts


def compute_tf(counts: np.ndarray, mode: str = "relative") -> np.ndarray:
    """
    Tính tần suất xuất hiện của từ (Term Frequency - TF) từ ma trận đếm thô.

    Các chế độ hỗ trợ:
        - "relative" (mặc định, theo giáo trình / Jurafsky & Martin):
            tf(t, d) = c(t, d) / sum_{t'} c(t', d)
            Tổng mỗi hàng bằng 1.0 (đối với văn bản không rỗng).
        - "raw":
            tf(t, d) = c(t, d) (số đếm thô, tương tự như cách dùng nội bộ của scikit-learn).
        - "log":
            tf(t, d) = 1 + log(c(t, d)) nếu c(t, d) > 0 ngược lại bằng 0 (thu nhỏ phi tuyến - sublinear scaling).

    Args:
        counts: Mảng numpy 2D chứa số lần xuất hiện của từ, kích thước (N, V).
        mode: Công thức tính TF ("relative", "raw", hoặc "log").

    Returns:
        Mảng numpy 2D chứa các giá trị TF kiểu float64.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if mode == "relative":
        doc_lengths = counts.sum(axis=1, keepdims=True)
        # Tránh chia cho 0 đối với các văn bản hoàn toàn rỗng
        safe_lengths = np.where(doc_lengths == 0, 1.0, doc_lengths)
        return counts / safe_lengths

    elif mode == "raw":
        return counts.copy()

    elif mode == "log":
        tf = np.zeros_like(counts, dtype=np.float64)
        positive_mask = counts > 0
        tf[positive_mask] = 1.0 + np.log(counts[positive_mask])
        return tf

    else:
        raise ValueError(f"Không hỗ trợ chế độ TF: '{mode}'. Hãy chọn 'relative', 'raw', hoặc 'log'.")

def compute_idf(
    corpus_or_counts: Union[List[str], np.ndarray],
    vocab: Optional[List[str]] = None,
    mode: str = "standard"
) -> np.ndarray:
    """
    Tính vector tần số nghịch đảo của văn bản (Inverse Document Frequency - IDF) trên toàn bộ corpus.

    Các công thức hỗ trợ:
        - "standard" (công thức giáo trình trong Phần A.4 & Bài tập 3):
            idf(t) = ln(N / df(t))
            Lưu ý: Nếu df(t) == N thì idf(t) = ln(1) = 0.0.
        - "smooth" (mặc định của scikit-learn với smooth_idf=True):
            idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
        - "sklearn_unsmoothed" (scikit-learn với smooth_idf=False):
            idf(t) = ln(N / df(t)) + 1.0

    Args:
        corpus_or_counts: Danh sách văn bản dạng chuỗi hoặc ma trận đếm đã tính trước.
        vocab: Danh sách từ vựng (bắt buộc nếu corpus_or_counts là danh sách chuỗi).
        mode: Quy ước tính IDF ("standard", "smooth", hoặc "sklearn_unsmoothed").

    Returns:
        Mảng numpy 1D kích thước (V,) chứa trọng số IDF cho từng từ.
    """
    if isinstance(corpus_or_counts, list):
        if vocab is None:
            raise ValueError("vocab phải được cung cấp khi corpus_or_counts là danh sách chuỗi.")
        counts = compute_counts(corpus_or_counts, vocab)
    else:
        counts = np.asarray(corpus_or_counts)

    num_docs = counts.shape[0]
    # Document frequency: số văn bản chứa từ t (count > 0)
    df = np.sum(counts > 0, axis=0).astype(np.float64)

    if mode == "standard":
        # Xử lý trường hợp df bằng 0 một cách an toàn: nếu df == 0, gán IDF = 0.0
        safe_df = np.where(df == 0, 1.0, df)
        idf = np.log(num_docs / safe_df)
        idf[df == 0] = 0.0
        return idf

    elif mode == "smooth":
        # Công thức mặc định của scikit-learn: ln((1 + N) / (1 + df)) + 1.0
        return np.log((1.0 + num_docs) / (1.0 + df)) + 1.0

    elif mode == "sklearn_unsmoothed":
        # Scikit-learn khi tắt làm mịn (smooth_idf=False): ln(N / df) + 1.0
        safe_df = np.where(df == 0, 1.0, df)
        return np.log(num_docs / safe_df) + 1.0

    else:
        raise ValueError(f"Không hỗ trợ chế độ IDF: '{mode}'. Hãy chọn 'standard', 'smooth', hoặc 'sklearn_unsmoothed'.")


def compute_tfidf(
    tf: np.ndarray,
    idf: np.ndarray,
    norm: Optional[str] = None
) -> np.ndarray:
    """
    Tính ma trận biểu diễn TF-IDF.

    tfidf(t, d) = tf(t, d) * idf(t)

    Chuẩn hóa tùy chọn:
        - None: Giá trị tf * idf thô, không chuẩn hóa (như trong Bài tập 4).
        - "l2": Chuẩn hóa vector đơn vị Euclid theo từng văn bản (v / ||v||_2), khớp với scikit-learn.

    Args:
        tf: Mảng numpy 2D chứa các giá trị TF, kích thước (N, V).
        idf: Mảng numpy 1D chứa các giá trị IDF, kích thước (V,).
        norm: Phương pháp chuẩn hóa (None hoặc "l2").

    Returns:
        Mảng numpy 2D biểu diễn các vector TF-IDF, kích thước (N, V).
    """
    tf = np.asarray(tf, dtype=np.float64)
    idf = np.asarray(idf, dtype=np.float64)

    # Nhân từng phần tử (element-wise) với cơ chế broadcasting theo hàng
    tfidf = tf * idf

    if norm is None:
        return tfidf
    elif norm == "l2":
        norms = np.linalg.norm(tfidf, ord=2, axis=1, keepdims=True)
        # Tránh lỗi chia cho 0 đối với các vector toàn số 0
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return tfidf / safe_norms
    else:
        raise ValueError(f"Không hỗ trợ phương pháp chuẩn hóa: '{norm}'. Hãy chọn None hoặc 'l2'.")


def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Tính độ tương đồng Cosine (Cosine Similarity) giữa hai vector (hoặc 2 ma trận 2D).

    cos(x, y) = (x . y) / (||x||_2 * ||y||_2)

    Xử lý linh hoạt cho cả cặp vector 1D đơn lẻ hoặc ma trận 2D theo cặp.

    Args:
        vec1: Mảng 1D hoặc 2D kích thước (V,) hoặc (N1, V).
        vec2: Mảng 1D hoặc 2D kích thước (V,) hoặc (N2, V).

    Returns:
        Giá trị kiểu float nếu cả 2 đầu vào là vector 1D, hoặc ma trận tương đồng 2D.
    """
    v1 = np.asarray(vec1, dtype=np.float64)
    v2 = np.asarray(vec2, dtype=np.float64)

    # Trường hợp 1: Cả hai đầu vào đều là vector 1D
    if v1.ndim == 1 and v2.ndim == 1:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        dot_prod = np.dot(v1, v2)
        sim = dot_prod / (norm1 * norm2)
        # Cắt giá trị trong khoảng [-1.0, 1.0] để tránh sai số số học
        return float(np.clip(sim, -1.0, 1.0))

    # Trường hợp 2: Ma trận 2D (tính độ tương đồng theo cặp theo lô)
    v1_2d = np.atleast_2d(v1)
    v2_2d = np.atleast_2d(v2)

    norm1 = np.linalg.norm(v1_2d, ord=2, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2_2d, ord=2, axis=1, keepdims=True)

    safe_norm1 = np.where(norm1 == 0.0, 1.0, norm1)
    safe_norm2 = np.where(norm2 == 0.0, 1.0, norm2)

    v1_normalized = v1_2d / safe_norm1
    v2_normalized = v2_2d / safe_norm2

    sim_matrix = np.dot(v1_normalized, v2_normalized.T)
    # Gán 0 cho các hàng/cột có chuẩn ban đầu bằng 0
    sim_matrix[norm1.ravel() == 0.0, :] = 0.0
    sim_matrix[:, norm2.ravel() == 0.0] = 0.0

    return np.clip(sim_matrix, -1.0, 1.0)


# ==============================================================================
# 8.4 UNIT TESTS (Kiểm thử trên Corpus mẫu 8.3 & Đối soát Part B)
# ==============================================================================

def run_unit_tests() -> None:
    """
    Thực thi các unit test nghiêm ngặt trên tập ngữ liệu mẫu từ Phần B:
        D1 = "cat eats fish"
        D2 = "dog eats fish"
        D3 = "cat likes fish"

    Kiểm tra từng hàm so với các kết quả tính toán giải tích của Part B (trong calculations.md)
    với ngưỡng sai số số học < 1e-9.
    """
    print("=" * 70)
    print("CHẠY UNIT TESTS (Đối soát với kết quả lý thuyết Part B)")
    print("=" * 70)

    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    # --- Test 1: build_vocabulary ---
    vocab = build_vocabulary(corpus)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Từ điển không khớp: nhận được {vocab}, kỳ vọng {expected_vocab}"
    print("[PASS] Test 1: build_vocabulary() ->", vocab)

    # --- Test 2: compute_counts (Bài tập 1) ---
    counts = compute_counts(corpus, vocab)
    expected_counts = np.array([
        [1, 0, 1, 1, 0],  # D1: cat=1, dog=0, eats=1, fish=1, likes=0
        [0, 1, 1, 1, 0],  # D2: cat=0, dog=1, eats=1, fish=1, likes=0
        [1, 0, 0, 1, 1]   # D3: cat=1, dog=0, eats=0, fish=1, likes=1
    ], dtype=np.int64)
    assert np.array_equal(counts, expected_counts), f"Ma trận đếm không khớp:\n{counts}\nKỳ vọng:\n{expected_counts}"
    print("[PASS] Test 2: compute_counts() khớp chính xác với Bài tập 1.")

    # --- Test 3: compute_tf (Bài tập 2) ---
    tf = compute_tf(counts, mode="relative")
    # Đối với D1: tf(cat)=1/3, tf(eats)=1/3, tf(fish)=1/3
    expected_tf_d1 = np.array([1/3, 0.0, 1/3, 1/3, 0.0])
    assert np.all(np.abs(tf[0] - expected_tf_d1) < 1e-9), f"TF(D1) không khớp: {tf[0]}"
    # Xác minh tổng sum_t tf(t, d) == 1.0 cho tất cả văn bản
    row_sums = tf.sum(axis=1)
    assert np.all(np.abs(row_sums - 1.0) < 1e-9), f"Tổng hàng TF khác 1.0: {row_sums}"
    print("[PASS] Test 3: compute_tf() khớp với Bài tập 2 và tổng hàng sum_t tf(t, d) == 1.0.")

    # --- Test 4: compute_idf (Bài tập 3) ---
    # idf = ln(N / df):
    # cat: ln(3/2) = ln(1.5)
    # dog: ln(3/1) = ln(3.0)
    # eats: ln(3/2) = ln(1.5)
    # fish: ln(3/3) = ln(1.0) = 0.0
    # likes: ln(3/1) = ln(3.0)
    expected_idf = np.array([
        np.log(1.5),
        np.log(3.0),
        np.log(1.5),
        0.0,
        np.log(3.0)
    ])
    idf = compute_idf(counts, mode="standard")
    assert np.all(np.abs(idf - expected_idf) < 1e-9), f"IDF không khớp:\n{idf}\nKỳ vọng:\n{expected_idf}"
    # Xác nhận riêng từ 'fish' có idf == 0.0 như phân tích trong Bài tập 3
    fish_idx = vocab.index("fish")
    assert abs(idf[fish_idx] - 0.0) < 1e-9, f"IDF của 'fish' phải bằng 0.0, nhận được {idf[fish_idx]}"
    print("[PASS] Test 4: compute_idf() khớp với Bài tập 3 (IDF của 'fish' đúng bằng 0.0).")

    # --- Test 5: compute_tfidf (Bài tập 4) ---
    tfidf = compute_tfidf(tf, idf, norm=None)
    # Đối với D1: [ (1/3)*ln(1.5), 0, (1/3)*ln(1.5), 0, 0 ]
    expected_tfidf_d1 = np.array([
        (1/3) * np.log(1.5),
        0.0,
        (1/3) * np.log(1.5),
        0.0,
        0.0
    ])
    assert np.all(np.abs(tfidf[0] - expected_tfidf_d1) < 1e-9), f"TF-IDF(D1) không khớp: {tfidf[0]}"
    # Kiểm tra trọng số của từ 'fish' bằng 0 trong tất cả văn bản
    assert np.all(tfidf[:, fish_idx] == 0.0), "Từ 'fish' phải có TF-IDF = 0 trên toàn bộ văn bản."
    print("[PASS] Test 5: compute_tfidf() khớp với Bài tập 4 (tf * idf chưa chuẩn hóa).")

    # --- Test 6: cosine_similarity trên vector TF-IDF của D1 và D2 (Khép kín Pipeline) ---
    # Dùng trực tiếp vector TF-IDF tfidf[0] (D1) và tfidf[1] (D2) từ Test 5
    sim_d1_d2 = cosine_similarity(tfidf[0], tfidf[1])
    # Kỳ vọng giải tích từ Part B:
    # dot_product = (ln(1.5)/3)^2
    # ||D1||_2 = sqrt(2) * (ln(1.5)/3)
    # ||D2||_2 = (1/3) * sqrt((ln 3)^2 + (ln 1.5)^2)
    # cos(D1, D2) = ln(1.5) / (sqrt(2) * sqrt((ln 3)^2 + (ln 1.5)^2)) ≈ 0.2448303
    expected_sim_d1_d2 = np.log(1.5) / (np.sqrt(2.0) * np.sqrt(np.log(3.0)**2 + np.log(1.5)**2))
    assert abs(sim_d1_d2 - expected_sim_d1_d2) < 1e-9, f"Cosine similarity D1-D2 không khớp: {sim_d1_d2}"
    print(f"[PASS] Test 6: cosine_similarity(D1, D2) trực tiếp từ vector TF-IDF -> sim ≈ {sim_d1_d2:.6f}")

    # Kiểm tra bổ trợ với vector mẫu Bài tập 5: x=[1, 1, 1], y=[1, 1, 0] -> 2/sqrt(6)
    sim_toy = cosine_similarity(np.array([1, 1, 1]), np.array([1, 1, 0]))
    assert abs(sim_toy - 2.0 / np.sqrt(6.0)) < 1e-9, f"Vector mẫu Bài tập 5 không khớp: {sim_toy}"

    # --- Test 7: cosine_similarity toàn bộ ma trận TF-IDF (Batch 2D) ---
    # Kiểm tra tính năng batch 2D và tính nhất quán trên toàn bộ corpus D1, D2, D3
    sim_matrix = cosine_similarity(tfidf, tfidf)
    # 1. Đường chéo chính (tự tương đồng) phải bằng 1.0
    assert np.all(np.abs(np.diag(sim_matrix) - 1.0) < 1e-9), "Cosine similarity của vector với chính nó phải bằng 1.0"
    # 2. Ma trận tương đồng phải có tính chất đối xứng: sim(Di, Dj) == sim(Dj, Di)
    assert np.all(np.abs(sim_matrix - sim_matrix.T) < 1e-9), "Ma trận Cosine Similarity phải đối xứng"
    # 3. sim_matrix[0, 1] tính theo batch 2D phải khớp chính xác với sim_d1_d2 tính theo 1D
    assert abs(sim_matrix[0, 1] - sim_d1_d2) < 1e-9, "Kết quả tương đồng batch 2D phải khớp với vector 1D"
    print(f"[PASS] Test 7: Khép kín toàn bộ Pipeline trên ma trận TF-IDF 2D (batch đối xứng, đường chéo = 1.0).")

    print("\nTẤT CẢ 7 UNIT TEST ĐÃ VƯỢT QUA THÀNH CÔNG! (Sai số < 1e-9)")
    print("=" * 70)


# ==============================================================================
# 8.5 SO SÁNH VỚI THƯ VIỆN SCIKIT-LEARN
# ==============================================================================

def compare_with_sklearn(corpus: Optional[List[str]] = None) -> None:
    """
    Thực hiện so sánh chi tiết giữa hàm tự cài đặt và thư viện chuẩn scikit-learn
    (CountVectorizer, TfidfVectorizer, cosine_similarity).

    Phân tích và giải thích các điểm khác biệt về:
    1. Thứ tự và chỉ mục của từ điển (vocabulary)
    2. Biểu diễn ma trận đếm
    3. Quy ước làm mịn IDF (smoothing):
        - Giáo trình: ln(N / df)
        - Sklearn smooth_idf=True: ln((1 + N) / (1 + df)) + 1
        - Sklearn smooth_idf=False: ln(N / df) + 1
    4. Trọng số TF và chuẩn hóa L2:
        - Giáo trình: tf(t, d) = c(t, d) / doc_len, TF-IDF không chuẩn hóa
        - Sklearn: dùng trực tiếp số đếm thô, sau đó chuẩn hóa L2 theo từng hàng vector
    5. Tính tương đương của độ tương đồng Cosine
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as skl_cos_sim

    if corpus is None:
        corpus = [
            "cat eats fish",
            "dog eats fish",
            "cat likes fish"
        ]

    print("\n" + "=" * 70)
    print("PHẦN E (8.5) — SO SÁNH HỆ THỐNG VỚI SCIKIT-LEARN")
    print("=" * 70)

    # 1. So sánh từ điển & ma trận đếm
    vocab_student = build_vocabulary(corpus)
    counts_student = compute_counts(corpus, vocab_student)

    cv = CountVectorizer()
    counts_sklearn = cv.fit_transform(corpus).toarray()
    vocab_sklearn = cv.get_feature_names_out().tolist()

    print("\n[1] So sánh Vocabulary & Count Matrix:")
    print(f"  - Từ điển tự cài đặt : {vocab_student}")
    print(f"  - Từ điển Sklearn     : {vocab_sklearn}")
    vocab_match = (vocab_student == vocab_sklearn)
    counts_match = np.array_equal(counts_student, counts_sklearn)
    print(f"  -> Trùng khớp từ điển: {vocab_match}")
    print(f"  -> Trùng khớp ma trận đếm: {counts_match}")
    assert vocab_match and counts_match, "Ma trận đếm CountVectorizer không khớp!"

    # 2. So sánh các công thức tính IDF
    print("\n[2] So sánh Tần số nghịch đảo văn bản (IDF):")
    idf_textbook = compute_idf(counts_student, mode="standard")
    idf_smooth = compute_idf(counts_student, mode="smooth")
    idf_sklearn_unsmoothed = compute_idf(counts_student, mode="sklearn_unsmoothed")

    # Mặc định của Sklearn (smooth_idf=True)
    tfidf_sk_default = TfidfVectorizer(norm=None)
    tfidf_sk_default.fit(corpus)
    sk_idf_default = tfidf_sk_default.idf_

    # Sklearn khi tắt làm mịn (smooth_idf=False)
    tfidf_sk_unsmoothed = TfidfVectorizer(smooth_idf=False, norm=None)
    tfidf_sk_unsmoothed.fit(corpus)
    sk_idf_unsmoothed = tfidf_sk_unsmoothed.idf_

    print(f"  - {'Từ':<8} | {'Giáo trình ln(N/df)':<20} | {'Sklearn (smooth=True)':<22} | {'Tự cài đặt smooth':<18}")
    print("  " + "-" * 72)
    for idx, term in enumerate(vocab_student):
        print(f"  - {term:<8} | {idf_textbook[idx]:<20.6f} | {sk_idf_default[idx]:<22.6f} | {idf_smooth[idx]:<18.6f}")

    assert np.allclose(idf_smooth, sk_idf_default, atol=1e-9), "Smooth IDF không khớp với sklearn!"
    assert np.allclose(idf_sklearn_unsmoothed, sk_idf_unsmoothed, atol=1e-9), "Unsmoothed IDF không khớp với sklearn!"
    print("  -> Chế độ 'smooth' tự cài đặt hoàn toàn trùng khớp với Sklearn smooth_idf=True.")
    print("  -> Chế độ 'sklearn_unsmoothed' tự cài đặt hoàn toàn trùng khớp với Sklearn smooth_idf=False.")

    # 3. Giải thích sự khác biệt giữa công thức Giáo trình và Mặc định của Sklearn
    print("\n[3] Phân tích nguyên nhân khác biệt về mặt cấu trúc:")
    print("  * Nguyên nhân A (Làm mịn IDF & Hằng số offset +1.0 bên ngoài phép log):")
    print("    - Công thức giáo trình: idf = ln(N / df). Khi một từ xuất hiện trong mọi văn bản (df = N),")
    print("      tỷ số N / df = 1 dẫn tới ln(1) = 0.0. Hậu quả: toàn bộ trọng số TF-IDF của từ này bị triệt tiêu")
    print("      hoàn toàn về 0, làm mất đi đặc trưng chung hữu ích giữa các tài liệu liên quan trong bài toán truy hồi.")
    print("    - Công thức Sklearn (smooth_idf=True): idf = ln((1 + N) / (1 + df)) + 1.0.")
    print("      + Hạng tử làm mịn Laplace (1 + N) / (1 + df) đóng vai trò giả định có thêm 1 văn bản chứa mọi từ,")
    print("        vừa tránh chia cho 0 khi df = 0, vừa kẹp tỷ số trong khoảng [1, 1+N].")
    print("      + Hằng số offset +1.0 bên ngoài log thiết lập cận dưới nghiêm ngặt: ngay cả khi từ xuất hiện ở mọi")
    print("        văn bản (df = N), ln((1+N)/(1+N)) = ln(1) = 0, nhưng nhờ +1.0 nên IDF luôn >= 1.0 > 0.")
    print("        Điều này bảo toàn một trọng số nền tảng (baseline weight) cho từ khóa thay vì xóa sạch nó.")
    print("  * Nguyên nhân B (Định nghĩa TF & Chuẩn hóa vector):")
    print("    - Giáo trình: Dùng tần suất tương đối tf(t, d) = c(t, d) / doc_length (tổng mỗi văn bản bằng 1).")
    print("    - Sklearn   : Dùng trực tiếp số đếm thô c(t, d), sau đó chuẩn hóa L2 trên vector TF-IDF cuối cùng.")

    # 4. Kiểm tra sự tương đương số học chính xác khi cấu hình đồng nhất
    print("\n[4] Kiểm chứng tương đương số học chính xác:")
    # Tái hiện thủ công quy trình xử lý của scikit-learn:
    # tf = số đếm thô (raw counts), idf = smooth idf, norm = l2
    tf_raw = compute_tf(counts_student, mode="raw")
    tfidf_student_as_sklearn = compute_tfidf(tf_raw, idf_smooth, norm="l2")

    tfidf_sk_official = TfidfVectorizer(smooth_idf=True, norm="l2")
    tfidf_sk_official_matrix = tfidf_sk_official.fit_transform(corpus).toarray()

    max_diff = np.max(np.abs(tfidf_student_as_sklearn - tfidf_sk_official_matrix))
    print(f"  - Độ lệch tuyệt đối lớn nhất so với TfidfVectorizer chính thức: {max_diff:.2e}")
    assert max_diff < 1e-9, f"Quy trình xử lý không khớp! Độ lệch lớn nhất: {max_diff}"
    print("  -> KHỚP HOÀN TOÀN! Khi cấu hình cùng quy ước, hàm tự cài đặt cho kết quả")
    print("     trùng khớp với scikit-learn mà không có sai số số học (< 1e-9).")

    # 5. So sánh độ tương đồng Cosine
    print("\n[5] So sánh độ tương đồng Cosine:")
    vec_a = tfidf_student_as_sklearn[0]
    vec_b = tfidf_student_as_sklearn[1]
    sim_student = cosine_similarity(vec_a, vec_b)
    sim_sklearn = skl_cos_sim(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0]
    print(f"  - Cosine tự cài đặt cos_sim(D1, D2) : {sim_student:.8f}")
    print(f"  - Cosine Sklearn cos_sim(D1, D2)    : {sim_sklearn:.8f}")
    assert abs(sim_student - sim_sklearn) < 1e-9, "Độ tương đồng Cosine không khớp!"
    print("  -> Độ tương đồng Cosine cho kết quả hoàn toàn đồng nhất.")

    print("\n" + "=" * 70)
    print("HOÀN THÀNH SO SÁNH: MỌI CÔNG THỨC ĐÃ ĐƯỢC PHÂN TÍCH VÀ KIỂM CHỨNG CHẶT CHẼ.")
    print("=" * 70)


# ==============================================================================
# ĐIỂM THỰC THI CHÍNH (ENTRY POINT)
# ==============================================================================

if __name__ == "__main__":
    # 1. Chạy unit tests theo calculations.md
    run_unit_tests()

    # 2. Chạy kiểm tra so sánh song song với scikit-learn
    compare_with_sklearn()
