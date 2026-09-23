"""
MAT3561 - Natural Language Processing and Applications
LAB 01 — From Text Processing to Search
Part E — Core Implementation

Author: Vu Tien Dat (MSSV: 23000111)
Course: MAT3561 - Natural Language Processing & Applications
Instructor: M.Sc. Pham Ngoc Hai (Lab PM) / Dr. Le Hong Phuong (Theory)

This module implements the core components of text representation from scratch:
1. build_vocabulary()
2. compute_counts()
3. compute_tf()
4. compute_idf()
5. compute_tfidf()
6. cosine_similarity()

It includes a comprehensive unit test suite validating against manual calculations
from calculations.md (Exercises 1-5), and a systematic comparison with scikit-learn.
"""

from typing import List, Union, Optional
import numpy as np


# ==============================================================================
# 1. CORE FUNCTIONS IMPLEMENTATION
# ==============================================================================

def build_vocabulary(corpus: List[str]) -> List[str]:
    """
    Construct a deterministic, alphabetically sorted vocabulary from a text corpus.

    Args:
        corpus: List of raw string documents.

    Returns:
        Sorted list of unique terms (tokens) present across all documents.
    """
    unique_terms = set()
    for doc in corpus:
        # Standard lowercased whitespace tokenization
        tokens = doc.lower().split()
        unique_terms.update(tokens)
    return sorted(list(unique_terms))


def compute_counts(corpus: List[str], vocab: List[str]) -> np.ndarray:
    """
    Construct the Bag-of-Words count matrix for a corpus given a vocabulary.

    Args:
        corpus: List of document strings.
        vocab: List of vocabulary terms (determines column ordering).

    Returns:
        2D numpy array of shape (N, V) where element (i, j) is the count of
        term j in document i.
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
    Compute Term Frequency (TF) from raw count matrix.

    Modes supported:
        - "relative" (default, textbook / Jurafsky & Martin):
            tf(t, d) = c(t, d) / sum_{t'} c(t', d)
            Row sum equals 1.0 (for non-empty documents).
        - "raw":
            tf(t, d) = c(t, d) (raw term counts, as used internally by scikit-learn).
        - "log":
            tf(t, d) = 1 + log(c(t, d)) if c(t, d) > 0 else 0 (sublinear TF scaling).

    Args:
        counts: 2D numpy array of term counts, shape (N, V).
        mode: TF formulation ("relative", "raw", or "log").

    Returns:
        2D numpy array of TF values with float64 precision.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if mode == "relative":
        doc_lengths = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero for completely empty documents
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
        raise ValueError(f"Unsupported TF mode: '{mode}'. Choose 'relative', 'raw', or 'log'.")


def compute_idf(
    corpus_or_counts: Union[List[str], np.ndarray],
    vocab: Optional[List[str]] = None,
    mode: str = "standard"
) -> np.ndarray:
    """
    Compute Inverse Document Frequency (IDF) vector across the corpus.

    Formulas supported:
        - "standard" (textbook formula in Part A.4 & Exercise 3):
            idf(t) = ln(N / df(t))
            Note: If df(t) == N, idf(t) = ln(1) = 0.0.
        - "smooth" (scikit-learn default with smooth_idf=True):
            idf(t) = ln((1 + N) / (1 + df(t))) + 1.0
        - "sklearn_unsmoothed" (scikit-learn with smooth_idf=False):
            idf(t) = ln(N / df(t)) + 1.0

    Args:
        corpus_or_counts: Either a list of document strings or a precomputed count matrix.
        vocab: List of vocabulary terms (required if corpus_or_counts is a list of strings).
        mode: IDF convention ("standard", "smooth", or "sklearn_unsmoothed").

    Returns:
        1D numpy array of shape (V,) containing IDF weights for each term.
    """
    if isinstance(corpus_or_counts, list):
        if vocab is None:
            raise ValueError("vocab must be provided when corpus_or_counts is a list of strings.")
        counts = compute_counts(corpus_or_counts, vocab)
    else:
        counts = np.asarray(corpus_or_counts)

    num_docs = counts.shape[0]
    # Document frequency: number of documents containing term t (count > 0)
    df = np.sum(counts > 0, axis=0).astype(np.float64)

    if mode == "standard":
        # Handle zero df defensively: if df == 0, set IDF to 0.0
        safe_df = np.where(df == 0, 1.0, df)
        idf = np.log(num_docs / safe_df)
        idf[df == 0] = 0.0
        return idf

    elif mode == "smooth":
        # Scikit-learn default: ln((1 + N) / (1 + df)) + 1.0
        return np.log((1.0 + num_docs) / (1.0 + df)) + 1.0

    elif mode == "sklearn_unsmoothed":
        # Scikit-learn with smooth_idf=False: ln(N / df) + 1.0
        safe_df = np.where(df == 0, 1.0, df)
        return np.log(num_docs / safe_df) + 1.0

    else:
        raise ValueError(f"Unsupported IDF mode: '{mode}'. Choose 'standard', 'smooth', or 'sklearn_unsmoothed'.")


def compute_tfidf(
    tf: np.ndarray,
    idf: np.ndarray,
    norm: Optional[str] = None
) -> np.ndarray:
    """
    Compute the TF-IDF representation matrix.

    tfidf(t, d) = tf(t, d) * idf(t)

    Optional Normalization:
        - None: Raw unnormalized tf * idf (as derived in Exercise 4).
        - "l2": Euclidean unit norm per document vector (v / ||v||_2), matching scikit-learn.

    Args:
        tf: 2D numpy array of TF values, shape (N, V).
        idf: 1D numpy array of IDF values, shape (V,).
        norm: Normalization scheme (None or "l2").

    Returns:
        2D numpy array of TF-IDF vectors, shape (N, V).
    """
    tf = np.asarray(tf, dtype=np.float64)
    idf = np.asarray(idf, dtype=np.float64)

    # Element-wise multiplication with broadcasting across rows
    tfidf = tf * idf

    if norm is None:
        return tfidf
    elif norm == "l2":
        norms = np.linalg.norm(tfidf, ord=2, axis=1, keepdims=True)
        # Avoid division by zero for all-zero vectors
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        return tfidf / safe_norms
    else:
        raise ValueError(f"Unsupported norm: '{norm}'. Choose None or 'l2'.")


def cosine_similarity(
    vec1: np.ndarray,
    vec2: np.ndarray
) -> Union[float, np.ndarray]:
    """
    Compute Cosine Similarity between two vectors (or 2D matrices).

    cos(x, y) = (x . y) / (||x||_2 * ||y||_2)

    Handles single 1D vector pairs or pairwise 2D matrices robustly.

    Args:
        vec1: 1D or 2D array of shape (V,) or (N1, V).
        vec2: 1D or 2D array of shape (V,) or (N2, V).

    Returns:
        Float value if both inputs are 1D vectors, or a 2D similarity matrix.
    """
    v1 = np.asarray(vec1, dtype=np.float64)
    v2 = np.asarray(vec2, dtype=np.float64)

    # Case 1: Both inputs are 1D vectors
    if v1.ndim == 1 and v2.ndim == 1:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        dot_prod = np.dot(v1, v2)
        sim = dot_prod / (norm1 * norm2)
        # Numerical clipping to [-1.0, 1.0]
        return float(np.clip(sim, -1.0, 1.0))

    # Case 2: 2D matrices (batch pairwise similarity)
    v1_2d = np.atleast_2d(v1)
    v2_2d = np.atleast_2d(v2)

    norm1 = np.linalg.norm(v1_2d, ord=2, axis=1, keepdims=True)
    norm2 = np.linalg.norm(v2_2d, ord=2, axis=1, keepdims=True)

    safe_norm1 = np.where(norm1 == 0.0, 1.0, norm1)
    safe_norm2 = np.where(norm2 == 0.0, 1.0, norm2)

    v1_normalized = v1_2d / safe_norm1
    v2_normalized = v2_2d / safe_norm2

    sim_matrix = np.dot(v1_normalized, v2_normalized.T)
    # Zero out rows/cols where original norm was 0
    sim_matrix[norm1.ravel() == 0.0, :] = 0.0
    sim_matrix[:, norm2.ravel() == 0.0] = 0.0

    return np.clip(sim_matrix, -1.0, 1.0)


# ==============================================================================
# 2. UNIT TESTS (VALIDATION AGAINST calculations.md)
# ==============================================================================

def run_unit_tests() -> None:
    """
    Execute rigorous unit tests on the toy corpus from Part B:
        D1 = "cat eats fish"
        D2 = "dog eats fish"
        D3 = "cat likes fish"

    Verifies each function against exact analytical derivations in calculations.md
    with numerical tolerance threshold < 1e-9.
    """
    print("=" * 70)
    print("RUNNING UNIT TESTS (Validated against calculations.md)")
    print("=" * 70)

    corpus = [
        "cat eats fish",
        "dog eats fish",
        "cat likes fish"
    ]

    # --- Test 1: build_vocabulary ---
    vocab = build_vocabulary(corpus)
    expected_vocab = ["cat", "dog", "eats", "fish", "likes"]
    assert vocab == expected_vocab, f"Vocabulary mismatch: got {vocab}, expected {expected_vocab}"
    print("[PASS] Test 1: build_vocabulary() ->", vocab)

    # --- Test 2: compute_counts (Exercise 1) ---
    counts = compute_counts(corpus, vocab)
    expected_counts = np.array([
        [1, 0, 1, 1, 0],  # D1: cat=1, dog=0, eats=1, fish=1, likes=0
        [0, 1, 1, 1, 0],  # D2: cat=0, dog=1, eats=1, fish=1, likes=0
        [1, 0, 0, 1, 1]   # D3: cat=1, dog=0, eats=0, fish=1, likes=1
    ], dtype=np.int64)
    assert np.array_equal(counts, expected_counts), f"Count matrix mismatch:\n{counts}\nExpected:\n{expected_counts}"
    print("[PASS] Test 2: compute_counts() matches Exercise 1 exactly.")

    # --- Test 3: compute_tf (Exercise 2) ---
    tf = compute_tf(counts, mode="relative")
    # For D1: tf(cat)=1/3, tf(eats)=1/3, tf(fish)=1/3
    expected_tf_d1 = np.array([1/3, 0.0, 1/3, 1/3, 0.0])
    assert np.all(np.abs(tf[0] - expected_tf_d1) < 1e-9), f"TF(D1) mismatch: {tf[0]}"
    # Verify sum_t tf(t, d) == 1.0 for all documents
    row_sums = tf.sum(axis=1)
    assert np.all(np.abs(row_sums - 1.0) < 1e-9), f"TF row sum is not 1.0: {row_sums}"
    print("[PASS] Test 3: compute_tf() matches Exercise 2 and sum_t tf(t, d) == 1.0.")

    # --- Test 4: compute_idf (Exercise 3) ---
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
    assert np.all(np.abs(idf - expected_idf) < 1e-9), f"IDF mismatch:\n{idf}\nExpected:\n{expected_idf}"
    # Specifically assert fish has idf == 0.0 as analyzed in Exercise 3
    fish_idx = vocab.index("fish")
    assert abs(idf[fish_idx] - 0.0) < 1e-9, f"IDF for 'fish' should be 0.0, got {idf[fish_idx]}"
    print("[PASS] Test 4: compute_idf() matches Exercise 3 (fish IDF is exactly 0.0).")

    # --- Test 5: compute_tfidf (Exercise 4) ---
    tfidf = compute_tfidf(tf, idf, norm=None)
    # For D1: [ (1/3)*ln(1.5), 0, (1/3)*ln(1.5), 0, 0 ]
    expected_tfidf_d1 = np.array([
        (1/3) * np.log(1.5),
        0.0,
        (1/3) * np.log(1.5),
        0.0,
        0.0
    ])
    assert np.all(np.abs(tfidf[0] - expected_tfidf_d1) < 1e-9), f"TF-IDF(D1) mismatch: {tfidf[0]}"
    # Check that fish weight is 0 in all documents
    assert np.all(tfidf[:, fish_idx] == 0.0), "Term 'fish' must have TF-IDF = 0 across all docs."
    print("[PASS] Test 5: compute_tfidf() matches Exercise 4 (unnormalized tf * idf).")

    # --- Test 6: cosine_similarity (Exercise 5) ---
    x = np.array([1, 1, 1])
    y = np.array([1, 1, 0])
    sim = cosine_similarity(x, y)
    expected_sim = 2.0 / np.sqrt(6.0)  # approx 0.816496580927726
    assert abs(sim - expected_sim) < 1e-9, f"Cosine similarity mismatch: got {sim}, expected {expected_sim}"
    print(f"[PASS] Test 6: cosine_similarity() matches Exercise 5 (cos(x, y) = 2/sqrt(6) ≈ {sim:.6f}).")

    print("\nALL 6 UNIT TESTS PASSED SUCCESSFULLY! (Tolerance < 1e-9)")
    print("=" * 70)


# ==============================================================================
# 3. COMPARISON WITH SCIKIT-LEARN (Part E Section 8.5)
# ==============================================================================

def compare_with_sklearn(corpus: Optional[List[str]] = None) -> None:
    """
    Conduct an in-depth comparison between Student Implementation and Reference Implementation
    from scikit-learn (CountVectorizer, TfidfVectorizer, cosine_similarity).

    Analyzes and explains differences in:
    1. Vocabulary ordering & indexing
    2. Count representations
    3. IDF smoothing conventions:
        - Textbook: ln(N / df)
        - Sklearn smooth_idf=True: ln((1 + N) / (1 + df)) + 1
        - Sklearn smooth_idf=False: ln(N / df) + 1
    4. TF weighting and L2 normalization:
        - Textbook: tf(t, d) = c(t, d) / doc_len, unnormalized TF-IDF
        - Sklearn: raw counts, followed by row-wise L2 normalization
    5. Cosine similarity equivalence
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
    print("PART E (8.5) — SYSTEMATIC COMPARISON WITH SCIKIT-LEARN")
    print("=" * 70)

    # 1. Compare Vocabulary & Count Matrix
    vocab_student = build_vocabulary(corpus)
    counts_student = compute_counts(corpus, vocab_student)

    cv = CountVectorizer()
    counts_sklearn = cv.fit_transform(corpus).toarray()
    vocab_sklearn = cv.get_feature_names_out().tolist()

    print("\n[1] Vocabulary & CountMatrix Comparison:")
    print(f"  - Student Vocab : {vocab_student}")
    print(f"  - Sklearn Vocab : {vocab_sklearn}")
    vocab_match = (vocab_student == vocab_sklearn)
    counts_match = np.array_equal(counts_student, counts_sklearn)
    print(f"  -> Vocab match  : {vocab_match}")
    print(f"  -> Counts match : {counts_match}")
    assert vocab_match and counts_match, "CountVectorizer mismatch!"

    # 2. Compare IDF Formulations
    print("\n[2] Inverse Document Frequency (IDF) Comparison:")
    idf_textbook = compute_idf(counts_student, mode="standard")
    idf_smooth = compute_idf(counts_student, mode="smooth")
    idf_sklearn_unsmoothed = compute_idf(counts_student, mode="sklearn_unsmoothed")

    # Sklearn default (smooth_idf=True)
    tfidf_sk_default = TfidfVectorizer(norm=None)
    tfidf_sk_default.fit(corpus)
    sk_idf_default = tfidf_sk_default.idf_

    # Sklearn without smoothing (smooth_idf=False)
    tfidf_sk_unsmoothed = TfidfVectorizer(smooth_idf=False, norm=None)
    tfidf_sk_unsmoothed.fit(corpus)
    sk_idf_unsmoothed = tfidf_sk_unsmoothed.idf_

    print(f"  - {'Term':<8} | {'Textbook ln(N/df)':<18} | {'Sklearn (smooth=True)':<22} | {'Student smooth':<15}")
    print("  " + "-" * 68)
    for idx, term in enumerate(vocab_student):
        print(f"  - {term:<8} | {idf_textbook[idx]:<18.6f} | {sk_idf_default[idx]:<22.6f} | {idf_smooth[idx]:<15.6f}")

    assert np.allclose(idf_smooth, sk_idf_default, atol=1e-9), "Smooth IDF mismatch with sklearn!"
    assert np.allclose(idf_sklearn_unsmoothed, sk_idf_unsmoothed, atol=1e-9), "Unsmoothed IDF mismatch with sklearn!"
    print("  -> Student 'smooth' mode matches Sklearn smooth_idf=True identically.")
    print("  -> Student 'sklearn_unsmoothed' mode matches Sklearn smooth_idf=False identically.")

    # 3. Explain the Discrepancy between Textbook and Sklearn Default
    print("\n[3] Architectural Explanation of Differences:")
    print("  * Reason A (IDF Smoothing & Offset +1):")
    print("    - Textbook formula: idf = ln(N / df). For terms appearing in all documents (df = N), idf = 0.0.")
    print("    - Sklearn formula : idf = ln((1 + N) / (1 + df)) + 1.0. An offset of +1 is added to guarantee")
    print("      that terms appearing in all documents do not get zeroed out completely.")
    print("  * Reason B (TF Definition & Normalization):")
    print("    - Textbook: Uses relative term frequency tf(t, d) = c(t, d) / doc_length (sum to 1 per doc).")
    print("    - Sklearn : Uses raw counts c(t, d) directly, followed by L2-normalization on the final vector.")

    # 4. Exact Numerical Alignment when matching configurations
    print("\n[4] Exact Numerical Equivalence Verification:")
    # Replicate scikit-learn's exact pipeline manually:
    # tf = raw counts, idf = smooth idf, norm = l2
    tf_raw = compute_tf(counts_student, mode="raw")
    tfidf_student_as_sklearn = compute_tfidf(tf_raw, idf_smooth, norm="l2")

    tfidf_sk_official = TfidfVectorizer(smooth_idf=True, norm="l2")
    tfidf_sk_official_matrix = tfidf_sk_official.fit_transform(corpus).toarray()

    max_diff = np.max(np.abs(tfidf_student_as_sklearn - tfidf_sk_official_matrix))
    print(f"  - Maximum absolute discrepancy with official TfidfVectorizer: {max_diff:.2e}")
    assert max_diff < 1e-9, f"Pipelines do not match! Max diff: {max_diff}"
    print("  -> PERFECT MATCH! When configured with matching conventions, student implementation")
    print("     reproduces scikit-learn output with 0 numerical deviation (< 1e-9).")

    # 5. Cosine Similarity Comparison
    print("\n[5] Cosine Similarity Equivalence:")
    vec_a = tfidf_student_as_sklearn[0]
    vec_b = tfidf_student_as_sklearn[1]
    sim_student = cosine_similarity(vec_a, vec_b)
    sim_sklearn = skl_cos_sim(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0, 0]
    print(f"  - Student cos_sim(D1, D2) : {sim_student:.8f}")
    print(f"  - Sklearn cos_sim(D1, D2) : {sim_sklearn:.8f}")
    assert abs(sim_student - sim_sklearn) < 1e-9, "Cosine similarity mismatch!"
    print("  -> Cosine similarity produces identical results.")

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE: ALL FORMULATIONS RIGOROUSLY UNDERSTOOD & VERIFIED.")
    print("=" * 70)


# ==============================================================================
# 4. ENTRY POINT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Run unit tests on calculations.md
    run_unit_tests()

    # 2. Run side-by-side comparison with scikit-learn
    compare_with_sklearn()
