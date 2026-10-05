"""Co-occurrence Word Representations from Scratch (LAB 03 - Sections 10 & 11).

- build_vocabulary(): Xây dựng từ điển với ngưỡng lọc tần số và stop words
- build_cooccurrence_matrix(): Xây dựng ma trận đếm đồng xuất hiện theo cửa sổ k
- cosine_similarity(): Tính toán độ tương đồng cosine chuẩn hóa giữa hai vector
- most_similar(): Tìm top-K từ có vector tương đồng nhất
- run_experiment_1(): Thực nghiệm khảo sát ảnh hưởng của window size (1, 2, 5)

"""

from collections import Counter, defaultdict
import math
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Cố định random seed theo quy định tại AGENTS.md
random.seed(42)
np.random.seed(42)
try:
    import torch
    torch.manual_seed(42)
except ImportError:
    pass


# Biểu thức chính quy tiền xử lý token
TOKEN_REGEX = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")


def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản và chuyển về chữ thường.

    Args:
        text: Chuỗi văn bản đầu vào.

    Returns:
        Danh sách các token từ dạng chữ thường.
    """
    return TOKEN_REGEX.findall(text.lower())


def build_vocabulary(
    corpus: Sequence[Union[str, Sequence[str]]],
    min_count: int = 1,
    stop_words: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Xây dựng bảng từ vựng (Vocabulary mapping) từ ngữ liệu.

    Args:
        corpus: Danh sách các văn bản (str) hoặc danh sách các token (list[str]).
        min_count: Tần số xuất hiện tối thiểu để giữ lại từ trong từ vựng.
        stop_words: Danh sách các từ dừng cần loại bỏ (nếu có).

    Returns:
        Từ điển word_to_id ánh xạ từ vựng sang chỉ số nguyên liên tục [0, |V| - 1].
    """
    stop_set = set(stop_words) if stop_words else set()
    counter: Counter = Counter()

    for item in corpus:
        tokens = tokenize(item) if isinstance(item, str) else [t.lower() for t in item]
        for token in tokens:
            if token not in stop_set:
                counter[token] += 1

    # Lọc theo min_count và sắp xếp bảng chữ cái để đảm bảo tính tái lập (reproducibility)
    vocab: Dict[str, int] = {}
    idx = 0
    for word in sorted(counter.keys()):
        if counter[word] >= min_count:
            vocab[word] = idx
            idx += 1

    return vocab


def build_cooccurrence_matrix(
    corpus: Sequence[Union[str, Sequence[str]]],
    vocabulary: Dict[str, int],
    window_size: int = 1,
    symmetric: bool = True,
) -> np.ndarray:
    """Xây dựng ma trận đồng xuất hiện Word-Context Matrix X.

    Args:
        corpus: Tập ngữ liệu gồm các câu/văn bản.
        vocabulary: Bảng từ điển word_to_id.
        window_size: Bán kính cửa sổ ngữ cảnh k (xét k từ trước và k từ sau).
        symmetric: True nếu cửa sổ đối xứng hai phía; False nếu chỉ xét từ phía sau.

    Returns:
        Ma trận số nguyên numpy.ndarray kích thước (|V|, |V|).
    """
    vocab_size = len(vocabulary)
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float64)

    for item in corpus:
        tokens = tokenize(item) if isinstance(item, str) else [t.lower() for t in item]
        # Lọc các token nằm trong vocabulary và lưu vị trí
        indexed_tokens: List[Tuple[int, int]] = [
            (pos, vocabulary[tok]) for pos, tok in enumerate(tokens) if tok in vocabulary
        ]

        # Duyệt qua các từ mục tiêu
        for i, (pos_target, target_id) in enumerate(indexed_tokens):
            # Xét các từ ngữ cảnh trong phạm vi cửa sổ
            for j in range(len(indexed_tokens)):
                if i == j:
                    continue
                pos_context, context_id = indexed_tokens[j]
                distance = pos_context - pos_target

                if symmetric:
                    if 1 <= abs(distance) <= window_size:
                        matrix[target_id, context_id] += 1.0
                else:
                    if 1 <= distance <= window_size:
                        matrix[target_id, context_id] += 1.0

    return matrix


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính độ tương đồng Cosine Similarity giữa hai vector đặc trưng.

    cos(a, b) = (a . b) / (||a||_2 * ||b||_2)

    Args:
        vec_a: Vector số thực chiều d.
        vec_b: Vector số thực chiều d.

    Returns:
        Giá trị tương đồng thuộc đoạn [-1.0, 1.0], hoặc 0.0 nếu có vector zero.
    """
    dot_product = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    sim = dot_product / (norm_a * norm_b)
    # Cắt gọn trong phạm vi [-1.0, 1.0] để tránh sai số dấu phẩy động
    return float(np.clip(sim, -1.0, 1.0))


def most_similar(
    word: str,
    matrix: np.ndarray,
    vocabulary: Dict[str, int],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Tìm Top-K từ có vector đồng xuất hiện tương đồng nhất với từ mục tiêu.

    Args:
        word: Từ mục tiêu cần tìm kiếm.
        matrix: Ma trận đồng xuất hiện X kích thước (|V|, |V|).
        vocabulary: Bảng ánh xạ word_to_id.
        top_k: Số lượng từ tương đồng cần trả về.

    Returns:
        Danh sách các cặp (từ_tương_đồng, điểm_cosine) giảm dần theo điểm số.
    """
    word_clean = word.lower()
    if word_clean not in vocabulary:
        raise KeyError(f"Từ '{word}' không tồn tại trong từ vựng (OOV).")

    target_idx = vocabulary[word_clean]
    target_vec = matrix[target_idx]

    id_to_word = {idx: w for w, idx in vocabulary.items()}
    results: List[Tuple[str, float]] = []

    for other_word, other_idx in vocabulary.items():
        if other_idx == target_idx:
            continue
        sim = cosine_similarity(target_vec, matrix[other_idx])
        results.append((other_word, sim))

    # Sắp xếp theo cosine giảm dần, nếu bằng nhau thì theo thứ tự từ điển
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:top_k]


def run_experiment_1(corpus: Optional[List[str]] = None) -> Dict[str, dict]:
    """Thực nghiệm 1: Word-context representation với các kích thước cửa sổ khác nhau.

    Khảo sát window = 1, window = 2, window = 5:
    - Kích thước từ vựng (Vocabulary size)
    - Kích thước ma trận (Matrix size)
    - Số lượng phần tử khác 0 (Number of non-zero entries)
    - Tỷ lệ thưa (Sparsity %)
    - Độ tương đồng Cosine giữa các cặp từ mẫu

    Args:
        corpus: Ngữ liệu thử nghiệm (nếu None sẽ dùng toy corpus mẫu mở rộng).

    Returns:
        Từ điển chứa số liệu phân tích của các cửa sổ.
    """
    if corpus is None:
        corpus = [
            "the doctor treated the patient in the hospital",
            "the physician treated the sick patient with medicine",
            "the doctor and the nurse work in the hospital clinic",
            "the patient visited the doctor at the clinic today",
            "the physician prescribed effective therapy for the disease",
            "a computer requires software and memory to process data",
            "he bought a new fast computer and installed software",
            "the children played football in the large football stadium",
            "he drove his red car to the hospital parking lot",
            "she likes to eat fresh yellow banana and sweet apple",
            "the monkey ate a sweet ripe banana in the tree",
        ]

    stop_words = ["the", "a", "an", "and", "in", "with", "at", "for", "to", "his", "she", "he"]
    vocab = build_vocabulary(corpus, min_count=1, stop_words=stop_words)
    vocab_size = len(vocab)

    windows = [1, 2, 5]
    results = {}

    target_pairs = [
        ("doctor", "physician"),
        ("doctor", "hospital"),
        ("doctor", "patient"),
        ("doctor", "computer"),
        ("doctor", "banana"),
    ]

    for w in windows:
        mat = build_cooccurrence_matrix(corpus, vocab, window_size=w)
        non_zero = int(np.count_nonzero(mat))
        total_entries = vocab_size * vocab_size
        sparsity = (1.0 - (non_zero / total_entries)) * 100.0

        pair_sims = {}
        for w1, w2 in target_pairs:
            if w1 in vocab and w2 in vocab:
                sim = cosine_similarity(mat[vocab[w1]], mat[vocab[w2]])
                pair_sims[f"{w1}-{w2}"] = round(sim, 4)
            else:
                pair_sims[f"{w1}-{w2}"] = None

        results[f"window_{w}"] = {
            "vocab_size": vocab_size,
            "matrix_shape": mat.shape,
            "non_zero_count": non_zero,
            "sparsity_pct": round(sparsity, 2),
            "pair_similarities": pair_sims,
        }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM TRA CÀI ĐẶT CO-OCCURRENCE MATRIX (LAB 03)")
    print("=" * 60)

    # 1. Kiểm tra với toy corpus trong Bài 6
    toy_corpus = [
        "the cat eats fish",
        "the dog eats fish",
        "the cat likes milk",
        "the dog likes meat",
    ]
    toy_vocab = build_vocabulary(toy_corpus, min_count=1, stop_words=["the"])
    toy_matrix = build_cooccurrence_matrix(toy_corpus, toy_vocab, window_size=1)

    print("\n1. Toy Vocabulary (|V| = {}):".format(len(toy_vocab)))
    print(toy_vocab)

    print("\n2. Co-occurrence Matrix (window=1):")
    print("Words:", list(toy_vocab.keys()))
    print(toy_matrix.astype(int))

    cos_cat_dog = cosine_similarity(toy_matrix[toy_vocab["cat"]], toy_matrix[toy_vocab["dog"]])
    print(f"\n3. Cosine(cat, dog) = {cos_cat_dog:.4f}")
    assert math.isclose(cos_cat_dog, 1.0, rel_tol=1e-5), "Cosine(cat, dog) phải bằng 1.0!"

    # 2. Chạy Thực nghiệm 1
    print("\n" + "=" * 60)
    print("CHẠY EXPERIMENT 1 — WORD-CONTEXT REPRESENTATION (Mục 10)")
    print("=" * 60)
    exp1_res = run_experiment_1()
    for w_name, data in exp1_res.items():
        print(f"\n>>> {w_name.upper()}:")
        print(f"  - Vocab size: {data['vocab_size']}")
        print(f"  - Matrix shape: {data['matrix_shape']}")
        print(f"  - Non-zero count: {data['non_zero_count']} (Sparsity: {data['sparsity_pct']}%)")
        print("  - Pair similarities:")
        for pair, sim in data["pair_similarities"].items():
            print(f"      * {pair}: {sim}")
