"""
LAB 07 — TRANSFORMER ARCHITECTURE: POSITIONAL ENCODING MODULE
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026
Học viên: Vũ Tiến Đạt - MSSV: 23000111
Giảng viên lý thuyết: TS. Lê Hồng Phương
Giảng viên thực hành: ThS. Phạm Ngọc Hải

Triển khai Mục 9, 10, 11, 12 trong W7.pdf:
- Sinusoidal Positional Encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(...)
- Bài tập tính PE với d = 4, pos = 0, 1, 2 (Mục 10)
- Experiment 1 & Permutation Experiment: "dog bites man" vs "man bites dog" (Mục 11 & 12)
"""

import numpy as np


def get_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    """
    Tính ma trận Sinusoidal Positional Encoding (Mục 9).
    
    Args:
      max_len: Độ dài chuỗi tối đa T
      d_model: Chiều biểu diễn của mô hình d
      
    Returns:
      pe: Ma trận kích thước (max_len, d_model)
    """
    pe = np.zeros((max_len, d_model), dtype=np.float64)
    position = np.arange(0, max_len, dtype=np.float64)[:, np.newaxis]
    div_term = np.power(10000.0, np.arange(0, d_model, 2, dtype=np.float64) / d_model)
    
    pe[:, 0::2] = np.sin(position / div_term)
    pe[:, 1::2] = np.cos(position / div_term)
    return pe


def add_positional_encoding(x: np.ndarray) -> np.ndarray:
    """
    Cộng Positional Encoding vào Input Embeddings: X' = X + PE.
    Hỗ trợ đầu vào 2D (T, d_model) hoặc 3D (B, T, d_model).
    """
    seq_len = x.shape[-2]
    d_model = x.shape[-1]
    pe = get_positional_encoding(seq_len, d_model)
    return x + pe


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention_forward(x: np.ndarray) -> np.ndarray:
    """Self-Attention đơn giản phục vụ thực nghiệm hoán vị (Mục 11 & 12)."""
    d_k = x.shape[-1]
    scores = (x @ np.swapaxes(x, -1, -2)) / np.sqrt(d_k)
    weights = stable_softmax(scores, axis=-1)
    return weights @ x


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 68)
    print("KIỂM CHỨNG POSITIONAL_ENCODING.PY (MỤC 9, 10, 11, 12 - LAB 07)")
    print("=" * 68)
    
    # 1. Mục 10: Bài tập tính PE cho d = 4, pos = 0, 1, 2
    pe_3x4 = get_positional_encoding(max_len=3, d_model=4)
    print("\n1. [Mục 10] Positional Encoding (d = 4, pos = 0, 1, 2):")
    for pos in range(3):
        print(f"   PE(pos={pos}) = {np.round(pe_3x4[pos], 4)}")
        
    # 2. Mục 12: Permutation Experiment ("dog bites man" vs "man bites dog")
    vocab_emb = {
        "dog":   np.array([1.0, 0.0, 0.0, 1.0]),
        "bites": np.array([0.0, 1.0, 1.0, 0.0]),
        "man":   np.array([1.0, 1.0, 0.0, 0.0]),
    }
    S1 = np.array([vocab_emb["dog"], vocab_emb["bites"], vocab_emb["man"]])
    S2 = np.array([vocab_emb["man"], vocab_emb["bites"], vocab_emb["dog"]])
    
    # Experiment A: Không có Positional Encoding
    out_s1_no_pe = self_attention_forward(S1).mean(axis=0)
    out_s2_no_pe = self_attention_forward(S2).mean(axis=0)
    diff_no_pe = np.linalg.norm(out_s1_no_pe - out_s2_no_pe)
    
    # Experiment B: Có Positional Encoding
    out_s1_with_pe = self_attention_forward(add_positional_encoding(S1)).mean(axis=0)
    out_s2_with_pe = self_attention_forward(add_positional_encoding(S2)).mean(axis=0)
    diff_with_pe = np.linalg.norm(out_s1_with_pe - out_s2_with_pe)
    
    print("\n2. [Mục 12] Permutation Experiment ('dog bites man' vs 'man bites dog'):")
    print(f"   - Exp A (Không PE) | Mean S1: {np.round(out_s1_no_pe, 4)}")
    print(f"                      | Mean S2: {np.round(out_s2_no_pe, 4)}")
    print(f"                      | Khoảng cách L2 giữa S1 & S2: {diff_no_pe:.6f} (Trùng hệt nhau!)")
    print(f"   - Exp B (Có PE)    | Mean S1: {np.round(out_s1_with_pe, 4)}")
    print(f"                      | Mean S2: {np.round(out_s2_with_pe, 4)}")
    print(f"                      | Khoảng cách L2 giữa S1 & S2: {diff_with_pe:.6f} (Phân biệt rõ rệt!)")
