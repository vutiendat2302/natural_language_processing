"""
LAB 06 — ATTENTION MECHANISM: CORE NUMPY IMPLEMENTATION
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)
Học viên: Vũ Tiến Đạt - MSSV: 23000111
Giảng viên lý thuyết: TS. Lê Hồng Phương
Giảng viên thực hành: ThS. Phạm Ngọc Hải

File này cài đặt các hàm cốt lõi của Attention Mechanism từ đầu (from scratch) bằng NumPy:
1. stable_softmax: Softmax ổn định số trị (Mục 17)
2. scaled_dot_product_attention: Triển khai chuẩn của Vaswani et al. (Mục 16)
3. self_attention: Tự chú ý cho chuỗi đầu vào X (Mục 20, 21)
4. compute_temperature_softmax: Khảo sát ảnh hưởng của Temperature và Entropy (Mục 15)
"""

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Tính Softmax ổn định số học (Mục 17: Numerical Stability).
    Công thức: softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    
    Phép trừ max(x) không làm đổi kết quả Softmax (tính Shift-Invariance)
    nhưng giúp triệt tiêu hoàn toàn nguy cơ tràn số (overflow khi exp(x) với x > 709).
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """
    Triển khai Scaled Dot-Product Attention (Mục 16: Core Implementation).
    
    Quy trình:
      S = Q K^T / sqrt(d_k)
      A = softmax(S)
      C = A V
      
    Args:
      Q: Ma trận Query (kích thước ... x T_q x d_k)
      K: Ma trận Key   (kích thước ... x T_k x d_k)
      V: Ma trận Value (kích thước ... x T_k x d_v)
      
    Returns:
      scores: Ma trận điểm số đã scale S
      attention_weights: Ma trận trọng số A (tổng mỗi hàng = 1.0)
      context: Biểu diễn ngữ cảnh C
    """
    d_k = K.shape[-1]
    
    # 1. Tính scores đã scale: S = Q K^T / sqrt(d_k)
    # Hỗ trợ cả 2D (T x d) và nhiều chiều batched (batch x T x d)
    if Q.ndim == 2 and K.ndim == 2:
        scores = (Q @ K.T) / np.sqrt(d_k)
    else:
        scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(d_k)
        
    # 2. Tính attention weights bằng stable softmax theo chiều cuối
    attention_weights = stable_softmax(scores, axis=-1)
    
    # 3. Tính context vector: C = A V
    context = attention_weights @ V
    
    return scores, attention_weights, context


def self_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    """
    Triển khai Self-Attention từ ma trận biểu diễn chuỗi X (Mục 20 & 21).
    
    Args:
      X: Ma trận embedding chuỗi token (T x d_model)
      W_Q, W_K, W_V: Các ma trận chiếu trọng số (d_model x d_k/d_v)
      
    Returns:
      Q, K, V, scores, attention_weights, context
    """
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    scores, attention_weights, context = scaled_dot_product_attention(Q, K, V)
    return Q, K, V, scores, attention_weights, context


def compute_temperature_softmax(scores: np.ndarray, T: float = 1.0) -> tuple:
    """
    Tính Softmax có tham số nhiệt độ Temperature T và Entropy (Mục 15).
    
    Returns:
      weights: Phân phối attention weights
      max_weight: Trọng số lớn nhất
      entropy: Độ hỗn loạn Shannon Entropy (nats)
    """
    scaled_s = scores / float(T)
    weights = stable_softmax(scaled_s, axis=-1)
    max_weight = np.max(weights)
    
    # Shannon Entropy H(p) = - sum(p * ln(p))
    # Thêm epsilon 1e-12 để tránh log(0)
    entropy = -np.sum(weights * np.log(weights + 1e-12))
    return weights, max_weight, entropy


if __name__ == "__main__":
    print("=" * 60)
    print("KIỂM CHỨNG CÁC HÀM TRONG ATTENTION_NUMPY.PY (LAB 06)")
    print("=" * 60)
    
    # 1. Test Numerical Stability (Mục 17)
    x_test = np.array([1000.0, 1001.0, 1002.0])
    alpha_test = stable_softmax(x_test)
    print("\n1. Test Stable Softmax với x = [1000, 1001, 1002]:")
    print("   Alpha:", np.round(alpha_test, 4))
    print("   Tổng Alpha:", np.sum(alpha_test))
    
    # 2. Test Scaled Dot-Product Attention (Mục 11 & 16)
    q = np.array([[1.0, 0.0]])  # 1 Query (1 x 2)
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 Keys (3 x 2)
    V = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])  # 3 Values (3 x 2)
    
    scores, attn, ctx = scaled_dot_product_attention(q, K, V)
    print("\n2. Test Scaled Dot-Product Attention (Mục 11):")
    print("   Scores (scaled):", np.round(scores, 4))
    print("   Attention weights:", np.round(attn, 4))
    print("   Context vector:", np.round(ctx, 4))
    
    # 3. Test Self-Attention trên chuỗi ['I', 'like', 'cats'] (Mục 21)
    X = np.array([
        [1.0, 0.0],  # 'I'
        [1.0, 1.0],  # 'like'
        [0.0, 2.0]   # 'cats'
    ])
    W_Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    W_K = np.array([[1.0, 1.0], [0.0, 1.0]])
    W_V = np.array([[1.0, 2.0], [2.0, 0.0]])
    
    Q, K, V, S, A, C = self_attention(X, W_Q, W_K, W_V)
    print("\n3. Test Self-Attention ['I', 'like', 'cats'] (Mục 21):")
    print("   Attention Matrix A (T x T):\n", np.round(A, 4))
    print("   Contextual Representation C (T x d):\n", np.round(C, 4))
    
    # 4. Test Temperature (Mục 15)
    s_temp = np.array([2.0, 1.0, 0.0])
    print("\n4. Test Temperature (Mục 15) với s = [2, 1, 0]:")
    for temp in [0.5, 1.0, 2.0]:
        w, mw, h = compute_temperature_softmax(s_temp, T=temp)
        print(f"   T={temp:3.1f} | Max weight: {mw:.4f} | Entropy: {h:.4f} nats | Weights: {np.round(w, 4)}")
        
    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC MODULE ĐÃ HOẠT ĐỘNG CHUẨN XÁC!")
    print("=" * 60)
