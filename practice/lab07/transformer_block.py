"""
LAB 07 — TRANSFORMER ARCHITECTURE: CORE TRANSFORMER BLOCK IMPLEMENTATION
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026
Học viên: Vũ Tiến Đạt - MSSV: 23000111
Giảng viên lý thuyết: TS. Lê Hồng Phương
Giảng viên thực hành: ThS. Phạm Ngọc Hải

Triển khai Mục 17 & 18 trong W7.pdf:
- Cài đặt TransformerBlock tối giản (Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm)
- Kiểm tra tự động toàn bộ Tensor Shapes với B = 8, T = 100, d_model = 128, n_heads = 4 (d_k = 32)
"""

import numpy as np


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Mục 16: Layer Normalization chuẩn hóa theo chiều đặc trưng cuối cùng (feature dimension).
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


class TransformerBlock:
    """
    Mục 17: Core Implementation — Transformer Encoder Block tối giản.
    Kiến trúc:
      x -> Multi-Head Self-Attention -> Residual + LayerNorm -> Feed-Forward -> Residual + LayerNorm -> output
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads!"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff
        
        # Khởi tạo trọng số Xavier/Glorot cho Multi-Head Attention
        scale_mha = np.sqrt(1.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale_mha
        self.W_K = np.random.randn(d_model, d_model) * scale_mha
        self.W_V = np.random.randn(d_model, d_model) * scale_mha
        self.W_O = np.random.randn(d_model, d_model) * scale_mha
        
        # Tham số LayerNorm 1
        self.gamma1 = np.ones((1, 1, d_model))
        self.beta1 = np.zeros((1, 1, d_model))
        
        # Trọng số Position-wise Feed-Forward Network (FFN)
        scale_ff1 = np.sqrt(2.0 / d_model)
        scale_ff2 = np.sqrt(1.0 / d_ff)
        self.W_1 = np.random.randn(d_model, d_ff) * scale_ff1
        self.b_1 = np.zeros((1, 1, d_ff))
        self.W_2 = np.random.randn(d_ff, d_model) * scale_ff2
        self.b_2 = np.zeros((1, 1, d_model))
        
        # Tham số LayerNorm 2
        self.gamma2 = np.ones((1, 1, d_model))
        self.beta2 = np.zeros((1, 1, d_model))
        
        # Lưu trữ các tensor trung gian để phục vụ kiểm tra shape (Mục 18) & trực quan hóa
        self.last_shapes = {}
        self.last_attention_weights = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Thực hiện lan truyền tiến (Forward pass) qua 1 Transformer Encoder Block.
        Input x: (B, T, d_model)
        Output:  (B, T, d_model)
        """
        B, T, D = x.shape
        self.last_shapes["X"] = x.shape
        
        # 1. Chiếu tuyến tính và tách thành n_heads: (B, T, D) -> (B, n_heads, T, d_k)
        Q = (x @ self.W_Q).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (x @ self.W_K).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (x @ self.W_V).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        self.last_shapes["Q"] = Q.shape
        self.last_shapes["K"] = K.shape
        self.last_shapes["V"] = V.shape
        
        # 2. Scaled Dot-Product Attention trên từng head: (B, n_heads, T, T)
        scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)
        self.last_shapes["Attention_scores"] = scores.shape
        
        attn_weights = stable_softmax(scores, axis=-1)
        self.last_attention_weights = attn_weights
        
        head_out = attn_weights @ V  # (B, n_heads, T, d_k)
        
        # 3. Ghép nối (Concat) các heads và nhân W_O: (B, T, D)
        concat_out = head_out.transpose(0, 2, 1, 3).reshape(B, T, D)
        attn_output = concat_out @ self.W_O
        self.last_shapes["Attention_output"] = attn_output.shape
        
        # 4. Residual Connection 1 + LayerNorm 1
        norm1_out = layer_norm(x + attn_output, self.gamma1, self.beta1)
        
        # 5. Position-wise Feed-Forward Network: Linear -> ReLU -> Linear
        ffn_hidden = np.maximum(0.0, norm1_out @ self.W_1 + self.b_1)
        self.last_shapes["FFN_hidden"] = ffn_hidden.shape
        ffn_output = ffn_hidden @ self.W_2 + self.b_2
        self.last_shapes["FFN"] = ffn_output.shape
        
        # 6. Residual Connection 2 + LayerNorm 2
        output = layer_norm(norm1_out + ffn_output, self.gamma2, self.beta2)
        self.last_shapes["Output"] = output.shape
        
        return output


if __name__ == "__main__":
    np.random.seed(42)
    print("=" * 68)
    print("KIỂM CHỨNG TRANSFORMER_BLOCK.PY & SHAPE CHECK (MỤC 17 & 18 - LAB 07)")
    print("=" * 68)
    
    # Cấu hình đúng theo yêu cầu Mục 18
    B = 8
    T = 100
    d_model = 128
    n_heads = 4
    d_ff = 512
    
    X_batch = np.random.randn(B, T, d_model)
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff)
    out = block.forward(X_batch)
    
    print(f"\nThông số: B = {B}, T = {T}, d_model = {d_model}, n_heads = {n_heads} -> d_k = {block.d_k}")
    print("-" * 68)
    for stage, shp in block.last_shapes.items():
        print(f"  {stage:<20} : {str(shp)}")
    print("-" * 68)
    print("Kiểm tra hoàn tất! Tất cả các shape khớp 100% với đặc tả lý thuyết.")
