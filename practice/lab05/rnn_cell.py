"""Module: rnn_cell.py
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561)
Sinh viên thực hiện: Vũ Tiến Đạt (MSSV: 23000111)

Mục 14: Core implementation - Simple RNN Cell
Triển khai SimpleRNNCell từ đầu và kiểm chứng tính tương đương chính xác
với nn.RNNCell của PyTorch.
"""

import math
import torch
import torch.nn as nn


class SimpleRNNCell(nn.Module):
    """Triển khai thủ công Simple Recurrent Neural Network Cell.
    
    Công thức toán học chuyển trạng thái:
        h_t = tanh(x_t @ W_ih.T + h_{t-1} @ W_hh.T + b_ih + b_hh)
    
    Quy ước kích thước (Batch-first):
        - x_t: Tensor (batch_size, input_size)
        - h_prev: Tensor (batch_size, hidden_size)
        - h_next: Tensor (batch_size, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        # Khởi tạo trọng số W_xh (Input to Hidden) kích thước (hidden_size, input_size)
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        # Khởi tạo trọng số W_hh (Hidden to Hidden) kích thước (hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))

        if bias:
            # PyTorch nn.RNNCell có 2 vector bias riêng: bias_ih và bias_hh
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Khởi tạo trọng số theo chuẩn Uniform(-1/sqrt(hidden_size), 1/sqrt(hidden_size)) của PyTorch."""
        std = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)
        if self.use_bias:
            nn.init.uniform_(self.bias_ih, -std, std)
            nn.init.uniform_(self.bias_hh, -std, std)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor = None) -> torch.Tensor:
        """Thực hiện một bước tính toán thuận (forward pass) qua RNN Cell.
        
        Args:
            x_t: Tensor đầu vào tại timestep t, shape (batch_size, input_size)
            h_prev: Hidden state từ bước t-1, shape (batch_size, hidden_size).
                   Nếu None, mặc định khởi tạo tensor số 0.
                   
        Returns:
            h_next: Hidden state tại timestep t, shape (batch_size, hidden_size)
        """
        batch_size = x_t.size(0)
        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=x_t.device, dtype=x_t.dtype)

        # 1. Tính đóng góp từ input hiện tại: x_t @ W_ih^T
        input_contrib = torch.matmul(x_t, self.weight_ih.t())

        # 2. Tính đóng góp từ hidden state quá khứ: h_prev @ W_hh^T
        hidden_contrib = torch.matmul(h_prev, self.weight_hh.t())

        # 3. Tính pre-activation sum (tổng kích hoạt trước phi tuyến)
        pre_activation = input_contrib + hidden_contrib
        if self.use_bias:
            pre_activation = pre_activation + self.bias_ih + self.bias_hh

        # 4. Kích hoạt phi tuyến qua hàm hyperbolic tangent (tanh)
        h_next = torch.tanh(pre_activation)
        return h_next


def verify_with_pytorch():
    """Hàm kiểm tra tính tương đương toán học giữa SimpleRNNCell tự cài đặt
    và nn.RNNCell có sẵn của PyTorch trên cùng một tập trọng số và dữ liệu đầu vào.
    """
    print("=" * 60)
    print("KIỂM TRA TÍNH TƯƠNG ĐƯƠNG GIỮA SimpleRNNCell VÀ torch.nn.RNNCell")
    print("=" * 60)

    # 1. Thiết lập cố định seed để tái lập
    torch.manual_seed(42)

    batch_size = 3
    input_size = 4
    hidden_size = 5

    # 2. Khởi tạo 2 cell
    custom_cell = SimpleRNNCell(input_size, hidden_size)
    torch_cell = nn.RNNCell(input_size, hidden_size)

    # 3. Sao chép chính xác toàn bộ trọng số và bias từ torch_cell sang custom_cell
    with torch.no_grad():
        custom_cell.weight_ih.copy_(torch_cell.weight_ih)
        custom_cell.weight_hh.copy_(torch_cell.weight_hh)
        custom_cell.bias_ih.copy_(torch_cell.bias_ih)
        custom_cell.bias_hh.copy_(torch_cell.bias_hh)

    # 4. Tạo dữ liệu giả lập ngẫu nhiên
    x_t = torch.randn(batch_size, input_size)
    h_prev = torch.randn(batch_size, hidden_size)

    # 5. Forward qua 2 cell
    h_custom = custom_cell(x_t, h_prev)
    h_torch = torch_cell(x_t, h_prev)

    # 6. So sánh kết quả
    diff = torch.abs(h_custom - h_torch).max().item()
    is_close = torch.allclose(h_custom, h_torch, atol=1e-7)

    print(f"Batch size: {batch_size}, Input size: {input_size}, Hidden size: {hidden_size}")
    print(f"\n[Custom Cell Output]:\n{h_custom.detach().numpy()}")
    print(f"\n[PyTorch nn.RNNCell Output]:\n{h_torch.detach().numpy()}")
    print(f"\nĐộ lệch tuyệt đối lớn nhất (Max Absolute Difference): {diff:.2e}")
    print(f"Kết quả trùng khớp hoàn toàn (tolerance 1e-7): {is_close}")
    print("=" * 60)
    assert is_close, "LỖI: Kết quả custom cell không khớp với PyTorch!"
    return is_close


if __name__ == "__main__":
    verify_with_pytorch()
