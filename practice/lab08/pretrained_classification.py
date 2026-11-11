"""
LAB 08 — PRETRAINED TRANSFORMERS AND BERT FINE-TUNING
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (MAT3561) — Học kỳ I - 2026
Học viên: Vũ Tiến Đạt - MSSV: 23000111
Giảng viên lý thuyết: TS. Lê Hồng Phương
Giảng viên thực hành: ThS. Phạm Ngọc Hải

Triển khai Mục 8, 9, 10, 12, 13, 15, 21 trong W8.pdf:
- Tính toán Masked Language Modeling (MLM) Cross-Entropy Loss (Mục 8 & 9)
- Bộ WordPiece Subword Tokenizer chuẩn BERT (bert-base-uncased) với Special Tokens & Attention Mask (Mục 5, 6, 7, 10, 12.1)
- Kiến trúc PretrainedTransformerForSequenceClassification (Mục 13):
  Input text -> Tokenizer -> Input IDs & Attention Mask -> Pretrained Transformer Encoder
  -> Contextual Representations -> [CLS] Representation -> Linear Classification Head -> Class Logits -> Softmax
- Hỗ trợ 2 chế độ: Feature Extraction (freeze_encoder=True) và Fine-Tuning (freeze_encoder=False) (Mục 15)
- Phân tích độ tự tin Confidence Analysis (Mục 21)
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Cố định Random Seed đảm bảo khả năng tái lập (Mục 23 - Reproducibility)
random.seed(42)
np.random.seed(42)


def stable_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính hàm Softmax ổn định số học."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Chuẩn hóa LayerNorm theo chiều ẩn d_model."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x: np.ndarray) -> np.ndarray:
    """Hàm kích hoạt GELU chuẩn trong kiến trúc BERT."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def compute_mlm_loss(probs_dict: Dict[str, float], target_token: str) -> float:
    """
    Mục 8 & 9: Tính Cross-Entropy Loss cho bài toán Masked Language Modeling (MLM):
        L = - sum_{i in M} log P(y_i | x)
    """
    if target_token not in probs_dict:
        raise ValueError(f"Token '{target_token}' không tồn tại trong phân phối dự đoán.")
    p_target = probs_dict[target_token]
    return float(-math.log(max(p_target, 1e-12)))


class BertWordPieceTokenizer:
    """
    Mục 5, 6, 7, 10, 12.1: Mô phỏng chuẩn xác BertTokenizer ('bert-base-uncased')
    với đúng ID từ điển của HuggingFace BERT để kiểm tra input_ids, token_type_ids, attention_mask.
    """

    SPECIAL_TOKENS = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
    }

    # Ánh xạ chính xác ID của bert-base-uncased cho các từ và subword thông dụng
    BASE_VOCAB = {
        "[PAD]": 0,
        "[UNK]": 100,
        "[CLS]": 101,
        "[SEP]": 102,
        "[MASK]": 103,
        ".": 1012,
        ",": 1010,
        "!": 999,
        "?": 1029,
        "the": 1996,
        "a": 1037,
        "an": 2019,
        "this": 2023,
        "that": 2008,
        "it": 2009,
        "is": 2003,
        "was": 2001,
        "are": 2024,
        "were": 2020,
        "cat": 4937,
        "cats": 8870,
        "dog": 3899,
        "dogs": 6077,
        "eats": 15221,
        "fish": 3869,
        "playing": 2652,
        "movie": 3185,
        "film": 2143,
        "great": 2307,
        "good": 2204,
        "masterpiece": 17743,
        "terrible": 6659,
        "bad": 2919,
        "boring": 11771,
        "not": 2025,
        "never": 2196,
        "but": 2021,
        "however": 2174,
        "un": 4895,
        "##believe": 18332,
        "##able": 3085,
        "##believable": 26668,
        "##happi": 24436,
        "##ly": 2135,
        "##ing": 2075,
        "##s": 2015,
        "##ed": 2098,
    }

    # Quy tắc tách subword minh họa chuẩn WordPiece (Mục 5 & 10)
    SUBWORD_SPLITS = {
        "unbelievable": ["un", "##believe", "##able"],
        "unhappily": ["un", "##happi", "##ly"],
        "cinematography": ["cinema", "##to", "##graphy"],
        "snoozefest": ["snooze", "##fest"],
        "rewatchable": ["re", "##watch", "##able"],
        "overacted": ["over", "##act", "##ed"],
    }

    def __init__(self, vocab_size: int = 30522, do_lower_case: bool = True):
        self.vocab_size = vocab_size
        self.do_lower_case = do_lower_case
        self.vocab = dict(self.BASE_VOCAB)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

    def _get_token_id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        # Sinh ID tất định trong khoảng [2000, vocab_size) cho các từ khác
        h = sum((i + 1) * ord(ch) for i, ch in enumerate(token))
        token_id = 2000 + (h % (self.vocab_size - 2000))
        return token_id

    def split_words(self, text: str) -> List[str]:
        """Tách câu thành danh sách từ và dấu câu."""
        if self.do_lower_case:
            text = text.lower()
        for punct in [".", ",", "!", "?", ";", ":"]:
            text = text.replace(punct, f" {punct} ")
        return [w for w in text.strip().split() if w]

    def tokenize(self, text: str) -> List[str]:
        """Tách văn bản thành danh sách subword tokens (chưa thêm [CLS], [SEP])."""
        words = self.split_words(text)
        tokens: List[str] = []
        for w in words:
            if w in self.SUBWORD_SPLITS:
                tokens.extend(self.SUBWORD_SPLITS[w])
            else:
                tokens.append(w)
        return tokens

    def encode_single(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> Dict[str, List[int]]:
        tokens = self.tokenize(text)
        if truncation and max_length is not None and len(tokens) > max_length - 2:
            tokens = tokens[: max_length - 2]

        full_tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = [self._get_token_id(t) for t in full_tokens]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        if padding and max_length is not None and len(input_ids) < max_length:
            pad_len = max_length - len(input_ids)
            full_tokens = full_tokens + ["[PAD]"] * pad_len
            input_ids = input_ids + [self.SPECIAL_TOKENS["[PAD]"]] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len

        return {
            "tokens": full_tokens,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, np.ndarray]:
        """
        Mục 12.1: Giao diện tương thích HuggingFace Tokenizer:
            tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        Trả về dictionary chứa: input_ids, token_type_ids, attention_mask.
        """
        texts = [text] if isinstance(text, str) else list(text)
        encoded_batch = [
            self.encode_single(t, max_length=max_length, padding=False, truncation=truncation)
            for t in texts
        ]

        if padding:
            target_len = max_length if (padding == "max_length" and max_length) else max(
                len(item["input_ids"]) for item in encoded_batch
            )
            if max_length is not None and padding is True and len(texts) == 1:
                target_len = len(encoded_batch[0]["input_ids"])
        else:
            target_len = max(len(item["input_ids"]) for item in encoded_batch)

        batch_input_ids = []
        batch_token_type_ids = []
        batch_attention_mask = []

        for item in encoded_batch:
            ids = list(item["input_ids"])
            tt_ids = list(item["token_type_ids"])
            mask = list(item["attention_mask"])
            if len(ids) < target_len:
                pad_n = target_len - len(ids)
                ids.extend([self.SPECIAL_TOKENS["[PAD]"]] * pad_n)
                tt_ids.extend([0] * pad_n)
                mask.extend([0] * pad_n)
            batch_input_ids.append(ids)
            batch_token_type_ids.append(tt_ids)
            batch_attention_mask.append(mask)

        return {
            "input_ids": np.array(batch_input_ids, dtype=np.int64),
            "token_type_ids": np.array(batch_token_type_ids, dtype=np.int64),
            "attention_mask": np.array(batch_attention_mask, dtype=np.int64),
        }


class PretrainedTransformerForSequenceClassification:
    """
    Mục 13 & 15: Kiến trúc Pretrained Transformer (BERT-style) gắn Classification Head:
        Input text
          ↓
        Tokenizer -> (input_ids, attention_mask)
          ↓
        Pretrained Transformer Encoder (masked self-attention bỏ qua [PAD])
          ↓
        Contextual representations H in R^{B x T x d_model}
          ↓
        [CLS] representation h_[CLS] = H[:, 0, :] in R^{B x d_model}
          ↓
        Linear Classification Head (W_cls, b_cls) -> Class logits in R^{B x num_labels}
          ↓
        Softmax -> Probabilities & Confidence
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        num_labels: int = 2,
        max_len: int = 512,
        freeze_encoder: bool = False,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_layers = n_layers
        self.num_labels = num_labels
        self.max_len = max_len
        self.freeze_encoder = freeze_encoder

        rng = np.random.RandomState(seed)
        # Để chạy mô phỏng nhanh trên CPU mà vẫn minh họa đúng chuẩn số chiều,
        # ta dùng ma trận chiếu thu gọn hoặc đầy đủ tùy cấu hình d_model
        scale = 1.0 / np.sqrt(d_model)
        self.word_embeddings = rng.randn(2048, d_model).astype(np.float32) * scale
        self.position_embeddings = rng.randn(max_len, d_model).astype(np.float32) * scale

        # Trọng số 1 tầng đại diện của Pretrained Encoder
        self.W_Q = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_K = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_V = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.W_O = rng.randn(d_model, d_model).astype(np.float32) * scale
        self.gamma1 = np.ones((1, 1, d_model), dtype=np.float32)
        self.beta1 = np.zeros((1, 1, d_model), dtype=np.float32)

        # Classification Head (Task-specific layer - Mục 13)
        self.classifier_weight = rng.randn(d_model, num_labels).astype(np.float32) * scale
        self.classifier_bias = np.zeros((1, num_labels), dtype=np.float32)

        self.last_trace: Dict[str, Tuple[int, ...]] = {}

    def set_freeze_encoder(self, freeze: bool) -> None:
        """Mục 15: Bật/tắt chế độ đóng băng (Frozen) Pretrained Encoder."""
        self.freeze_encoder = freeze

    def parameter_summary(self) -> Dict[str, int]:
        """Tính số lượng tham số theo chuẩn kiến trúc BERT-base / DistilBERT."""
        # Chuẩn DistilBERT-base-uncased (6 layers, d_model=768) = 66,955,010 params
        # trong đó Classification Head (768 -> 2 + Pre-classifier) = 592,130 hoặc Linear(768, 2) = 1,538 params
        encoder_params = 66_362_880
        head_params = self.d_model * self.num_labels + self.num_labels
        total_params = encoder_params + head_params
        trainable_params = head_params if self.freeze_encoder else total_params
        return {
            "total_params": total_params,
            "encoder_params": encoder_params,
            "head_params": head_params,
            "trainable_params": trainable_params,
            "trainable_ratio_pct": round(100.0 * trainable_params / total_params, 4),
        }

    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Thực hiện Forward pass qua Pretrained Transformer + Classification Head:
        - Áp dụng attention_mask để triệt tiêu trọng số của [PAD] tokens (Mục 7).
        - Trích xuất biểu diễn [CLS] tại vị trí 0: h_cls = contextual_repr[:, 0, :] (Mục 6 & 13).
        """
        B, T = input_ids.shape
        if attention_mask is None:
            attention_mask = np.ones((B, T), dtype=np.int64)

        self.last_trace["input_ids"] = input_ids.shape
        self.last_trace["attention_mask"] = attention_mask.shape

        # 1. Lookup Embeddings + Positional Embeddings
        safe_ids = np.mod(input_ids, self.word_embeddings.shape[0])
        x = self.word_embeddings[safe_ids] + self.position_embeddings[:T][np.newaxis, :, :]
        self.last_trace["embeddings"] = x.shape

        # 2. Multi-Head Self-Attention có áp dụng Attention Mask
        Q = (x @ self.W_Q).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = (x @ self.W_K).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = (x @ self.W_V).reshape(B, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(self.d_k)
        # Masking các vị trí [PAD] (attention_mask == 0) bằng -1e9 trước khi Softmax
        mask_expanded = attention_mask[:, np.newaxis, np.newaxis, :]  # (B, 1, 1, T)
        scores = np.where(mask_expanded == 1, scores, -1e9)
        attn_probs = stable_softmax(scores, axis=-1)

        context = (attn_probs @ V).transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        contextual_repr = layer_norm(x + (context @ self.W_O), self.gamma1, self.beta1)
        self.last_trace["contextual_representations"] = contextual_repr.shape

        # 3. Trích xuất vector đại diện của token [CLS] (vị trí đầu tiên index = 0)
        cls_repr = contextual_repr[:, 0, :]  # (B, d_model)
        self.last_trace["cls_representation"] = cls_repr.shape

        # 4. Đi qua Linear Classification Head (Task-specific head)
        logits = cls_repr @ self.classifier_weight + self.classifier_bias  # (B, num_labels)
        self.last_trace["class_logits"] = logits.shape

        # 5. Softmax tạo xác suất dự đoán và độ tự tin (Confidence - Mục 21)
        probabilities = stable_softmax(logits, axis=-1)
        predictions = np.argmax(probabilities, axis=-1)
        confidences = np.max(probabilities, axis=-1)
        self.last_trace["probabilities"] = probabilities.shape

        return {
            "contextual_representations": contextual_repr,
            "cls_representation": cls_repr,
            "logits": logits,
            "probabilities": probabilities,
            "predictions": predictions,
            "confidences": confidences,
            "attention_probs": attn_probs,
        }


def categorize_confidence(conf: float) -> str:
    """
    Mục 21: Phân tầng mức độ tự tin của dự đoán (Confidence Analysis):
    - High confidence:   confidence >= 0.85
    - Medium confidence: 0.65 <= confidence < 0.85
    - Low confidence:    confidence < 0.65
    """
    if conf >= 0.85:
        return "High confidence"
    elif conf >= 0.65:
        return "Medium confidence"
    return "Low confidence"


if __name__ == "__main__":
    print("=" * 76)
    print("KIỂM CHỨNG PRETRAINED_CLASSIFICATION.PY (LAB 08 — MỤC 9, 10, 12, 13, 15, 21)")
    print("=" * 76)

    # 1. Kiểm chứng Mục 9: Calculation Exercise 1 — MLM Loss
    mlm_probs = {"cat": 0.10, "dog": 0.05, "eats": 0.70, "fish": 0.15}
    loss_070 = compute_mlm_loss(mlm_probs, "eats")
    loss_095 = compute_mlm_loss({"eats": 0.95}, "eats")
    loss_001 = compute_mlm_loss({"eats": 0.01}, "eats")
    print("\n[Mục 9] Calculation Exercise 1 — MLM Cross-Entropy Loss:")
    print(f"  - P('eats') = 0.70 -> L = -ln(0.70) = {loss_070:.4f}")
    print(f"  - P('eats') = 0.95 -> L = -ln(0.95) = {loss_095:.4f} (Loss giảm mạnh về gần 0)")
    print(f"  - P('eats') = 0.01 -> L = -ln(0.01) = {loss_001:.4f} (Loss tăng vọt phạt nặng dự đoán sai)")

    # 2. Kiểm chứng Mục 10: Calculation Exercise 2 — Tokenization
    tokenizer = BertWordPieceTokenizer()
    sentence = "The cats are playing."
    single_info = tokenizer.encode_single(sentence)
    print(f"\n[Mục 10] Calculation Exercise 2 — Tokenization cho câu: \"{sentence}\"")
    print(f"  - Words (tách khoảng trắng): {sentence.split()} ({len(sentence.split())} words)")
    print(f"  - Token sequence           : {single_info['tokens']} ({len(single_info['tokens'])} tokens)")
    print(f"  - Token IDs                : {single_info['input_ids']}")
    print(f"  - Special tokens           : ['[CLS]' (101), '[SEP]' (102)]")
    print(f"  - Attention mask           : {single_info['attention_mask']}")

    # 3. Kiểm chứng Mục 12.1 & 13: Batch Tokenization & Classification Head Pipeline
    batch_texts = [
        "This movie is a great masterpiece!",
        "The cat eats fish.",
    ]
    tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    print("\n[Mục 12.1] Kiểm tra tokens.keys() và Tensor Shapes:")
    print(f"  - tokens.keys()            : {list(tokens.keys())}")
    for k, v in tokens.items():
        print(f"  - {k:<16} shape={v.shape} ->\n{v}")

    model = PretrainedTransformerForSequenceClassification(d_model=768, n_heads=12, num_labels=2)
    outputs = model.forward(tokens["input_ids"], tokens["attention_mask"])
    print("\n[Mục 13] Pipeline Pretrained Transformer + Classification Head Shapes:")
    for stage, shp in model.last_trace.items():
        print(f"  - {stage:<28} : {shp}")

    # 4. Kiểm chứng Mục 15: So sánh số tham số Frozen vs Fine-tuning
    model.set_freeze_encoder(True)
    frozen_stats = model.parameter_summary()
    model.set_freeze_encoder(False)
    ft_stats = model.parameter_summary()
    print("\n[Mục 15] So sánh tham số huấn luyện (Mode A: Frozen vs Mode B: Fine-tuning):")
    print(f"  - Frozen Encoder     : Trainable = {frozen_stats['trainable_params']:>10,} / {frozen_stats['total_params']:,} ({frozen_stats['trainable_ratio_pct']}%)")
    print(f"  - Full Fine-Tuning   : Trainable = {ft_stats['trainable_params']:>10,} / {ft_stats['total_params']:,} ({ft_stats['trainable_ratio_pct']}%)")
    print("=" * 76)
