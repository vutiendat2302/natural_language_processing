"""Mô hình ngôn ngữ N-gram từ đầu (N-Gram Language Models from Scratch).

Cài đặt các mô hình Unigram, Bigram, Trigram theo:
- Maximum Likelihood Estimation (MLE)
- Laplace (Add-1) Smoothing
Hỗ trợ tính toán xác suất chuỗi, log-probability, Perplexity (PPL),
đánh giá mô hình (evaluate), sinh từ tiếp theo (next-word prediction)
và xếp hạng câu ứng viên (sentence ranking).

"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Biểu thức chính quy tách câu và tách từ chuẩn
SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?…])\s+|\n+")
TOKEN_REGEX = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def split_sentences(text: str) -> List[str]:
    """Tách văn bản thô thành danh sách các câu dựa trên dấu kết thúc câu hoặc dấu xuống dòng."""
    return [s.strip() for s in SENT_SPLIT_REGEX.split(text) if s.strip()]


def tokenize(text: str) -> List[str]:
    """Tách từ theo chữ thường, giữ lại số và các từ có dấu nháy đơn tiếng Anh (ví dụ: don't)."""
    return TOKEN_REGEX.findall(text.lower())


class NGramLanguageModel:
    """Mô hình ngôn ngữ N-gram với MLE hoặc Laplace Smoothing."""

    def __init__(
        self,
        n: int = 1,
        min_count: int = 1,
        smoothing: str = "mle",
    ):
        """Khởi tạo mô hình N-gram.

        Args:
            n: Bậc của mô hình (1: Unigram, 2: Bigram, 3: Trigram, ...).
            min_count: Ngưỡng tần số tối thiểu; từ xuất hiện ít hơn ngưỡng này sẽ thành <unk>.
            smoothing: Phương pháp làm mịn ('mle' hoặc 'laplace').
        """
        if n < 1:
            raise ValueError(f"Bậc n phải >= 1, nhận được: {n}")
        if min_count < 1:
            raise ValueError(f"min_count phải >= 1, nhận được: {min_count}")

        self.n: int = n
        self.min_count: int = min_count
        self.smoothing: str = smoothing.lower()
        if self.smoothing not in ("mle", "laplace"):
            raise ValueError(
                f"smoothing phải là 'mle' hoặc 'laplace', nhận được: {smoothing}"
            )

        # Các ký hiệu đặc biệt
        self.start_token: str = "<s>"
        self.end_token: str = "</s>"
        self.unk_token: str = "<unk>"

        # Cấu trúc lưu trữ dữ liệu huấn luyện
        self.vocab: set = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.continuations: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.total_tokens: int = 0
        self.is_fitted: bool = False

    @property
    def vocab_size(self) -> int:
        """Kích thước tập từ vựng V (bao gồm </s> và <unk> nếu có, không gồm <s>)."""
        return len(self.vocab)

    def build_vocabulary(
        self, corpus: Sequence[Sequence[str]]
    ) -> "NGramLanguageModel":
        """Xây dựng tập từ vựng từ corpus huấn luyện.

        Các từ có tần số < min_count được gom vào <unk>.
        Quy ước: </s> luôn thuộc từ vựng; <s> không thuộc từ vựng;
        <unk> thuộc từ vựng nếu có từ bị lọc hoặc min_count > 1.
        """
        raw_counts: Counter = Counter()
        for sent in corpus:
            raw_counts.update(sent)

        # Giữ lại các từ đạt ngưỡng tần số
        self.vocab = {w for w, c in raw_counts.items() if c >= self.min_count}

        # Bổ sung <unk> nếu có từ bị lọc hoặc min_count > 1
        has_unk = (
            any(c < self.min_count for c in raw_counts.values())
            or self.min_count > 1
        )
        if has_unk:
            self.vocab.add(self.unk_token)

        # Token </s> luôn là một thành phần hợp lệ trong tập từ dự đoán
        self.vocab.add(self.end_token)
        return self

    def _map_oov(self, sentence: Sequence[str]) -> List[str]:
        """Ánh xạ các từ không nằm trong tập từ vựng (OOV) thành <unk>."""
        return [w if w in self.vocab else self.unk_token for w in sentence]

    def count_ngrams(
        self, corpus: Sequence[Sequence[str]]
    ) -> "NGramLanguageModel":
        """Đếm số lần xuất hiện của các N-gram và ngữ cảnh (context) tương ứng."""
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.continuations = defaultdict(Counter)
        self.total_tokens = 0

        for sent in corpus:
            tokens = self._map_oov(sent)
            if not tokens:
                continue

            if self.n == 1:
                # Unigram: xét từng token kèm </s>, context là tuple rỗng ()
                tokens_with_end = tokens + [self.end_token]
                for token in tokens_with_end:
                    self.ngram_counts[(token,)] += 1
                    self.context_counts[()] += 1
                    self.continuations[()][token] += 1
                    self.total_tokens += 1
            else:
                # N >= 2: thêm (n - 1) <s> ở đầu và 1 </s> ở cuối
                padded = (
                    [self.start_token] * (self.n - 1)
                    + tokens
                    + [self.end_token]
                )
                for i in range(len(padded) - self.n + 1):
                    window = padded[i : i + self.n]
                    ngram = tuple(window)
                    context = tuple(window[:-1])
                    word = window[-1]
                    self.ngram_counts[ngram] += 1
                    self.context_counts[context] += 1
                    self.continuations[context][word] += 1
                    self.total_tokens += 1

        return self

    def fit(
        self, corpus: Sequence[Union[str, Sequence[str]]]
    ) -> "NGramLanguageModel":
        """Huấn luyện mô hình N-gram: xây dựng vocab và đếm n-gram counts."""
        tokenized_corpus: List[List[str]] = []
        for doc in corpus:
            if isinstance(doc, str):
                tokens = tokenize(doc)
            else:
                tokens = list(doc)
            if tokens:
                tokenized_corpus.append(tokens)

        self.build_vocabulary(tokenized_corpus)
        self.count_ngrams(tokenized_corpus)
        self.is_fitted = True
        return self

    def _normalize_context(
        self, context: Union[str, Sequence[str]]
    ) -> Tuple[str, ...]:
        """Chuẩn hóa ngữ cảnh: lấy n-1 từ cuối, pad <s> nếu thiếu, map OOV sang <unk>."""
        if self.n == 1:
            return ()

        if isinstance(context, str):
            tokens = tokenize(context)
        else:
            tokens = list(context)

        req_len = self.n - 1
        if len(tokens) < req_len:
            tokens = [self.start_token] * (req_len - len(tokens)) + tokens
        else:
            tokens = tokens[-req_len:]

        # Ánh xạ OOV cho ngữ cảnh (giữ nguyên token <s>)
        normalized = []
        for t in tokens:
            if t == self.start_token:
                normalized.append(self.start_token)
            elif t in self.vocab:
                normalized.append(t)
            else:
                normalized.append(self.unk_token)
        return tuple(normalized)

    def _prob_from_counts(self, ngram_count: int, context_count: int) -> float:
        """Tính xác suất từ số đếm n-gram và số đếm context theo MLE hoặc Laplace."""
        if self.smoothing == "laplace":
            # Laplace: (C(h,w) + 1) / (C(h) + V)
            # Đối với context chưa thấy (context_count = 0), trả về 1 / V
            return (ngram_count + 1) / (context_count + self.vocab_size)

        # Mặc định: MLE
        if context_count == 0 or ngram_count == 0:
            return 0.0
        return ngram_count / context_count

    def probability(
        self, context: Union[str, Sequence[str]], word: str
    ) -> float:
        """Tính xác suất có điều kiện P(word | context).

        Chỉ dùng n-1 từ cuối của context; nếu ngắn hơn thì pad <s> bên trái.
        Context hoặc word ngoài vocab được ánh xạ sang <unk>.
        """
        target_word = (
            word
            if (word in self.vocab or word == self.end_token)
            else self.unk_token
        )
        ctx = self._normalize_context(context)

        if self.n == 1:
            c_ngram = self.ngram_counts.get((target_word,), 0)
            c_ctx = self.context_counts.get((), 0)
            return self._prob_from_counts(c_ngram, c_ctx)

        c_ngram = self.ngram_counts.get(ctx + (target_word,), 0)
        c_ctx = self.context_counts.get(ctx, 0)
        return self._prob_from_counts(c_ngram, c_ctx)

    def log_probability(
        self, context: Union[str, Sequence[str]], word: str
    ) -> float:
        """Tính log-xác suất tự nhiên ln P(word | context).

        Quy ước: Trả về float('-inf') khi xác suất p <= 0.0.
        """
        p = self.probability(context, word)
        if p <= 0.0:
            return float("-inf")
        return math.log(p)

    def sentence_log_probability(
        self, sentence: Union[str, Sequence[str]]
    ) -> float:
        """Tính log-xác suất của cả câu: ln P(S) = sum ln P(w_t | context_t).

        Tự động thêm padding và ánh xạ OOV.
        Nếu gặp bất kỳ vị trí nào có xác suất 0.0, trả về ngay float('-inf').
        """
        if isinstance(sentence, str):
            tokens = tokenize(sentence)
        else:
            tokens = list(sentence)

        if not tokens:
            return float("-inf")

        mapped_tokens = self._map_oov(tokens)

        if self.n == 1:
            tokens_to_score = mapped_tokens + [self.end_token]
            total_lp = 0.0
            for t in tokens_to_score:
                lp = self.log_probability((), t)
                if math.isinf(lp) and lp < 0:
                    return float("-inf")
                total_lp += lp
            return total_lp

        padded = (
            [self.start_token] * (self.n - 1)
            + mapped_tokens
            + [self.end_token]
        )
        total_lp = 0.0
        for i in range(len(padded) - self.n + 1):
            ctx = tuple(padded[i : i + self.n - 1])
            w = padded[i + self.n - 1]
            lp = self.log_probability(ctx, w)
            if math.isinf(lp) and lp < 0:
                return float("-inf")
            total_lp += lp
        return total_lp

    def sentence_probability(self, sentence: Union[str, Sequence[str]]) -> float:
        """Tính xác suất P(S) của cả câu bằng tích xác suất chuỗi.

        Quy ước: math.exp(sentence_log_probability(sentence)).
        Nếu sentence_log_probability là -inf, trả về đúng 0.0.
        """
        lp = self.sentence_log_probability(sentence)
        if math.isinf(lp) and lp < 0:
            return 0.0
        return math.exp(lp)

    def next_word_distribution(
        self, context: Union[str, Sequence[str]], top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """Trả về phân phối xác suất các từ tiếp theo dựa trên context đã cho.

        Quy ước:
        - Sắp xếp giảm dần theo xác suất; tie-break: sắp xếp theo thứ tự bảng chữ cái nếu cùng xác suất.
        - Với MLE: context chưa từng thấy trong train thì trả về danh sách rỗng [].
        - Với Laplace: trả phân phối trên TOÀN BỘ vocab (mọi từ đều có xác suất > 0).
        - Nếu có top_k, chỉ trả về top_k phần tử đầu tiên.
        """
        ctx = self._normalize_context(context)

        if self.smoothing == "laplace":
            c_ctx = (
                self.context_counts.get(ctx, 0)
                if self.n > 1
                else self.context_counts.get((), 0)
            )
            seen_counts = self.continuations.get(ctx, {})

            dist = []
            for w in self.vocab:
                cnt = seen_counts.get(w, 0)
                prob = (cnt + 1) / (c_ctx + self.vocab_size)
                dist.append((w, prob))

            dist.sort(key=lambda item: (-item[1], item[0]))
            if top_k is not None:
                return dist[:top_k]
            return dist

        # Mô hình MLE
        if ctx not in self.continuations:
            return []

        c_ctx = self.context_counts[ctx]
        if c_ctx == 0:
            return []

        words_dict = self.continuations[ctx]
        dist = [
            (w, self._prob_from_counts(cnt, c_ctx))
            for w, cnt in words_dict.items()
        ]
        dist.sort(key=lambda item: (-item[1], item[0]))
        if top_k is not None:
            return dist[:top_k]
        return dist


# Các hàm tiện ích cấp module (Module-level helper functions)
def train_unigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    min_count: int = 1,
    smoothing: str = "mle",
) -> NGramLanguageModel:
    """Huấn luyện mô hình Unigram (n=1) với phương pháp smoothing chỉ định."""
    return NGramLanguageModel(
        n=1, min_count=min_count, smoothing=smoothing
    ).fit(corpus)


def train_bigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    min_count: int = 1,
    smoothing: str = "mle",
) -> NGramLanguageModel:
    """Huấn luyện mô hình Bigram (n=2) với phương pháp smoothing chỉ định."""
    return NGramLanguageModel(
        n=2, min_count=min_count, smoothing=smoothing
    ).fit(corpus)


def train_trigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    min_count: int = 1,
    smoothing: str = "mle",
) -> NGramLanguageModel:
    """Huấn luyện mô hình Trigram (n=3) với phương pháp smoothing chỉ định."""
    return NGramLanguageModel(
        n=3, min_count=min_count, smoothing=smoothing
    ).fit(corpus)


def perplexity_from_probs(probs: Sequence[float]) -> float:
    """Tính chỉ số Perplexity từ danh sách xác suất của chuỗi tokens: PPL = exp(- 1/N * sum ln p_i).

    Nếu danh sách rỗng hoặc có ít nhất một xác suất <= 0.0, trả về float('inf').
    """
    if not probs:
        return float("inf")
    total_log_prob = 0.0
    for p in probs:
        if p <= 0.0:
            return float("inf")
        total_log_prob += math.log(p)
    avg_log_prob = total_log_prob / len(probs)
    return math.exp(-avg_log_prob)


def evaluate(
    model: NGramLanguageModel, sentences: Sequence[Union[str, Sequence[str]]]
) -> Dict[str, Union[float, int]]:
    """Đánh giá toàn diện mô hình trên tập câu: Perplexity, zero_rate, avg_logprob.

    Returns:
        Dict gồm:
        - 'ppl': Perplexity chuẩn (trả float('inf') nếu có n-gram gặp xác suất 0.0).
        - 'N': Tổng số vị trí token được đánh giá (bao gồm </s>, không bao gồm <s>).
        - 'avg_logprob': Log-xác suất trung bình mỗi token (float('-inf') nếu có zero).
        - 'n_zero': Tổng số vị trí có xác suất bằng 0.0.
        - 'zero_rate': Tỷ lệ n_zero / N.
        - 'ppl_excluding_zeros': Perplexity tính trên tập các token có p > 0.
          Lưu ý: ppl_excluding_zeros là trường phụ, KHÔNG so sánh được giữa các mô hình khác nhau
          vì tập token được đưa vào đánh giá đã bị lọc khác nhau.
    """
    N = 0
    n_zero = 0
    sum_nonzero_logprob = 0.0

    for sent in sentences:
        if isinstance(sent, str):
            tokens = tokenize(sent)
        else:
            tokens = list(sent)
        if not tokens:
            continue

        mapped_tokens = model._map_oov(tokens)

        if model.n == 1:
            tokens_to_score = mapped_tokens + [model.end_token]
            for t in tokens_to_score:
                N += 1
                p = model.probability((), t)
                if p <= 0.0:
                    n_zero += 1
                else:
                    sum_nonzero_logprob += math.log(p)
        else:
            padded = (
                [model.start_token] * (model.n - 1)
                + mapped_tokens
                + [model.end_token]
            )
            for i in range(len(padded) - model.n + 1):
                N += 1
                ctx = tuple(padded[i : i + model.n - 1])
                w = padded[i + model.n - 1]
                p = model.probability(ctx, w)
                if p <= 0.0:
                    n_zero += 1
                else:
                    sum_nonzero_logprob += math.log(p)

    zero_rate = n_zero / N if N > 0 else 0.0
    if n_zero > 0:
        ppl = float("inf")
        avg_logprob = float("-inf")
    else:
        avg_logprob = sum_nonzero_logprob / N if N > 0 else 0.0
        ppl = math.exp(-avg_logprob) if N > 0 else float("inf")

    nonzero_count = N - n_zero
    if nonzero_count > 0:
        ppl_excluding_zeros = math.exp(-(sum_nonzero_logprob / nonzero_count))
    else:
        ppl_excluding_zeros = float("inf")

    return {
        "ppl": ppl,
        "N": N,
        "avg_logprob": avg_logprob,
        "n_zero": n_zero,
        "zero_rate": zero_rate,
        "ppl_excluding_zeros": ppl_excluding_zeros,
    }


def score_continuation(
    model: NGramLanguageModel,
    context_tokens: Union[str, Sequence[str]],
    candidate_tokens: Union[str, Sequence[str]],
    add_eos: bool = True,
) -> Tuple[float, float, int]:
    """Tính log P(candidate | context) và log-prob chuẩn hóa theo số token.

    Args:
        model: Mô hình ngôn ngữ N-gram đã fit.
        context_tokens: Ngữ cảnh ban đầu (chuỗi hoặc danh sách token).
        candidate_tokens: Đoạn tiếp nối cần chấm điểm.
        add_eos: Có tính thêm xác suất chuyển sang </s> ở cuối hay không.

    Returns:
        Tuple: (total_log_prob, avg_log_prob, num_tokens_scored)
        Nếu gặp xác suất 0.0, total_log_prob và avg_log_prob là -inf.
    """
    if isinstance(context_tokens, str):
        ctx_list = tokenize(context_tokens)
    else:
        ctx_list = list(context_tokens)

    if isinstance(candidate_tokens, str):
        cand_list = tokenize(candidate_tokens)
    else:
        cand_list = list(candidate_tokens)

    # Lịch sử gồm ngữ cảnh ban đầu
    history = list(ctx_list)
    tokens_to_score = list(cand_list)
    if add_eos:
        tokens_to_score.append(model.end_token)

    total_lp = 0.0
    num_tokens = len(tokens_to_score)
    if num_tokens == 0:
        return 0.0, 0.0, 0

    for token in tokens_to_score:
        lp = model.log_probability(history, token)
        if math.isinf(lp) and lp < 0:
            return float("-inf"), float("-inf"), num_tokens
        total_lp += lp
        history.append(token)

    avg_lp = total_lp / num_tokens
    return total_lp, avg_lp, num_tokens


def rank_candidates(
    model: NGramLanguageModel,
    context: Union[str, Sequence[str]],
    candidates: Sequence[Union[str, Sequence[str]]],
    add_eos: bool = True,
) -> List[Dict[str, Union[str, float, int]]]:
    """Xếp hạng các câu ứng viên candidate dựa trên log-prob và avg-log-prob.

    Các candidate bằng nhau xếp đồng hạng, tie-break theo thứ tự bảng chữ cái.
    """
    results = []
    for cand in candidates:
        cand_str = cand if isinstance(cand, str) else " ".join(cand)
        tot_lp, avg_lp, n_tok = score_continuation(
            model, context, cand, add_eos=add_eos
        )
        results.append({
            "candidate": cand_str,
            "log_prob": tot_lp,
            "avg_log_prob": avg_lp,
            "tokens": n_tok,
        })

    # Xếp hạng theo total log_prob (giảm dần, tie-break candidate chữ cái)
    results.sort(key=lambda x: (-x["log_prob"], x["candidate"]))
    cur_rank = 1
    for i, r in enumerate(results):
        if i > 0 and r["log_prob"] == results[i - 1]["log_prob"]:
            r["rank_total"] = results[i - 1]["rank_total"]
        else:
            r["rank_total"] = i + 1

    # Xếp hạng theo avg log_prob (giảm dần, tie-break candidate chữ cái)
    results.sort(key=lambda x: (-x["avg_log_prob"], x["candidate"]))
    for i, r in enumerate(results):
        if i > 0 and r["avg_log_prob"] == results[i - 1]["avg_log_prob"]:
            r["rank_avg"] = results[i - 1]["rank_avg"]
        else:
            r["rank_avg"] = i + 1

    # Trả về sắp xếp theo rank_total làm chuẩn
    results.sort(key=lambda x: (x["rank_total"], -x["avg_log_prob"], x["candidate"]))
    return results
