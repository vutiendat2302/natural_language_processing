"""N-Gram Language Models from Scratch.

This module implements Unigram, Bigram, and Trigram language models with
Maximum Likelihood Estimation (MLE) and Add-k (Laplace) smoothing, as well as
log-probability evaluation, perplexity measurement, next-word prediction,
and sentence candidate ranking.

Author: Vu Tien Dat (MSSV: 23000111)
Course: MAT3561 - Natural Language Processing and Applications
"""

from collections import Counter, defaultdict
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Tokenize a string into a list of word tokens.

    Args:
        text: Input raw string.
        lowercase: Whether to convert text to lowercase.

    Returns:
        List of alphanumeric word tokens.
    """
    if lowercase:
        text = text.lower()
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text)


def build_vocabulary(
    tokenized_sentences: Sequence[Sequence[str]],
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    unk_token: str = "<unk>",
) -> Dict[str, int]:
    """Extract vocabulary and frequency counts from tokenized sentences.

    Args:
        tokenized_sentences: Sequence of token lists.
        min_freq: Minimum frequency threshold to retain a word.
        max_vocab_size: Maximum vocabulary capacity (ranked by frequency).
        unk_token: Special out-of-vocabulary representation token.

    Returns:
        Mapping from word token to its corpus frequency count.
    """
    counter: Counter = Counter()
    for sentence in tokenized_sentences:
        counter.update(sentence)

    # Filter by minimum frequency
    filtered = {w: c for w, c in counter.items() if c >= min_freq}

    if max_vocab_size is not None and len(filtered) > max_vocab_size:
        sorted_words = sorted(
            filtered.items(), key=lambda item: item[1], reverse=True
        )
        filtered = dict(sorted_words[:max_vocab_size])

    if unk_token not in filtered:
        filtered[unk_token] = 0

    return filtered


def count_ngrams(
    tokenized_sentences: Sequence[Sequence[str]],
    n: int,
    use_boundary: bool = True,
    start_token: str = "<s>",
    end_token: str = "</s>",
) -> Tuple[Counter, Counter]:
    """Count n-grams and their conditioning (n-1)-gram contexts.

    Args:
        tokenized_sentences: Tokenized sentences.
        n: N-gram order (1 for unigram, 2 for bigram, 3 for trigram).
        use_boundary: Whether to append/prepend boundary markers.
        start_token: Boundary prefix token.
        end_token: Boundary suffix token.

    Returns:
        Tuple of (ngram_counts, context_counts).
    """
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()

    for sent in tokenized_sentences:
        if not sent:
            continue
        if n == 1:
            tokens = list(sent) + ([end_token] if use_boundary else [])
            for token in tokens:
                ngram_counts[(token,)] += 1
                context_counts[()] += 1
        else:
            padding = [start_token] * (n - 1) if use_boundary else []
            tokens = padding + list(sent) + ([end_token] if use_boundary else [])
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                context = tuple(tokens[i : i + n - 1])
                ngram_counts[ngram] += 1
                context_counts[context] += 1

    return ngram_counts, context_counts


class NGramLanguageModel:
    """N-Gram Language Model with MLE and Laplace/Add-k Smoothing."""

    def __init__(
        self,
        n: int = 2,
        smoothing: Optional[str] = None,
        k: float = 1.0,
        use_boundary: bool = True,
        unk_token: str = "<unk>",
        start_token: str = "<s>",
        end_token: str = "</s>",
    ):
        """Initialize the N-gram Language Model.

        Args:
            n: Order of the model (1=Unigram, 2=Bigram, 3=Trigram, ...).
            smoothing: Smoothing method. Supported: None (MLE), 'laplace', 'add-k'.
            k: Smoothing pseudo-count parameter (k=1.0 for standard Laplace).
            use_boundary: Whether to include <s> and </s> tokens.
            unk_token: Out-of-vocabulary representation token.
            start_token: Start-of-sentence marker.
            end_token: End-of-sentence marker.
        """
        if n < 1:
            raise ValueError(f"Order n must be >= 1, got {n}")
        self.n = n
        self.smoothing = smoothing.lower() if smoothing else None
        self.k = float(k)
        self.use_boundary = use_boundary
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token

        self.vocab: set = set()
        self.ngram_counts: Counter = Counter()
        self.context_counts: Counter = Counter()
        self.total_tokens: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        corpus: Sequence[Union[str, Sequence[str]]],
        vocab: Optional[set] = None,
    ) -> "NGramLanguageModel":
        """Train n-gram counts and vocabulary on the input corpus.

        Args:
            corpus: Collection of sentences, either raw strings or lists of tokens.
            vocab: Optional predefined vocabulary set.

        Returns:
            Self (fitted model).
        """
        # Tokenize if needed
        tokenized_corpus: List[List[str]] = []
        for doc in corpus:
            if isinstance(doc, str):
                tokens = tokenize(doc)
            else:
                tokens = list(doc)
            if tokens:
                tokenized_corpus.append(tokens)

        # Build vocabulary
        if vocab is not None:
            self.vocab = set(vocab)
        else:
            token_counts: Counter = Counter()
            for sent in tokenized_corpus:
                token_counts.update(sent)
            self.vocab = set(token_counts.keys())

        if self.use_boundary:
            self.vocab.add(self.end_token)
            if self.n > 1:
                self.vocab.add(self.start_token)
        self.vocab.add(self.unk_token)

        # Replace unknown tokens
        sanitized_corpus: List[List[str]] = []
        for sent in tokenized_corpus:
            sanitized = [w if w in self.vocab else self.unk_token for w in sent]
            sanitized_corpus.append(sanitized)

        # Count n-grams and unigrams
        self.unigram_counts: Counter = Counter()
        for sent in sanitized_corpus:
            for token in sent:
                self.unigram_counts[token] += 1
            if self.use_boundary:
                self.unigram_counts[self.end_token] += 1

        self.ngram_counts, self.context_counts = count_ngrams(
            sanitized_corpus,
            self.n,
            use_boundary=self.use_boundary,
            start_token=self.start_token,
            end_token=self.end_token,
        )
        if self.use_boundary:
            self.total_tokens = sum(len(s) + 1 for s in sanitized_corpus)
        else:
            self.total_tokens = sum(len(s) for s in sanitized_corpus)
        self.is_fitted = True
        return self

    @property
    def vocab_size(self) -> int:
        """Return effective vocabulary size |V|."""
        return len(self.vocab)

    def _normalize_context(
        self, context: Union[str, Sequence[str]]
    ) -> Tuple[str, ...]:
        """Convert input context to a normalized tuple of length up to n-1."""
        if self.n == 1:
            return ()

        if isinstance(context, str):
            tokens = tokenize(context)
        else:
            tokens = list(context)

        # Sanitize known tokens
        sanitized = [w if w in self.vocab else self.unk_token for w in tokens]

        # Truncate or pad
        required_len = self.n - 1
        if len(sanitized) < required_len:
            if self.use_boundary:
                padding = [self.start_token] * (required_len - len(sanitized))
                return tuple(padding + sanitized)
            else:
                return tuple(sanitized)
        return tuple(sanitized[-required_len:])

    def probability(
        self, context: Union[str, Sequence[str]], word: str
    ) -> float:
        """Compute conditional probability P(word | context).

        Args:
            context: History tokens or string prefix.
            word: Target continuation word.

        Returns:
            Probability value in [0.0, 1.0].
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted with .fit() before computing probability."
            )

        target_word = word if word in self.vocab else self.unk_token
        ctx = self._normalize_context(context)

        if self.n == 1 or len(ctx) == 0:
            count_w = self.unigram_counts[target_word]
            total_n = self.total_tokens

            if self.smoothing in ("laplace", "add-k"):
                return (count_w + self.k) / (total_n + self.k * self.vocab_size)
            return count_w / total_n if total_n > 0 else 0.0

        target_ngram = ctx + (target_word,)
        count_cw = self.ngram_counts[target_ngram]
        count_c = self.context_counts[ctx]

        if self.smoothing in ("laplace", "add-k"):
            return (count_cw + self.k) / (count_c + self.k * self.vocab_size)

        # Standard MLE
        return count_cw / count_c if count_c > 0 else 0.0

    def log_probability(
        self, context: Union[str, Sequence[str]], word: str
    ) -> float:
        """Compute natural log-probability ln P(word | context)."""
        prob = self.probability(context, word)
        if prob <= 0.0:
            return -float("inf")
        return math.log(prob)

    def sentence_probability(
        self, sentence: Union[str, Sequence[str]], add_boundary: Optional[bool] = None
    ) -> float:
        """Compute the joint probability of a sentence using the chain rule.

        P(S) = prod P(w_t | context_t)
        """
        log_prob = self.sentence_log_probability(
            sentence, add_boundary=add_boundary
        )
        if math.isinf(log_prob) and log_prob < 0:
            return 0.0
        return math.exp(log_prob)

    def sentence_log_probability(
        self, sentence: Union[str, Sequence[str]], add_boundary: Optional[bool] = None
    ) -> float:
        """Compute the log-probability of a sentence to avoid numerical underflow.

        ln P(S) = sum ln P(w_t | context_t)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before scoring sentences.")

        if isinstance(sentence, str):
            tokens = tokenize(sentence)
        else:
            tokens = list(sentence)

        if not tokens:
            return -float("inf")

        if add_boundary is None:
            add_boundary = self.use_boundary

        sanitized = [w if w in self.vocab else self.unk_token for w in tokens]
        if add_boundary:
            tokens_to_eval = sanitized + [self.end_token]
        else:
            tokens_to_eval = sanitized

        total_log_prob = 0.0
        if self.use_boundary and self.n > 1:
            history: List[str] = [self.start_token] * (self.n - 1)
        else:
            history = []

        for token in tokens_to_eval:
            lp = self.log_probability(history, token)
            if math.isinf(lp) and lp < 0:
                return -float("inf")
            total_log_prob += lp
            history.append(token)
            if len(history) > (self.n - 1):
                history = history[-(self.n - 1) :]

        return total_log_prob

    def perplexity(
        self,
        sentences: Sequence[Union[str, Sequence[str]]],
        add_boundary: bool = True,
    ) -> float:
        """Compute perplexity over a test corpus.

        PP(W) = exp(- 1/N * sum ln P(w_i | context_i))

        Returns float('inf') if any transition has zero probability.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted before computing perplexity."
            )

        total_log_prob = 0.0
        total_tokens = 0

        for sent in sentences:
            if isinstance(sent, str):
                tokens = tokenize(sent)
            else:
                tokens = list(sent)
            if not tokens:
                continue

            num_tokens = len(tokens) + (1 if add_boundary else 0)
            log_p = self.sentence_log_probability(sent, add_boundary=add_boundary)

            if math.isinf(log_p) and log_p < 0:
                return float("inf")

            total_log_prob += log_p
            total_tokens += num_tokens

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -total_log_prob / total_tokens
        try:
            return math.exp(cross_entropy)
        except OverflowError:
            return float("inf")

    def next_word_distribution(
        self, context: Union[str, Sequence[str]]
    ) -> Dict[str, float]:
        """Compute the full next-word conditional probability distribution over V."""
        dist = {}
        for word in self.vocab:
            if word == self.start_token:
                continue
            dist[word] = self.probability(context, word)
        return dist

    def predict_next_words(
        self, context: Union[str, Sequence[str]], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict the top-k most likely words to follow the given context."""
        dist = self.next_word_distribution(context)
        # Exclude special boundary tokens from candidate generation
        filtered = {
            w: p
            for w, p in dist.items()
            if w not in (self.start_token, self.unk_token)
        }
        sorted_candidates = sorted(
            filtered.items(), key=lambda item: item[1], reverse=True
        )
        return sorted_candidates[:top_k]

    def score_continuation(
        self,
        context: Union[str, Sequence[str]],
        continuation: Union[str, Sequence[str]],
    ) -> float:
        """Compute conditional log probability ln P(continuation | context)."""
        if isinstance(context, str):
            ctx_tokens = tokenize(context)
        else:
            ctx_tokens = list(context)

        if isinstance(continuation, str):
            cont_tokens = tokenize(continuation)
        else:
            cont_tokens = list(continuation)

        history = list(ctx_tokens)
        total_log_prob = 0.0

        for token in cont_tokens:
            lp = self.log_probability(history, token)
            if math.isinf(lp) and lp < 0:
                return -float("inf")
            total_log_prob += lp
            history.append(token)

        return total_log_prob

    def rank_sentences(
        self,
        context: Union[str, Sequence[str]],
        candidates: Sequence[Union[str, Sequence[str]]],
    ) -> List[Tuple[int, Union[str, Sequence[str]], float]]:
        """Rank candidate continuations given a prefix context based on log probability."""
        scored = []
        for idx, cand in enumerate(candidates):
            score = self.score_continuation(context, cand)
            scored.append((idx, cand, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored


# Functional API requested in Section 14
def train_unigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    smoothing: Optional[str] = None,
    k: float = 1.0,
) -> NGramLanguageModel:
    """Train a Unigram language model."""
    model = NGramLanguageModel(n=1, smoothing=smoothing, k=k)
    model.fit(corpus)
    return model


def train_bigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    smoothing: Optional[str] = None,
    k: float = 1.0,
) -> NGramLanguageModel:
    """Train a Bigram language model."""
    model = NGramLanguageModel(n=2, smoothing=smoothing, k=k)
    model.fit(corpus)
    return model


def train_trigram(
    corpus: Sequence[Union[str, Sequence[str]]],
    smoothing: Optional[str] = None,
    k: float = 1.0,
) -> NGramLanguageModel:
    """Train a Trigram language model."""
    model = NGramLanguageModel(n=3, smoothing=smoothing, k=k)
    model.fit(corpus)
    return model


def probability(
    model: NGramLanguageModel, context: Union[str, Sequence[str]], word: str
) -> float:
    """Functional wrapper for probability calculation."""
    return model.probability(context, word)


def sentence_probability(
    model: NGramLanguageModel, sentence: Union[str, Sequence[str]]
) -> float:
    """Functional wrapper for sentence probability calculation."""
    return model.sentence_probability(sentence)


def sentence_log_probability(
    model: NGramLanguageModel, sentence: Union[str, Sequence[str]]
) -> float:
    """Functional wrapper for sentence log-probability calculation."""
    return model.sentence_log_probability(sentence)
