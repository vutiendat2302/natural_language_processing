"""Script to run all Lab 02 NLP experiments, compute statistics, generate plots,

and save results.csv and experiments.ipynb.

Author: Vu Tien Dat (MSSV: 23000111)
"""

from collections import Counter
import json
import math
import os
import random
import re
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ngram_lm import NGramLanguageModel, tokenize

# Fix random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("EXPERIMENT 1: CORPUS STATISTICS")
print("=" * 60)

DATA_PATH = "30K.json"
MAX_DOCS = 10000

t0 = time.time()
documents = []
sentences = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i in range(MAX_DOCS):
        line = f.readline()
        if not line:
            break
        data = json.loads(line)
        text = data.get("text", "")
        documents.append(text)
        # Sentence splitting by newlines and sentence punctuation
        raw_sents = re.split(r"[\n.?!]+", text)
        for s in raw_sents:
            tokens = tokenize(s)
            if len(tokens) >= 2:
                sentences.append(tokens)

print(
    f"Loaded {len(documents):,} documents, extracted {len(sentences):,} sentences in {time.time() - t0:.2f}s"
)

# Extract N-grams and counts
total_tokens = sum(len(s) for s in sentences)
unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()

for s in sentences:
    unigram_counts.update(s)
    for j in range(len(s) - 1):
        bigram_counts[(s[j], s[j + 1])] += 1
    for j in range(len(s) - 2):
        trigram_counts[(s[j], s[j + 1], s[j + 2])] += 1

vocab_size = len(unigram_counts)
num_unigrams = len(unigram_counts)
num_bigrams = len(bigram_counts)
num_trigrams = len(trigram_counts)

hapax_unigram = sum(1 for c in unigram_counts.values() if c == 1)
hapax_bigram = sum(1 for c in bigram_counts.values() if c == 1)
hapax_trigram = sum(1 for c in trigram_counts.values() if c == 1)

corpus_stats = {
    "Metric": [
        "Total Documents",
        "Total Sentences",
        "Total Word Tokens",
        "Vocabulary Size |V|",
        "Unique Unigrams",
        "Hapax Unigrams (count=1)",
        "Hapax Unigram Ratio (%)",
        "Unique Bigrams",
        "Hapax Bigrams (count=1)",
        "Hapax Bigram Ratio (%)",
        "Unique Trigrams",
        "Hapax Trigrams (count=1)",
        "Hapax Trigram Ratio (%)",
    ],
    "Value": [
        len(documents),
        len(sentences),
        total_tokens,
        vocab_size,
        num_unigrams,
        hapax_unigram,
        f"{hapax_unigram / num_unigrams * 100:.2f}%",
        num_bigrams,
        hapax_bigram,
        f"{hapax_bigram / num_bigrams * 100:.2f}%",
        num_trigrams,
        hapax_trigram,
        f"{hapax_trigram / num_trigrams * 100:.2f}%",
    ],
}
df_corpus = pd.DataFrame(corpus_stats)
print(df_corpus.to_string(index=False))

# Plot frequency distribution (Zipf log-log plot)
fig, ax = plt.subplots(figsize=(8, 5))
ranks_uni = np.arange(1, min(5000, len(unigram_counts)) + 1)
counts_uni = [c for _, c in unigram_counts.most_common(len(ranks_uni))]

ranks_bi = np.arange(1, min(5000, len(bigram_counts)) + 1)
counts_bi = [c for _, c in bigram_counts.most_common(len(ranks_bi))]

ranks_tri = np.arange(1, min(5000, len(trigram_counts)) + 1)
counts_tri = [c for _, c in trigram_counts.most_common(len(ranks_tri))]

ax.loglog(ranks_uni, counts_uni, label="Unigrams", color="#1f77b4", linewidth=2)
ax.loglog(ranks_bi, counts_bi, label="Bigrams", color="#ff7f0e", linewidth=2)
ax.loglog(ranks_tri, counts_tri, label="Trigrams", color="#2ca02c", linewidth=2)

ax.set_title("N-gram Frequency Distribution (Log-Log Zipf Plot)", fontsize=13)
ax.set_xlabel("Log Rank", fontsize=11)
ax.set_ylabel("Log Frequency", fontsize=11)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig("ngram_distribution.png", dpi=200)
print("Saved ngram_distribution.png")

print("\n" + "=" * 60)
print("EXPERIMENT 2 & 3: MODEL TRAINING & PERPLEXITY EVALUATION")
print("=" * 60)

# Split Train (80%), Val (10%), Test (10%)
random.shuffle(sentences)
# Sample 20,000 sentences for balanced training & evaluation
SAMPLE_SIZE = min(25000, len(sentences))
sampled_sentences = sentences[:SAMPLE_SIZE]

n_tr = int(len(sampled_sentences) * 0.8)
n_va = int(len(sampled_sentences) * 0.1)
train_data = sampled_sentences[:n_tr]
val_data = sampled_sentences[n_tr : n_tr + n_va]
test_data = sampled_sentences[n_tr + n_va :]

print(
    f"Splits -> Train: {len(train_data):,} sents, Val: {len(val_data):,} sents, Test: {len(test_data):,} sents"
)

# Benchmark subset of train for perplexity evaluation to avoid long execution
eval_train = train_data[:1000]

models = {
    "Unigram MLE": NGramLanguageModel(n=1, smoothing=None),
    "Unigram Laplace": NGramLanguageModel(n=1, smoothing="laplace", k=1.0),
    "Bigram MLE": NGramLanguageModel(n=2, smoothing=None),
    "Bigram Laplace": NGramLanguageModel(n=2, smoothing="laplace", k=1.0),
    "Trigram MLE": NGramLanguageModel(n=3, smoothing=None),
    "Trigram Laplace": NGramLanguageModel(n=3, smoothing="laplace", k=1.0),
}

results = []
for name, model in models.items():
    t_start = time.time()
    model.fit(train_data)
    fit_time = time.time() - t_start

    tr_ppl = model.perplexity(eval_train)
    va_ppl = model.perplexity(val_data)
    te_ppl = model.perplexity(test_data)

    results.append(
        {
            "Model": name,
            "Train PPL": (
                round(tr_ppl, 2) if not math.isinf(tr_ppl) else "inf"
            ),
            "Validation PPL": (
                round(va_ppl, 2) if not math.isinf(va_ppl) else "inf"
            ),
            "Test PPL": round(te_ppl, 2) if not math.isinf(te_ppl) else "inf",
            "Fit Time (s)": round(fit_time, 2),
        }
    )
    print(
        f"{name:16s} | Train: {str(results[-1]['Train PPL']):>10} | Valid: {str(results[-1]['Validation PPL']):>10} | Test: {str(results[-1]['Test PPL']):>10} | Time: {fit_time:.2f}s"
    )

df_ppl = pd.DataFrame(results)

print("\n" + "=" * 60)
print("EXPERIMENT 4: NEXT-WORD PREDICTION (Mục 20)")
print("=" * 60)

test_contexts = [
    "in the",
    "you can",
    "the cat",
    "machine learning",
    "natural language",
]

eval_model = models["Bigram Laplace"]
prediction_records = []

for ctx in test_contexts:
    top_preds = eval_model.predict_next_words(ctx, top_k=3)
    preds_str = ", ".join([f"{w} ({p:.4f})" for w, p in top_preds])

    # Find actual next words in corpus if present
    ctx_tokens = tokenize(ctx)
    next_words_actual = Counter()
    for s in sentences:
        for idx in range(len(s) - len(ctx_tokens)):
            if s[idx : idx + len(ctx_tokens)] == ctx_tokens:
                next_words_actual[s[idx + len(ctx_tokens)]] += 1
    actual_top = (
        ", ".join([f"{w} ({c})" for w, c in next_words_actual.most_common(2)])
        if next_words_actual
        else "None observed"
    )

    prediction_records.append(
        {
            "Context": ctx,
            "Predictions (Top 3 with P)": preds_str,
            "Actual In-Corpus Continuations": actual_top,
        }
    )
    print(f"Context: '{ctx}'")
    print(f"  Predicted: {preds_str}")
    print(f"  In-Corpus: {actual_top}")

df_predictions = pd.DataFrame(prediction_records)

print("\n" + "=" * 60)
print("EXPERIMENT 5: CANDIDATE SENTENCE RANKING (Mục 21)")
print("=" * 60)

ranking_context = "machine learning"
candidates = [
    "is useful for nlp",
    "banana computer quickly",
    "studies language models",
]

ranking_results = eval_model.rank_sentences(ranking_context, candidates)
ranking_records = []
for rank, (idx, cand, score) in enumerate(ranking_results, 1):
    ranking_records.append(
        {
            "Rank": rank,
            "Context": ranking_context,
            "Candidate Continuation": cand,
            "Conditional Log Probability": (
                round(score, 2) if not math.isinf(score) else "-inf"
            ),
        }
    )
    print(f"Rank {rank}: '{cand}' (Log-prob: {score:.2f})")

df_rankings = pd.DataFrame(ranking_records)

# Write results.csv
with open("results.csv", "w", encoding="utf-8") as f:
    f.write("# PART 1: CORPUS STATISTICS\n")
    df_corpus.to_csv(f, index=False)
    f.write("\n# PART 2: PERPLEXITY EVALUATION ACROSS N-GRAM MODELS\n")
    df_ppl.to_csv(f, index=False)
    f.write("\n# PART 3: NEXT-WORD PREDICTION BENCHMARK\n")
    df_predictions.to_csv(f, index=False)
    f.write("\n# PART 4: SENTENCE RANKING CANDIDATE CONTINUATIONS\n")
    df_rankings.to_csv(f, index=False)

print("\nWrote results.csv successfully!")
