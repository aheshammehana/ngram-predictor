import json
from collections import Counter, defaultdict


class NGramModel:
    """
    Builds and stores n-gram language model probabilities
    with backoff from highest order to unigrams.
    """

    def __init__(self, ngram_order: int, unk_threshold: int):
        """
        Initialize the model.

        :param ngram_order: Maximum n-gram order (e.g. 4)
        :param unk_threshold: Minimum frequency for a word to stay in vocab
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold

        self.vocab = set()
        self.ngram_counts = defaultdict(Counter)
        self.probabilities = {}

    # -------------------------------------------------
    # Vocabulary
    # -------------------------------------------------

    def build_vocab(self, token_file: str):
        """
        Build vocabulary from tokenized training file.
        Words appearing fewer times than unk_threshold
        are replaced by <UNK>.
        """
        word_counts = Counter()

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                word_counts.update(tokens)

        self.vocab = set()

        for word, count in word_counts.items():
            if count >= self.unk_threshold:
                self.vocab.add(word)

        self.vocab.add("<UNK>")

    def save_vocab(self, vocab_path: str):
        """
        Save vocabulary to vocab.json.
        """
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab), f, indent=2)

    # -------------------------------------------------
    # N-gram counts
    # -------------------------------------------------

    def build_ngram_counts(self, token_file: str):
        """
        Build n-gram counts for all orders from 1 to ngram_order.
        """
        self.ngram_counts = defaultdict(Counter)

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()

                # Replace OOV words with <UNK>
                tokens = [
                    token if token in self.vocab else "<UNK>"
                    for token in tokens
                ]

                for n in range(1, self.ngram_order + 1):
                    if len(tokens) < n:
                        continue

                    for i in range(len(tokens) - n + 1):
                        ngram = tuple(tokens[i : i + n])
                        self.ngram_counts[n][ngram] += 1

    # -------------------------------------------------
    # Probabilities (MLE)
    # -------------------------------------------------

    def build_probabilities(self):
        """
        Compute Maximum Likelihood Estimation (MLE)
        probabilities from n-gram counts.
        """
        self.probabilities = {}

        # Unigrams
        total = sum(self.ngram_counts[1].values())
        self.probabilities[1] = {
            ngram: count / total
            for ngram, count in self.ngram_counts[1].items()
        }

        # Higher-order n-grams
        for n in range(2, self.ngram_order + 1):
            self.probabilities[n] = {}

            for ngram, count in self.ngram_counts[n].items():
                prefix = ngram[:-1]
                prefix_count = self.ngram_counts[n - 1][prefix]

                if prefix_count > 0:
                    self.probabilities[n][ngram] = count / prefix_count

    # -------------------------------------------------
    # Serialization
    # -------------------------------------------------

    def save_model(self, model_path: str):
        """
        Save probability tables to model.json.
        """
        output = {}

        for n, probs in self.probabilities.items():
            output[f"{n}gram"] = {
                " ".join(ngram): prob
                for ngram, prob in probs.items()
            }

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)