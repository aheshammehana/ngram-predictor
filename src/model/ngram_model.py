import json
from collections import Counter, defaultdict


class NGramModel:
    """
    Builds and stores n-gram language model probabilities
    with backoff from highest order to unigrams.
    """

    def __init__(self, ngram_order, unk_threshold):
        """
        Initialize the model.
        """
        self.ngram_order = ngram_order
        self.unk_threshold = unk_threshold

        self.vocab = set()
        self.ngram_counts = defaultdict(Counter)
        self.probabilities = dict()

    def build_vocab(self, token_file):
        """
        Build vocabulary from tokenized training file.
        Words that appear fewer times than unk_threshold
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

    def save_vocab(self, vocab_path):
        """
        Save vocabulary to a JSON file.
        """
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(sorted(self.vocab), f, indent=2)

    def build_ngram_counts(self, token_file):
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
                        ngram = tuple(tokens[i:i + n])
                        self.ngram_counts[n][ngram] += 1


def main():
    print("Running ngram_model.py")

    model = NGramModel(ngram_order=4, unk_threshold=3)
    model.build_vocab("data/processed/train_tokens.txt")
    model.build_ngram_counts("data/processed/train_tokens.txt")

    print("Unigram count size:", len(model.ngram_counts[1]))
    print("Bigram count size:", len(model.ngram_counts[2]))
    print("Trigram count size:", len(model.ngram_counts[3]))


if __name__ == "__main__":
    main()