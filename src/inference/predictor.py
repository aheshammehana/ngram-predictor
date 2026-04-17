import json


class Predictor:
    """
    Performs next-word prediction using a trained NGramModel
    with backoff.
    """

    def __init__(self, model_path, vocab_path, normalizer):
        """
        Initialize the Predictor.

        :param model_path: Path to model.json
        :param vocab_path: Path to vocab.json
        :param normalizer: Normalizer instance
        """
        self.normalizer = normalizer

        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = set(json.load(f))

        # Load model probabilities
        with open(model_path, "r", encoding="utf-8") as f:
            self.model = json.load(f)

    def normalize_input(self, text):
        """
        Normalize user input text using the shared Normalizer.
        """
        return self.normalizer.normalize(text)
    
    def get_context(self, text, ngram_order):
        """
        Extract the last ngram_order - 1 tokens from normalized input.
        """
        normalized = self.normalize_input(text)
        tokens = normalized.split()

        if len(tokens) == 0:
            return []

        return tokens[-(ngram_order - 1):]
    
    def map_oov(self, tokens):
        """
        Replace out-of-vocabulary tokens with <UNK>.
        """
        return [
            token if token in self.vocab else "<UNK>"
            for token in tokens
        ]
    
    def lookup_with_backoff(self, context):
        """
        Perform backoff lookup.
        Try highest-order context first, then back off.
        Returns a dict {word: probability}.
        """

        # Maximum order based on available context
        max_order = len(context) + 1

        for order in range(max_order, 0, -1):
            key = f"{order}gram"

            if key not in self.model:
                continue

            # Unigram case
            if order == 1:
                return self.model["1gram"]

            # Context length for this order
            prefix_length = order - 1
            prefix = context[-prefix_length:]
            prefix_str = " ".join(prefix)

            candidates = {}

            for ngram_str, prob in self.model[key].items():
                ngram_parts = ngram_str.split()

                if " ".join(ngram_parts[:-1]) == prefix_str:
                    next_word = ngram_parts[-1]
                    candidates[next_word] = prob

            if candidates:
                return candidates

        return {}
    
    def predict_next(self, text, k):
        """
        Predict the next k words given input text.
        """

        # Step 1: get context
        context = self.get_context(
            text,
            ngram_order=max(int(key.replace("gram", "")) for key in self.model)
        )

        # Step 2: map OOV tokens
        context = self.map_oov(context)

        # Step 3: backoff lookup
        candidates = self.lookup_with_backoff(context)

        if not candidates:
            return []

        # Step 4: sort by probability (descending)
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: item[1],
            reverse=True
        )

        # Step 5: return top-k words
        return [word for word, _ in sorted_candidates[:k]]


