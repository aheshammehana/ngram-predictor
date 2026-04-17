import argparse
import os

from dotenv import load_dotenv
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor
from src.data_prep.normalizer import Normalizer


def run_dataprep():
    """
    Run data preparation:
    - Load raw text files
    - Normalize text
    - Sentence tokenize
    - Word tokenize
    - Save train_tokens.txt
    """

    # Read paths from .env
    train_raw_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens_path = os.getenv("TRAIN_TOKENS")

    if train_raw_dir is None or train_tokens_path is None:
        raise RuntimeError(
            "Environment variables not loaded. Check config/.env"
        )

    normalizer = Normalizer()

    # 1. Load raw text files
    raw_texts = normalizer.load(train_raw_dir)

    all_sentences = []

    for text in raw_texts:
        text = normalizer.strip_gutenberg(text)

        sentences = normalizer.sentence_tokenize(text)

        for sentence in sentences:
            sentence = normalizer.normalize(sentence)
            tokens = normalizer.word_tokenize(sentence)
            if tokens:
                all_sentences.append(tokens)

        # 5. Word tokenize each sentence
        for sentence in sentences:
            tokens = normalizer.word_tokenize(sentence)
            if tokens:
                all_sentences.append(tokens)

    # 6. Save tokenized sentences
    normalizer.save(all_sentences, train_tokens_path)

def run_model():
    model = NGramModel(
        ngram_order=int(os.getenv("NGRAM_ORDER")),
        unk_threshold=int(os.getenv("UNK_THRESHOLD"))
    )

    token_file = os.getenv("TRAIN_TOKENS")
    vocab_path = os.getenv("VOCAB")
    model_path = os.getenv("MODEL")

    model.build_vocab(token_file)
    model.save_vocab(vocab_path)

    model.build_ngram_counts(token_file)
    model.build_probabilities()
    model.save_model(model_path)

    print("Model and vocabulary saved.")

def main():
    # Load environment variables
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["dataprep", "model", "inference"],
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "dataprep":
        run_dataprep()
    elif args.step == "model":
        run_model()
    elif args.step == "inference":
        run_inference()    


def run_inference():
    normalizer = Normalizer()

    predictor = Predictor(
        model_path=os.getenv("MODEL"),
        vocab_path=os.getenv("VOCAB"),
        normalizer=normalizer
    )

    k = int(os.getenv("TOP_K"))

    print("Type text (or 'quit' to exit):")
    while True:
        text = input("> ").strip()
        if text.lower() == "quit":
            break

        predictions = predictor.predict_next(text, k)
        print("Predictions:", predictions)


if __name__ == "__main__":
    main()