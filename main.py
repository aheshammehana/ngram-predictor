import argparse
import os

from dotenv import load_dotenv
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


def main():
    # Load environment variables
    load_dotenv("config/.env")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["dataprep"],
        help="Pipeline step to run"
    )
    args = parser.parse_args()

    if args.step == "dataprep":
        run_dataprep()


if __name__ == "__main__":
    main()