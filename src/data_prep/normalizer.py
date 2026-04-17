import os 
import string
import re

from matplotlib import text

class Normalizer:
    """
    Responsible for loading, cleaning, normalizing, tokenizing,
    and saving text data for the n-gram language model.
    """

    def load(self, folder_path):
        """
        Load all .txt files from a folder.
        """
        
        texts = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    texts.append(f.read())

        return "\n".join(texts)


    def strip_gutenberg(self, text):
        """
        Remove Project Gutenberg header and footer from raw text.
        """
        
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

        start_index = text.find(start_marker)
        end_index = text.find(end_marker)

        if start_index != -1:
            text = text[start_index + len(start_marker):]

        if end_index != -1:
            text = text[:end_index]

        return text.strip()


    def lowercase(self, text):
        """
        Convert text to lowercase.
        """
        return text.lower()

    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        """
        return text.translate(str.maketrans("", "", string.punctuation))

    def remove_numbers(self, text):
        """
        Remove numbers from text.
        """
        return ''.join(char for char in text if not char.isdigit())

    def remove_whitespace(self, text):
        """
        Remove extra whitespace and blank lines.
        """
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def normalize(self, text):
        """
        Apply all normalization steps in order:
        lowercase → remove punctuation → remove numbers → remove whitespace.
        """
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

    def sentence_tokenize(self, text):
        """
        Split text into a list of sentences.
        """
        sentences = text.split("\n")
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def word_tokenize(self, sentence):
        """
        Split a sentence into tokens (words).
        """
        return sentence.split()

    def save(self, sentences, filepath):
        """
        Save tokenized sentences to a file.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")


def main():
    print("Normalizer module")


if __name__ == "__main__":
    main()