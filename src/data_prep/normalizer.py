import os 

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
        pass

    def lowercase(self, text):
        """
        Convert text to lowercase.
        """
        pass

    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        """
        pass

    def remove_numbers(self, text):
        """
        Remove numbers from text.
        """
        pass

    def remove_whitespace(self, text):
        """
        Remove extra whitespace and blank lines.
        """
        pass

    def normalize(self, text):
        """
        Apply all normalization steps in order:
        lowercase → remove punctuation → remove numbers → remove whitespace.
        """
        pass

    def sentence_tokenize(self, text):
        """
        Split text into a list of sentences.
        """
        pass

    def word_tokenize(self, sentence):
        """
        Split a sentence into tokens (words).
        """
        pass

    def save(self, sentences, filepath):
        """
        Save tokenized sentences to a file.
        """
        pass


def main():
    print("Normalizer module")


if __name__ == "__main__":
    main()