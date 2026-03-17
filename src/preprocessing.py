"""
Fake News Detection - Text Preprocessing Module

This module contains functions for preprocessing text data using NLP techniques
including tokenization, lemmatization, stopword removal, and punctuation removal.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    """
    A class for preprocessing text data for fake news detection.
    """

    def __init__(self):
        """
        Initialize the TextPreprocessor with necessary NLP tools.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # IMPORTANT: Be careful with custom stopwords removal
        # Some words that might indicate fake news (exclusive, shocking, breaking, etc.) should be kept
        # Only remove truly generic words that don't discriminate between real and fake news
        custom_stopwords = ['would', 'could', 'also', 'one', 'two', 'three',
                           'first', 'second', 'third', 'old', 'very', 'much']
        self.stop_words.update(custom_stopwords)

    def to_lowercase(self, text):
        """
        Convert text to lowercase.

        Args:
            text (str): Input text

        Returns:
            str: Text converted to lowercase
        """
        return text.lower()

    def remove_punctuation(self, text):
        """
        Remove punctuation from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with punctuation removed
        """
        # Remove punctuation using regex
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def remove_stopwords(self, text):
        """
        Remove stopwords from text.

        Args:
            text (str): Input text

        Returns:
            str: Text with stopwords removed
        """
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(filtered_tokens)

    def lemmatize_text(self, text):
        """
        Lemmatize words in the text.

        Args:
            text (str): Input text

        Returns:
            str: Text with lemmatized words
        """
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(lemmatized_tokens)

    def preprocess_text(self, text):
        """
        Apply complete preprocessing pipeline to text.

        Args:
            text (str): Input text

        Returns:
            str: Fully preprocessed text
        """
        if not isinstance(text, str):
            return ""

        # Apply preprocessing steps in sequence
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)

        return text

    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.

        Args:
            texts (list): List of text strings

        Returns:
            list: List of preprocessed text strings
        """
        return [self.preprocess_text(text) for text in texts]