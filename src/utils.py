import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from unidecode import unidecode


PATH_STOPWORDS_FILE = "data/utils/stopwords.txt"

def text_preprocessing(documents: List[str]) -> List[str]:
    """Achieves a list of preprocessing tasks to prepare text for the natural language model.

    Args:
        documents (List[str]): List of documents (strings) to process?

    Returns:
        List[str]: List of processed documents (strings).
    """
    # Load stopwords if exists, otherwise create it
    if os.path.exists(PATH_STOPWORDS_FILE):
        with open(PATH_STOPWORDS_FILE, "r") as f:
            stop_words = set(word for word in f.read().split("\n"))
    else:
        os.makedirs(os.path.dirname(PATH_STOPWORDS_FILE), exist_ok=True)
        stop_words = set(unidecode(word.lower()) for word in set(stopwords.words()))
        with open(PATH_STOPWORDS_FILE, "w") as f:
            f.write("\n".join(stop_words))

    # Preprocess documents
    lemm = WordNetLemmatizer()
    processed_documents = []
    for document in documents:
        document = unidecode(document.lower())

        tokens = []
        for token in word_tokenize(document):
            # Handle compound words
            tokens.extend(token.split("-"))

        processed_tokens = []
        for token in tokens:
            if token not in stop_words and token.isalnum():
                processed_tokens.append(lemm.lemmatize(token.strip()))

        processed_documents.append(" ".join(processed_tokens))

    return processed_documents


if __name__ == "__main__":
    documents = [
        "I am a data scientist",
        "I am a data engineer",
        "I am a software engineer",
        "I am a data analyst",
        "I am a software developer",
    ]
    processed_documents = text_preprocessing(documents)
    print(processed_documents)
