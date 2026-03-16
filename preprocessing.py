"""
Text preprocessing module.

Handles downloading, loading, cleaning, and tokenizing the two source texts:
- Chinese Room Argument (CRA) by Searle (2001)
- Minds, Brains, and Programs (MBP) by Searle (1980)
"""

import re
import string

import gdown
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# Google Drive file IDs for the source texts
CRA_FILE_ID = "1-kk1AfKs6v-LzBJmQSKZGsU83SNOFh3c"
MBP_FILE_ID = "1-kHLLzWxtSe6suWgjVdKL-eQRf778B2f"


def download_text(file_id, output_path):
    """Download a text file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    with open(output_path, "r") as f:
        return f.read()


def remove_punctuation_and_stopwords(text):
    """Remove punctuation and English stopwords from text."""
    text = "".join(char for char in text if char not in string.punctuation)
    text = " ".join(word for word in text.split() if word.lower() not in STOP_WORDS)
    return text


def tokenize(text):
    """Split cleaned text into a list of word tokens."""
    return text.split()


def count_text_stats(text):
    """Count sentences, words, and unique words in raw text.

    Returns:
        tuple: (num_sentences, num_words, num_unique_words)
    """
    sentences = re.split(r"[.!?]", text)
    num_sentences = len([s for s in sentences if s.strip()])
    words = text.split()
    return num_sentences, len(words), len(set(words))


def load_cra():
    """Download and preprocess the Chinese Room Argument text.

    Returns:
        tuple: (raw_text, cleaned_text, tokens)
    """
    raw = download_text(CRA_FILE_ID, "cra.txt")
    raw = raw.replace("\n", " ")
    cleaned = remove_punctuation_and_stopwords(raw)
    tokens = tokenize(cleaned)
    return raw, cleaned, tokens


def load_mbp():
    """Download and preprocess the Minds, Brains, and Programs text.

    Returns:
        tuple: (raw_text, cleaned_text, tokens)
    """
    raw = download_text(MBP_FILE_ID, "mbp.txt")

    # Remove page numbers and author/title artifacts
    for num in range(236, 253):
        raw = raw.replace(str(num), "")
    raw = raw.replace("John R. Searle", "")
    raw = raw.replace("JOHN R. SEARLE", "")
    raw = raw.replace("MINDS, BRAINS, AND PROGRAMS", "")
    raw = raw.replace("\n", " ")

    cleaned = remove_punctuation_and_stopwords(raw)
    tokens = tokenize(cleaned)
    return raw, cleaned, tokens
