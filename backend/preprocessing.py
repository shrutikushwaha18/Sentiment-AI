"""
preprocessing.py
-----------------
Shared text preprocessing pipeline used identically at training time and at
inference time, so the model always sees text prepared the same way.

Pipeline: lowercase -> strip punctuation/special chars -> collapse whitespace
          -> remove stopword-ish noise -> tokenize -> stem
"""

import re

# A small, curated stopword list. We keep negations ("not", "no", "never")
# because they flip sentiment and are important signal for the model.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "to", "of", "in", "on", "at", "for", "with", "as",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "we", "they", "them", "his", "her", "their", "our", "your", "my",
    "so", "just", "than", "then", "there", "here", "do", "does", "did",
    "have", "has", "had", "am", "will", "would", "can", "could", "should",
    "im", "ive", "youre",
}

SUFFIX_RULES = [
    ("ational", "ate"), ("tional", "tion"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("ization", "ize"),
    ("ational", "ate"), ("ingly", ""), ("edly", ""),
    ("ing", ""), ("edly", ""), ("ed", ""), ("ies", "y"),
    ("ied", "y"), ("ies", "y"), ("es", ""), ("s", ""),
    ("ly", ""), ("ful", ""), ("ness", ""), ("ment", ""),
]


def simple_stem(word: str) -> str:
    """A lightweight, dependency-free suffix-stripping stemmer.

    This is not a full Porter/Snowball implementation, but it is
    deterministic and consistent between training and inference, which is
    what actually matters for the model's vocabulary to line up.
    """
    if len(word) <= 3:
        return word
    for suffix, replacement in SUFFIX_RULES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/special characters, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list:
    """Split cleaned text into tokens."""
    if not text:
        return []
    return text.split(" ")


def remove_noise(tokens: list) -> list:
    """Drop stopwords/noise tokens, keep negations."""
    return [t for t in tokens if t and t not in STOPWORDS]


def stem_tokens(tokens: list) -> list:
    return [simple_stem(t) for t in tokens]


def preprocess(text: str, keep_stopwords: bool = False) -> list:
    """Full pipeline: clean -> tokenize -> (optionally remove noise) -> stem.

    Returns a list of stemmed tokens ready for vocabulary lookup.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    if not keep_stopwords:
        tokens = remove_noise(tokens)
    tokens = stem_tokens(tokens)
    return tokens
