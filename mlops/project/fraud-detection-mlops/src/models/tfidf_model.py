"""
tfidf_model.py
--------------
TF-IDF bag-of-n-grams embedding baseline.

Each account's token_string (space-separated token sequence) is treated as a
"document". The TF-IDF vectorizer learns a vocabulary of up to max_features
n-grams from the training corpus and represents each account as a dense vector.

Input  : token_string  (str, space-separated)
Output : np.ndarray of shape (n_accounts, max_features), dtype float32
MLP input_dim : max_features (default 5000)
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

VARIANT_NAME = "tfidf"


class TFIDFEmbedder:
    """
    TF-IDF bag-of-n-grams embedder for account token sequences.

    Parameters
    ----------
    max_features : int
        Maximum vocabulary size (number of TF-IDF features).
    ngram_range  : tuple
        (min_n, max_n) for n-gram extraction.
    sublinear_tf : bool
        Apply log(1 + tf) scaling. Helps with the short, repetitive vocabulary.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 3),
        sublinear_tf: bool = True,
    ) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.embedding_dim: Optional[int] = None

    def fit(self, token_strings: List[str]) -> "TFIDFEmbedder":
        """
        Fit TF-IDF vocabulary on training token_strings.

        Parameters
        ----------
        token_strings : list of str
            Each element is one account's token sequence as a single space-
            separated string, e.g. "[CLS] [ACCT] SEG_MICRO [TXN] AMT_HIGH ...".
        """
        logger.info(
            "Fitting TF-IDF (max_features=%d, ngrams=%s) on %d sequences",
            self.max_features, self.ngram_range, len(token_strings),
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            analyzer="word",
            # Use \S+ so tokens like [CLS], [TXN], [SEP] are kept intact.
            # The default pattern strips leading brackets.
            token_pattern=r"\S+",
        )
        self.vectorizer.fit(token_strings)
        self.embedding_dim = len(self.vectorizer.vocabulary_)
        logger.info("TF-IDF vocabulary size: %d", self.embedding_dim)
        return self

    def transform(self, token_strings: List[str]) -> np.ndarray:
        """
        Transform token_strings to dense TF-IDF vectors.

        Returns
        -------
        np.ndarray of shape (n, embedding_dim), dtype float32
        """
        if self.vectorizer is None:
            raise RuntimeError("Call fit() before transform().")
        sparse = self.vectorizer.transform(token_strings)
        return sparse.toarray().astype(np.float32)

    def fit_transform(self, token_strings: List[str]) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        self.fit(token_strings)
        return self.transform(token_strings)

    def save(self, path: "str | Path") -> None:
        """Pickle the entire embedder (vectorizer included)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("TFIDFEmbedder saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path") -> "TFIDFEmbedder":
        """Load a pickled TFIDFEmbedder."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("TFIDFEmbedder loaded from %s", path)
        return obj
