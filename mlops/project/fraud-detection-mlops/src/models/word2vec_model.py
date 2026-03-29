"""
word2vec_model.py
-----------------
Word2Vec (skip-gram) embedding for account token sequences.

Each transaction token is embedded by Word2Vec. An account's embedding is
the mean-pool of all token vectors in its sequence — including structural
tokens ([CLS], [ACCT], [TXN]) which learn positional/role semantics.

Input  : token_sequence  (List[str], from token_string.split())
Output : np.ndarray of shape (n_accounts, vector_size), dtype float32
MLP input_dim : vector_size (default 128)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

VARIANT_NAME = "word2vec"


class Word2VecEmbedder:
    """
    Word2Vec skip-gram embedder for account token sequences (mean pooling).

    Parameters
    ----------
    vector_size : int    embedding dimension
    window      : int    context window size
    min_count   : int    minimum token frequency (1 = keep all 34 vocab tokens)
    epochs      : int    training epochs
    sg          : int    1 = skip-gram, 0 = CBOW
    seed        : int    for reproducibility (also sets workers=1)
    """

    def __init__(
        self,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 20,
        sg: int = 1,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.seed = seed
        self.model = None  # gensim Word2Vec
        self.embedding_dim = vector_size

    def fit(self, token_sequences: List[List[str]]) -> "Word2VecEmbedder":
        """
        Train Word2Vec on a list of token sequences.

        Parameters
        ----------
        token_sequences : list of list of str
            Each inner list is one account's tokenised sequence.
        """
        from gensim.models import Word2Vec

        logger.info(
            "Training Word2Vec (sg=%d, dim=%d, window=%d, epochs=%d) on %d sequences",
            self.sg, self.vector_size, self.window, self.epochs, len(token_sequences),
        )
        self.model = Word2Vec(
            sentences=token_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,  # workers=1 ensures determinism with a fixed seed
        )
        logger.info("Word2Vec trained. Vocab size: %d", len(self.model.wv))
        return self

    def _embed_sequence(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool token vectors for a single account sequence."""
        wv = self.model.wv
        vecs = [wv[tok] for tok in tokens if tok in wv]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def transform(self, token_sequences: List[List[str]]) -> np.ndarray:
        """
        Transform token sequences to mean-pooled Word2Vec embeddings.

        Returns
        -------
        np.ndarray of shape (n, vector_size), dtype float32
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")
        return np.array([self._embed_sequence(seq) for seq in token_sequences], dtype=np.float32)

    def fit_transform(self, token_sequences: List[List[str]]) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        self.fit(token_sequences)
        return self.transform(token_sequences)

    def save(self, path: "str | Path") -> None:
        """Save the gensim Word2Vec model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Word2VecEmbedder saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path", **kwargs) -> "Word2VecEmbedder":
        """Load a saved Word2VecEmbedder from a gensim model file."""
        from gensim.models import Word2Vec

        obj = cls(**kwargs)
        obj.model = Word2Vec.load(str(path))
        obj.vector_size = obj.model.vector_size
        obj.embedding_dim = obj.vector_size
        logger.info("Word2VecEmbedder loaded from %s", path)
        return obj
