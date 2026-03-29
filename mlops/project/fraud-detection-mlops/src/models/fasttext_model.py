"""
fasttext_model.py
-----------------
FastText embedding for account token sequences (mean pooling).

FastText extends Word2Vec by also learning character n-gram sub-embeddings.
For our discrete 34-token vocabulary the subword benefit is marginal, but it
provides (a) a strict fair comparison and (b) true OOV robustness at inference.

Interface is 100% identical to Word2VecEmbedder — train.py treats them the same.

Input  : token_sequence  (List[str], from token_string.split())
Output : np.ndarray of shape (n_accounts, vector_size), dtype float32
MLP input_dim : vector_size (default 128)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

VARIANT_NAME = "fasttext"


class FastTextEmbedder:
    """
    FastText embedder for account token sequences (mean pooling).

    Parameters
    ----------
    vector_size : int    embedding dimension
    window      : int    context window size
    min_count   : int    minimum token frequency
    epochs      : int    training epochs
    seed        : int    random seed
    """

    def __init__(
        self,
        vector_size: int = 128,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 20,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model = None  # gensim FastText
        self.embedding_dim = vector_size

    def fit(self, token_sequences: List[List[str]]) -> "FastTextEmbedder":
        """
        Train FastText on a list of token sequences.

        Parameters
        ----------
        token_sequences : list of list of str
            Each inner list is one account's tokenised sequence.
        """
        from gensim.models import FastText

        logger.info(
            "Training FastText (dim=%d, window=%d, epochs=%d) on %d sequences",
            self.vector_size, self.window, self.epochs, len(token_sequences),
        )
        self.model = FastText(
            sentences=token_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,  # determinism
        )
        logger.info("FastText trained. Vocab size: %d", len(self.model.wv))
        return self

    def _embed_sequence(self, tokens: List[str]) -> np.ndarray:
        """Mean-pool FastText vectors for a single account sequence."""
        # FastText handles OOV via subword; self.model.wv[tok] always succeeds
        vecs = [self.model.wv[tok] for tok in tokens]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def transform(self, token_sequences: List[List[str]]) -> np.ndarray:
        """
        Transform token sequences to mean-pooled FastText embeddings.

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
        """Save the gensim FastText model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("FastTextEmbedder saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path", **kwargs) -> "FastTextEmbedder":
        """Load a saved FastTextEmbedder from a gensim model file."""
        from gensim.models import FastText

        obj = cls(**kwargs)
        obj.model = FastText.load(str(path))
        obj.vector_size = obj.model.vector_size
        obj.embedding_dim = obj.vector_size
        logger.info("FastTextEmbedder loaded from %s", path)
        return obj
