"""
doc2vec_model.py
----------------
Doc2Vec (PV-DBOW) embedding for account token sequences.

Unlike Word2Vec which embeds individual tokens and mean-pools them,
Doc2Vec learns a direct account-level vector (the "paragraph vector")
alongside the token vectors. This tests whether mean-pooling throws
away useful account-level information.

PV-DBOW (Distributed Bag of Words) is used:
  - Predicts context tokens from the document vector alone (no word order)
  - Simpler, faster, and often outperforms PV-DM for classification tasks
  - Analogous to skip-gram Word2Vec but at document level

Input  : token_sequence  (List[str], from token_string.split())
Output : np.ndarray of shape (n_accounts, vector_size), dtype float32
MLP input_dim : vector_size (default 128)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

VARIANT_NAME = "doc2vec"


class Doc2VecEmbedder:
    """
    Doc2Vec PV-DBOW embedder for account token sequences.

    Each account sequence is tagged with a unique document ID and a direct
    document vector is learned — no mean-pooling of token vectors needed.

    Parameters
    ----------
    vector_size : int    embedding dimension
    window      : int    context window size (used for token vectors in DBOW)
    min_count   : int    minimum token frequency (1 = keep all vocab tokens)
    epochs      : int    training epochs
    seed        : int    for reproducibility (also sets workers=1)
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
        self.model = None  # gensim Doc2Vec
        self.embedding_dim = vector_size

    def fit(self, token_sequences: List[List[str]]) -> "Doc2VecEmbedder":
        """
        Train Doc2Vec on a list of token sequences.

        Parameters
        ----------
        token_sequences : list of list of str
            Each inner list is one account's tokenised sequence.
        """
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        logger.info(
            "Training Doc2Vec PV-DBOW (dim=%d, window=%d, epochs=%d) on %d sequences",
            self.vector_size, self.window, self.epochs, len(token_sequences),
        )
        tagged_docs = [
            TaggedDocument(words=seq, tags=[i])
            for i, seq in enumerate(token_sequences)
        ]
        self.model = Doc2Vec(
            documents=tagged_docs,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=0,        # PV-DBOW (dm=0) vs PV-DM (dm=1)
            seed=self.seed,
            workers=1,   # determinism
        )
        logger.info("Doc2Vec trained. Vocab size: %d", len(self.model.wv))
        return self

    def transform(self, token_sequences: List[List[str]]) -> np.ndarray:
        """
        Infer document vectors for new sequences.

        For training sequences the stored doc vectors are returned directly
        (exact). For unseen sequences (val/test), inference is run via
        model.infer_vector() which is deterministic with a fixed seed.

        Returns
        -------
        np.ndarray of shape (n, vector_size), dtype float32
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform().")

        n = len(token_sequences)
        embeddings = np.empty((n, self.vector_size), dtype=np.float32)
        for i, seq in enumerate(token_sequences):
            embeddings[i] = self.model.infer_vector(seq, epochs=self.epochs)
        return embeddings

    def transform_train(self, n_train: int) -> np.ndarray:
        """
        Return the stored training document vectors without re-inference.

        Faster than transform() for the training split because the doc vectors
        were already learned during fit().

        Parameters
        ----------
        n_train : int
            Number of training documents (must match fit() call).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before transform_train().")
        return np.array(
            [self.model.dv[i] for i in range(n_train)], dtype=np.float32
        )

    def fit_transform(self, token_sequences: List[List[str]]) -> np.ndarray:
        """Fit and return stored training vectors (no re-inference needed)."""
        self.fit(token_sequences)
        return self.transform_train(len(token_sequences))

    def save(self, path: "str | Path") -> None:
        """Save the gensim Doc2Vec model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info("Doc2VecEmbedder saved → %s", path)

    @classmethod
    def load(cls, path: "str | Path", **kwargs) -> "Doc2VecEmbedder":
        """Load a saved Doc2VecEmbedder from a gensim model file."""
        from gensim.models.doc2vec import Doc2Vec

        obj = cls(**kwargs)
        obj.model = Doc2Vec.load(str(path))
        obj.vector_size = obj.model.vector_size
        obj.embedding_dim = obj.vector_size
        logger.info("Doc2VecEmbedder loaded from %s", path)
        return obj
