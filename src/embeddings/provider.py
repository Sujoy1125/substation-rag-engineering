"""Pluggable embedding provider abstraction.

Retrieval code depends only on the `EmbeddingProvider` interface, never on
a specific library, so the underlying model can be swapped (local model,
hosted API) without touching retrieval/index code.

`ModelUnavailableError` is raised deterministically (never silently
swallowed into an empty/zero vector) when weights cannot be loaded — a
missing model must fail loudly, not produce fake embeddings.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


class ModelUnavailableError(RuntimeError):
    """Raised when the embedding model's weights cannot be loaded
    (network blocked, not downloaded yet, corrupted cache, etc.).
    Callers must not catch this and substitute fake vectors."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    dimension: int
    device: str
    provider: str  # "local" | "hosted"


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    @property
    @abstractmethod
    def config(self) -> EmbeddingConfig: ...

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Returns an (n_texts, dimension) float32 array."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """Returns a (dimension,) float32 array."""
        ...

    def is_available(self) -> bool:
        """Best-effort check without raising — used for graceful
        degradation / clear error messages upstream."""
        try:
            self.embed_query("availability check")
            return True
        except Exception:
            return False


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Wraps a local sentence-transformers model. Requires the model
    weights to already be present in the local HF cache — this class
    does not silently fall back to anything if they aren't."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        self._model_name = model_name
        self._device = device
        self._model = None  # lazy-loaded so constructing this object never fails
        self._dimension: int | None = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ModelUnavailableError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        try:
            self._model = SentenceTransformer(self._model_name, device=self._device)
            if hasattr(self._model, "get_embedding_dimension"):
                # sentence-transformers renamed this method; prefer the new
                # name when present to avoid the FutureWarning, but fall
                # back to the old one so this still works on any
                # sentence-transformers>=3.0 install per requirements.txt.
                self._dimension = self._model.get_embedding_dimension()
            else:
                self._dimension = self._model.get_sentence_embedding_dimension()
        except Exception as e:
            raise ModelUnavailableError(
                f"Could not load model weights for '{self._model_name}'. "
                f"This typically means the model has not been downloaded to "
                f"the local Hugging Face cache and this environment has no "
                f"network access to huggingface.co. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e

    @property
    def config(self) -> EmbeddingConfig:
        self._ensure_loaded()
        return EmbeddingConfig(
            model_name=self._model_name,
            dimension=self._dimension,
            device=self._device,
            provider="local",
        )

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        return np.asarray(
            self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0],
            dtype=np.float32,
        )


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake embeddings for SOFTWARE UNIT TESTS ONLY.

    Explicitly NOT usable for retrieval-quality benchmarking: vectors are
    a hashed bag-of-words projection with no real semantic content, so
    Recall@K computed against them would be meaningless. Any benchmark
    entry-point in this codebase must refuse to accept this provider —
    enforced by `assert_not_mock()` below, called from benchmark code.
    """

    def __init__(self, dimension: int = 32, seed: int = 0):
        self._dimension = dimension
        self._seed = seed

    @property
    def config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name="mock-hash-embedding",
            dimension=self._dimension,
            device="cpu",
            provider="mock",
        )

    def _hash_vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((self._seed, text))) % (2**32))
        v = rng.standard_normal(self._dimension).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._hash_vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._hash_vec(text)


def assert_not_mock(provider: EmbeddingProvider) -> None:
    if provider.config.provider == "mock":
        raise ValueError(
            "MockEmbeddingProvider produces non-semantic hashed vectors and "
            "must never be used to compute retrieval-quality metrics "
            "(Recall@K, MRR). It is for software-interface unit tests only."
        )
