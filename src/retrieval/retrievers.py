"""Retriever implementations.

NOTE ON SCOPE: this environment's network egress allowlist does not include
huggingface.co or api.openai.com (both return 403 from the sandbox egress
proxy), so no embedding model — hosted or local-downloaded — can currently
be reached. Only lexical (non-embedding) retrieval can be exercised here.
This module therefore implements two lexical baselines (BM25, TF-IDF
cosine) and a pluggable interface so a dense retriever can be dropped in
later without touching the benchmark harness. Dense-retrieval numbers are
NOT fabricated or substituted with a lexical stand-in.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.common.chunk import Chunk
from src.embeddings.provider import EmbeddingProvider, ModelUnavailableError, assert_not_mock

_TOKEN_RE = re.compile(r"[A-Za-z0-9°%/.]+")


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedResult:
    chunk: Chunk
    score: float
    rank: int


class Retriever(ABC):
    name: str

    @abstractmethod
    def index(self, chunks: List[Chunk]) -> None: ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]: ...


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        corpus = [tokenize(c.searchable_text()) for c in chunks]
        self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        assert self._bm25 is not None, "call index() first"
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            RetrievedResult(chunk=self._chunks[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(ranked)
        ]


class TfidfRetriever(Retriever):
    """Lexical baseline using TF-IDF + cosine similarity. Not a substitute
    for semantic/dense retrieval — included as a second, differently-biased
    lexical method for comparison against BM25."""

    name = "tfidf"

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
        self._matrix = None

    def index(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        texts = [c.searchable_text() for c in chunks]
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        assert self._matrix is not None, "call index() first"
        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix)[0]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [
            RetrievedResult(chunk=self._chunks[i], score=float(sims[i]), rank=rank + 1)
            for rank, i in enumerate(ranked)
        ]


class DenseRetriever(Retriever):
    """Cosine-similarity retrieval over an EmbeddingProvider's vectors.

    Refuses to index/retrieve with a mock provider (see assert_not_mock) —
    this class is only ever wired to a real embedding model for benchmark
    purposes. Raises ModelUnavailableError at index() time if the
    underlying model's weights can't be loaded, rather than silently
    producing degenerate results.
    """

    name = "dense"

    def __init__(self, provider: EmbeddingProvider, allow_mock: bool = False):
        self._provider = provider
        self._allow_mock = allow_mock
        self._chunks: List[Chunk] = []
        self._matrix: "np.ndarray | None" = None

    def index(self, chunks: List[Chunk]) -> None:
        if not self._allow_mock:
            assert_not_mock(self._provider)
        self._chunks = chunks
        texts = [c.searchable_text() for c in chunks]
        self._matrix = self._provider.embed_documents(texts)  # raises ModelUnavailableError if blocked

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        assert self._matrix is not None, "call index() first"
        qvec = self._provider.embed_query(query)
        sims = self._matrix @ qvec  # vectors are pre-normalized -> dot product == cosine
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [
            RetrievedResult(chunk=self._chunks[i], score=float(sims[i]), rank=rank + 1)
            for rank, i in enumerate(ranked)
        ]


class HybridRRFRetriever(Retriever):
    """Combines two retrievers via Reciprocal Rank Fusion — chosen over
    raw score addition because BM25 and cosine-similarity scores live on
    incomparable scales; RRF only uses each retriever's rank ordering,
    which sidesteps that normalization problem entirely.

    RRF score for a chunk = sum over retrievers of 1 / (k_const + rank).
    """

    def __init__(self, retrievers: List[Retriever], k_const: int = 60, name: str = "hybrid_rrf"):
        assert len(retrievers) >= 2
        self._retrievers = retrievers
        self._k_const = k_const
        self.name = name

    def index(self, chunks: List[Chunk]) -> None:
        for r in self._retrievers:
            r.index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        # Pull a wider candidate pool from each retriever than top_k so
        # fusion has enough material to reorder from.
        pool_k = max(top_k * 3, 20)
        chunk_by_id: dict[str, Chunk] = {}
        rrf_scores: dict[str, float] = {}

        for retriever in self._retrievers:
            for r in retriever.retrieve(query, top_k=pool_k):
                chunk_by_id[r.chunk.chunk_id] = r.chunk
                rrf_scores[r.chunk.chunk_id] = rrf_scores.get(r.chunk.chunk_id, 0.0) + 1.0 / (
                    self._k_const + r.rank
                )

        ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]
        return [
            RetrievedResult(chunk=chunk_by_id[cid], score=rrf_scores[cid], rank=rank + 1)
            for rank, cid in enumerate(ranked_ids)
        ]
