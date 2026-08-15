"""Document-diversity reranking.

Wraps any base Retriever (e.g. EquipmentAwareRetriever(BM25Retriever()))
and reorders its own top-N candidate pool so that no more than `cap`
chunks from the same document_id occupy the final top_k. This does not
change the base retriever's scoring in any way -- it only reorders an
already-scored pool by walking it in score order and deferring
over-cap candidates to the end (still in score order), so nothing is
dropped, only reordered relative to document concentration.

Approved configuration (from the controlled experiment):
BM25 -> EquipmentAwareRetriever -> DocumentDiversityReranker -> top-10,
cap=2, pool_k=30.
"""
from __future__ import annotations

from typing import List

from src.retrieval.retrievers import RetrievedResult, Retriever


class DocumentDiversityReranker(Retriever):
    """Reranks a base retriever's wide candidate pool to cap how many
    chunks from a single document_id can appear in the final top_k."""

    def __init__(self, base: Retriever, cap: int = 2, pool_k: int = 30, name: str | None = None):
        self._base = base
        self._cap = cap
        self._pool_k = pool_k
        self.name = name or f"{base.name}_diversity_cap{cap}"

    def index(self, chunks) -> None:
        self._base.index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        pool = self._base.retrieve(query, top_k=self._pool_k)

        kept: List[RetrievedResult] = []
        deferred: List[RetrievedResult] = []
        doc_count: dict[str, int] = {}

        for r in pool:
            doc_id = r.chunk.document_id
            if doc_count.get(doc_id, 0) < self._cap:
                kept.append(r)
                doc_count[doc_id] = doc_count.get(doc_id, 0) + 1
            else:
                deferred.append(r)

        ordered = kept + deferred
        top = ordered[:top_k]
        return [
            RetrievedResult(chunk=r.chunk, score=r.score, rank=i + 1)
            for i, r in enumerate(top)
        ]