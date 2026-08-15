"""EquipmentAwareRetrieverV3 — isolated experimental variant.

Per the controlled-experiment rules, `equipment_aware_v2.py` is NOT modified.
This is a separate implementation so V2 and V3 can be A/B'd on both datasets.

THE HYPOTHESIS
--------------
Diagnosing the four standing `evaluation_v2` retrieval failures showed that all
four gold chunks are present in the KB and ARE retrieved — at ranks 10, 18, 19
and 33. Nothing is missing; they are merely out-ranked. For two of them the
reason is visible in the metadata:

    V2-011  gold chunk D03-C0006  Equipment = "NOT VERIFIED"
    V2-013  gold chunk D04-C0007  Equipment = "NOT VERIFIED"

**46% of KB_v1.1 (801 of 1745 chunks) has a sentinel Equipment field.** V2
multiplies a chunk's score by 1.25 when its Equipment matches the query's, and
leaves it unchanged otherwise. "Otherwise" silently covers two very different
situations:

    Equipment = "Circuit Breaker", query about transformers
        -> positive evidence this chunk is about something else

    Equipment = "NOT VERIFIED"
        -> no evidence either way; the field was never filled in

Treating those identically means a correctly-relevant chunk is demoted 25%
relative to its competitors because of a gap in the *annotation*, not because
of anything about its content. Absence of evidence is being scored as evidence
of absence, across nearly half the corpus.

THE CHANGE
----------
Three-way instead of two-way, preserving V2's ordering while adding the signal
that was being thrown away:

    match           x (1 + boost)        as V2
    unknown         x 1.0                as V2 — genuinely neutral
    known mismatch  x 1 / (1 + boost)    NEW — explicit demotion

A match still beats an unknown by the same 25%. What changes is that a chunk
explicitly tagged with *different* equipment now ranks below one whose tag is
simply missing, which is the ordering the metadata actually supports.

Extraction and alias handling are reused from V2 unchanged, so any measured
difference is attributable to this scoring change alone.

WHETHER THIS IS AN IMPROVEMENT IS AN EMPIRICAL QUESTION, settled by
`experiments/retrieval_baseline_final.py` on BOTH D09 and evaluation_v2 — not
by how reasonable the argument above sounds. See docs/RETRIEVAL_BASELINE_V2.md
for the measured outcome.
"""
from __future__ import annotations

from typing import List, Set

from src.common.chunk import Chunk
from src.retrieval.equipment_aware import EQUIPMENT_ALIASES
from src.retrieval.equipment_aware_v2 import (
    _chunk_mentions_equipment_v2,
    _normalize,
    extract_equipment_v2,
)
from src.retrieval.retrievers import RetrievedResult, Retriever


def chunk_equipment_state(chunk: Chunk, equipment_types: Set[str]) -> str:
    """One of 'match', 'unknown', 'mismatch'.

    'unknown' means the KB never recorded this chunk's equipment — 46% of
    KB_v1.1. It must not be scored the same as a chunk we positively know is
    about different equipment.
    """
    if not equipment_types:
        return "unknown"  # no equipment in the query: nothing to compare against
    if chunk.is_sentinel(chunk.equipment):
        return "unknown"
    if _chunk_mentions_equipment_v2(chunk, equipment_types):
        return "match"
    return "mismatch"


class EquipmentAwareRetrieverV3(Retriever):
    """V2's extraction and aliasing, with a three-way score adjustment."""

    def __init__(self, base: Retriever, boost: float = 0.25, name: str | None = None):
        self._base = base
        self._boost = boost
        self.name = name or f"{base.name}_equipment_aware_v3"

    def index(self, chunks: List[Chunk]) -> None:
        self._base.index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        equipment = extract_equipment_v2(query)
        pool_k = max(top_k * 3, 20)
        candidates = self._base.retrieve(query, top_k=pool_k)

        factor = 1.0 + self._boost
        rescored = []
        for r in candidates:
            state = chunk_equipment_state(r.chunk, equipment)
            if state == "match":
                score = r.score * factor
            elif state == "mismatch":
                score = r.score / factor
            else:
                score = r.score
            rescored.append((score, r.chunk))

        rescored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedResult(chunk=chunk, score=float(score), rank=i + 1)
            for i, (score, chunk) in enumerate(rescored[:top_k])
        ]
