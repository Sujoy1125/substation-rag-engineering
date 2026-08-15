"""Equipment-aware retrieval.

Per Section 9 of the task: equipment_inventory.xlsx is a retrieval
enrichment reference, not the primary evidence source, and equipment
metadata must never override actual retrieved evidence. This module
therefore implements a *soft boost* (re-scores candidates whose chunk-level
Equipment field mentions an equipment type extracted from the query) rather
than a hard filter — a hard filter risks false negatives whenever the
query's equipment wording doesn't exactly match the free-text Equipment
column (which, on inspection, is inconsistent: e.g. "Transformer, Reactor,
Bushing" as one literal string), so boosting is the safer choice.

Equipment extraction itself is deterministic (substring/alias match
against the 11 canonical equipment types), per the instruction to prefer
deterministic extraction over an LLM call for every query.
"""
from __future__ import annotations

from typing import Dict, List, Set

from src.common.chunk import Chunk
from src.retrieval.retrievers import RetrievedResult, Retriever

# Canonical equipment types, taken from equipment_inventory.xlsx "Equipment
# Type" column, with a small deterministic alias set derived by inspecting
# how the chunk-level `Equipment` free-text field and the eval questions
# actually phrase things.
EQUIPMENT_ALIASES: Dict[str, List[str]] = {
    "Transformer": ["transformer", "oltc", "tap changer", "bushing", "reactor", "silica gel", "breather"],
    "Circuit Breaker": ["circuit breaker", "breaker", "vcb", "vacuum interrupter", "ocb", "sf6 breaker"],
    "Isolator / Disconnector": ["isolator", "disconnector"],
    "CT": ["current transformer", " ct "],
    "PT / CVT": ["potential transformer", "cvt", " pt "],
    "Surge Arrester": ["surge arrester", "lightning arrester"],
    "Busbar": ["busbar", "bus bar"],
    "Battery Bank": ["battery"],
    "Protection Relay": ["protection relay", "relay"],
    "Earthing System": ["earthing", "earth switch", "grounding"],
    "Switchgear": ["switchgear"],
}


def extract_equipment(query: str) -> Set[str]:
    q = f" {query.lower()} "
    found = set()
    for canonical, aliases in EQUIPMENT_ALIASES.items():
        for alias in aliases:
            if alias in q:
                found.add(canonical)
                break
    return found


def _chunk_mentions_equipment(chunk: Chunk, equipment_types: Set[str]) -> bool:
    if not equipment_types or chunk.is_sentinel(chunk.equipment):
        return False
    field = chunk.equipment.lower()
    for eq in equipment_types:
        if eq.lower().split(" / ")[0] in field:  # "CT" / "PT / CVT" -> match first token
            return True
    return False


class EquipmentAwareRetriever(Retriever):
    """Wraps a base retriever; re-scores (does not filter) using a soft
    multiplicative boost for chunks whose Equipment field matches
    equipment extracted from the query."""

    def __init__(self, base: Retriever, boost: float = 0.25, name: str | None = None):
        self._base = base
        self._boost = boost
        self.name = name or f"{base.name}_equipment_aware"

    def index(self, chunks: List[Chunk]) -> None:
        self._base.index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        equipment = extract_equipment(query)
        pool_k = max(top_k * 3, 20)
        candidates = self._base.retrieve(query, top_k=pool_k)

        rescored = []
        for r in candidates:
            score = r.score
            if _chunk_mentions_equipment(r.chunk, equipment):
                score = score * (1.0 + self._boost)
            rescored.append((score, r.chunk))

        rescored.sort(key=lambda x: x[0], reverse=True)
        top = rescored[:top_k]
        return [
            RetrievedResult(chunk=chunk, score=float(score), rank=i + 1)
            for i, (score, chunk) in enumerate(top)
        ]
