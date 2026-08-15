"""EquipmentAwareRetrieverV2 — isolated experimental variant.

Per the controlled-experiment rules: `equipment_aware.py` /
`EquipmentAwareRetriever` is NOT modified. This module is a separate,
parallel implementation so both can be A/B'd. It targets exactly the two
confirmed bugs and nothing else:

Bug A — query-side hyphen handling: extract_equipment() matched
"surge arrester" (space) but not "surge-arrester" (hyphen). Fixed by
normalizing hyphens/underscores to spaces before substring matching, on
both the query and the alias table lookups (so an alias itself could
contain a hyphen and still match either query spelling).

Bug B — canonical-vs-granular chunk-side matching: _chunk_mentions_equipment()
required the literal canonical family name (e.g. "Transformer") to appear
in the chunk's Equipment field, so a chunk tagged only "OLTC" or "Bushing"
never matched even though those terms alias to "Transformer" on the query
side. Fixed by checking the chunk's Equipment field against the *same*
alias list used for query extraction (canonical name OR any of its
aliases), not just the bare canonical name.

Everything else (boost factor, pool_k, base retriever wrapping, class
shape) is unchanged from the original so the comparison isolates these two
fixes only.
"""
from __future__ import annotations

import re
from typing import Dict, List, Set

from src.common.chunk import Chunk
from src.retrieval.equipment_aware import EQUIPMENT_ALIASES
from src.retrieval.retrievers import RetrievedResult, Retriever

_HYPHEN_RE = re.compile(r"[-_]")


def _normalize(text: str) -> str:
    """Normalize hyphens/underscores to spaces so 'surge-arrester' and
    'surge_arrester' match the same as 'surge arrester'. Does not touch
    any other tokenization behavior."""
    return _HYPHEN_RE.sub(" ", text)


def extract_equipment_v2(query: str) -> Set[str]:
    q = f" {_normalize(query.lower())} "
    found = set()
    for canonical, aliases in EQUIPMENT_ALIASES.items():
        for alias in aliases:
            if _normalize(alias) in q:
                found.add(canonical)
                break
    return found


def _chunk_mentions_equipment_v2(chunk: Chunk, equipment_types: Set[str]) -> bool:
    """Alias-aware chunk-side check: a chunk matches a canonical equipment
    type if its Equipment field contains the canonical name itself OR any
    granular alias belonging to that canonical family (e.g. a chunk tagged
    'OLTC' matches canonical 'Transformer', since OLTC is one of
    Transformer's aliases)."""
    if not equipment_types or chunk.is_sentinel(chunk.equipment):
        return False
    field = _normalize(chunk.equipment.lower())
    for eq in equipment_types:
        canonical_token = eq.lower().split(" / ")[0]
        if canonical_token in field:
            return True
        for alias in EQUIPMENT_ALIASES.get(eq, []):
            if _normalize(alias.strip()) in field:
                return True
    return False


class EquipmentAwareRetrieverV2(Retriever):
    """Same shape/boost as the original EquipmentAwareRetriever; only the
    extraction/matching logic changes (Bug A + Bug B fixes)."""

    def __init__(self, base: Retriever, boost: float = 0.25, name: str | None = None):
        self._base = base
        self._boost = boost
        self.name = name or f"{base.name}_equipment_aware_v2"

    def index(self, chunks: List[Chunk]) -> None:
        self._base.index(chunks)

    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedResult]:
        equipment = extract_equipment_v2(query)
        pool_k = max(top_k * 3, 20)
        candidates = self._base.retrieve(query, top_k=pool_k)

        rescored = []
        for r in candidates:
            score = r.score
            if _chunk_mentions_equipment_v2(r.chunk, equipment):
                score = score * (1.0 + self._boost)
            rescored.append((score, r.chunk))

        rescored.sort(key=lambda x: x[0], reverse=True)
        top = rescored[:top_k]
        return [
            RetrievedResult(chunk=chunk, score=float(score), rank=i + 1)
            for i, (score, chunk) in enumerate(top)
        ]
