"""Citation resolution and rendering.

The single invariant this module enforces:

    A citation is built from a Chunk object. It is never built from, parsed
    out of, or influenced by generated text.

The model's only contribution to a citation is a label (`E3`). The label is
looked up in the evidence context that was actually sent for this question; if
it is not there, the citation is rejected and recorded as invalid. Every page
number, section name and document title a user sees therefore comes from
`knowledge_chunks.xlsx`, which makes an invented page reference structurally
impossible rather than merely discouraged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from src.common.chunk import Chunk
from src.generation.context import EvidenceContext


@dataclass(frozen=True)
class Citation:
    """A resolved reference to one KB chunk. All fields come from the Chunk."""

    label: str
    chunk_id: str
    document_id: str
    document_title: str
    page: str
    section: str
    organization: str
    authority_level: str
    retrieval_rank: int
    retrieval_score: float

    @classmethod
    def from_chunk(
        cls,
        label: str,
        chunk: Chunk,
        retrieval_rank: int = 0,
        retrieval_score: float = 0.0,
    ) -> "Citation":
        return cls(
            label=label,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_title="" if chunk.is_sentinel(chunk.document_title) else chunk.document_title,
            page="" if chunk.is_sentinel(chunk.pdf_page) else chunk.pdf_page,
            section="" if chunk.is_sentinel(chunk.source_section) else chunk.source_section,
            organization="" if chunk.is_sentinel(chunk.organization) else chunk.organization,
            authority_level="" if chunk.is_sentinel(chunk.authority_level) else chunk.authority_level,
            retrieval_rank=retrieval_rank,
            retrieval_score=retrieval_score,
        )

    def short(self) -> str:
        """Compact inline form: `[D09, p.123]`, degrading gracefully when the
        KB has no page for the chunk."""
        return f"[{self.document_id}, {self.page}]" if self.page else f"[{self.document_id}]"

    def full(self) -> str:
        """Reference-list form, with everything the KB actually holds."""
        bits = [self.document_id]
        if self.document_title:
            bits.append(self.document_title)
        if self.section:
            bits.append(self.section)
        if self.page:
            bits.append(self.page)
        bits.append(f"chunk {self.chunk_id}")
        return " — ".join(bits)


def resolve_labels(
    labels: Sequence[str],
    context: EvidenceContext,
) -> Tuple[List[Citation], List[str]]:
    """Map evidence labels to Citations against the context actually sent.

    Returns `(citations, invalid_labels)`. Order follows the input, duplicates
    are dropped, and matching is case-insensitive and whitespace-tolerant
    (`e3`, ` E3 ` all resolve) — a formatting wobble is not the same failure
    as citing evidence that was never supplied, and only the latter should
    show up as invalid.
    """
    by_label = {k.upper(): v for k, v in context.by_label().items()}
    citations: List[Citation] = []
    invalid: List[str] = []
    seen: set[str] = set()

    for raw in labels:
        if raw is None:
            continue
        key = str(raw).strip().strip("[]").upper()
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        item = by_label.get(key)
        if item is None:
            invalid.append(str(raw).strip())
            continue
        citations.append(
            Citation.from_chunk(
                label=item.label,
                chunk=item.chunk,
                retrieval_rank=item.rank,
                retrieval_score=item.score,
            )
        )
    return citations, invalid


def render_reference_list(citations: Sequence[Citation]) -> str:
    """Numbered reference block, deduplicated by chunk, in evidence order."""
    seen: set[str] = set()
    lines: List[str] = []
    for c in citations:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        lines.append(f"{len(lines) + 1}. {c.full()}")
    return "\n".join(lines)
