"""Evidence context builder.

Turns retrieved chunks into (a) the text block shown to the LLM and (b) a
label -> Chunk map the citation layer resolves against.

Two decisions here matter more than they look:

1. **The model cites by label, never by page.** Each chunk is presented as
   `[E1]`, `[E2]`, ... and the model is asked to cite those labels. Document
   ids, page numbers and section names are shown for the model's *reasoning*,
   but the citation string a user finally sees is rebuilt from the `Chunk`
   object in `citation.py`. A model that invents `[E9]` when only `[E1]`–`[E5]`
   exist produces a detectably invalid label, not a plausible fake page number.

2. **Sentinels are dropped, not rendered.** KB_v1.1 uses `NOT VERIFIED`,
   `NOT COVERED`, `NOT APPLICABLE`, `N/A` to mean "checked, nothing there".
   Printing those into the prompt invites the model to treat them as content.
   `Chunk.is_sentinel` decides; this module only omits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from src.common.chunk import Chunk
from src.retrieval.retrievers import RetrievedResult

# Field label -> Chunk attribute, in the order they are rendered. Ordered so
# the substantive content (verified information, procedure, limits) comes
# before the qualifiers.
_FIELDS: Sequence[tuple[str, str]] = (
    ("Verified information", "verified_information"),
    ("Procedure", "procedure"),
    ("Frequency", "frequency"),
    ("Technical limit / value", "technical_limit_value"),
    ("Safety information", "safety_information"),
    ("Troubleshooting / failure information", "troubleshooting_failure_information"),
    ("Applicability", "applicability"),
)


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved chunk as presented to the model."""

    label: str  # "E1", "E2", ...
    chunk: Chunk
    score: float
    rank: int


@dataclass(frozen=True)
class EvidenceContext:
    items: List[EvidenceItem]
    text: str

    @property
    def labels(self) -> List[str]:
        return [i.label for i in self.items]

    def by_label(self) -> Dict[str, EvidenceItem]:
        return {i.label: i for i in self.items}

    def is_empty(self) -> bool:
        return not self.items

    def document_ids(self) -> List[str]:
        seen: List[str] = []
        for i in self.items:
            if i.chunk.document_id not in seen:
                seen.append(i.chunk.document_id)
        return seen


def _render_item(item: EvidenceItem) -> str:
    c = item.chunk
    header_bits = [f"[{item.label}]"]
    header_bits.append(f"Document: {c.document_id}")
    if not c.is_sentinel(c.document_title):
        header_bits.append(f"Title: {c.document_title}")
    if not c.is_sentinel(c.pdf_page):
        header_bits.append(f"Page: {c.pdf_page}")
    if not c.is_sentinel(c.source_section):
        header_bits.append(f"Section: {c.source_section}")
    if not c.is_sentinel(c.equipment):
        eq = c.equipment
        if not c.is_sentinel(c.equipment_subtype):
            eq = f"{eq} / {c.equipment_subtype}"
        header_bits.append(f"Equipment: {eq}")
    if not c.is_sentinel(c.topic):
        header_bits.append(f"Topic: {c.topic}")
    if not c.is_sentinel(c.authority_level):
        header_bits.append(f"Authority: {c.authority_level}")
    if not c.is_sentinel(c.organization):
        header_bits.append(f"Organization: {c.organization}")

    lines = ["\n".join(header_bits[:1] + [f"  {b}" for b in header_bits[1:]])]

    fields: List[tuple[str, str]] = []
    # Several KB fields legitimately carry the same extracted sentence — a
    # failure-analysis paragraph is often stored under both "Verified
    # Information" and "Troubleshooting / Failure Information". Printing it
    # twice wastes prompt budget and, worse, makes one source look like two
    # corroborating statements. Show it once, under every field it belongs to.
    seen_values: dict[str, int] = {}
    for label, attr in _FIELDS:
        value = getattr(c, attr, "")
        if not value or c.is_sentinel(value):
            continue
        key = " ".join(value.split()).lower()
        if key in seen_values:
            idx = seen_values[key]
            existing_label, existing_value = fields[idx]
            fields[idx] = (f"{existing_label} / {label}", existing_value)
            continue
        seen_values[key] = len(fields)
        fields.append((label, value))

    body: List[str] = [f"  {label}: {value}" for label, value in fields]
    if not body:
        # A chunk whose every content field is a sentinel carries no evidence.
        # Say so explicitly rather than presenting an empty block the model
        # might fill in from its own knowledge.
        body.append("  (no content fields populated in the knowledge base for this chunk)")

    lines.append("\n".join(body))
    return "\n".join(lines)


def build_context(
    results: Sequence[RetrievedResult],
    max_items: int | None = None,
) -> EvidenceContext:
    """Build the evidence block from retrieval results, in rank order.

    `max_items` truncates from the bottom of the ranking. Truncation is a
    context-budget decision only; it never reorders, and the labels always
    follow retrieval rank so `E1` is the top-ranked chunk.
    """
    selected = list(results)[: max_items if max_items is not None else len(results)]

    items = [
        EvidenceItem(label=f"E{i}", chunk=r.chunk, score=r.score, rank=r.rank)
        for i, r in enumerate(selected, start=1)
    ]

    if not items:
        return EvidenceContext(items=[], text="(no evidence retrieved)")

    blocks = [_render_item(i) for i in items]
    return EvidenceContext(items=items, text="\n\n".join(blocks))
