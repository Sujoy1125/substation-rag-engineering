"""Canonical internal representation of a KB_v1 knowledge chunk.

This is the single shared shape every downstream module (retrieval,
generation, citation, confidence) consumes. It intentionally mirrors the
21-column schema of KB_v1/knowledge_chunks.xlsx field-for-field rather than
collapsing anything into a single blob, so no metadata is lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Sentinel values used throughout KB_v1 to mean "checked, nothing there" —
# these must be preserved and recognized, not treated as missing/null data.
SENTINELS = {"NOT VERIFIED", "NOT COVERED", "NOT APPLICABLE", "N/A", ""}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    original_filename: str
    document_title: str
    organization: str
    authority_level: str
    equipment: str
    equipment_subtype: str
    topic: str
    subtopic: str
    knowledge_type: str
    verified_information: str
    procedure: str
    frequency: str
    technical_limit_value: str
    safety_information: str
    troubleshooting_failure_information: str
    applicability: str
    pdf_page: str
    source_section: str
    notes: str

    def is_sentinel(self, value: str) -> bool:
        return value is None or value.strip() in SENTINELS

    def searchable_text(self) -> str:
        """Text fields concatenated for lexical/dense indexing.

        Sentinel-valued fields are excluded so "NOT VERIFIED" / "NOT COVERED"
        don't pollute the index with noise tokens repeated ~1700 times.
        """
        parts = [
            self.document_title,
            self.equipment,
            self.equipment_subtype,
            self.topic,
            self.subtopic,
            self.knowledge_type,
            self.verified_information,
            self.procedure,
            self.frequency,
            self.technical_limit_value,
            self.safety_information,
            self.troubleshooting_failure_information,
            self.applicability,
        ]
        return " ".join(p for p in parts if p and not self.is_sentinel(p))

    def citation(self) -> str:
        """Deterministic citation string built only from chunk metadata —
        never from LLM-generated text."""
        section = "" if self.is_sentinel(self.source_section) else f", {self.source_section}"
        return f"[{self.document_title}, {self.pdf_page}{section}, {self.chunk_id}]"
