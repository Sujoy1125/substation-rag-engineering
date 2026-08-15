"""Parsing and validation of the model's JSON reply.

Everything in this module is deterministic post-processing. It assumes the
model output is untrusted: malformed JSON, a fenced block, prose wrapped
around the object, invented evidence labels, a claimed ANSWER with no
supporting labels at all. Each of those is a real thing models do, and each is
handled here so the rest of the system can work with a validated object.

WHAT THIS MODULE DOES NOT DO: it does not gate. `GroundingSignals` measures
things — citation coverage, invalid-label count, evidence agreement — and
`AnswerStatus` records what happened, but no weighted score and no threshold
is applied. Those weights must be calibrated against evaluation_v2 rather than
guessed, and inventing them here would bake uncalibrated constants into the
foundation.

The one hard rule applied at this layer is structural, not statistical: a
reply claiming status ANSWER with zero valid citations is downgraded to
UNSUPPORTED. That is not a tuned threshold — an answer citing nothing that was
actually retrieved is, by this system's own definition, not evidence-grounded.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

from src.citation.citations import Citation, resolve_labels
from src.generation.context import EvidenceContext

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class AnswerStatus(str, Enum):
    ANSWER = "ANSWER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    # Assigned by this module, never by the model:
    UNSUPPORTED = "UNSUPPORTED"  # claimed an answer, cited nothing valid
    PARSE_ERROR = "PARSE_ERROR"  # reply was not usable JSON


class MalformedAnswerError(ValueError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """Recover the JSON object from a model reply.

    Tries, in order: the whole string; a fenced block; the outermost
    brace-balanced span. Raises MalformedAnswerError if none parse — the
    caller records a PARSE_ERROR rather than guessing at intent.
    """
    if text is None:
        raise MalformedAnswerError("empty model reply")
    s = text.strip()
    if not s:
        raise MalformedAnswerError("empty model reply")

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = _FENCE_RE.search(s)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    start = s.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    raise MalformedAnswerError(f"could not parse JSON from model reply: {s[:200]!r}")


@dataclass(frozen=True)
class Claim:
    text: str
    citations: List[Citation]
    invalid_labels: List[str]

    @property
    def is_supported(self) -> bool:
        return bool(self.citations)


@dataclass
class GroundingSignals:
    """Observable, uncalibrated signals for the future confidence layer.

    Recorded now, weighted later. Nothing here is combined into a score.
    """

    n_claims: int = 0
    n_supported_claims: int = 0
    n_invalid_labels: int = 0
    invalid_labels: List[str] = field(default_factory=list)
    n_evidence_items: int = 0
    n_evidence_cited: int = 0
    distinct_documents_cited: int = 0
    distinct_documents_retrieved: int = 0
    top_retrieval_score: float = 0.0
    min_cited_rank: int | None = None
    max_cited_rank: int | None = None
    authority_levels_cited: List[str] = field(default_factory=list)
    conflict_reported: bool = False

    @property
    def citation_coverage(self) -> float:
        """Fraction of claims carrying at least one valid citation."""
        return self.n_supported_claims / self.n_claims if self.n_claims else 0.0

    @property
    def evidence_utilisation(self) -> float:
        """Fraction of supplied evidence items the answer actually used. Low
        values mean retrieval returned more than the answer needed; high
        values with few items mean the answer leaned on everything it had."""
        return self.n_evidence_cited / self.n_evidence_items if self.n_evidence_items else 0.0


@dataclass
class GeneratedAnswer:
    question: str
    status: AnswerStatus
    answer_text: str
    claims: List[Claim]
    citations: List[Citation]
    clarification_question: str
    missing_information: str
    conflict_present: bool
    conflict_description: str
    conflict_citations: List[Citation]
    signals: GroundingSignals
    raw_model_text: str = ""
    parse_error: str = ""
    downgrade_reason: str = ""

    @property
    def is_answer(self) -> bool:
        return self.status is AnswerStatus.ANSWER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "status": self.status.value,
            "answer": self.answer_text,
            "claims": [
                {
                    "text": c.text,
                    "citations": [x.short() for x in c.citations],
                    "chunk_ids": [x.chunk_id for x in c.citations],
                    "invalid_labels": c.invalid_labels,
                }
                for c in self.claims
            ],
            "citations": [
                {
                    "label": c.label,
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "document_title": c.document_title,
                    "page": c.page,
                    "section": c.section,
                    "authority_level": c.authority_level,
                    "retrieval_rank": c.retrieval_rank,
                    "short": c.short(),
                    "full": c.full(),
                }
                for c in self.citations
            ],
            "clarification_question": self.clarification_question,
            "missing_information": self.missing_information,
            "conflict": {
                "present": self.conflict_present,
                "description": self.conflict_description,
                "citations": [c.short() for c in self.conflict_citations],
            },
            "signals": {
                "n_claims": self.signals.n_claims,
                "n_supported_claims": self.signals.n_supported_claims,
                "citation_coverage": round(self.signals.citation_coverage, 4),
                "n_invalid_labels": self.signals.n_invalid_labels,
                "invalid_labels": self.signals.invalid_labels,
                "n_evidence_items": self.signals.n_evidence_items,
                "n_evidence_cited": self.signals.n_evidence_cited,
                "evidence_utilisation": round(self.signals.evidence_utilisation, 4),
                "distinct_documents_cited": self.signals.distinct_documents_cited,
                "distinct_documents_retrieved": self.signals.distinct_documents_retrieved,
                "top_retrieval_score": round(self.signals.top_retrieval_score, 4),
                "min_cited_rank": self.signals.min_cited_rank,
                "max_cited_rank": self.signals.max_cited_rank,
                "authority_levels_cited": self.signals.authority_levels_cited,
                "conflict_reported": self.signals.conflict_reported,
            },
            "downgrade_reason": self.downgrade_reason,
            "parse_error": self.parse_error,
        }


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _parse_status(value: Any) -> AnswerStatus:
    raw = _as_str(value).upper().replace(" ", "_").replace("-", "_")
    for s in (
        AnswerStatus.ANSWER,
        AnswerStatus.INSUFFICIENT_EVIDENCE,
        AnswerStatus.NEEDS_CLARIFICATION,
    ):
        if raw == s.value:
            return s
    # Tolerate near-misses the models actually emit, but nothing looser: an
    # unrecognised status is treated as abstention, which is the safe default.
    if "CLARIF" in raw:
        return AnswerStatus.NEEDS_CLARIFICATION
    if "INSUFFICIENT" in raw or "UNANSWER" in raw or "NO_EVIDENCE" in raw:
        return AnswerStatus.INSUFFICIENT_EVIDENCE
    if raw == "":
        return AnswerStatus.INSUFFICIENT_EVIDENCE
    return AnswerStatus.INSUFFICIENT_EVIDENCE


def build_answer(
    question: str,
    raw_model_text: str,
    context: EvidenceContext,
) -> GeneratedAnswer:
    """Validate a model reply against the evidence it was given."""
    signals = GroundingSignals(
        n_evidence_items=len(context.items),
        distinct_documents_retrieved=len(context.document_ids()),
        top_retrieval_score=context.items[0].score if context.items else 0.0,
    )

    try:
        payload = extract_json(raw_model_text)
    except MalformedAnswerError as e:
        return GeneratedAnswer(
            question=question,
            status=AnswerStatus.PARSE_ERROR,
            answer_text="",
            claims=[],
            citations=[],
            clarification_question="",
            missing_information="",
            conflict_present=False,
            conflict_description="",
            conflict_citations=[],
            signals=signals,
            raw_model_text=raw_model_text or "",
            parse_error=str(e),
        )

    status = _parse_status(payload.get("status"))
    answer_text = _as_str(payload.get("answer"))
    clarification_question = _as_str(payload.get("clarification_question"))
    missing_information = _as_str(payload.get("missing_information"))

    claims: List[Claim] = []
    all_citations: List[Citation] = []
    all_invalid: List[str] = []
    seen_chunk_ids: set[str] = set()

    for raw_claim in _as_list(payload.get("claims")):
        if isinstance(raw_claim, dict):
            text = _as_str(raw_claim.get("text") or raw_claim.get("claim"))
            labels = _as_list(
                raw_claim.get("evidence_labels")
                or raw_claim.get("evidence")
                or raw_claim.get("labels")
            )
        else:
            text = _as_str(raw_claim)
            labels = []
        if not text:
            continue
        cites, invalid = resolve_labels(labels, context)
        claims.append(Claim(text=text, citations=cites, invalid_labels=invalid))
        all_invalid.extend(invalid)
        for c in cites:
            if c.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(c.chunk_id)
                all_citations.append(c)

    conflict_raw = payload.get("conflict") or {}
    if not isinstance(conflict_raw, dict):
        conflict_raw = {}
    conflict_present = bool(conflict_raw.get("present"))
    conflict_description = _as_str(conflict_raw.get("description"))
    conflict_citations, conflict_invalid = resolve_labels(
        _as_list(conflict_raw.get("evidence_labels")), context
    )
    all_invalid.extend(conflict_invalid)
    # A "conflict" with no description is a checkbox, not a finding.
    if conflict_present and not conflict_description:
        conflict_present = False

    signals.n_claims = len(claims)
    signals.n_supported_claims = sum(1 for c in claims if c.is_supported)
    signals.invalid_labels = list(dict.fromkeys(all_invalid))
    signals.n_invalid_labels = len(signals.invalid_labels)
    signals.n_evidence_cited = len(all_citations)
    signals.distinct_documents_cited = len({c.document_id for c in all_citations})
    signals.authority_levels_cited = sorted(
        {c.authority_level for c in all_citations if c.authority_level}
    )
    signals.conflict_reported = conflict_present
    if all_citations:
        ranks = [c.retrieval_rank for c in all_citations if c.retrieval_rank]
        if ranks:
            signals.min_cited_rank = min(ranks)
            signals.max_cited_rank = max(ranks)

    downgrade_reason = ""
    if status is AnswerStatus.ANSWER:
        if not all_citations:
            status = AnswerStatus.UNSUPPORTED
            downgrade_reason = (
                "model returned ANSWER but cited no evidence label present in the "
                "context supplied for this question"
            )
        elif not answer_text:
            status = AnswerStatus.UNSUPPORTED
            downgrade_reason = "model returned ANSWER with empty answer text"

    return GeneratedAnswer(
        question=question,
        status=status,
        answer_text=answer_text,
        claims=claims,
        citations=all_citations,
        clarification_question=clarification_question,
        missing_information=missing_information,
        conflict_present=conflict_present,
        conflict_description=conflict_description,
        conflict_citations=conflict_citations,
        signals=signals,
        raw_model_text=raw_model_text or "",
        downgrade_reason=downgrade_reason,
    )
