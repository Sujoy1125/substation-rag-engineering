"""Confidence signals: observable quantities, normalised to [0, 1].

Every signal here is computed from things the pipeline already recorded — the
retrieval scores, the citation resolution, the model's own status. Nothing is
asked of the model and nothing is invented.

TWO RULES THIS MODULE OBEYS
---------------------------
1. **No weights.** Extraction produces a vector of named signals. Combining
   them is `gate.py`'s job, and the combination coefficients come from
   calibration, not from this file.

2. **Every signal is oriented so that higher = more confident**, and bounded to
   [0, 1]. Unbounded or inconsistently-oriented signals make a linear
   combination impossible to reason about and let one feature silently
   dominate through scale alone.

WHY NORMALISATION IS NOT TRIVIAL HERE
-------------------------------------
BM25 scores are unbounded and corpus-dependent — a raw score of 30 means
nothing on its own. `retrieval_strength` therefore uses the *gap* between the
top hit and the rest of the retrieved set, which is scale-free: a top chunk
that scores far above its neighbours is a confident retrieval, whether the
absolute numbers are 8 or 80. A flat score profile means BM25 found many
equally-plausible chunks, which is exactly the situation where a confident
answer is least warranted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

from src.generation.answer import AnswerStatus, GeneratedAnswer
from src.generation.pipeline import PipelineResult

# Signals are listed here in one place so the gate, the calibrator and the
# report all agree on the feature order without duplicating the list.
SIGNAL_NAMES: List[str] = [
    "retrieval_strength",
    "evidence_concentration",
    "citation_coverage",
    "citation_validity",
    "evidence_utilisation",
    "top_rank_cited",
    "source_authority",
    "answer_specificity",
]


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


@dataclass(frozen=True)
class ConfidenceSignals:
    """All signals in [0, 1], higher = more confident."""

    retrieval_strength: float = 0.0
    evidence_concentration: float = 0.0
    citation_coverage: float = 0.0
    citation_validity: float = 0.0
    evidence_utilisation: float = 0.0
    top_rank_cited: float = 0.0
    source_authority: float = 0.0
    answer_specificity: float = 0.0

    def as_vector(self) -> List[float]:
        return [getattr(self, n) for n in SIGNAL_NAMES]

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def _retrieval_strength(scores: List[float]) -> float:
    """How far the top hit stands above the rest of the pool.

    Scale-free by construction, because BM25 scores are unbounded and mean
    nothing in absolute terms. A flat profile — many chunks scoring alike —
    signals that retrieval could not discriminate, which is when a confident
    answer is least warranted.
    """
    positive = [s for s in scores if s > 0]
    if len(positive) < 2:
        return 1.0 if positive else 0.0
    top = positive[0]
    rest_mean = sum(positive[1:]) / len(positive[1:])
    if top <= 0:
        return 0.0
    return _clamp((top - rest_mean) / top)


def _evidence_concentration(document_ids: List[str]) -> float:
    """Whether the retrieved evidence agrees on a source.

    All top-K chunks from one document is a coherent body of evidence. One
    chunk each from five different documents usually means the query matched
    a common phrase rather than a topic. Computed as the share held by the
    most common document.
    """
    if not document_ids:
        return 0.0
    counts: Dict[str, int] = {}
    for d in document_ids:
        counts[d] = counts.get(d, 0) + 1
    return _clamp(max(counts.values()) / len(document_ids))


def _top_rank_cited(cited_ranks: List[int]) -> float:
    """Did the answer rest on highly-ranked evidence?

    Citing rank 1 is stronger than citing rank 5. Mapped so rank 1 -> 1.0 and
    decaying with the best rank actually used.
    """
    ranks = [r for r in cited_ranks if r and r > 0]
    if not ranks:
        return 0.0
    return _clamp(1.0 / min(ranks))


def _source_authority(levels: List[str]) -> float:
    """KB authority level of the cited chunks.

    KB_v1.1 records HIGH throughout the current corpus, so this signal is
    near-constant today and calibration should be expected to give it little
    or no weight. It is extracted anyway because it costs nothing and becomes
    meaningful the moment a lower-authority source enters the KB — at which
    point the gate should already know how to use it.
    """
    if not levels:
        return 0.0
    ranking = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
    vals = [ranking.get(l.strip().upper(), 0.5) for l in levels if l]
    return _clamp(sum(vals) / len(vals)) if vals else 0.0


def _answer_specificity(answer_text: str) -> float:
    """Whether the answer commits to something checkable.

    In this domain a useful answer usually contains a figure, an interval or a
    unit. "Inspect periodically as required" is evasive; "every five years" is
    an answer. Deliberately crude — a digit-bearing token ratio — because
    anything cleverer would need its own validation.
    """
    if not answer_text:
        return 0.0
    tokens = answer_text.split()
    if not tokens:
        return 0.0
    with_digit = sum(1 for t in tokens if any(ch.isdigit() for ch in t))
    # 1 numeric token in 12 is already a specific answer; saturate there
    # rather than rewarding walls of numbers.
    return _clamp((with_digit / len(tokens)) / (1.0 / 12.0))


def extract_signals(result: PipelineResult) -> ConfidenceSignals:
    """Compute all signals from a completed pipeline run.

    A run that never reached the model yields all-zero signals: there is
    nothing to be confident about, and it must not be scored as a confident
    abstention.
    """
    answer: GeneratedAnswer = result.answer
    if answer.status is AnswerStatus.LLM_ERROR:
        return ConfidenceSignals()

    scores = [r.score for r in result.retrieved]
    doc_ids = [r.chunk.document_id for r in result.retrieved]
    sig = answer.signals

    return ConfidenceSignals(
        retrieval_strength=_retrieval_strength(scores),
        evidence_concentration=_evidence_concentration(doc_ids),
        citation_coverage=_clamp(sig.citation_coverage),
        # One invented label is a strong negative signal, so this falls away
        # fast rather than degrading linearly.
        citation_validity=_clamp(1.0 / (1.0 + sig.n_invalid_labels)),
        evidence_utilisation=_clamp(sig.evidence_utilisation),
        top_rank_cited=_top_rank_cited([c.retrieval_rank for c in answer.citations]),
        source_authority=_source_authority(sig.authority_levels_cited),
        answer_specificity=_answer_specificity(answer.answer_text),
    )
