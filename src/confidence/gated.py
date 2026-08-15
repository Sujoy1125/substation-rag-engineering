"""Applying gate decisions to pipeline results, so both systems can be scored
by the same scorers.

THE PROBLEM THIS SOLVES
-----------------------
`src/evaluation/generation_eval.py` scores a `PipelineResult` by looking at its
`AnswerStatus`. The gate, however, produces a `Decision`. Comparing gated
against ungated therefore needs the gate's verdict expressed in the vocabulary
the scorers already speak — otherwise the comparison runs through two different
scoring paths and any difference between them is confounded with the thing
being measured.

So a gated result is the same result with its status rewritten:

    Decision.ANSWER   -> AnswerStatus.ANSWER                (unchanged)
    Decision.ABSTAIN  -> AnswerStatus.INSUFFICIENT_EVIDENCE
    Decision.CLARIFY  -> AnswerStatus.NEEDS_CLARIFICATION

Identical scorers, identical metrics, one variable changed. That is the whole
point of the comparison.

WHY AN OVERRULED ANSWER LOSES ITS CITATIONS
-------------------------------------------
When the gate turns an ANSWER into an ABSTAIN, the answer text and citations
are cleared from the gated view. A system that abstains does not show the user
an answer, and it does not cite anything; leaving the citations attached would
let the gated report claim citation credit for text it never displayed, which
would flatter exactly the number the comparison exists to test.

The original result is kept intact alongside, so nothing is lost for
diagnostics — `GatedResult` holds both plus the gate's reasoning.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from src.confidence.gate import ConfidenceGate, Decision, GateOutcome
from src.generation.answer import AnswerStatus
from src.generation.pipeline import PipelineResult

_DECISION_TO_STATUS: Dict[Decision, AnswerStatus] = {
    Decision.ANSWER: AnswerStatus.ANSWER,
    Decision.ABSTAIN: AnswerStatus.INSUFFICIENT_EVIDENCE,
    Decision.CLARIFY: AnswerStatus.NEEDS_CLARIFICATION,
}


@dataclass
class GatedResult:
    """One question under both systems."""

    ungated: PipelineResult
    gated: PipelineResult
    outcome: GateOutcome

    @property
    def changed(self) -> bool:
        return self.ungated.answer.status is not self.gated.answer.status

    def to_dict(self) -> Dict:
        return {
            "question": self.ungated.question,
            "ungated_status": self.ungated.answer.status.value,
            "gated_status": self.gated.answer.status.value,
            "changed_by_gate": self.changed,
            "gate": self.outcome.to_dict(),
        }


def apply_gate(result: PipelineResult, outcome: GateOutcome) -> PipelineResult:
    """Return a copy of `result` rewritten to reflect the gate's decision.

    The input is not mutated: the ungated result stays available for the
    side-by-side comparison.
    """
    gated = copy.deepcopy(result)
    new_status = _DECISION_TO_STATUS[outcome.decision]
    gated.answer.status = new_status

    if outcome.decision is not Decision.ANSWER:
        # An abstaining system displays no answer and cites nothing. Keeping
        # the citations would let the gated report take credit for text the
        # user never saw.
        gated.answer.answer_text = ""
        gated.answer.claims = []
        gated.answer.citations = []
        conf = "" if outcome.confidence is None else f" (confidence {outcome.confidence:.2f})"
        gated.answer.downgrade_reason = f"withheld by confidence gate{conf}: {outcome.reason}"

    if outcome.decision is Decision.CLARIFY and not gated.answer.clarification_question:
        # The gate can route to CLARIFY on an answer the model never framed as
        # a question. Say what is actually known rather than inventing a
        # specific clarification the evidence does not support.
        gated.answer.clarification_question = (
            "The retrieved evidence is related but not conclusive for this question. "
            "Could you narrow it — equipment type, voltage class, or the specific "
            "quantity you need?"
        )

    return gated


def run_gate(
    gate: ConfidenceGate,
    results: Sequence[PipelineResult],
) -> List[GatedResult]:
    out: List[GatedResult] = []
    for r in results:
        outcome = gate.decide(r)
        out.append(GatedResult(ungated=r, gated=apply_gate(r, outcome), outcome=outcome))
    return out


def split_results(gated: Sequence[GatedResult]) -> Tuple[List[PipelineResult], List[PipelineResult]]:
    """(ungated_results, gated_results) in the original order."""
    return [g.ungated for g in gated], [g.gated for g in gated]


def summarise_changes(gated: Sequence[GatedResult]) -> Dict[str, int]:
    """What the gate actually did, as a transition count. A gate that changes
    nothing is not a gate, and a gate that changes everything to ABSTAIN is
    not useful — this makes either visible at a glance."""
    counts: Dict[str, int] = {}
    for g in gated:
        key = f"{g.ungated.answer.status.value} -> {g.gated.answer.status.value}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
