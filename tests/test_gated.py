"""Tests for the gated evaluation path.

This is the code that produces the project's headline claim, so the properties
worth defending are the ones that would let a gate look better than it is:

  - an overruled answer must not keep citation credit for text never shown
  - both systems must be scored by the same scorers over the same runs
  - a gate that abstains from everything must show zero coverage, not a
    flattering safety score
  - the ungated result must survive intact for the comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.confidence.gate import ConfidenceGate, ConfidenceModel, Decision
from src.confidence.gated import apply_gate, run_gate, split_results, summarise_changes
from src.confidence.signals import SIGNAL_NAMES
from src.evaluation.generation_eval import build_report, score_answerable, score_unanswerable
from src.generation.answer import AnswerStatus
from tests.test_generation_eval import (
    OIL_QUERY,
    answerable_q,
    dead_pipeline,
    pipeline_with,
    reply_citing,
    unanswerable_q,
)


def model(answer_threshold, clarify_threshold):
    return ConfidenceModel(
        weights={n: 1.0 for n in SIGNAL_NAMES},
        answer_threshold=answer_threshold,
        clarify_threshold=clarify_threshold,
        fitted_on="test",
        fitted_n_questions=40,
    )


PERMISSIVE = lambda: model(0.0, 0.0)      # gate lets everything through
STRICT = lambda: model(1.01, 1.01)        # gate blocks everything


def answered_result():
    return pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY)


# --------------------------------------------------------------------------
# applying a decision
# --------------------------------------------------------------------------


def test_permissive_gate_leaves_an_answer_untouched():
    r = answered_result()
    g = run_gate(ConfidenceGate(PERMISSIVE()), [r])[0]
    assert g.gated.answer.status is AnswerStatus.ANSWER
    assert g.gated.answer.answer_text == r.answer.answer_text
    assert g.gated.answer.citations
    assert not g.changed


def test_overruled_answer_loses_its_citations():
    """Otherwise the gated report claims citation credit for text the user
    was never shown — flattering exactly the number under test."""
    g = run_gate(ConfidenceGate(STRICT()), [answered_result()])[0]
    assert g.gated.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert g.gated.answer.answer_text == ""
    assert g.gated.answer.citations == []
    assert g.gated.answer.claims == []
    assert "withheld by confidence gate" in g.gated.answer.downgrade_reason


def test_ungated_result_survives_intact():
    """The comparison needs both sides; applying the gate must not mutate the
    original."""
    r = answered_result()
    original_text = r.answer.answer_text
    original_citations = len(r.answer.citations)
    run_gate(ConfidenceGate(STRICT()), [r])
    assert r.answer.status is AnswerStatus.ANSWER
    assert r.answer.answer_text == original_text
    assert len(r.answer.citations) == original_citations


def test_gate_never_overrides_a_model_refusal_in_the_gated_view():
    reply = json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []})
    g = run_gate(ConfidenceGate(PERMISSIVE()), [pipeline_with(reply).answer(OIL_QUERY)])[0]
    assert g.gated.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_clarify_decision_gets_a_question_without_inventing_specifics():
    r = answered_result()
    m = model(answer_threshold=1.01, clarify_threshold=0.0)  # forces CLARIFY
    g = run_gate(ConfidenceGate(m), [r])[0]
    assert g.gated.answer.status is AnswerStatus.NEEDS_CLARIFICATION
    q = g.gated.answer.clarification_question
    assert q and "narrow it" in q
    # must not fabricate a specific technical distinction the evidence lacks
    assert "kV" not in q


def test_unreached_model_stays_abstained_under_the_gate():
    g = run_gate(ConfidenceGate(PERMISSIVE()), [dead_pipeline().answer(OIL_QUERY)])[0]
    assert g.gated.answer.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert g.outcome.confidence is None


# --------------------------------------------------------------------------
# comparison bookkeeping
# --------------------------------------------------------------------------


def test_split_results_preserves_order_and_length():
    rs = [answered_result(), answered_result()]
    gated = run_gate(ConfidenceGate(STRICT()), rs)
    ungated, gated_only = split_results(gated)
    assert len(ungated) == len(gated_only) == 2
    assert ungated[0] is rs[0]


def test_transition_summary_reports_what_changed():
    gated = run_gate(ConfidenceGate(STRICT()), [answered_result(), answered_result()])
    changes = summarise_changes(gated)
    assert changes == {"ANSWER -> INSUFFICIENT_EVIDENCE": 2}


def test_no_op_gate_is_visible_as_no_change():
    """A gate that changes nothing is not a gate; the summary must make that
    obvious rather than hiding it in equal-looking metrics."""
    gated = run_gate(ConfidenceGate(PERMISSIVE()), [answered_result()])
    assert summarise_changes(gated) == {"ANSWER -> ANSWER": 1}


# --------------------------------------------------------------------------
# the property the whole comparison exists to test
# --------------------------------------------------------------------------


def test_abstaining_from_everything_shows_zero_coverage_not_a_safety_win():
    """A gate can trivially drive unsafe assertions to zero by refusing to
    answer. The paired coverage number is what stops that reading as success."""
    a_result = answered_result()
    u_result = pipeline_with(reply_citing(["E1"], answer="Weekly.")).answer(OIL_QUERY)

    ungated = build_report(
        "ungated",
        [score_answerable(answerable_q(), a_result)],
        [score_unanswerable(unanswerable_q(), u_result)],
        [],
    )

    gate = ConfidenceGate(STRICT())
    a_g = run_gate(gate, [a_result])[0]
    u_g = run_gate(gate, [u_result])[0]
    gated = build_report(
        "gated",
        [score_answerable(answerable_q(), a_g.gated)],
        [score_unanswerable(unanswerable_q(), u_g.gated)],
        [],
    )

    # the gate does remove the unsafe assertion...
    assert gated.safety.n_unsafe_assertions < ungated.safety.n_unsafe_assertions
    # ...but it pays for it in coverage, and the report shows both
    assert gated.safety.answer_coverage == 0.0
    assert gated.safety.useful_answer_rate == 0.0


def test_both_reports_use_the_same_scorers_over_the_same_runs():
    """One variable changed. If the two sides ran through different scoring
    paths, any difference would be confounded with the measurement."""
    r = answered_result()
    g = run_gate(ConfidenceGate(PERMISSIVE()), [r])[0]
    ungated = build_report("u", [score_answerable(answerable_q(), r)], [], [])
    gated = build_report("g", [score_answerable(answerable_q(), g.gated)], [], [])
    # permissive gate changes nothing, so every metric must match exactly
    assert ungated.answerable == gated.answerable
    assert ungated.safety == gated.safety


def test_gated_result_serialises():
    g = run_gate(ConfidenceGate(STRICT()), [answered_result()])[0]
    payload = g.to_dict()
    json.dumps(payload)
    assert payload["changed_by_gate"] is True
    assert payload["gate"]["decision"] == "ABSTAIN"
