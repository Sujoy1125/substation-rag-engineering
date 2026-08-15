"""Tests for the confidence layer.

The properties worth defending here are the ones that would let an
uncalibrated or over-permissive gate pass for a working one:

  - a gate with no fitted weights must refuse to decide, not guess
  - the gate must never talk the system INTO answering something the model
    declined
  - signals must stay in [0,1] and oriented so higher = more confident
  - calibration must refuse data that cannot support a fit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.confidence.calibrate import (
    LabelledExample,
    NotEnoughDataError,
    calibrate,
    fit_weights,
    sweep_thresholds,
)
from src.confidence.gate import (
    ConfidenceGate,
    ConfidenceModel,
    Decision,
    UncalibratedGateError,
)
from src.confidence.signals import SIGNAL_NAMES, ConfidenceSignals, extract_signals
from src.generation.answer import AnswerStatus
from tests.test_generation_eval import (
    OIL_QUERY,
    dead_pipeline,
    pipeline_with,
    reply_citing,
)


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------


def test_all_signals_are_bounded_and_oriented():
    """A signal outside [0,1], or one where higher means less confident, makes
    a weighted sum impossible to reason about."""
    result = pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY)
    s = extract_signals(result)
    for name in SIGNAL_NAMES:
        v = getattr(s, name)
        assert 0.0 <= v <= 1.0, f"{name} = {v} is outside [0, 1]"


def test_unreached_model_yields_no_confidence():
    """A question that never reached the model must not score as a confident
    anything — there is nothing to be confident about."""
    s = extract_signals(dead_pipeline().answer(OIL_QUERY))
    assert s.as_vector() == [0.0] * len(SIGNAL_NAMES)


def test_invalid_labels_drive_citation_validity_down():
    clean = extract_signals(pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY))
    invented = extract_signals(pipeline_with(reply_citing(["E1", "E9"])).answer(OIL_QUERY))
    assert invented.citation_validity < clean.citation_validity


def test_citing_the_top_chunk_beats_citing_a_lower_one():
    top = extract_signals(pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY))
    low = extract_signals(pipeline_with(reply_citing(["E3"])).answer(OIL_QUERY))
    assert top.top_rank_cited > low.top_rank_cited


def test_specific_answers_score_above_evasive_ones():
    """'every five years' is an answer; 'periodically as required' is not."""
    specific = pipeline_with(
        reply_citing(["E1"], answer="Test BDV every 12 months, minimum 50 kV.")
    ).answer(OIL_QUERY)
    vague = pipeline_with(
        reply_citing(["E1"], answer="Test it periodically as and when required.")
    ).answer(OIL_QUERY)
    assert extract_signals(specific).answer_specificity > extract_signals(vague).answer_specificity


def test_signals_serialise():
    s = extract_signals(pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY))
    json.dumps(s.to_dict())


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def calibrated_model(answer_threshold=0.6, clarify_threshold=0.3):
    return ConfidenceModel(
        weights={n: 1.0 for n in SIGNAL_NAMES},
        answer_threshold=answer_threshold,
        clarify_threshold=clarify_threshold,
        fitted_on="test",
        fitted_n_questions=40,
    )


def test_uncalibrated_gate_refuses_to_decide():
    """Guessed constants are indistinguishable from fitted ones in a write-up.
    A gate that refuses to run cannot be mistaken for a calibrated one."""
    gate = ConfidenceGate(ConfidenceModel())
    assert ConfidenceModel().is_calibrated is False
    with pytest.raises(UncalibratedGateError):
        gate.decide(pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY))


def test_gate_never_overrides_a_model_refusal():
    """Letting confidence talk the system INTO answering a question the model
    declined would invert the whole safety property."""
    reply = json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []})
    result = pipeline_with(reply).answer(OIL_QUERY)
    # thresholds at zero: everything would score "confident enough"
    outcome = ConfidenceGate(calibrated_model(0.0, 0.0)).decide(result)
    assert outcome.decision is Decision.ABSTAIN
    assert "does not overrule" in outcome.reason


def test_gate_never_overrides_a_clarification_request():
    reply = json.dumps(
        {"status": "NEEDS_CLARIFICATION", "clarification_question": "Which limit?", "claims": []}
    )
    outcome = ConfidenceGate(calibrated_model(0.0, 0.0)).decide(
        pipeline_with(reply).answer(OIL_QUERY)
    )
    assert outcome.decision is Decision.CLARIFY


def test_unsupported_answer_abstains_regardless_of_confidence():
    result = pipeline_with(reply_citing([], answer="Annually.")).answer(OIL_QUERY)
    assert result.answer.status is AnswerStatus.UNSUPPORTED
    assert ConfidenceGate(calibrated_model(0.0, 0.0)).decide(result).decision is Decision.ABSTAIN


def test_unreached_model_abstains_and_is_not_scored():
    outcome = ConfidenceGate(calibrated_model()).decide(dead_pipeline().answer(OIL_QUERY))
    assert outcome.decision is Decision.ABSTAIN
    assert outcome.confidence is None  # never asked, so never scored
    assert "could not be reached" in outcome.reason


def test_high_threshold_forces_abstention_on_an_attempted_answer():
    result = pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY)
    outcome = ConfidenceGate(calibrated_model(1.01, 1.01)).decide(result)
    assert outcome.decision is Decision.ABSTAIN
    assert outcome.confidence is not None


def test_middle_band_routes_to_clarify():
    result = pipeline_with(reply_citing(["E1"])).answer(OIL_QUERY)
    conf = calibrated_model().score(extract_signals(result))
    # place the answer line just above the actual score, clarify line below
    tuned = calibrated_model(answer_threshold=min(1.0, conf + 0.01), clarify_threshold=max(0.0, conf - 0.01))
    assert ConfidenceGate(tuned).decide(result).decision is Decision.CLARIFY


def test_score_is_bounded():
    model = ConfidenceModel(
        weights={n: 5.0 for n in SIGNAL_NAMES}, answer_threshold=0.5, clarify_threshold=0.2
    )
    s = ConfidenceSignals(**{n: 1.0 for n in SIGNAL_NAMES})
    assert 0.0 <= model.score(s) <= 1.0


def test_model_round_trips_through_json(tmp_path):
    m = calibrated_model()
    p = tmp_path / "model.json"
    m.save(p)
    assert ConfidenceModel.load(p).to_dict() == m.to_dict()


def test_loading_a_missing_model_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError):
        ConfidenceModel.load(tmp_path / "nope.json")


def test_shipped_model_is_uncalibrated():
    """The committed model must have no invented weights in it."""
    assert ConfidenceModel().is_calibrated is False


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------


def make_examples(n=40, positive_fraction=0.5):
    out = []
    for i in range(n):
        positive = i < int(n * positive_fraction)
        level = 0.9 if positive else 0.1
        out.append(
            LabelledExample(
                question_id=f"Q{i:03}",
                question_class="answerable",
                signals=ConfidenceSignals(**{k: level for k in SIGNAL_NAMES}),
                should_answer=positive,
                assertion_would_be_unsafe=not positive,
            )
        )
    return out


def test_fit_refuses_too_little_data():
    with pytest.raises(NotEnoughDataError):
        fit_weights(make_examples(n=10))


def test_fit_refuses_single_class_labels():
    with pytest.raises(NotEnoughDataError):
        fit_weights(make_examples(n=40, positive_fraction=1.0))


def test_fit_produces_one_weight_per_signal():
    w = fit_weights(make_examples())
    assert set(w) == set(SIGNAL_NAMES)


def test_separable_data_yields_a_usable_gate():
    model, achieved = calibrate(make_examples(), max_unsafe_rate=0.05)
    assert model.is_calibrated
    assert achieved["calibration_unsafe_assertion_rate"] <= 0.05
    assert model.fitted_n_questions == 40


def test_stricter_policy_never_increases_unsafe_rate():
    loose, loose_rates = calibrate(make_examples(), max_unsafe_rate=0.50)
    strict, strict_rates = calibrate(make_examples(), max_unsafe_rate=0.00)
    assert strict_rates["calibration_unsafe_assertion_rate"] <= loose_rates[
        "calibration_unsafe_assertion_rate"
    ]
    assert strict.answer_threshold >= loose.answer_threshold


def test_calibrated_model_records_what_it_was_fitted_on():
    """A saved model must never be mistakable for one fitted elsewhere."""
    model, _ = calibrate(make_examples(), max_unsafe_rate=0.05)
    assert "calibration" in model.fitted_on
    assert "in-sample" in model.notes
    assert model.fitted_n_questions == 40


def test_threshold_sweep_rejects_an_impossible_policy():
    with pytest.raises(ValueError):
        sweep_thresholds({n: 1.0 for n in SIGNAL_NAMES}, make_examples(), max_unsafe_rate=1.5)
