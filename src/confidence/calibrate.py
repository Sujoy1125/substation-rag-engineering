"""Fitting the confidence model — weights and thresholds — from measured data.

Runs on the **calibration split only**. Nothing here may see the holdout.

WHAT IS FITTED, AND FROM WHAT
-----------------------------
Each calibration question contributes one labelled example:

    signals        the [0,1] vector from signals.py
    should_answer  did the system's answer deserve to be asserted?

`should_answer` is derived from ground truth, not from the model's opinion:

    answerable   True  only if the answer cited the gold location
                 False if it answered citing the wrong place
    unanswerable False — no answer should be asserted
    ambiguous    False — the correct behaviour is to ask

Weights come from logistic regression on those labels, then normalised. The
coefficients are what the data says each signal is worth; they are not chosen
to look balanced. A signal that turns out not to matter gets a small weight,
and that is a finding worth reporting rather than something to correct.

THRESHOLDS ARE A POLICY CHOICE, AND ARE TREATED AS ONE
------------------------------------------------------
Once every question has a score, where to put the two cut lines is not a
statistical question — it is a decision about how much risk is acceptable.
`max_unsafe_rate` must be passed explicitly; there is no default, because a
default would quietly become the project's safety policy without anyone
choosing it. Thresholds are then swept to maximise useful answers subject to
that stated ceiling.

This ordering matters: fit the weights to the data, then set the operating
point by policy. Doing both at once produces numbers nobody can explain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.confidence.gate import ConfidenceModel
from src.confidence.signals import SIGNAL_NAMES, ConfidenceSignals

# Threshold sweep granularity. 0.01 is finer than the data can justify with
# 40 calibration questions, but the sweep is cheap and a coarser grid would
# quantise the operating point for no reason.
_GRID = [i / 100 for i in range(0, 101)]


@dataclass
class LabelledExample:
    question_id: str
    question_class: str  # answerable | unanswerable | ambiguous
    signals: ConfidenceSignals
    should_answer: bool
    # True when asserting this answer would be an unsafe assertion — an
    # answerable question answered without citing the gold location, or any
    # assertion on an unanswerable or ambiguous question.
    assertion_would_be_unsafe: bool


class NotEnoughDataError(RuntimeError):
    """Too few, or too one-sided, examples to fit anything meaningful."""


def fit_weights(examples: Sequence[LabelledExample]) -> Dict[str, float]:
    """Logistic regression coefficients over the signal vector.

    Raises rather than returning something arbitrary when the data cannot
    support a fit — a model fitted on 3 examples, or on labels that are all
    True, is worse than no model because it looks like one.
    """
    if len(examples) < 20:
        raise NotEnoughDataError(
            f"only {len(examples)} calibration examples; refusing to fit weights. "
            "Run the full calibration split first."
        )
    labels = [e.should_answer for e in examples]
    if len(set(labels)) < 2:
        raise NotEnoughDataError(
            "all calibration examples have the same label, so no signal can be "
            "distinguished from any other. Check the scoring, not the fit."
        )

    from sklearn.linear_model import LogisticRegression

    X = [e.signals.as_vector() for e in examples]
    y = [1 if e.should_answer else 0 for e in examples]

    # Regularised: 40 questions and 8 features overfits trivially otherwise.
    # liblinear is deterministic on data this size, which keeps the fit
    # reproducible.
    clf = LogisticRegression(C=1.0, solver="liblinear", random_state=0)
    clf.fit(X, y)

    return {name: float(c) for name, c in zip(SIGNAL_NAMES, clf.coef_[0])}


def _rates(
    examples: Sequence[LabelledExample],
    scores: Sequence[float],
    answer_threshold: float,
    clarify_threshold: float,
) -> Tuple[float, float]:
    """(unsafe_assertion_rate, useful_answer_rate) at these thresholds."""
    n = len(examples)
    if n == 0:
        return 0.0, 0.0
    unsafe = useful = 0
    for e, s in zip(examples, scores):
        asserted = s >= answer_threshold
        if not asserted:
            continue
        if e.assertion_would_be_unsafe:
            unsafe += 1
        if e.should_answer:
            useful += 1
    return unsafe / n, useful / n


def sweep_thresholds(
    weights: Dict[str, float],
    examples: Sequence[LabelledExample],
    max_unsafe_rate: float,
) -> Tuple[float, float, Dict[str, float]]:
    """Pick the operating point: highest useful-answer rate whose unsafe rate
    stays within the stated ceiling.

    Returns (answer_threshold, clarify_threshold, achieved_rates).
    """
    if not 0.0 <= max_unsafe_rate <= 1.0:
        raise ValueError("max_unsafe_rate must be in [0, 1]")

    scorer = ConfidenceModel(weights=dict(weights), answer_threshold=0.0, clarify_threshold=0.0)
    scores = [scorer.score(e.signals) for e in examples]

    best = None
    for a_thr in _GRID:
        unsafe, useful = _rates(examples, scores, a_thr, 0.0)
        if unsafe > max_unsafe_rate:
            continue
        # Prefer more useful answers; break ties toward the lower threshold,
        # which answers more questions for the same measured safety.
        key = (useful, -a_thr)
        if best is None or key > best[0]:
            best = (key, a_thr, unsafe, useful)

    if best is None:
        # Even answering nothing exceeds the ceiling — impossible unless the
        # ceiling is negative, but fail loudly rather than silently picking 1.0.
        raise NotEnoughDataError(
            f"no threshold satisfies max_unsafe_rate={max_unsafe_rate}; "
            "check the labels."
        )

    _, answer_threshold, unsafe, useful = best

    # The clarify line sits below the answer line, catching questions with
    # real but insufficient evidence. Placed at the median score of the
    # examples that fall short of the answer threshold — a data-derived split
    # of "found something" from "found nothing", not a chosen constant.
    below = sorted(s for s in scores if s < answer_threshold)
    clarify_threshold = below[len(below) // 2] if below else 0.0

    return (
        answer_threshold,
        clarify_threshold,
        {
            "calibration_unsafe_assertion_rate": round(unsafe, 4),
            "calibration_useful_answer_rate": round(useful, 4),
            "max_unsafe_rate_policy": max_unsafe_rate,
        },
    )


def calibrate(
    examples: Sequence[LabelledExample],
    max_unsafe_rate: float,
    fitted_on: str = "evaluation_v2 calibration split",
) -> Tuple[ConfidenceModel, Dict[str, float]]:
    """Fit weights, then set the operating point. Calibration data only."""
    weights = fit_weights(examples)
    answer_thr, clarify_thr, achieved = sweep_thresholds(weights, examples, max_unsafe_rate)

    model = ConfidenceModel(
        weights=weights,
        answer_threshold=answer_thr,
        clarify_threshold=clarify_thr,
        fitted_on=fitted_on,
        fitted_n_questions=len(examples),
        notes=(
            f"Weights: logistic regression (C=1.0, liblinear) on the signal vector. "
            f"Thresholds: swept to maximise useful answers subject to the stated "
            f"policy ceiling max_unsafe_rate={max_unsafe_rate}. "
            f"Calibration-set rates are in-sample and must not be reported as results "
            f"— report on the holdout split."
        ),
    )
    return model, achieved
