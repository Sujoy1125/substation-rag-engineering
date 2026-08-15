"""Fit the confidence model on the calibration split.

STEP 9. Reads a saved run of

    python experiments/run_generation_eval.py --live --split calibration

and fits weights + thresholds from it, writing configs/confidence_model_v1.json.

    python experiments/calibrate_confidence.py \
        --from experiments/generation_eval_<stamp>.json \
        --max-unsafe-rate 0.05

`--max-unsafe-rate` has no default and must be stated. It is a policy choice —
how often the system may assert something it should not have — and a default
would quietly become the project's safety policy without anyone deciding it.

This script REFUSES a run that was not on the calibration split. Fitting on
holdout data, or on all 57, destroys the only out-of-sample evidence the
project has; see docs/EVALUATION_PLAN.md.

Labels come from ground truth, never from the model's own opinion:

    answerable    should_answer = answered AND cited the gold location
    unanswerable  should_answer = False   (any assertion is unsafe)
    ambiguous     should_answer = False   (the right behaviour is to ask)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.confidence.calibrate import LabelledExample, NotEnoughDataError, calibrate
from src.confidence.gate import DEFAULT_MODEL_PATH, ConfidenceModel
from src.confidence.signals import SIGNAL_NAMES, ConfidenceSignals


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="source", required=True, help="generation_eval_*.json from a calibration run")
    p.add_argument(
        "--max-unsafe-rate",
        type=float,
        required=True,
        help="POLICY: highest acceptable unsafe-assertion rate on the calibration set (0-1). "
        "No default — this must be a stated decision.",
    )
    p.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--force-split", action="store_true", help="fit on a non-calibration run (destroys the holdout; do not use)")
    return p.parse_args()


def build_examples(payload: dict):
    report = payload["report"]
    raw_signals = payload.get("confidence_signals", {})
    if not raw_signals:
        raise SystemExit(
            "This run has no 'confidence_signals' block — it predates the confidence "
            "layer. Re-run the calibration evaluation to regenerate it."
        )

    def signals_for(qid: str) -> ConfidenceSignals:
        d = raw_signals.get(qid, {})
        return ConfidenceSignals(**{n: float(d.get(n, 0.0)) for n in SIGNAL_NAMES})

    examples = []

    for s in report["answerable_scores"]:
        # Correct only if it both answered AND pointed at the gold location.
        deserved = bool(s["answered"] and s["page_level_cited"])
        examples.append(
            LabelledExample(
                question_id=s["question_id"],
                question_class="answerable",
                signals=signals_for(s["question_id"]),
                should_answer=deserved,
                assertion_would_be_unsafe=bool(s["answered"] and not s["page_level_cited"]),
            )
        )

    for s in report["unanswerable_scores"]:
        examples.append(
            LabelledExample(
                question_id=s["question_id"],
                question_class="unanswerable",
                signals=signals_for(s["question_id"]),
                should_answer=False,
                assertion_would_be_unsafe=True,
            )
        )

    for s in report["ambiguous_scores"]:
        examples.append(
            LabelledExample(
                question_id=s["question_id"],
                question_class="ambiguous",
                signals=signals_for(s["question_id"]),
                should_answer=False,
                assertion_would_be_unsafe=True,
            )
        )

    return examples


def main() -> int:
    args = parse_args()
    payload = json.loads(Path(args.source).read_text(encoding="utf-8"))

    split = payload.get("split", "unknown")
    if split != "calibration" and not args.force_split:
        print(
            f"REFUSING: this run used --split {split!r}, not 'calibration'.\n\n"
            "Fitting on the holdout — or on all 57 — turns the holdout into training\n"
            "data and destroys the only out-of-sample evidence the project has. The\n"
            "final result would then be unreportable.\n\n"
            "Produce a calibration run first:\n"
            "  python experiments/run_generation_eval.py --live --split calibration"
        )
        return 2

    if not payload["report"].get("results_are_valid", True):
        print(
            f"REFUSING: that run is marked invalid "
            f"({payload['report'].get('n_llm_errors')} questions never reached the model). "
            "Fix the connection and re-run."
        )
        return 2

    examples = build_examples(payload)
    print(f"Loaded {len(examples)} calibration examples from {Path(args.source).name}")
    n_pos = sum(1 for e in examples if e.should_answer)
    print(f"  should_answer=True: {n_pos}   should_answer=False: {len(examples) - n_pos}")
    print(f"  policy: max_unsafe_rate = {args.max_unsafe_rate}")

    try:
        model, achieved = calibrate(examples, max_unsafe_rate=args.max_unsafe_rate)
    except NotEnoughDataError as e:
        print(f"\nCannot calibrate: {e}")
        return 3

    print("\n--- fitted weights (logistic regression coefficients) ---")
    for name in SIGNAL_NAMES:
        w = model.weights[name]
        bar = "#" * int(abs(w) * 20)
        print(f"  {name:<24} {w:+.4f}  {bar}")

    print("\n--- operating point ---")
    print(f"  answer_threshold   {model.answer_threshold:.2f}")
    print(f"  clarify_threshold  {model.clarify_threshold:.2f}")
    print("\n--- achieved ON THE CALIBRATION SET (in-sample, not results) ---")
    for k, v in achieved.items():
        print(f"  {k:<38} {v}")

    model.save(args.out)
    print(f"\nSaved: {Path(args.out).relative_to(REPO_ROOT)}")
    print(
        "\nThese calibration-set numbers are in-sample and must not be quoted as\n"
        "results. Report by running, ONCE:\n"
        "  python experiments/run_generation_eval.py --live --split holdout"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
