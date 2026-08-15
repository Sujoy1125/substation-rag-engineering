"""The confidence gate: answer / abstain / clarify.

    GeneratedAnswer + retrieval  ->  signals  ->  score  ->  decision

WEIGHTS AND THRESHOLDS ARE DELIBERATELY UNSET IN THIS FILE.

`ConfidenceModel` ships with every weight at 0.0 and both thresholds at None,
and `decide()` refuses to run in that state. That is not an oversight — it is
the point. Numbers like 0.4 / 0.2 / 0.1 chosen because they look balanced are
indistinguishable, in a write-up, from numbers fitted to data, and a reviewer
cannot tell which they are being shown. An uncalibrated gate that refuses to
run cannot be mistaken for a calibrated one.

Weights come from `calibrate.py`, fitted on the calibration split only
(`evaluation_v2/split_v1.json`), and are saved to a versioned JSON file that
records what it was fitted on.

WHAT THE GATE MAY AND MAY NOT OVERRIDE
--------------------------------------
The gate decides how much to trust an attempted answer. It does not
second-guess the model's own refusals:

    model said INSUFFICIENT_EVIDENCE  -> ABSTAIN, always
    model said NEEDS_CLARIFICATION    -> CLARIFY, always
    code said UNSUPPORTED             -> ABSTAIN, always (structural, pre-gate)
    code said PARSE_ERROR / LLM_ERROR -> ABSTAIN, always

Only `ANSWER` reaches the score. Letting a confidence score talk the system
*into* answering a question the model declined would invert the safety
property the whole design exists to provide.

Between the two thresholds sits a deliberate middle band: evidence good enough
to have found something, not good enough to assert it. That band routes to
CLARIFY rather than ANSWER or ABSTAIN, because "I found related material but
cannot tell if it answers your question" is more useful to a maintenance
engineer than either a confident guess or a flat refusal.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from src.confidence.signals import SIGNAL_NAMES, ConfidenceSignals, extract_signals
from src.generation.answer import AnswerStatus
from src.generation.pipeline import PipelineResult

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "configs" / "confidence_model_v1.json"


class Decision(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    CLARIFY = "CLARIFY"


class UncalibratedGateError(RuntimeError):
    """The gate was asked to decide before its weights were fitted."""


@dataclass
class ConfidenceModel:
    """Weights and thresholds. Empty until calibrated."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in SIGNAL_NAMES}
    )
    answer_threshold: Optional[float] = None
    clarify_threshold: Optional[float] = None

    # Provenance, so a saved model can never be mistaken for one fitted
    # somewhere else or on the wrong data.
    fitted_on: str = ""
    fitted_n_questions: int = 0
    notes: str = ""

    @property
    def is_calibrated(self) -> bool:
        return (
            self.answer_threshold is not None
            and self.clarify_threshold is not None
            and any(w != 0.0 for w in self.weights.values())
        )

    def score(self, signals: ConfidenceSignals) -> float:
        """Weighted sum, normalised by total weight so the result stays in
        [0, 1] and stays comparable across differently-scaled fits."""
        total = sum(abs(w) for w in self.weights.values())
        if total == 0:
            return 0.0
        s = sum(self.weights.get(n, 0.0) * getattr(signals, n) for n in SIGNAL_NAMES)
        return max(0.0, min(1.0, s / total))

    def to_dict(self) -> Dict:
        return {
            "version": 1,
            "weights": {k: round(v, 6) for k, v in self.weights.items()},
            "answer_threshold": self.answer_threshold,
            "clarify_threshold": self.clarify_threshold,
            "fitted_on": self.fitted_on,
            "fitted_n_questions": self.fitted_n_questions,
            "notes": self.notes,
            "is_calibrated": self.is_calibrated,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "ConfidenceModel":
        return cls(
            weights=dict(payload.get("weights", {})),
            answer_threshold=payload.get("answer_threshold"),
            clarify_threshold=payload.get("clarify_threshold"),
            fitted_on=payload.get("fitted_on", ""),
            fitted_n_questions=payload.get("fitted_n_questions", 0),
            notes=payload.get("notes", ""),
        )

    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "ConfidenceModel":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"No confidence model at {p}. Fit one on the calibration split:\n"
                f"    python experiments/calibrate_confidence.py --from <eval json>"
            )
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


@dataclass
class GateOutcome:
    decision: Decision
    confidence: Optional[float]
    signals: ConfidenceSignals
    reason: str
    model_status: str

    def to_dict(self) -> Dict:
        return {
            "decision": self.decision.value,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "reason": self.reason,
            "model_status": self.model_status,
            "signals": self.signals.to_dict(),
        }


# Statuses the gate never overrides, and what each becomes.
_FORCED: Dict[AnswerStatus, tuple] = {
    AnswerStatus.INSUFFICIENT_EVIDENCE: (
        Decision.ABSTAIN,
        "model reported insufficient evidence; the gate does not overrule a refusal",
    ),
    AnswerStatus.NEEDS_CLARIFICATION: (
        Decision.CLARIFY,
        "model asked for clarification; the gate does not overrule a refusal",
    ),
    AnswerStatus.UNSUPPORTED: (
        Decision.ABSTAIN,
        "answer cited no evidence actually retrieved (structural, decided before the gate)",
    ),
    AnswerStatus.PARSE_ERROR: (
        Decision.ABSTAIN,
        "model reply could not be parsed",
    ),
    AnswerStatus.LLM_ERROR: (
        Decision.ABSTAIN,
        "model could not be reached — nothing was evaluated",
    ),
}


class ConfidenceGate:
    def __init__(self, model: ConfidenceModel):
        self.model = model

    def decide(self, result: PipelineResult) -> GateOutcome:
        status = result.answer.status
        signals = extract_signals(result)

        forced = _FORCED.get(status)
        if forced is not None:
            decision, reason = forced
            return GateOutcome(
                decision=decision,
                confidence=None,  # not scored: the model already decided
                signals=signals,
                reason=reason,
                model_status=status.value,
            )

        if not self.model.is_calibrated:
            raise UncalibratedGateError(
                "ConfidenceGate has no fitted weights or thresholds. Guessed "
                "constants would be indistinguishable from calibrated ones in a "
                "write-up. Fit on the calibration split first:\n"
                "    python experiments/calibrate_confidence.py --from <eval json>"
            )

        confidence = self.model.score(signals)
        if confidence >= self.model.answer_threshold:
            decision, reason = Decision.ANSWER, "confidence above the answer threshold"
        elif confidence >= self.model.clarify_threshold:
            decision, reason = (
                Decision.CLARIFY,
                "evidence found but not strong enough to assert; asking is more "
                "useful than guessing or refusing",
            )
        else:
            decision, reason = Decision.ABSTAIN, "confidence below the clarify threshold"

        return GateOutcome(
            decision=decision,
            confidence=confidence,
            signals=signals,
            reason=reason,
            model_status=status.value,
        )
