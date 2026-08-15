"""Frozen calibration / holdout split of evaluation_v2.

WHY THIS EXISTS
---------------
The confidence gate has knobs: how much retrieval score counts, how much
citation coverage counts, where the answer/abstain line sits. Those knobs get
set by looking at which values perform best. If the final result is then
reported on the same questions the knobs were fitted to, the number is
optimistic by an unknown amount — it measures how well the gate was fitted to
57 specific questions, not how well it works. Held-out questions turn a
flattering number into a defensible one.

THE SPLIT MUST BE FROZEN BEFORE ANY GATE RESULT IS SEEN. A holdout chosen
after looking at performance is not a holdout. `split_v1.json` is written once
and committed; `load_split()` reads that file and only computes a fresh split
if it is missing. Nothing here re-randomises an existing split.

    calibration   40 questions   tune the gate here, look as much as you like
    holdout       17 questions   touch once, at the end, to report

STRATIFICATION
--------------
The answerable set is stratified by (difficulty, document) so the holdout is
not accidentally all Easy questions from one document — with 44 questions an
unstratified draw can easily land that way, and the reported number would then
say more about the draw than the gate. Unanswerable and ambiguous are small
(7 and 6) and are stratified by category alone.

Selection within a stratum is by SHA-256 of the question id: deterministic,
independent of row order in the workbook, and not tunable by anyone hoping for
a friendlier draw.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # allow `python src/evaluation/splits.py`
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SPLIT_PATH = REPO_ROOT / "evaluation_v2" / "split_v1.json"

# 30%: enough held-out questions for the reported rates to mean something,
# while leaving 40 to calibrate on. Below ~15 held out, a single question
# swings the rate by more than the effect being measured.
DEFAULT_HOLDOUT_FRACTION = 0.30


def _stable_rank(question_id: str) -> str:
    """Deterministic per-question ordering key. SHA-256 rather than hash()
    because Python randomises string hashing per process."""
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest()


def _apportion(stratum_sizes: Dict[str, int], target_total: int) -> Dict[str, int]:
    """Largest-remainder apportionment: give each stratum its proportional
    share of the holdout, then hand the leftover seats to the strata with the
    biggest fractional claim. Keeps the holdout's composition close to the
    full set's rather than over-drawing from whichever stratum happens to be
    largest."""
    total = sum(stratum_sizes.values())
    if total == 0 or target_total <= 0:
        return {k: 0 for k in stratum_sizes}

    exact = {k: n * target_total / total for k, n in stratum_sizes.items()}
    counts = {k: int(v) for k, v in exact.items()}
    remainder = target_total - sum(counts.values())

    # Ties broken by stratum name so the result never depends on dict order.
    by_fraction = sorted(
        stratum_sizes,
        key=lambda k: (-(exact[k] - counts[k]), k),
    )
    for k in by_fraction:
        if remainder <= 0:
            break
        if counts[k] < stratum_sizes[k]:
            counts[k] += 1
            remainder -= 1
    return counts


def _split_stratified(
    items: Sequence[Tuple[str, str]],
    fraction: float,
) -> Tuple[List[str], List[str]]:
    """`items` is (question_id, stratum_key). Returns (calibration, holdout)
    question ids, both sorted."""
    strata: Dict[str, List[str]] = {}
    for qid, key in items:
        strata.setdefault(key, []).append(qid)

    target = round(len(items) * fraction)
    quotas = _apportion({k: len(v) for k, v in strata.items()}, target)

    holdout: List[str] = []
    for key, qids in strata.items():
        ordered = sorted(qids, key=_stable_rank)
        holdout.extend(ordered[: quotas[key]])

    holdout_set = set(holdout)
    calibration = [qid for qid, _ in items if qid not in holdout_set]
    return sorted(calibration), sorted(holdout)


@dataclass
class Split:
    """Which question ids belong to which side. Ids only — the questions
    themselves stay in the frozen workbooks."""

    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION
    answerable_calibration: List[str] = field(default_factory=list)
    answerable_holdout: List[str] = field(default_factory=list)
    unanswerable_calibration: List[str] = field(default_factory=list)
    unanswerable_holdout: List[str] = field(default_factory=list)
    ambiguous_calibration: List[str] = field(default_factory=list)
    ambiguous_holdout: List[str] = field(default_factory=list)

    def calibration_ids(self) -> set:
        return set(
            self.answerable_calibration
            + self.unanswerable_calibration
            + self.ambiguous_calibration
        )

    def holdout_ids(self) -> set:
        return set(
            self.answerable_holdout + self.unanswerable_holdout + self.ambiguous_holdout
        )

    def ids_for(self, side: str) -> set | None:
        """`side` is 'calibration', 'holdout' or 'all'. None means no filter."""
        side = side.strip().lower()
        if side == "calibration":
            return self.calibration_ids()
        if side == "holdout":
            return self.holdout_ids()
        if side == "all":
            return None
        raise ValueError(f"unknown split side {side!r}; use calibration | holdout | all")

    def summary(self) -> str:
        return (
            f"calibration {len(self.calibration_ids())} "
            f"(A{len(self.answerable_calibration)} "
            f"U{len(self.unanswerable_calibration)} "
            f"M{len(self.ambiguous_calibration)})  |  "
            f"holdout {len(self.holdout_ids())} "
            f"(A{len(self.answerable_holdout)} "
            f"U{len(self.unanswerable_holdout)} "
            f"M{len(self.ambiguous_holdout)})"
        )

    def to_dict(self) -> Dict:
        return {
            "split": "evaluation_v2 calibration / holdout",
            "version": 1,
            "holdout_fraction": self.holdout_fraction,
            "note": (
                "FROZEN. Calibrate the confidence gate on the calibration ids only. "
                "Report final results on the holdout ids. Do not regenerate this "
                "file after any gate performance has been observed — a holdout "
                "chosen with knowledge of the results is not a holdout."
            ),
            "answerable": {
                "calibration": self.answerable_calibration,
                "holdout": self.answerable_holdout,
            },
            "unanswerable": {
                "calibration": self.unanswerable_calibration,
                "holdout": self.unanswerable_holdout,
            },
            "ambiguous": {
                "calibration": self.ambiguous_calibration,
                "holdout": self.ambiguous_holdout,
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "Split":
        return cls(
            holdout_fraction=payload.get("holdout_fraction", DEFAULT_HOLDOUT_FRACTION),
            answerable_calibration=list(payload["answerable"]["calibration"]),
            answerable_holdout=list(payload["answerable"]["holdout"]),
            unanswerable_calibration=list(payload["unanswerable"]["calibration"]),
            unanswerable_holdout=list(payload["unanswerable"]["holdout"]),
            ambiguous_calibration=list(payload["ambiguous"]["calibration"]),
            ambiguous_holdout=list(payload["ambiguous"]["holdout"]),
        )


def build_split(
    answerable,
    unanswerable,
    ambiguous,
    fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> Split:
    """Compute the split. Deterministic: same inputs always give the same
    result, regardless of row order or process."""
    a_items = [
        (q.question_id, f"{q.gold.difficulty}|{q.gold.expected_document_id}")
        for q in answerable
    ]
    u_items = [(q.question_id, q.category or "uncategorised") for q in unanswerable]
    m_items = [(q.question_id, q.category or "uncategorised") for q in ambiguous]

    a_cal, a_hold = _split_stratified(a_items, fraction)
    u_cal, u_hold = _split_stratified(u_items, fraction)
    m_cal, m_hold = _split_stratified(m_items, fraction)

    return Split(
        holdout_fraction=fraction,
        answerable_calibration=a_cal,
        answerable_holdout=a_hold,
        unanswerable_calibration=u_cal,
        unanswerable_holdout=u_hold,
        ambiguous_calibration=m_cal,
        ambiguous_holdout=m_hold,
    )


def load_split(path: str | Path = DEFAULT_SPLIT_PATH) -> Split:
    """Read the frozen split. Raises if it does not exist — this must fail
    loudly rather than quietly inventing a new split at report time."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No frozen split at {p}. Create it once with:\n"
            f"    python src/evaluation/splits.py --freeze\n"
            f"and commit it BEFORE observing any confidence-gate results."
        )
    return Split.from_dict(json.loads(p.read_text(encoding="utf-8")))


def freeze(path: str | Path = DEFAULT_SPLIT_PATH, fraction: float = DEFAULT_HOLDOUT_FRACTION) -> Split:
    """Write the split file. Refuses to overwrite an existing one."""
    from src.evaluation.eval_loader import load_all

    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"{p} already exists. Regenerating a frozen split after gate results "
            f"have been seen invalidates the holdout. Delete it deliberately if "
            f"you are certain no gate performance has been observed."
        )
    answerable, unanswerable, ambiguous = load_all()
    split = build_split(answerable, unanswerable, ambiguous, fraction=fraction)
    p.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--freeze", action="store_true", help="write the split file (once)")
    parser.add_argument("--show", action="store_true", help="print the existing split")
    args = parser.parse_args()

    if args.freeze:
        split = freeze()
        print(f"Wrote {DEFAULT_SPLIT_PATH.relative_to(REPO_ROOT)}")
        print(split.summary())
        print("\nCommit this file now, before running any confidence-gate experiment.")
        return 0

    split = load_split()
    print(split.summary())
    if args.show:
        print("\nholdout — answerable:  ", ", ".join(split.answerable_holdout))
        print("holdout — unanswerable:", ", ".join(split.unanswerable_holdout))
        print("holdout — ambiguous:   ", ", ".join(split.ambiguous_holdout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
