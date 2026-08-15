"""Read-only loaders for the frozen evaluation_v2 question sets.

Before this module existed, `experiments/eval_v2_regression_v2.py` parsed
`answerable.xlsx` inline with a positional row unpack. That worked, but it
(a) could not be reused by the generation/confidence layers and (b) failed
silently-ish if a column were ever inserted. These loaders validate the
header explicitly and expose all three question classes, because the
answer / abstain / clarify evaluation needs all three:

    answerable    -> system must ANSWER, grounded + cited
    unanswerable  -> system must ABSTAIN (evidence genuinely absent from KB)
    ambiguous     -> system must ASK FOR CLARIFICATION

The three classes are deliberately kept as distinct types. Collapsing
"ambiguous" into "unanswerable" would destroy the distinction the whole
confidence gate is meant to demonstrate.

Nothing here writes back to the workbooks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import openpyxl

from src.retrieval.gold_questions import GoldQuestion

ANSWERABLE_HEADER = (
    "Question ID",
    "Question",
    "Expected Answer",
    "Expected Document",
    "Document ID (KB)",
    "Expected Page",
    "Expected Section",
    "Difficulty",
    "Answer Available",
    "Expected Chunk ID(s)",
    "Evidence Basis",
)

UNANSWERABLE_HEADER = (
    "Question ID",
    "Question",
    "Why It's Unanswerable (from current KB)",
    "Answer Available",
    "Category",
    "Risk If Hallucinated",
)

AMBIGUOUS_HEADER = (
    "Question ID",
    "Question",
    "Why It's Ambiguous",
    "Ideal System Behavior",
    "Category",
)


@dataclass(frozen=True)
class AnswerableQuestion:
    """An evaluation_v2 answerable question.

    `gold` carries the retrieval-facing fields in the exact shape
    `src.retrieval.benchmark.run_benchmark` already consumes, so the
    retrieval harness needs no changes. The two extra columns
    (`expected_chunk_ids`, `evidence_basis`) are what the generation and
    citation layers need and are kept alongside rather than folded in.
    """

    gold: GoldQuestion
    expected_chunk_ids: List[str]
    evidence_basis: str

    @property
    def question_id(self) -> str:
        return self.gold.question_id

    @property
    def question(self) -> str:
        return self.gold.question


@dataclass(frozen=True)
class UnanswerableQuestion:
    question_id: str
    question: str
    why_unanswerable: str
    category: str
    risk_if_hallucinated: str


@dataclass(frozen=True)
class AmbiguousQuestion:
    question_id: str
    question: str
    why_ambiguous: str
    ideal_system_behavior: str
    category: str


def _rows(xlsx_path: str, expected_header: tuple):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    header = tuple(next(it))
    if header != expected_header:
        raise ValueError(
            f"{Path(xlsx_path).name}: unexpected header.\n"
            f"  expected: {expected_header}\n"
            f"  actual:   {header}"
        )
    for row in it:
        if row is None or row[0] is None:
            continue
        yield ["" if v is None else str(v).strip() for v in row]


def _split_chunk_ids(raw: str) -> List[str]:
    """Expected Chunk ID(s) may hold one id or several separated by
    comma / semicolon / pipe. Empty and sentinel-ish values become []."""
    if not raw or raw.upper() in {"N/A", "NONE", "NOT APPLICABLE"}:
        return []
    parts = raw.replace(";", ",").replace("|", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def load_answerable(xlsx_path: str) -> List[AnswerableQuestion]:
    out: List[AnswerableQuestion] = []
    for r in _rows(xlsx_path, ANSWERABLE_HEADER):
        out.append(
            AnswerableQuestion(
                gold=GoldQuestion(
                    question_id=r[0],
                    question=r[1],
                    expected_answer=r[2],
                    expected_document_ref=r[3],
                    expected_document_id=r[4],
                    expected_page=r[5],
                    expected_section=r[6],
                    difficulty=r[7],
                ),
                expected_chunk_ids=_split_chunk_ids(r[9]),
                evidence_basis=r[10],
            )
        )
    return out


def load_unanswerable(xlsx_path: str) -> List[UnanswerableQuestion]:
    return [
        UnanswerableQuestion(
            question_id=r[0],
            question=r[1],
            why_unanswerable=r[2],
            category=r[4],
            risk_if_hallucinated=r[5],
        )
        for r in _rows(xlsx_path, UNANSWERABLE_HEADER)
    ]


def load_ambiguous(xlsx_path: str) -> List[AmbiguousQuestion]:
    return [
        AmbiguousQuestion(
            question_id=r[0],
            question=r[1],
            why_ambiguous=r[2],
            ideal_system_behavior=r[3],
            category=r[4],
        )
        for r in _rows(xlsx_path, AMBIGUOUS_HEADER)
    ]


def load_gold_for_retrieval(xlsx_path: str) -> List[GoldQuestion]:
    """Answerable questions in the shape run_benchmark() expects."""
    return [a.gold for a in load_answerable(xlsx_path)]


DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[2] / "evaluation_v2"


def load_all(eval_dir: str | Path = DEFAULT_EVAL_DIR):
    eval_dir = Path(eval_dir)
    return (
        load_answerable(str(eval_dir / "answerable.xlsx")),
        load_unanswerable(str(eval_dir / "unanswerable.xlsx")),
        load_ambiguous(str(eval_dir / "ambiguous.xlsx")),
    )


if __name__ == "__main__":
    a, u, amb = load_all()
    print(f"answerable:   {len(a)}")
    print(f"unanswerable: {len(u)}")
    print(f"ambiguous:    {len(amb)}")
