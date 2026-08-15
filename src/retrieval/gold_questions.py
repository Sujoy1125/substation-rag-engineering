"""Loads the frozen 20_Answerable sheet from rag_test_55.xlsx.

Read-only. Never writes to the benchmark file. Used exclusively to measure
retrieval recall/MRR — per ENGINEERING_HANDOFF.md Section 5, this sheet is
not to be used for confidence-threshold tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import openpyxl


@dataclass(frozen=True)
class GoldQuestion:
    question_id: str
    question: str
    expected_answer: str
    expected_document_ref: str
    expected_document_id: str  # KB Document ID, e.g. "D09"
    expected_page: str
    expected_section: str
    difficulty: str


def load_gold_answerable(xlsx_path: str) -> List[GoldQuestion]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb["20_Answerable"]
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    expected_header = (
        "Question ID", "Question", "Expected Answer", "Expected Document",
        "Document ID (KB)", "Expected Page", "Expected Section", "Difficulty",
        "Answer Available",
    )
    assert tuple(header) == expected_header, f"unexpected header: {header}"

    out: List[GoldQuestion] = []
    for row in rows:
        if row is None or row[0] is None:
            continue
        out.append(
            GoldQuestion(
                question_id=str(row[0]).strip(),
                question=str(row[1]).strip(),
                expected_answer=str(row[2]).strip(),
                expected_document_ref=str(row[3]).strip(),
                expected_document_id=str(row[4]).strip(),
                expected_page=str(row[5]).strip(),
                expected_section=str(row[6]).strip(),
                difficulty=str(row[7]).strip(),
            )
        )
    return out


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[2] / "KB_v1" / "rag_test_55.xlsx"
    qs = load_gold_answerable(str(path))
    print(f"Loaded {len(qs)} answerable gold questions")
    for q in qs[:3]:
        print(q)
