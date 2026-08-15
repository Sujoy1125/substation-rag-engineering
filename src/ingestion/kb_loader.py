"""Loader for KB_v1/knowledge_chunks.xlsx.

Read-only: never writes back to the source workbook. Produces a
deterministic list[Chunk] plus a validation report of any malformed rows.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # allow `src.common` imports
from src.common.chunk import Chunk

EXPECTED_COLUMNS = [
    "Chunk ID",
    "Document ID",
    "Original Filename",
    "Document Title",
    "Organization",
    "Authority Level",
    "Equipment",
    "Equipment Subtype",
    "Topic",
    "Subtopic",
    "Knowledge Type",
    "Verified Information",
    "Procedure",
    "Frequency",
    "Technical Limit / Value",
    "Safety Information",
    "Troubleshooting / Failure Information",
    "Applicability",
    "PDF Page",
    "Source Section",
    "Notes",
]


@dataclass
class LoadReport:
    total_rows_seen: int
    loaded: int
    malformed_row_numbers: List[int]
    duplicate_chunk_ids: List[str]
    header_matches_expected: bool

    def ok(self) -> bool:
        return (
            self.header_matches_expected
            and not self.malformed_row_numbers
            and not self.duplicate_chunk_ids
        )


def load_chunks(xlsx_path: str, sheet: str = "knowledge_chunks") -> tuple[List[Chunk], LoadReport]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb[sheet]

    rows_iter = ws.iter_rows(values_only=True)
    header = next(rows_iter)
    header_matches_expected = list(header) == EXPECTED_COLUMNS

    chunks: List[Chunk] = []
    malformed: List[int] = []
    seen_ids: dict[str, int] = {}
    dup_ids: List[str] = []
    total_seen = 0

    for row_number, row in enumerate(rows_iter, start=2):
        if row is None or row[0] is None:
            continue  # trailing blank row (openpyxl sometimes yields these)
        total_seen += 1

        if len(row) != len(EXPECTED_COLUMNS):
            malformed.append(row_number)
            continue

        values = ["" if v is None else str(v) for v in row]
        chunk_id = values[0].strip()
        if not chunk_id:
            malformed.append(row_number)
            continue

        if chunk_id in seen_ids:
            dup_ids.append(chunk_id)
        else:
            seen_ids[chunk_id] = row_number

        try:
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=values[1].strip(),
                original_filename=values[2].strip(),
                document_title=values[3].strip(),
                organization=values[4].strip(),
                authority_level=values[5].strip(),
                equipment=values[6].strip(),
                equipment_subtype=values[7].strip(),
                topic=values[8].strip(),
                subtopic=values[9].strip(),
                knowledge_type=values[10].strip(),
                verified_information=values[11].strip(),
                procedure=values[12].strip(),
                frequency=values[13].strip(),
                technical_limit_value=values[14].strip(),
                safety_information=values[15].strip(),
                troubleshooting_failure_information=values[16].strip(),
                applicability=values[17].strip(),
                pdf_page=values[18].strip(),
                source_section=values[19].strip(),
                notes=values[20].strip(),
            )
        except Exception:
            malformed.append(row_number)
            continue

        chunks.append(chunk)

    report = LoadReport(
        total_rows_seen=total_seen,
        loaded=len(chunks),
        malformed_row_numbers=malformed,
        duplicate_chunk_ids=dup_ids,
        header_matches_expected=header_matches_expected,
    )
    return chunks, report


if __name__ == "__main__":
    path = Path(__file__).resolve().parents[2] / "KB_v1" / "knowledge_chunks.xlsx"
    chunks, report = load_chunks(str(path))
    print(f"Header matches expected schema: {report.header_matches_expected}")
    print(f"Rows seen: {report.total_rows_seen}")
    print(f"Chunks loaded: {report.loaded}")
    print(f"Malformed rows: {report.malformed_row_numbers}")
    print(f"Duplicate chunk IDs: {report.duplicate_chunk_ids}")
    print(f"PASS: {report.ok()}")
