"""Validates evaluation_v2 against the actual KB_v1.1 chunks:
- unique question IDs, no duplicate questions
- every referenced chunk ID actually exists in knowledge_chunks.xlsx
- every referenced document ID actually exists
- the referenced page overlaps the chunk's actual PDF page (chunk-evidence
  is real, not hallucinated)
- cross-file consistency (doc ID on the chunk matches doc ID on the question)
"""
import re
import sys
from pathlib import Path
import openpyxl
import pandas as pd

KB = Path(__file__).resolve().parents[1] / "KB_v1.1_extracted" / "KB_v1.1_final"
OUT = Path(__file__).resolve().parent

chunks_df = pd.read_excel(KB / "knowledge_chunks.xlsx")
chunks_by_id = {row["Chunk ID"]: row for _, row in chunks_df.iterrows()}
valid_doc_ids = set(chunks_df["Document ID"].unique())

_NUM_RE = re.compile(r"\d+")
def page_nums(s):
    nums = [int(n) for n in _NUM_RE.findall(str(s))]
    if len(nums) >= 2:
        return set(range(min(nums), max(nums) + 1))
    return set(nums)

errors = []
warnings = []

def load_sheet(path, header_len):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header, data = rows[0], rows[1:]
    return header, [r for r in data if r and r[0]]

# --- answerable ---
header, rows = load_sheet(OUT / "answerable.xlsx", 11)
seen_ids = set()
seen_questions = set()
for r in rows:
    qid, question, ans, expdoc, docid, page, section, diff, avail, chunkid, evidence = r
    if qid in seen_ids:
        errors.append(f"[answerable] duplicate question_id {qid}")
    seen_ids.add(qid)
    qnorm = question.strip().lower()
    if qnorm in seen_questions:
        errors.append(f"[answerable] duplicate question text for {qid}")
    seen_questions.add(qnorm)

    if docid not in valid_doc_ids:
        errors.append(f"[answerable] {qid}: document id {docid} not in KB")
        continue

    if chunkid not in chunks_by_id:
        errors.append(f"[answerable] {qid}: chunk id {chunkid} does not exist in knowledge_chunks.xlsx")
        continue

    chunk_row = chunks_by_id[chunkid]
    if chunk_row["Document ID"] != docid:
        errors.append(f"[answerable] {qid}: chunk {chunkid} belongs to {chunk_row['Document ID']}, not {docid}")

    chunk_pages = page_nums(chunk_row["PDF Page"])
    q_pages = page_nums(page)
    if not (chunk_pages & q_pages):
        errors.append(f"[answerable] {qid}: expected page {page} does not overlap chunk {chunkid}'s actual page {chunk_row['PDF Page']}")

print(f"Answerable: {len(rows)} questions, {len(seen_ids)} unique IDs")

# --- unanswerable ---
header, rows = load_sheet(OUT / "unanswerable.xlsx", 6)
seen_ids2 = set()
for r in rows:
    qid = r[0]
    if qid in seen_ids2:
        errors.append(f"[unanswerable] duplicate question_id {qid}")
    seen_ids2.add(qid)
print(f"Unanswerable: {len(rows)} questions, {len(seen_ids2)} unique IDs")

# --- ambiguous ---
header, rows = load_sheet(OUT / "ambiguous.xlsx", 5)
seen_ids3 = set()
for r in rows:
    qid = r[0]
    if qid in seen_ids3:
        errors.append(f"[ambiguous] duplicate question_id {qid}")
    seen_ids3.add(qid)
print(f"Ambiguous: {len(rows)} questions, {len(seen_ids3)} unique IDs")

# cross-class ID collision check
all_ids = seen_ids | seen_ids2 | seen_ids3
if len(all_ids) != len(seen_ids) + len(seen_ids2) + len(seen_ids3):
    errors.append("Cross-class question_id collision detected")

report_lines = []
report_lines.append("=" * 60)
report_lines.append(f"EVALUATION_V2 VALIDATION REPORT")
report_lines.append("=" * 60)
report_lines.append(f"Answerable questions:   {len(seen_ids)}")
report_lines.append(f"Unanswerable questions: {len(seen_ids2)}")
report_lines.append(f"Ambiguous questions:    {len(seen_ids3)}")
report_lines.append(f"Total:                  {len(all_ids)}")
report_lines.append("")
if errors:
    report_lines.append(f"ERRORS ({len(errors)}):")
    for e in errors:
        report_lines.append(f"  - {e}")
else:
    report_lines.append("ERRORS: 0")
if warnings:
    report_lines.append(f"\nWARNINGS ({len(warnings)}):")
    for w in warnings:
        report_lines.append(f"  - {w}")
else:
    report_lines.append("WARNINGS: 0")
report_lines.append("")
report_lines.append("RESULT: " + ("FAIL" if errors else "PASS"))

report = "\n".join(report_lines)
print("\n" + report)
(OUT / "validation_report.txt").write_text(report)

sys.exit(1 if errors else 0)
