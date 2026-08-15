# DATASET_MANIFEST.md — KB_v1

**Freeze date:** 2026-08-14
**Version:** v1 (first frozen research-layer snapshot)
**Validation status:** PASS — `validate_kb.py` run against this exact folder, 0 errors, 0 warnings.

## Source documents (9)

| ID | File | Authority | Chunks |
|---|---|---|---|
| D01 | 765_powerplants.pdf | HIGH | 23 |
| D02 | CEA Safety Requirements for Construction, Operation & Maintenance of Electrical Plants and Electric Lines, Amendment Regulations 2022 | HIGH | 40 |
| D03 | format_failure_tf.pdf | MEDIUM | 12 |
| D04 | Guidelines for O&M of Distribution Transformers | HIGH | 170 |
| D05 | reg_elec_plants_lines.pdf | HIGH | 54 |
| D06 | SS_Report_on_SSE_failure_July23_Dec24 (with covering letter) | HIGH | 463 |
| D07 | Tender_Part-1 (BHEL) | HIGH for BHEL procedure content | 133 |
| D08 | Transformer Manual, Amendment 01 | HIGH | 800 |
| D09 | CEA Guidelines for Benchmarking of O&M Norms for Distribution Utilities, 2024 | HIGH | 49 |

**Total knowledge chunks: 1,744**

## Equipment inventory
11 equipment types (Transformer, Circuit Breaker, Isolator/Disconnector, CT, PT/CVT, Surge
Arrester, Busbar, Battery Bank, Protection Relay, Earthing System, Switchgear), each with
source-cited maintenance activities, observable issues, and — where D09 provides a direct
schedule — sourced maintenance frequency. Frequencies not directly given by any source are
explicitly marked `NOT VERIFIED` rather than inferred.

## Evaluation dataset
- 20 Answerable questions — every one verified to have a covering evidence chunk in the KB
  (validated programmatically, see `validate_kb.py` check 7).
- 20 Unanswerable questions — each with a specific justification (manufacturer-specific value,
  out of document scope, etc.), distinguishing "genuinely not in any source" from "retriever
  might just miss it."
- 10 Ambiguous questions — each with a stated reason for ambiguity and an "ideal system
  behavior" (clarify vs. surface multiple checklists).
- 5 Conflicting/version questions — **intentionally not created.** No genuine same-topic,
  different-version source conflict has been found in the current 9-document corpus. This is
  documented, not defaulted or skipped silently (see `5_Conflicting_PENDING` sheet).

## D09 page-numbering convention (important — read before citing any D09 page)
D09's PDF contains an internal printed page label ("Page | N") in the footer of each page, which
is **offset by exactly +4** from the physical PDF page index used by any PDF viewer's "go to
page" function, across the entire 129-page document (verified against all 115 internal page
labels found in the file — zero exceptions to the +4 offset itself).

**All page citations in this frozen dataset (`knowledge_chunks.xlsx`, `equipment_inventory.xlsx`,
`rag_test_55.xlsx`) use the physical PDF page number** (what you'd see in a PDF viewer / Ctrl+G),
**not** the document's internal printed label. If you ever manually cross-reference against the
document's own footer text, subtract 4 from the physical page to get the internal label.

D01–D08 page citations were already in physical-PDF-page format from Member 1's original work
and required no correction.

## Known limitations (intentionally deferred, not blockers)
- D09 chunking covers Chapters 1, 2, 4, and 6 only. Chapters 3 (partially — only 3.1/3.2 DT
  intro), 5 (Asset Management), 7 (Staff Roles), and 8 (O&M Expense Guidelines) are not chunked.
  Not required for the current 20-question evaluation set; defer unless a specific evaluation
  need arises.
- Equipment types Busbar, standalone Protection Relay, and generic Switchgear have no D09-sourced
  maintenance frequency (D09 doesn't give a universal schedule for these) — correctly left
  `NOT VERIFIED` in `equipment_inventory.xlsx` rather than inferred.
- `questions_100_plus.xlsx` and `evaluator_questions.xlsx` are not part of this frozen snapshot —
  they still need the same availability reconciliation the 20-question set already received.
