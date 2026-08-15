# RESEARCH_FREEZE.md — Research → Engineering Handoff

This is the permanent record of what was verified, corrected, and decided before freezing
KB_v1. Read this once; after this point the dataset should not be silently modified — any
further correction creates KB_v1.1, KB_v1.2, etc.

## What was verified

1. **D09 is a real, public CEA document** (confirmed via web search + direct fetch of the
   source PDF from `cea.nic.in`), correctly catalogued as D09 with HIGH authority.
2. **All 1,744 knowledge chunks have unique IDs, non-empty content, and a valid source ID**
   that resolves to a real entry in `document_catalog.xlsx` (programmatic check, 0 failures).
3. **D09's page-numbering bug was root-caused, not patched.** The document's internal printed
   page label is offset from the physical PDF page by a constant +4 across the entire 129-page
   document — confirmed against all 115 internal page labels present in the file, zero
   exceptions to the offset itself. (An earlier working assumption that there were "2 exceptions"
   turned out to be an artifact of comparing single-page approximations against multi-page chunk
   ranges, not a real exception to the +4 offset — resolved during this pass.)
4. **All 49 D09 chunk page citations were corrected** using the verified offset, applied via a
   direct mapping built from the real PDF's own page-label text (not a blind find/replace).
5. **Every corrected D09 citation was spot-verified against the actual PDF text** — for each
   chunk, keyword terms from its Equipment/Topic/Section fields were checked against the actual
   text on the cited physical page ±1 (49/49 passed).
6. **All 20 of Member 3's independently-verified page corrections were cross-checked** against
   the chunk-level corrections above — 100% consistent (every question's verified page falls
   within its corresponding chunk's corrected page range).
7. **The correction was propagated** into `equipment_inventory.xlsx`'s D09 chunk/page references
   (rebuilt authoritatively from chunk IDs, not regex-guessed) — 8 rows updated, all now
   consistent with `knowledge_chunks.xlsx`.
8. **Every answerable question has real supporting evidence** — programmatically confirmed each
   of the 20 answerable questions' cited page is covered by an actual D09 chunk.
9. **Unanswerable question justifications were checked for specificity** — 19/20 give a specific,
   falsifiable reason (manufacturer-specific value, out of document scope, etc.); 1 (U016, battery
   market pricing) has a slightly generic justification but the underlying classification is
   correct — commercial pricing data is not maintenance-procedure content by definition, low risk.
10. **Source catalog completeness confirmed** — all 9 documents have a source URL and an
    authority level; no orphaned or missing references.
11. **Cross-file consistency validated end-to-end** via `validate_kb.py` — catalog → chunks →
    equipment inventory → evaluation questions, 0 errors on the frozen snapshot.

## What was corrected
- 49 D09 chunk `PDF Page` citations in `knowledge_chunks.xlsx` (internal label → physical page).
- 8 D09 chunk/page references in `equipment_inventory.xlsx`, rebuilt from corrected chunk IDs.
- (Member 3's `rag_test_55.xlsx` corrections were already done before this pass — no change
  needed here, only cross-validated.)

## Decisions made
- **Physical PDF page number is the one authoritative page-citation convention project-wide.**
  D01–D08 already used this convention; D09 is now aligned to match. Do not introduce a second
  convention later (e.g. citing D09 by its internal printed label) — always use physical page.
- **The 5-conflicting-questions sheet is intentionally left empty**, per the explicit team rule
  against fabricating conflicts. This is a decision, not an oversight — documented in the sheet
  itself and here.
- **Chapters 3 (partial), 5, 7, 8 of D09 remain unchunked.** This is a deliberate scope decision,
  not a gap discovered late — none of the current 50-question evaluation set requires them, and
  expanding coverage further is explicitly classified as deferrable (see Manifest).
- **`questions_100_plus.xlsx` and `evaluator_questions.xlsx` are excluded from this freeze.**
  They were flagged in a prior pass as needing the same reconciliation the 20-question set
  received, and that reconciliation hasn't happened yet. Freezing them now would freeze a known
  inaccuracy. They can be added to KB_v1.1 once reconciled.

## Definitions used for this evaluation set
- **Answerable:** the question's answer is explicitly present in a specific, citable chunk with
  a verified page reference.
- **Unanswerable:** the answer genuinely does not exist anywhere in the current 9-document
  corpus (not merely "the retriever might fail to find it") — verified by a stated, specific
  reason (out of scope, manufacturer-specific, not yet standardized, etc.).
- **Ambiguous:** the question has no single correct interpretation without further
  specification (equipment subtype, frequency tier, substation manning status, etc.) — verified
  by a stated reason plus an "ideal system behavior" (clarify or surface multiple options).
- **Conflicting:** two or more sources give different guidance for the same equipment/parameter,
  distinguishable by version/date. None currently confirmed to exist in this corpus.

## Known limitations (see DATASET_MANIFEST.md for full list)
D09 chapter coverage is partial; three equipment types have no sourced maintenance frequency;
the 100+ question bank is not yet reconciled. None of these block the current MVP evaluation
scope.

## Freeze rule
Do not silently modify any file in this `KB_v1/` folder from this point forward. Any further
correction — including fixing the `questions_100_plus.xlsx` reconciliation, expanding D09
chapter coverage, or adding a 10th document — creates a new versioned folder (`KB_v1.1/`, etc.)
with its own manifest and freeze note, so the evaluation results tied to KB_v1 stay reproducible.
