# evaluation_v2 — Manifest

## Scope
Generalization test set for the BM25 -> EquipmentAwareRetriever ->
DocumentDiversityReranker(cap=2, pool_k=30) pipeline, covering D01-D08
(D09 is excluded — it is the saturated, already-fixed benchmark in
`rag_test_55.xlsx`, frozen and untouched).

Does NOT modify KB_v1 or KB_v1.1. All files live under this directory.

## Contents
- `answerable.xlsx` — 44 questions. Schema matches `rag_test_55.xlsx`'s
  `20_Answerable` sheet, plus two added columns (`Expected Chunk ID(s)`,
  `Evidence Basis`) so every question's grounding is auditable without
  re-deriving it.
- `unanswerable.xlsx` — 7 questions. Schema matches `20_Unanswerable`.
  Each verified two ways: (1) the document's chunk Topic distribution
  genuinely has no chunks of the needed kind, not just a failed keyword
  search; (2) cross-checked against `document_summary.xlsx`'s "What
  information is NOT covered" column, and independently re-verified in
  this session against the actual chunk table (not just trusted from the
  summary) — including catching a near-miss on D08 (2 chunks tagged
  "Circuit Breaker" that turned out to be incidental interlock mentions,
  not breaker specifications, so the unanswerable claim held).
- `ambiguous.xlsx` — 6 questions. Schema matches `10_Ambiguous`. Each
  ambiguity is grounded in genuinely conflicting/conditional real chunk
  content (not manufactured vagueness) — e.g. V2-A02 (surge arrester
  test frequency) reflects three actually-different, condition-triggered
  actions in D06, not an invented distinction.
- `author_questions.py` — generates the three xlsx files from hardcoded,
  chunk-sourced content (source of truth: direct pandas inspection of
  `knowledge_chunks.xlsx`, captured to `/tmp/d0*.txt` this session).
- `validate_evaluation_v2.py` — validates unique IDs, no duplicate
  questions, every chunk ID exists in KB_v1.1, document ID on the chunk
  matches the question's claimed document, and the claimed page overlaps
  the chunk's actual PDF page. Run before any benchmarking.
- `validation_report.txt` — output of the above; result: PASS, 0 errors.

## Document coverage (answerable questions)
D01=4, D02=5, D03=3, D04=6, D05=5, D06=8, D07=6, D08=7 (44 total).
Distribution roughly follows chunk-count/topic-richness per document,
not forced equality — D06 and D08 (the two largest, richest documents)
get the most coverage; D03 (12 chunks total, a reporting form) gets the
fewest.

## What was deliberately NOT done
- No questions authored from document titles/summaries alone — every
  answerable question traces to one specific chunk's Verified
  Information/Procedure/Frequency/Technical Limit field, quoted or
  closely paraphrased.
- No dense/hybrid retrieval run — out of scope for this phase (KB_v1.1
  handoff already flagged these as unverified/blocked in the network-
  restricted sandbox; this session did not attempt to unblock them since
  the task is BM25+EquipmentAware+Diversity only).
