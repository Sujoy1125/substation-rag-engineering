# RESEARCH_FREEZE.md — KB_v1.1

## Transition
KB_v1 → KB_v1.1, dated 2026-08-15. KB_v1 remains frozen and unmodified in its own folder; KB_v1.1 is a versioned, targeted derivative, not a replacement.

## Exact corrections
1. `D09-C0009` (PDF p.26-30, unchanged ID/span) — enriched `Verified Information`/`Frequency` with OLTC oil leakage/seepage (weekly) and DCRM (5-year) content, source-grounded on PDF p.29-30.
2. `D09-C0020` (PDF p.53-57, unchanged ID/span) — enriched `Verified Information` with the IS 1180 maximum-allowable-losses statement, source-grounded on PDF p.54.
3. `D09-C0050` (new, PDF p.33) — added to close a confirmed chunk-coverage gap between `D09-C0009` (ends p.30) and `D09-C0010` (begins p.34); source-grounded on PDF p.33.
4. `D09-C0009` — same chunk as (1); DCRM was the second gap addressed on that page range (Q020).

No other chunk, in D09 or any other document, was touched. No gold question, question ID, expected document, or expected page was modified — `rag_test_55.xlsx` is byte-for-byte identical to KB_v1's copy (hash-verified).

## Validation result
`validate_kb.py` (unmodified, original project script) run against this exact KB_v1.1 folder: **0 errors, 0 warnings, PASS.** All checks passed: no duplicate chunk IDs, no orphan source references, no empty-content chunks, valid PDF Page formatting, complete catalog entries, valid equipment-inventory cross-references, and full answerable-question evidence coverage (including the newly-added `D09-C0050` closing the p.33 gap).

## Benchmark result
Same unmodified retrieval pipeline (`BM25 -> EquipmentAwareRetriever -> DocumentDiversityReranker(cap=2, pool_k=30)`), same `run_benchmark()`/`is_hit()`, same 20 gold questions:

| Metric | KB_v1 | KB_v1.1 |
|---|---|---|
| R@1  | 0.55 | 0.65 |
| R@3  | 0.65 | 0.85 |
| R@5  | 0.75 | 1.00 |
| R@10 | 0.80 | 1.00 |
| MRR  | 0.611 | 0.768 |

## Zero-regression result
0 hit → miss across all 20 answerable questions. 4 miss → hit (Q009, Q017, Q019, Q020), 1 rank-improved side effect (Q016), 15 unchanged.

## Remaining research limitations
- 20-question benchmark is entirely D09-sourced (~2.9% of the corpus); no evidence yet on D01-D08 retrieval quality or on whether cap=2/pool_k=30 generalizes outside this slice.
- No generation, citation rendering, or confidence-gating layer has been built or evaluated yet — KB_v1.1 only certifies the retrieval layer.

## Frozen-file list
```
knowledge_chunks.xlsx, rag_test_55.xlsx, equipment_inventory.xlsx,
document_catalog.xlsx, document_summary.xlsx,
maintenance_data_dictionary.xlsx, observable_issues.xlsx,
maintenance_terminology.md, maintenance_workflow.md, source_notes.md,
validate_kb.py, documents/ (9 source PDFs), DATASET_MANIFEST.md,
RESEARCH_FREEZE.md
```

## Versioning rule
KB_v1.1 is now frozen. Any further change to any file listed above requires a new version, KB_v1.2 or later, with its own manifest, freeze note, and full validation + regression re-run. KB_v1.1 must not be edited in place.

---

**KB_v1.1 is the first corrected research baseline for engineering/model development.**
