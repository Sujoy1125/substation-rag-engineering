# DATASET_MANIFEST.md — KB_v1.1

**Dataset version:** KB_v1.1
**Parent version:** KB_v1 (frozen, unmodified — preserved as-is; see KB_v1/DATASET_MANIFEST.md)
**Freeze date:** 2026-08-15
**Validation status:** PASS — `validate_kb.py` (unmodified, original project script) run against this exact folder, 0 errors, 0 warnings.

## Source documents (9)
Unchanged from KB_v1 — D01-D09, same files, same authority levels. D09 chunk count changes from 49 to 50 (see below); all other documents unchanged.

## Knowledge chunks
**1745** (KB_v1 had 1744; +1 new chunk, 2 chunks enriched — both changes confined to D09)

## Answerable evaluation questions
**20** (`rag_test_55.xlsx`, `20_Answerable` sheet — byte-for-byte identical to KB_v1's copy; confirmed via hash comparison, no gold question, ID, expected document, or expected page was touched)

## Retrieval configuration (validated pipeline)
```
BM25
  -> EquipmentAwareRetriever
  -> DocumentDiversityReranker
       cap = 2
       pool_k = 30
```

## Validated retrieval performance
| Metric | KB_v1 | KB_v1.1 |
|---|---|---|
| R@1  | 0.55 | 0.65 |
| R@3  | 0.65 | 0.85 |
| R@5  | 0.75 | 1.00 |
| R@10 | 0.80 | 1.00 |
| MRR  | 0.611 | 0.768 |

## Changes from KB_v1
- **Q009 indexing correction** — `D09-C0009` (unchanged ID, unchanged PDF p.26-30 span) enriched with OLTC oil leakage/seepage (weekly) and DCRM (5-year) content from PDF p.29-30, previously present in the source but omitted from the curated `Verified Information`/`Frequency` fields.
- **Q017 indexing correction** — `D09-C0020` (unchanged ID, unchanged PDF p.53-57 span) enriched with the IS 1180 maximum-allowable-losses statement from PDF p.54, previously omitted (original summary covered only the p.53 failure-causes content).
- **Q019 D09 coverage correction** — new chunk `D09-C0050` (PDF p.33 only) added to close a confirmed chunk-coverage gap: `D09-C0009` ends p.30, `D09-C0010` begins p.34, leaving p.31-33 uncovered in KB_v1. p.33 (Section 2.3 Circuit Breakers, VCB/SF6 voltage-level applicability) contains the needed evidence.
- **Q020 indexing correction** — same chunk/edit as Q009 (`D09-C0009`); DCRM was the second gap on that page range.

All four corrections verified directly against the source PDF (`documents/OM_Benchmarking_for_Discoms_CEA_Guidelines.pdf`) in this package, re-confirmed at freeze time.

## Regression
0 hit → miss regressions across all 20 answerable questions.
- miss → hit: Q009 (rank 2), Q017 (rank 4), Q019 (rank 1), Q020 (rank 1)
- rank improved (side effect, not separately targeted): Q016 (rank 9 → 2)
- all other 15 questions: unchanged rank

## Known limitations
- The answerable benchmark contains only 20 questions, all sourced from D09 (~2.9% of the 1745-chunk corpus). Retrieval performance on D01-D08 has not been separately evaluated.
- Broader KB generalization of the diversity-cap=2/pool_k=30 configuration has not yet been evaluated beyond this D09 slice.
- `equipment_inventory.xlsx`, `document_catalog.xlsx`, `document_summary.xlsx`, `maintenance_data_dictionary.xlsx`, `observable_issues.xlsx`, `maintenance_terminology.md`, `maintenance_workflow.md`, `source_notes.md`, and `documents/` are carried forward from KB_v1 byte-for-byte, unmodified.

## Files in this package
```
knowledge_chunks.xlsx        (modified — see Changes above)
rag_test_55.xlsx              (unchanged, hash-verified against KB_v1)
equipment_inventory.xlsx      (unchanged, carried from KB_v1)
document_catalog.xlsx         (unchanged, carried from KB_v1)
document_summary.xlsx         (unchanged, carried from KB_v1)
maintenance_data_dictionary.xlsx  (unchanged, carried from KB_v1)
observable_issues.xlsx        (unchanged, carried from KB_v1)
maintenance_terminology.md    (unchanged, carried from KB_v1)
maintenance_workflow.md       (unchanged, carried from KB_v1)
source_notes.md               (unchanged, carried from KB_v1)
validate_kb.py                (unchanged, carried from KB_v1)
documents/                    (unchanged, carried from KB_v1 — all 9 source PDFs)
DATASET_MANIFEST.md           (this file)
RESEARCH_FREEZE.md
```
