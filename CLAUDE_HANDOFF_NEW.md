# CLAUDE_HANDOFF.md — Substation RAG Engineering

## 1. Project objective
```
Authoritative technical documents
        -> Knowledge Base
        -> Retrieval
        -> Evidence-grounded generation
        -> Confidence gate
        -> Answer / Abstain / Clarify
```
Generation, confidence gating, and answer/abstain/clarify are NOT built yet.

## 2. Current architecture
Retrieval pipeline (validated, do not modify without new experimental evidence):
```
BM25 -> EquipmentAwareRetriever -> DocumentDiversityReranker(cap=2, pool_k=30) -> Top-10
```
- `src/retrieval/retrievers.py` — BM25Retriever, TfidfRetriever, DenseRetriever, HybridRRFRetriever (unmodified throughout this project's history)
- `src/retrieval/equipment_aware.py` — EquipmentAwareRetriever (unmodified)
- `src/retrieval/document_diversity.py` — DocumentDiversityReranker (new this project; cap/pool_k as above)
- `src/retrieval/benchmark.py` — run_benchmark(), is_hit(), page_numbers(), print_result() (unmodified except wiring in new retriever configs into main())
- `src/retrieval/gold_questions.py`, `src/ingestion/kb_loader.py` — unmodified
- `src/common/chunk.py` — Chunk dataclass, searchable_text(), citation() (unmodified)

## 3. Frozen dataset versions
- **KB_v1** — original frozen research dataset. Untouched, still exists as-is.
- **KB_v1.1** — corrected engineering baseline, now also FROZEN. Any further KB change must create **KB_v1.2**, never edit KB_v1.1 in place.

## 4. Exact KB_v1.1 state
- 1,745 knowledge chunks (KB_v1 had 1,744)
- 20 answerable evaluation questions in `rag_test_55.xlsx` — byte-for-byte (SHA-256) identical to KB_v1's copy; no gold question/ID/expected-doc/expected-page was touched
- All non-chunk KB_v1 files (`equipment_inventory.xlsx`, `document_catalog.xlsx`, `document_summary.xlsx`, `maintenance_data_dictionary.xlsx`, `observable_issues.xlsx`, `maintenance_terminology.md`, `maintenance_workflow.md`, `source_notes.md`, `validate_kb.py`, `documents/` — 9 source PDFs) copied forward byte-for-byte, unmodified
- `validate_kb.py` (unmodified, original project script) run against KB_v1.1: **0 errors, 0 warnings, exit code 0**

### D09 corrections in KB_v1.1 (all verified directly against the source PDF)
| Question | Chunk | Change | Grounded on |
|---|---|---|---|
| Q009 | D09-C0009 (unchanged span p.26-30) | Enriched Verified Information/Frequency with OLTC oil leakage/seepage (weekly) | PDF p.29 |
| Q020 | D09-C0009 (same chunk as Q009) | Enriched with DCRM periodicity (5 years) | PDF p.30 |
| Q017 | D09-C0020 (unchanged span p.53-57) | Enriched Verified Information with IS 1180 max-allowable-losses statement | PDF p.54 |
| Q019 | D09-C0050 (NEW chunk) | Created to close a chunk-coverage gap (D09-C0009 ends p.30, D09-C0010 begins p.34 — p.31-33 had zero coverage) | PDF p.33 |

## 5. Exact retrieval configuration
```python
DocumentDiversityReranker(
    EquipmentAwareRetriever(BM25Retriever()),
    cap=2,
    pool_k=30,
)
```
Registered in `benchmark.py` as retriever name `bm25_equipment_aware_diversity_cap2`, alongside unmodified `bm25`, `tfidf`, `bm25_equipment_aware`, `dense`, `hybrid_rrf`.

## 6. Exact benchmark results (20 answerable questions, same run_benchmark()/is_hit() throughout)
| Stage | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| BM25 | 0.55 | 0.65 | 0.65 | 0.75 | 0.597 |
| BM25 + EquipmentAware | 0.55 | 0.65 | 0.70 | 0.75 | 0.604 |
| + DocumentDiversityReranker (cap=2, pool_k=30), KB_v1 | 0.55 | 0.65 | 0.75 | 0.80 | 0.611 |
| Same pipeline, KB_v1.1 | **0.65** | **0.85** | **1.00** | **1.00** | **0.768** |

0 hit → miss regressions at every stage. 20/20 answerable questions now hit within top-10 on KB_v1.1.

## 7. Experiment history (do not repeat without new evidence)
1. **Title exclusion from BM25 index** — REJECTED. Removing `document_title` from searchable_text() dropped R@1 0.55→0.50 and did not recover any of the 5 known failures. Title-token overlap was not the dominant driver.
2. **Document-diversity reranking** — cap=2/pool_k=30 tested against cap=3/cap=4 (both no-ops). cap=2 accepted: R@5 0.70→0.75, MRR 0.604→0.611, 0 regressions. Recovered Q016 only; Q009/Q017/Q019/Q020 remained failing at this stage.
3. **KB_v1.1 corrections** — root-caused the 4 remaining failures as indexing gaps (Q009/Q017/Q020) and a true chunk-coverage gap (Q019), NOT ranking problems. Fixing them (see §4) closed all remaining failures without any further retrieval-algorithm change.

## 8. What has NOT been implemented yet
- Generation (no LLM answer-synthesis layer exists)
- Citation rendering in an end-user-facing sense (chunk.citation() exists as a method but is unused by any pipeline)
- Confidence gating / answer-abstain-clarify logic
- Unanswerable/ambiguous question evaluation (gold questions for these exist conceptually but are not wired into any benchmark)
- Evaluation beyond the 20 D09-only answerable questions

## 9. Current limitations
- The 20-question benchmark is 100% D09-sourced (~2.9% of the 1,745-chunk corpus). **UNVERIFIED**: whether the diversity-cap=2/pool_k=30 configuration generalizes to D01-D08.
- Dense retrieval and HybridRRF were exercised earlier in the project (via a real sentence-transformers model download) but were NOT re-run against KB_v1.1 — only the winning BM25+EquipmentAware+Diversity pipeline was. **UNVERIFIED**: dense/hybrid numbers on KB_v1.1.
- KB_v1.1's non-chunk files (equipment_inventory.xlsx etc.) were carried forward unmodified from KB_v1 but were not independently content-audited beyond what `validate_kb.py`'s existing checks cover.

## 10. Exact next task
```
Audit KB_v1.1 evaluation coverage
    -> Create evaluation_v2 (separate from KB_v1.1 — do not modify KB_v1.1)
    -> Target ~30-50 additional answerable, 5-10 unanswerable, 5-10 ambiguous
       questions across D01-D08 (exact counts depend on what evidence supports)
    -> Run the existing, unmodified retrieval pipeline against it
    -> Measure generalization
    -> Classify any failures using the same rigor as the D09 root-cause work
       (retrieval/ranking vs. indexing vs. KB coverage vs. evaluation vs.
       benchmark-implementation problem)
    -> Decide whether retrieval is sufficiently validated to move to generation
```
Do not start generation before this step unless evidence clearly shows it's unnecessary. Do not perform another retrieval optimization automatically — the D09 benchmark is saturated and can no longer distinguish retrieval changes; only a broader benchmark can.

## 11. Rules for the next Claude session
- KB_v1 and KB_v1.1 are both frozen — read-only. Any KB fix must create KB_v1.2.
- Do not modify `retrievers.py`, `equipment_aware.py`, `document_diversity.py`, `benchmark.py`'s `run_benchmark()`/`is_hit()`, or `gold_questions.py` without new experimental evidence justifying it.
- Any new experiment: controlled, one change at a time, benchmark before/after, report regressions, stop for approval before implementing.
- Never fabricate benchmark numbers. If blocked by environment/dependency issues, report the exact error rather than substituting synthetic results — unless explicitly told a temporary shim is acceptable for a controlled-experiment context (see §12).

## 12. Environment/dependency issues encountered
- The sandbox environment used for parts of this project has **no network egress** and **`rank_bm25` is not installed** (required by `retrievers.py`, `from rank_bm25 import BM25Okapi`). This blocks running `benchmark.py` directly in-sandbox.
- Workaround used (only for controlled-experiment turns, disclosed each time): a byte-for-byte reimplementation of `rank_bm25.BM25Okapi` (same formula, same defaults k1=1.5/b=0.75/epsilon=0.25) injected via `sys.modules` before importing the real, unmodified project code. Validated by reproducing the trusted human-run baseline numbers exactly (bit-for-bit on R@1/3/5/10/MRR and identical failure sets) before trusting it for new variants.
- All numbers reported as final/production results in this handoff (§6) came from the actual project owner's own venv (which has `rank_bm25` installed) running the real `benchmark.py`, OR from this Claude session re-running the same unmodified pipeline/loader files with the validated shim and cross-checking against the human-run numbers — not from unvalidated synthetic substitutes.
- `sentence-transformers`/dense retrieval requires a Hugging Face model download (`BAAI/bge-small-en-v1.5`); this succeeded in the project owner's environment (unauthenticated HF request) but has not been attempted in the network-restricted sandbox.

## 12b. evaluation_v2 — COMPLETE (this session)

`src/embeddings/provider.py` (previously missing, blocking all imports) was
supplied by the project owner and added unmodified. Verified the real
`benchmark.py` reproduces the exact KB_v1.1 D09 baseline (R@1 .65/R@3 .85/
R@5 1.00/R@10 1.00/MRR .768) bit-for-bit before trusting it on new data.

Built `evaluation_v2/` (44 answerable + 7 unanswerable + 6 ambiguous
questions across D01-D08, D09 excluded as saturated). Every answerable
question traces to one specific chunk's actual field content; validated
programmatically (chunk exists, doc ID matches, page overlaps) — 0 errors.
Unanswerable claims independently spot-checked against real chunk topic
distributions, not just document_summary.xlsx's say-so.

**Benchmark result (real pipeline, unmodified, run against evaluation_v2):**
R@1=0.75, R@3=0.82, R@5=0.86, R@10=0.91, MRR=0.795 (44 questions).
Does NOT match the saturated D09 numbers — retrieval generalizes
reasonably but not perfectly. 4 questions missed entirely in top-10;
2 more only hit at rank 7-9.

**Root cause of all 4 top-10 misses (isolated stage-by-stage — same
mechanism each time): a real bug in `EquipmentAwareRetriever`
(`src/retrieval/equipment_aware.py`), not a KB coverage gap or an
indexing gap on the base BM25 index.**
- All 4 gold chunks rank respectably in raw BM25 (position 8-12 of a
  30-pool) but get demoted by the equipment-aware re-scoring stage
  specifically (down to rank 16-30), which then keeps them out of the
  diversity reranker's top-10 window entirely.
- Two distinct failure modes inside `EquipmentAwareRetriever`:
  1. `extract_equipment()`'s substring match requires a literal space
     (`"surge arrester"`) and misses hyphenated query phrasing
     (`"surge-arrester"`) — query-side extraction bug.
  2. `EQUIPMENT_ALIASES` maps granular chunk-level equipment names
     (OLTC, Bushing, Reactor, etc.) many-to-one onto canonical
     `"Transformer"` for *query-side* extraction, but
     `_chunk_mentions_equipment()` requires the *chunk's* Equipment
     field to literally contain `"transformer"` as a substring — so a
     chunk tagged only `"OLTC, Bushing"` never receives the boost even
     when the query is genuinely about OLTC/bushings, while competing
     chunks literally tagged `"Transformer"` get boosted past it.
  3. A related, distinct issue (not a code bug, a KB metadata gap):
     2 of the 4 misses also have Equipment=`"NOT VERIFIED"` on the gold
     chunk itself (D03-C0006, D04-C0007) — these chunks can never
     receive the boost regardless of query wording, while competing
     chunks with populated Equipment fields do. This is a genuine
     KB_v1.1 metadata-completeness gap, not a retrieval-algorithm bug —
     candidate for KB_v1.2 if Equipment tagging is ever revisited, but
     NOT touched in this session (KB_v1.1 stays frozen).
- Classification: retrieval/ranking problem (A) for the code-level
  alias/extraction bugs; KB coverage/metadata problem (C, mild — content
  itself is present and correctly indexed, only the Equipment *metadata
  column* is sparse) for the NOT VERIFIED cases. **No evaluation-ground-
  truth (D) or benchmark-implementation (E) issues found** — re-verified
  each of the 4 failing questions' evidence against the source chunk text
  directly before concluding this.
- **Not touched: `equipment_aware.py`, `retrievers.py`,
  `document_diversity.py` remain byte-for-byte unmodified this session**,
  per the "diagnose before optimizing" rule. This is a proposal, not an
  applied fix.
- **Proposed controlled experiment (not yet run, needs approval):** two
  small, independently-testable changes to `extract_equipment()` /
  `_chunk_mentions_equipment()` — (i) normalize hyphens/word-boundaries
  before alias substring matching, (ii) when checking whether a chunk
  matches a canonical equipment type, also accept any of that canonical
  type's own alias tokens appearing in the chunk's Equipment field (not
  just the canonical name itself). Benchmark before/after on both
  evaluation_v2 AND the D09 sheet (regression check), report deltas,
  do not accept without an explicit 0-regression result on D09.

Full per-question results, by-document breakdown, and raw retrieved
chunk IDs for every question saved to
`experiments/evaluation_v2_benchmark_<timestamp>.json`.

**Decision reached: retrieval does NOT yet generalize cleanly enough to
freeze and move to generation.** The EquipmentAwareRetriever bug is
real, reproducible, and localized — worth fixing via the controlled
experiment above before declaring retrieval frozen. Diversity-cap=2 is
not implicated (bug is upstream of it) and needs no re-litigation yet.

## 12c. Exact next action for the next session
1. Get approval for the proposed `extract_equipment()` /
   `_chunk_mentions_equipment()` experiment above (or a different fix).
2. Implement it as a NEW retriever variant (don't overwrite the
   validated one in place) so both can be A/B'd.
3. Re-run both `experiments/sanity_check_baseline.py` (D09, must stay
   1.00/1.00/0.768, zero regressions) and
   `experiments/run_evaluation_v2_benchmark.py` (evaluation_v2, should
   improve on 0.86/0.91/0.795) before accepting.
4. Only after an accepted fix (or an explicit decision that the current
   pipeline is "good enough"): revisit whether retrieval is ready to
   freeze and generation work can start.

## 13. Commands to reproduce the current benchmark (owner's environment)
```
python src/retrieval/benchmark.py
```
Run from the project root, in the venv containing `rank_bm25`, `scikit-learn`, `sentence-transformers`, `torch`, `openpyxl`, `pandas`. Produces a timestamped JSON under `experiments/`. Point `load_chunks()`/`main()` at KB_v1.1's `knowledge_chunks.xlsx` and `rag_test_55.xlsx` (not KB_v1's) to reproduce the §6 KB_v1.1 row — **UNVERIFIED**: whether `benchmark.py`'s `main()` currently hardcodes a KB_v1 path or takes one as an argument; check before assuming.
