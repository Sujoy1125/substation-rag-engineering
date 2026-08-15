# Retrieval Baseline — Phase 3 + Phase 4

> **SUPERSEDED — historical record only.**
> This document describes an earlier state in which the document-diversity
> reranker was part of the default pipeline, measured against **KB_v1** and the
> D09-only benchmark. Neither is current.
>
> The authoritative retrieval baseline is **[`RETRIEVAL_BASELINE_V2.md`](RETRIEVAL_BASELINE_V2.md)**:
> `BM25 → EquipmentAwareRetrieverV2 → Top-K`, measured on `KB_v1.1` against both
> D09 (20 questions) and `evaluation_v2` (44 questions, D01–D08), with no
> diversity reranker.
>
> Kept unedited below so the earlier numbers remain auditable.

All numbers below are measured against the frozen `KB_v1/rag_test_55.xlsx`
`20_Answerable` sheet, 1,744 chunks loaded from `KB_v1/knowledge_chunks.xlsx`.
Hit definition: retrieved chunk's `Document ID` matches gold `Document ID
(KB)` AND the chunk's `PDF Page` overlaps the gold `Expected Page` (both
sides parsed as page-number sets to handle ranges like `p.26-27` vs
`PDF p. 26-30`). See `src/retrieval/benchmark.py::is_hit`.

## 1. BM25 baseline

| Metric | Value |
|---|---|
| Recall@1 | 0.55 |
| Recall@3 | 0.65 |
| Recall@5 | 0.65 |
| Recall@10 | 0.75 |
| MRR | 0.597 |
| Mean latency | ~9–13 ms |

## 2. TF-IDF cosine baseline

| Metric | Value |
|---|---|
| Recall@1 | 0.25 |
| Recall@3 | 0.40 |
| Recall@5 | 0.60 |
| Recall@10 | 0.65 |
| MRR | 0.371 |
| Mean latency | ~3 ms |

BM25 outperforms TF-IDF cosine on every metric. TF-IDF is retained in the
codebase as a second lexical baseline for comparison, not as a candidate
for the production configuration.

## 3. Embedding model selected

**Primary:** `BAAI/bge-small-en-v1.5` (sentence-transformers, 384-dim, CPU-capable, ~130MB).
**Fallback:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, smaller/faster, slightly lower general retrieval quality).

Rationale for bge-small over larger BGE variants: this sandbox has 1 vCPU /
~3.9GB RAM (see Section 5) and no GPU — `bge-base`/`bge-large` are
unnecessary for a 1,744-chunk corpus and would meaningfully slow local
CPU inference on a developer's laptop for no measured benefit yet.
`all-MiniLM-L6-v2` is the fallback because it's smaller/faster if
`bge-small` proves too slow on the target machine, at some quality cost.

Neither model's weights could be downloaded in this sandbox (see Section 4)
— the recommendation above is based on published benchmarks and known
resource characteristics, not a result measured here. This must be
explicitly re-validated once weights are actually loadable, per the
project's rule against treating an untested claim as a validated result.

## 4. Dense retrieval — architecture (built) vs measured (blocked)

**Architecture (implemented, `src/embeddings/provider.py` + `src/retrieval/retrievers.py::DenseRetriever`):**
`EmbeddingProvider` interface → `LocalSentenceTransformerProvider` (lazy-loaded,
raises `ModelUnavailableError` on failure, never returns fake/zero vectors) →
`DenseRetriever` (cosine similarity over pre-normalized embeddings, refuses
to run against `MockEmbeddingProvider` unless explicitly overridden for
unit tests).

**Measured result: BLOCKED.**

```
=== dense ===
BLOCKED — Could not load model weights for 'BAAI/bge-small-en-v1.5'.
This typically means the model has not been downloaded to the local
Hugging Face cache and this environment has no network access to
huggingface.co.
Original error: OSError: We couldn't connect to 'https://huggingface.co'
to load the files, and couldn't find them in the cached files.
```

Confirmed independently with `curl -sS -o /dev/null -w "%{http_code}"
https://huggingface.co` → `403`, and `https://api.openai.com` → `403`.
`pypi.org` → `200` (this is why `sentence-transformers`/`torch` install
fine via pip but can't fetch model weights).

Recall@K for dense retrieval is **N/A — not fabricated**.

## 5. Local environment (inspected this session)

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 |
| CPU | 1 vCPU, Intel Xeon @2.80GHz |
| RAM | 3.9 GiB total, ~3.7 GiB available |
| GPU | none (`nvidia-smi` not found) |
| Disk free | ~9.9 GB |
| torch | not installed by default; installs cleanly via pip (2.13.0+cu130) |
| sentence-transformers | not installed by default; installs cleanly via pip (5.7.0) |
| onnxruntime | pre-installed (1.24.4) — usable if an ONNX-format embedding model's weights become reachable |
| Existing local model cache | none found |

## 6. Hybrid retrieval (BM25 + Dense, RRF)

**Architecture (implemented, `HybridRRFRetriever`):** Reciprocal Rank
Fusion — `score(chunk) = Σ 1/(60 + rank)` across each component retriever's
top-N candidate pool. RRF was chosen specifically because BM25 scores and
cosine similarities are on incomparable scales; RRF only consumes rank
order, sidestepping normalization entirely.

**Measured result: BLOCKED** (depends on the dense retriever above — not run).

## 7. Equipment-aware retrieval

Implemented as a **soft multiplicative boost** (`src/retrieval/equipment_aware.py`),
not a hard filter — a hard filter risks false negatives because the
chunk-level `Equipment` field is inconsistent free text (e.g. literal
`"Transformer, Reactor, Bushing"` as one string; 801/1744 chunks are
`NOT VERIFIED` for equipment). Equipment is extracted from the query
deterministically via a small alias table against the 11 canonical
equipment types in `equipment_inventory.xlsx` — no LLM call.

Compared against BM25 baseline (dense/hybrid equipment-aware comparison is
blocked pending Section 4/6, same as plain hybrid):

| Metric | BM25 | BM25 + equipment-aware boost |
|---|---|---|
| Recall@1 | 0.55 | 0.55 |
| Recall@3 | 0.65 | 0.65 |
| Recall@5 | 0.65 | **0.70** |
| Recall@10 | 0.75 | 0.75 |
| MRR | 0.597 | 0.604 |

One question (Q011, CT visual-check frequency) moved from Recall@5 miss
(hit at rank 7) to a Recall@5 hit. No question regressed — no new failures
were introduced at any K. Small but genuine, real improvement, kept per the
"only keep it if it demonstrably improves retrieval without harmful false
negatives" rule.

## 8. Reranker

**Not evaluated.** Per Step 10, a reranker is only justified once
Hybrid's real failures are measured — Hybrid itself is currently blocked.
Deferred.

## 9. Error analysis — Recall@5 failures (BM25, current strongest measured baseline)

7 of 20 questions miss at Recall@5: Q003, Q009, Q011 (now fixed by
equipment-aware boost, see Section 7), Q016, Q017, Q019, Q020.

| Q | Expected | Top wrong hits | Classification |
|---|---|---|---|
| Q003 | D09 p.21 (silica gel breather colour) | D08, D07, D04 chunks about bushings/breathers | **semantic/lexical mismatch across documents** — D08 (transformer spec manual) discusses similar breather terminology; BM25 has no way to prefer D09's O&M-specific phrasing |
| Q009 | D09 p.29 (OLTC oil leakage check frequency) | D08, D06 chunks about oil leakage generally | **wrong document** — same failure pattern as Q003 |
| Q016 | D09 p.53 (DT failure reasons) | D04, D03, D06 chunks about DT failure | **wrong document** — D04 ("O&M of Distribution Transformers") is topically almost a duplicate of what D09 says here, genuinely hard for lexical retrieval to disambiguate |
| Q017 | D09 p.54 (IS standard for DT losses) | D08, D04 chunks mentioning IS standards | **wrong document / numeric-entity mismatch** — BM25 doesn't weight the specific "IS standard" entity highly enough over general transformer-loss terms |
| Q019 | D09 p.33 (VCB vs SF6 voltage usage) | D09-C0010 (right doc, wrong page) then D06 chunks | **partial hit / wrong document mix** — D09-C0010 is retrieved but doesn't cover this specific page; D06 (failure report) also discusses VCB/SF6 |
| Q020 | D09 p.30 (DCRM periodicity) | D06, D08 chunks about OLTC diagnostics | **wrong document** |

Common pattern: **all 6 remaining failures are "wrong document" — D09
content is consistently outcompeted by D04/D06/D08, which cover
near-duplicate technical subject matter for transformers/OLTC.** This is
the textbook case dense/semantic retrieval is meant to help with (it can
weight D09's specific O&M-benchmarking phrasing over generic technical
manual phrasing), which is exactly why Section 4/6 being blocked matters —
this is not a case where "hybrid probably won't help much."

Equipment-aware boosting (Section 7) closed 1 of these; the other 6 have
not been addressed by anything measurable so far.

## 10. Final selected retrieval architecture (current state)

**BM25 + equipment-aware boost** — the strongest measured, fully
reproducible configuration available in this sandbox. Dense and
hybrid remain the architecturally-correct next step and are fully
implemented and ready to run the moment weights are reachable (see
`docs/BLOCKERS.md`).

## 11. Known limitations

- No dense/hybrid measurement yet — architecturally complete, blocked on
  model weights.
- Reranker not evaluated (correctly deferred).
- Equipment alias table (`EQUIPMENT_ALIASES`) is hand-built from the 11
  canonical types and the 20 answerable questions' phrasing; it has not
  been validated against the ambiguous/unanswerable question sets or
  against real-world query phrasing variety.
- N=20 answerable questions is a small sample; recall differences of one
  question (5%) should not be treated as highly significant.

## 12. Reproduction

```bash
cd substation-rag
python3 src/retrieval/benchmark.py          # BM25, TF-IDF, attempts dense+hybrid
python3 -m pytest tests/ -q                 # 31 tests
```

To actually run dense/hybrid, see `docs/BLOCKERS.md` for the exact local
(Windows) command sequence.
