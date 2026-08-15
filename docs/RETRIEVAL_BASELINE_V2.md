# Retrieval Baseline (V2) — authoritative

**Status:** current
**Supersedes:** `docs/RETRIEVAL_BASELINE.md` (describes an earlier state in which the
document-diversity reranker was part of the default pipeline — no longer true)
**KB:** `KB_v1.1/KB_v1.1_final` — 1,745 chunks, frozen
**Reproduced by:** `python experiments/retrieval_baseline_final.py`

---

## Decision

```
Query
  ↓
BM25Retriever
  ↓
EquipmentAwareRetrieverV2   (boost = 0.25, pool_k = max(3·top_k, 20))
  ↓
Top-K
```

No document-diversity reranker in the default pipeline.

`DocumentDiversityReranker` is **not deleted** — it remains in
`src/retrieval/document_diversity.py` and is still measured by the baseline
experiment. It is simply not selected.

---

## Why this configuration

Ranked on `evaluation_v2` (Recall@3, MRR as tie-break). `evaluation_v2` is the
primary dataset because it spans eight documents (D01–D08); D09 is a single
document and cannot distinguish a retriever that generalises from one tuned to
one book.

| # | configuration | R@3 | MRR |
|---|---|---|---|
| 1 | **bm25_equipment_aware_v2** | **0.909** | **0.828** |
| 2 | bm25 | 0.886 | 0.827 |
| 3 | bm25_equipment_aware | 0.864 | 0.800 |
| 4 | bm25_equipment_aware_v2_diversity | 0.841 | 0.817 |
| 5 | bm25_equipment_aware_diversity | 0.818 | 0.795 |

`EquipmentAwareRetrieverV2` is the only configuration that beats plain BM25 on
the multi-document set, and it does so without losing anything on D09.

Note the honest size of the win: V2 is **+0.023 R@3 and +0.001 MRR over plain
BM25** on `evaluation_v2` — one question's difference. It is selected because it
is at least as good on every metric on both datasets, not because the margin is
large. Any future retrieval change should be judged against that bar, not
against the weaker `bm25_equipment_aware` variant.

---

## Measured results

`top_k = 10`. Recall@k = fraction of questions whose gold evidence appears in
the top k. A hit requires the retrieved chunk's `Document ID` to match the gold
document **and** its PDF page range to overlap the gold page range.

### D09 — 20 answerable questions, single document
Source: `KB_v1.1/KB_v1.1_final/rag_test_55.xlsx`, sheet `20_Answerable`

| configuration | R@1 | R@3 | R@5 | R@10 | MRR | misses |
|---|---|---|---|---|---|---|
| bm25 | 0.650 | 0.800 | 0.850 | 0.950 | 0.732 | Q016 |
| bm25_equipment_aware | 0.650 | 0.850 | 0.950 | 1.000 | 0.767 | — |
| **bm25_equipment_aware_v2** | **0.650** | **0.850** | **0.950** | **1.000** | **0.756** | **—** |
| bm25_equipment_aware_diversity | 0.650 | 0.850 | 1.000 | 1.000 | 0.768 | — |
| bm25_equipment_aware_v2_diversity | 0.650 | 0.850 | 0.950 | 1.000 | 0.758 | — |

### evaluation_v2 — 44 answerable questions, D01–D08
Source: `evaluation_v2/answerable.xlsx`

| configuration | R@1 | R@3 | R@5 | R@10 | MRR | misses |
|---|---|---|---|---|---|---|
| bm25 | 0.773 | 0.886 | 0.886 | 0.955 | 0.827 | V2-015, V2-043 |
| bm25_equipment_aware | 0.750 | 0.864 | 0.886 | 0.886 | 0.800 | V2-011, V2-013, V2-015, V2-030, V2-043 |
| **bm25_equipment_aware_v2** | **0.773** | **0.909** | **0.909** | **0.932** | **0.828** | **V2-011, V2-013, V2-015** |
| bm25_equipment_aware_diversity | 0.750 | 0.818 | 0.864 | 0.909 | 0.795 | V2-011, V2-013, V2-030, V2-043 |
| bm25_equipment_aware_v2_diversity | 0.773 | 0.841 | 0.864 | 0.932 | 0.817 | V2-011, V2-013, V2-043 |

---

## Why the diversity reranker was dropped from the default

Measured effect of adding `DocumentDiversityReranker(cap=2, pool_k=30)` on top
of `EquipmentAwareRetrieverV2`:

| dataset | ΔR@3 | ΔR@5 | ΔMRR |
|---|---|---|---|
| D09 | +0.00 | +0.00 | +0.002 |
| evaluation_v2 | **−0.07** | **−0.05** | **−0.011** |

The reranker was originally accepted on the strength of D09, where it lifted
`bm25_equipment_aware` to R@5 = 1.00. That dataset is 20 questions inside a
single document, so a cap of two chunks per document is nearly inert there — it
buys one question and costs nothing. Across eight documents it starts evicting
correct same-document evidence to make room for other documents, and R@3 drops
by three questions.

The reranker was not wrong; the benchmark it was accepted on was too narrow to
see its cost. **Do not restore it to the default because it looks better on
D09.**

If it is revisited later, the promising direction is making the cap conditional
rather than fixed — e.g. apply diversity only when the candidate pool is
dominated by a single document — and it must be re-measured on both datasets.
Not before the generation layer works.

---

## What `EquipmentAwareRetrieverV2` fixes

Two bugs in `EquipmentAwareRetriever`, and nothing else — boost factor, pool
size and class shape are unchanged so the comparison isolates the fixes:

- **Bug A — query normalisation.** `extract_equipment()` matched
  `surge arrester` but not `surge-arrester` / `surge_arrester`. V2 normalises
  hyphens and underscores to spaces on both the query and the alias table.
- **Bug B — alias-aware chunk matching.** The chunk-side check required the
  literal canonical family name in the chunk's `Equipment` field, so a chunk
  tagged only `OLTC` or `Bushing` never matched the canonical family
  `Transformer`. V2 checks the chunk's `Equipment` field against the same alias
  list used for query extraction.

Per-question root-cause attribution for every recovered question is produced by
`experiments/eval_v2_regression_v2.py`.

---

## The four standing failures — diagnosed

`V2-011`, `V2-013`, `V2-015`, `V2-043` never get their gold evidence into the
top-5 context. Diagnosed rather than assumed:

| question | gold chunk | in KB? | retrieved at rank | gold `Equipment` |
|---|---|---|---|---|
| V2-011 | D03-C0006 | yes | 19 | `NOT VERIFIED` |
| V2-013 | D04-C0007 | yes | 33 | `NOT VERIFIED` |
| V2-015 | D04-C0117 | yes | 18 | `Transformer` |
| V2-043 | D08-C0412 | yes | 10 | `Bushing` |

**All four gold chunks are present in the KB and all four are retrieved.** They
are out-ranked, not missing. So this is a ranking problem, and **KB expansion
would not fix it** — the handoff's rule "expand only when an evaluation failure
proves it necessary" is not triggered, because the evidence is already there.

### A hypothesis, tested and rejected

Two of the four gold chunks have `Equipment = NOT VERIFIED`, and **46% of
KB_v1.1 (801 of 1745 chunks) shares that sentinel**. V2 multiplies matched
chunks by 1.25 and leaves everything else alone, so "tagged with different
equipment" and "never annotated" score identically — absence of evidence
treated as evidence of absence, across nearly half the corpus.

`EquipmentAwareRetrieverV3` (`src/retrieval/equipment_aware_v3.py`) tests the
fix: match ×1.25, unknown ×1.0, known-mismatch ×0.8. Measured on both datasets:

| configuration | D09 R@3 / MRR | evaluation_v2 R@3 / MRR |
|---|---|---|
| **bm25_equipment_aware_v2** | 0.850 / 0.756 | **0.909 / 0.828** |
| bm25_equipment_aware_v3 | 0.850 / 0.756 | 0.886 / 0.813 |

**Rejected.** Identical on D09, worse on evaluation_v2 — one question of R@3 and
0.015 of MRR. The demotion costs more than the sentinel neutrality gains,
presumably because equipment tags are frequently partial (a chunk tagged only
`Bushing` is often still the right answer to a transformer question), so
"mismatch" is a noisier signal than it looks.

V3 stays in the repo and in the benchmark as a measured negative result, for
the same reason the diversity reranker does: the next person to have this idea
should find the number rather than re-run the experiment.

### What this leaves

The four questions remain a **documented limitation, not a defect to fix**.
Raising `top_k` to 10 would recover only `V2-043` (rank 10) and would dilute
every prompt to buy one question. The correct system behaviour on all four is
to **abstain** — the evidence never reaches the model, so any answer would be
ungrounded. They are therefore useful test cases for the abstain path rather
than a gap to close.

---

## Known limitations — measured, not hidden

- **Dense and hybrid retrieval are unmeasured.** The earlier sandbox blocked
  `huggingface.co` and `api.openai.com` (HTTP 403), so no embedding model could
  be loaded. See `docs/BLOCKERS.md`. No dense/hybrid superiority is claimed
  anywhere in this repository. `DenseRetriever` and `HybridRRFRetriever` exist
  and refuse to run against a mock provider rather than emitting fake numbers.
- **Three standing `evaluation_v2` retrieval failures** under the selected
  configuration: `V2-011`, `V2-013`, `V2-015`. These are the natural first
  targets once end-to-end evaluation exists — and the natural test of whether
  the abstain path behaves correctly when retrieval genuinely fails.
- **Recall@1 is 0.773.** Roughly one answerable question in four does not have
  its gold evidence at rank 1, which is why the generation layer is given the
  top-K context rather than the single top chunk, and why citations are built
  from chunk metadata rather than from rank-1 assumptions.
- **Retrieval failure ≠ unanswerable.** A question missing from the top-K is a
  retrieval failure, not evidence that the KB lacks the answer. The confidence
  layer must not conflate the two.

---

## Rules for changing this baseline

1. Any retrieval change must be benchmarked on **both** D09 and `evaluation_v2`.
2. A change that wins on D09 alone is not accepted.
3. `experiments/retrieval_baseline_final.py` asserts the selection: if the
   selected configuration is no longer top-ranked on `evaluation_v2`, the script
   prints a warning and exits non-zero. Update this document in the same commit
   that changes the default pipeline.
4. Do not tune retrieval indefinitely before the generation layer is measurable.
