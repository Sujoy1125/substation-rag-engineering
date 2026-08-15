# Substation O&M Assistant

Evidence-grounded question answering over a frozen knowledge base of substation
equipment maintenance and O&M documents. Every answer cites the source records
it was built from, and the system abstains rather than guessing when the
evidence does not support an answer.

**Smart India Hackathon 2026** · Knowledge base `KB_v1.1` (frozen) · 1,745
chunks from 9 Central Electricity Authority / Ministry of Power and BHEL
documents.

---

## Status — read this first

| Layer | State |
|---|---|
| Knowledge base | Frozen, validated, 0 errors |
| Retrieval | **Measured** on two datasets, six configurations, one reproducible command |
| Generation | Implemented, 198 offline tests, **no live run — no quality metrics exist** |
| Citation | Implemented and tested |
| Confidence gate | Implemented, **deliberately uncalibrated** — ships with zero weights and refuses to decide |
| Evaluation harness | Built and tested; the runs are pending |
| HTTP service and UI | Working |

**No number describing answer quality, hallucination rate, abstention accuracy
or the benefit of confidence gating appears anywhere in this repository**,
because those runs have not been performed. The harness, the frozen question
set and the scorers exist; the measurement does not. Where a capability could
not be measured, the code refuses to produce a number rather than producing an
approximate one.

---

## The problem

A general-purpose language model will answer any question about substation
maintenance immediately, fluently, and with no way for the reader to tell
whether it is right. In a domain where a wrong maintenance interval or a missed
isolation step has physical consequences, plausible is not the standard.

Three specific failures follow, and the design is a response to each:

1. **Fabrication** — inventing a figure, or attributing a real one to the wrong
   document and page.
2. **Answering the unanswerable** — the documents are silent; the model answers
   anyway from general knowledge.
3. **Answering the ambiguous** — "what is the maintenance interval?" has no
   single answer, and picking one silently is worse than asking.

## The approach

**The model never produces a citation. It produces a label.**

Retrieved extracts are presented as `[E1]`, `[E2]`, `[E3]`. The model refers to
them by label and nothing else. Afterwards the code looks each label up in the
evidence that was actually sent and rebuilds the reference — document, section,
page, chunk id — from the `Chunk` object. A fabricated page number is
*structurally impossible*, not merely discouraged: there is no channel through
which the model's text becomes a page number. An invented `[E9]` is a
detectably invalid label, recorded and counted.

Abstention is a first-class outcome with its own question class and its own
score. Ambiguity is met with a question rather than a guess.

---

## Quickstart

```bash
git clone https://github.com/Sujoy1125/substation-rag-engineering
cd substation-rag-engineering

python -m venv venv
venv\Scripts\activate              # Windows
source venv/bin/activate           # macOS / Linux

pip install -r requirements.txt    # ~100 MB, about a minute
python scripts/check_setup.py      # seven checks; no paid API call
python -m pytest -q                # 198 passed (or 194 + 4 skipped)

uvicorn src.api.service:build_default_app --factory --reload
```

Then open <http://127.0.0.1:8000/> for the interface or `/docs` for the API.

Full instructions, including how to package the project for a teammate without
shipping your API key: **[SETUP.md](SETUP.md)**.

### Without an API key

Most of the system works. This is worth understanding rather than treating as
a degraded mode:

| Works without a key | Needs a key |
|---|---|
| Retrieval over all 1,745 chunks | Generating an answer (`POST /ask`) |
| `GET /evidence/{chunk_id}` — source records | The evaluation runs |
| `GET /facets` — knowledge base coverage | |
| The full test suite | |
| `python scripts/ask.py "..." --evidence` | |

`POST /ask` returns **503 naming the specific cause** — out of credit, rate
limited, bad key, or network — because "the provider is unavailable" and "this
service is broken" need different reactions.

---

## Measured results — retrieval

Selected configuration: `BM25 → EquipmentAwareRetrieverV2`, boost 0.25 (matched
chunks scored ×1.25), candidate pool `max(3·top_k, 20)`, production `top_k = 5`.

Reproduce every row below with one command:

```bash
python experiments/retrieval_baseline_final.py
```

**evaluation_v2 — 44 answerable questions across D01–D08 (primary dataset)**

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| `bm25_equipment_aware_v2` **(selected)** | 0.773 | **0.909** | **0.909** | 0.932 | **0.828** |
| `bm25` | 0.773 | 0.886 | 0.886 | **0.955** | 0.827 |
| `bm25_equipment_aware` | 0.750 | 0.864 | 0.886 | 0.886 | 0.800 |
| `bm25_equipment_aware_v2_diversity` | 0.773 | 0.841 | 0.864 | 0.932 | 0.817 |
| `bm25_equipment_aware_diversity` | 0.750 | 0.818 | 0.864 | 0.909 | 0.795 |

**D09 — 20 answerable questions, single document (secondary)**

| Configuration | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| `bm25_equipment_aware_v2` **(selected)** | 0.650 | 0.850 | 0.950 | 1.000 | 0.756 |
| `bm25` | 0.650 | 0.800 | 0.850 | 0.950 | 0.732 |

**Read honestly:** on the primary dataset the selected retriever beats plain
BM25 by 0.023 Recall@3 — one question out of 44. It was selected because it is
at least as good on every metric on *both* datasets, not because the margin is
large. Plain BM25 is better at Recall@10.

### The ceiling on answer correctness

Gold evidence reaches the top-5 context for **40 of 44** answerable questions
(0.909), at mean rank 1.30 when present. Four questions — `V2-011`, `V2-013`,
`V2-015`, `V2-043` — never reach it, so they can only be answered correctly by
accident. Any correctness figure this project eventually reports must be read
against 0.909, not 1.000.

All four gold chunks are **present in the knowledge base and retrieved** — at
ranks 19, 33, 18 and 10. They are out-ranked, not missing. This is a ranking
problem; expanding the knowledge base would not fix it, and the correct system
behaviour on all four is to abstain.

### Two measured negative results, kept deliberately

**`DocumentDiversityReranker`** — capped chunks per document in the final
results. Accepted early on D09 alone, where it lifted Recall@5 to 1.00.
Re-measured across both datasets it costs **−0.07 Recall@3** on the
multi-document set. The reranker was not wrong; the benchmark it was accepted
on was too narrow to see its cost. Still in `src/retrieval/document_diversity.py`,
still measured, not selected.

**`EquipmentAwareRetrieverV3`** — three-way equipment scoring (match / unknown /
mismatch), on the reasoning that 46% of the corpus has `Equipment = NOT VERIFIED`
and V2 scores "tagged differently" and "never annotated" identically. Measured:
identical on D09, **worse on the primary dataset** (0.886 / 0.813 against V2's
0.909 / 0.828). Rejected, and kept in the repository and in the benchmark so the
next person to have this idea finds the number instead of re-running the
experiment.

---

## How this will be evaluated

Three question classes, scored separately and **never pooled into a single
accuracy number**:

| Class | n | Correct behaviour | Headline failure |
|---|---|---|---|
| Answerable | 44 | Answer, grounded and correctly cited | False answer — wrong location cited |
| Unanswerable | 7 | Abstain | Hallucination — answered anyway |
| Ambiguous | 6 | Ask for clarification | Answered instead of asking |

A system that answers everything scores 1.000 on the first and 0.000 on the
other two; one that abstains from everything does the reverse. Only the three
read together mean anything — which is the entire argument for confidence
gating.

**Two headline numbers, always reported together:**

```
unsafe assertion rate    answered when it should not have, over all 57
answer coverage          answered at all, over all 57
```

Refusing everything drives unsafe to 0.000 *and* coverage to 0.000. A test
constructs exactly that degenerate gate and asserts all three rates collapse.

**The split is frozen.** 40 calibration / 17 holdout, committed as
`evaluation_v2/split_v1.json`, stratified by difficulty and document, selected
by SHA-256 of each question id — deterministic, order-independent, and not
tunable by anyone hoping for a friendlier draw. A holdout chosen after looking
at performance is not a holdout. A test asserts the committed file still
matches what the code computes.

**The judge comes with a caveat.** Answer correctness is graded by an LLM judge,
which is itself uncalibrated: on a set where most answers are correct, a judge
that says CORRECT unconditionally scores ~0.90 raw agreement and 0.00 Cohen's
kappa (a test asserts this). So the protocol is judge all 44, hand-grade at
least 15, and report kappa beside every judged number.

---

## The confidence layer

Ships with **all weights 0.0 and both thresholds `None`**; `decide()` raises
`UncalibratedGateError`. A test asserts the shipped model is uncalibrated.

This is deliberate. Constants like `0.4 / 0.2 / 0.1`, chosen because they look
balanced, are — in a written report — indistinguishable from constants fitted to
data, and a reader has no way to tell them apart. The weights come from a live
calibration run, or they do not exist.

Eight signals, all bounded `[0,1]` and oriented so higher means more confident,
all computed from what the pipeline already recorded — nothing is asked of the
model:

`retrieval_strength` · `evidence_concentration` · `citation_coverage` ·
`citation_validity` · `evidence_utilisation` · `top_rank_cited` ·
`source_authority` · `answer_specificity`

The gate may never overrule a refusal. `INSUFFICIENT_EVIDENCE`,
`NEEDS_CLARIFICATION`, `UNSUPPORTED`, `PARSE_ERROR` and `LLM_ERROR` all force
their outcome; only `ANSWER` reaches the score. Two tests pin this using the
most permissive gate constructible — it can only ever make the system more
cautious.

Weights are fitted by logistic regression on the calibration split. Thresholds
are a **policy** choice: `--max-unsafe-rate` has no default, because a default
would quietly become the project's safety policy without anyone choosing it.

---

## Repository map

```
src/
  common/chunk.py            the Chunk type and the sentinel set
  ingestion/kb_loader.py     read-only loader for the frozen workbooks
  retrieval/                 BM25, equipment-aware v1/v2/v3, diversity, benchmark
  generation/                context, prompt, LLM clients, answer parsing, pipeline
  citation/citations.py      label resolution — the anti-fabrication invariant
  confidence/                signals, gate, calibration, gated comparison
  evaluation/                scorers, LLM judge, the frozen split
  api/service.py             FastAPI service; api/static/index.html is the UI
experiments/                 reproducible measurement scripts
scripts/                     ask, setup check, network diagnostic, SDK probe
tests/                       198 tests, all offline
docs/                        the written record — see below
KB_v1.1/                     the frozen knowledge base
evaluation_v2/               the frozen question sets and split
```

### Key commands

```bash
python experiments/retrieval_baseline_final.py     # the retrieval table above
python experiments/run_generation_eval.py          # dry run: retrieval ceiling, free
python experiments/run_generation_eval.py --live --limit 3
python scripts/ask.py "bushing inspection interval" --evidence
python scripts/check_setup.py
python scripts/diagnose_network.py                 # if generation fails
```

### Documentation

| Document | Contents |
|---|---|
| [SETUP.md](SETUP.md) | Installing and running, and packaging for a teammate |
| [docs/RETRIEVAL_BASELINE_V2.md](docs/RETRIEVAL_BASELINE_V2.md) | Authoritative retrieval results, both negative results, the four failures |
| [docs/GENERATION_LAYER.md](docs/GENERATION_LAYER.md) | Generation design and what is explicitly not verified |
| [docs/CONFIDENCE_LAYER.md](docs/CONFIDENCE_LAYER.md) | The eight signals, calibration design, why the weights are empty |
| [docs/EVALUATION_PLAN.md](docs/EVALUATION_PLAN.md) | Question classes, headline metrics, judge protocol, the split |
| [docs/DATA_STORE.md](docs/DATA_STORE.md) | Why there is no database, the designed pgvector schema, and its activation triggers |
| [docs/BLOCKERS.md](docs/BLOCKERS.md) | The dense-retrieval blocker |

---

## Known limitations

1. **No generation results exist.** The largest gap. Everything about answer
   quality is pending a live run.
2. **Dense and hybrid retrieval are unmeasured.** Implemented and unit-tested;
   `huggingface.co` returned HTTP 403 from our environment's egress proxy
   (confirmed twice), so no embedding weights could be downloaded. The dense
   retriever raises `ModelUnavailableError` rather than producing fake vectors,
   so no placeholder number can leak into a result. See
   [docs/BLOCKERS.md](docs/BLOCKERS.md).
3. **The margin over plain BM25 is one question** — 0.023 Recall@3 on 44.
4. **Recall@1 is 0.773**, which is why the model gets the top five chunks rather
   than the single best one.
5. **Four questions cannot be answered correctly except by accident.**
6. **The evaluation set is small** — 57 questions, 44 answerable. One question
   is 2.3%, so small differences should not be over-read.
7. **The D09 benchmark covers ~2.9% of the corpus** and is drawn from a single
   document.
8. **The equipment alias table is hand-built** and unvalidated against
   real-world query phrasing.
9. **`safety_information` is not uniformly precaution text** — on failure-report
   documents it can carry incident narrative. The interface labels that block
   "recorded in the cited source" rather than presenting it as advice.
10. **`source_authority` is near-constant** (1,733 of 1,745 chunks are HIGH), so
    calibration will likely give it little weight. A signal that turns out not to
    matter is a finding, not a bug.
11. **No database — by decision, not omission.** Retrieval is in-process over
    the frozen knowledge base: 510 ms to build the index, 1.8 ms median per
    query. pgvector's purpose is storing embeddings we cannot yet generate, and
    moving a frozen, hash-verified corpus into a mutable table would trade
    reproducibility for infrastructure we do not use. The integration is
    designed and its activation triggers are stated in
    [docs/DATA_STORE.md](docs/DATA_STORE.md).

---

## What remains

| Step | Command | Produces |
|---|---|---|
| 5 | `run_generation_eval.py --live --limit 3` | Three answers read by hand. Not a measurement. |
| 6 | `--live --split all --judge --review-sheet` | First real numbers across all 57 questions |
| 6c | hand-grade ≥15, then `--agreement FILE` | Cohen's kappa — without it, judged numbers cannot be quoted |
| 9 | `--live --split calibration` → `calibrate_confidence.py` | Fitted weights and thresholds |
| 10 | `--live --split holdout --gated` | The headline claim. **Run once.** |

Re-running the holdout after adjusting anything turns it into training data.

---

## Design decisions worth knowing

- **No LangChain, no agent framework.** Generation is one request with one
  prompt; an orchestration layer would add a dependency and indirection without
  removing work, and would make the prompt and the parsing harder to inspect.
- **The model is never asked how confident it is.** A self-reported number is
  uncalibrated, and once it exists every later stage is tempted to trust it. A
  test asserts the prompt never requests one.
- **Sentinels are dropped from the prompt, shown in the UI.** `NOT VERIFIED` as
  context invites the model to treat it as content; hiding it from a human
  auditing a citation would make coverage look better than it is.
- **`LLM_ERROR` is kept strictly separate from `PARSE_ERROR`.** An early run had
  all nine API calls fail and the report printed "unsafe assertions 0.000",
  which reads as a perfect safety score. A run containing unreached questions is
  now marked invalid, and the runner aborts after three consecutive failures.
- **Failed experiments stay in the repository** with their measurements
  attached, so nobody repeats them and the history does not look like every idea
  worked.
