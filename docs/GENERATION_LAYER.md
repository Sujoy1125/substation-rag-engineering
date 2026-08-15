# Generation Layer

**Status:** implemented, plumbing-verified offline. **Not yet evaluated against
a live model** — no generation quality metric is claimed anywhere.
**Retrieval baseline in use:** `BM25 → EquipmentAwareRetrieverV2 → Top-K`
(see [`RETRIEVAL_BASELINE_V2.md`](RETRIEVAL_BASELINE_V2.md))

---

## Pipeline

```
question
   │
   ▼
EquipmentAwareRetrieverV2(BM25Retriever)      src/retrieval/
   │  top-K RetrievedResult  (K = 5)
   ▼
build_context()                               src/generation/context.py
   │  EvidenceContext — chunks labelled [E1]…[EK], sentinels dropped
   ▼
build_messages()                              src/generation/prompt.py
   │  evidence-only system prompt + JSON output contract
   ▼
LLMClient.complete()                          src/generation/llm.py
   │  raw JSON reply (untrusted)
   ▼
build_answer()                                src/generation/answer.py
   │  labels verified, citations resolved from Chunk objects
   ▼
GeneratedAnswer
   ANSWER │ INSUFFICIENT_EVIDENCE │ NEEDS_CLARIFICATION │ UNSUPPORTED │ PARSE_ERROR
```

`RAGPipeline` (`src/generation/pipeline.py`) wires this together. The retriever
is injected, not hard-wired, so changing the baseline is a one-line change at
the call site.

---

## The one idea this layer is built around

**The model never produces a citation. It produces a label.**

Each retrieved chunk is presented as `[E1]`, `[E2]`, … The model cites those
labels and nothing else — it is explicitly forbidden to write a page number or
section name as a citation. Afterwards, `src/citation/citations.py` looks each
label up in the evidence context *that was actually sent for that question* and
rebuilds the reference from the `Chunk` object: document id, title, page,
section, chunk id.

Consequences:

- A fabricated page number is **structurally impossible**, not merely
  discouraged. There is no path from generated text into a citation string.
- A fabricated *label* (`[E9]` when only `E1`–`E5` exist) is detectable exactly,
  and is recorded in `signals.invalid_labels`.
- Citations survive paraphrase. The model rewording a claim cannot corrupt the
  reference attached to it.

---

## Files

| File | Responsibility |
|---|---|
| `src/generation/context.py` | Chunks → labelled evidence block. Drops sentinel fields; merges duplicated field text. |
| `src/generation/prompt.py` | System prompt and JSON output contract. |
| `src/generation/llm.py` | `LLMClient` ABC; `OpenAIClient`, `AnthropicClient`, `ScriptedClient` (tests only). |
| `src/generation/answer.py` | Parses and validates the reply. Produces `GeneratedAnswer` + `GroundingSignals`. |
| `src/generation/pipeline.py` | `RAGPipeline` — retrieve → context → generate → cite. |
| `src/citation/citations.py` | `Citation`, label resolution, reference rendering. |
| `src/evaluation/eval_loader.py` | Loaders for all three evaluation_v2 question classes. |
| `experiments/generation_smoke.py` | Subset plumbing check, `--dry-run` (no key) or `--live`. |
| `scripts/ask.py` | Single-question CLI. `--evidence` inspects retrieval with no key. |

---

## Design decisions and why

### Sentinels are dropped, not rendered
KB_v1.1 uses `NOT VERIFIED`, `NOT COVERED`, `NOT APPLICABLE`, `N/A` to mean
"checked, nothing there". Rendering them into the prompt invites the model to
read them as content. A chunk whose every content field is a sentinel is shown
as explicitly empty rather than as a blank block the model might fill from its
own knowledge.

### Duplicated field text is shown once
A single extracted paragraph is often stored under both *Verified Information*
and *Troubleshooting / Failure Information*. Printing it twice wastes prompt
budget and makes one source look like two corroborating statements. Both field
labels are shown against one copy of the text. This cut the evidence block by
roughly 25% on the questions checked.

### `top_k = 5`
Recall@3 and Recall@5 are both 0.909 on `evaluation_v2` — no question has its
sole gold hit at rank 4 or 5 — and ranks 6–10 add 0.023. Five is the smallest K
that loses nothing to ranks 1–5, and it keeps the prompt short enough that a
weak answer cannot hide inside a wall of marginally related extracts.

### The prompt never asks for a confidence score
A self-reported confidence is an uncalibrated number that downstream stages
would be tempted to trust. Confidence must be computed from observable signals
and calibrated against `evaluation_v2`. A test asserts the prompt contains no
such request.

### Empty retrieval short-circuits before the model
If retrieval returns nothing, the pipeline abstains without spending a request.
Sending an empty evidence block is a direct invitation to answer from parametric
knowledge — the exact failure this system exists to prevent.

### No orchestration framework
Generation is one request with one prompt. LangChain or an agent framework
would add a dependency and a layer of indirection without removing any work.

---

## Statuses

| Status | Set by | Meaning |
|---|---|---|
| `ANSWER` | model | Grounded answer with at least one valid citation. |
| `INSUFFICIENT_EVIDENCE` | model | Evidence does not support an answer. Abstention. |
| `NEEDS_CLARIFICATION` | model | Question cannot be safely answered as asked. |
| `UNSUPPORTED` | **code** | Model claimed `ANSWER` but cited no valid label, or returned empty text. |
| `PARSE_ERROR` | **code** | Model replied, but not with usable JSON. |
| `LLM_ERROR` | **code** | Model never reached — network failure, bad key, rate limit. |

`LLM_ERROR` is kept strictly separate from `PARSE_ERROR`: one is a statement
about the plumbing, the other about the model's output, and conflating them
sends you debugging the wrong layer. A question that returns `LLM_ERROR` was
never actually asked, so `EvaluationReport.is_valid` goes false and the report
prints an INVALID banner rather than rates — otherwise an all-failed run
displays `unsafe assertions 0.000`, which reads as a perfect safety score.
The runner also aborts after three consecutive failures rather than burning
57 API calls against a dead connection.

An unrecognised status string degrades to `INSUFFICIENT_EVIDENCE` — abstention
is the safe default.

`UNSUPPORTED` is a **structural** rule, not a tuned threshold: an answer citing
nothing that was retrieved is, by this system's own definition, not
evidence-grounded. It is the only place the code overrides the model, and it
involves no calibrated constant.

---

## Grounding signals — measured, not yet weighted

`GeneratedAnswer.signals` records, per question:

```
n_claims                      n_evidence_items          top_retrieval_score
n_supported_claims            n_evidence_cited          min_cited_rank
citation_coverage             evidence_utilisation      max_cited_rank
n_invalid_labels              distinct_documents_cited  authority_levels_cited
invalid_labels                distinct_documents_retrieved   conflict_reported
```

**No confidence score is computed and no threshold is applied.** These are the
raw inputs the confidence layer will combine once its weights are calibrated
against `evaluation_v2`. Inventing weights now would bake uncalibrated constants
into the foundation, and the resulting gate could not be defended.

---

## Conflicts and ambiguity

The prompt asks for genuine conflicts to be reported with both sets of labels
rather than silently resolved, and explicitly warns that extracts covering
different equipment, voltage classes or conditions are a *distinction*, not a
conflict. A conflict flagged with no description is discarded by
`build_answer()` — a checkbox is not a finding. There is no conflict question
set, and none should be fabricated to create one.

Ambiguity returns `NEEDS_CLARIFICATION` with a specific question. Ambiguous is
kept strictly distinct from unanswerable: they are different system behaviours
and collapsing them would destroy the distinction the confidence gate exists to
demonstrate.

---

## Running it

Dry run — no API key, no cost. Verifies retrieval, context assembly and prompt
construction against the frozen KB:

```powershell
python experiments\generation_smoke.py -n 5
python experiments\generation_smoke.py --ids V2-001 --show-prompt
python scripts\ask.py --evidence "transformer oil BDV test frequency"
```

Live — requires `OPENAI_API_KEY` in `.env`:

```powershell
python experiments\generation_smoke.py --live -n 5
python scripts\ask.py "How often should transformer oil BDV be tested?"
```

Tests (offline, deterministic, no API cost — they run against `ScriptedClient`):

```powershell
python -m pytest -q
```

---

## What is verified, and what is not

**Verified.** 42 offline tests covering context assembly, sentinel handling,
field deduplication, prompt content, JSON recovery (plain / fenced /
prose-wrapped / brace-in-string / garbage), label resolution (valid, invalid,
case and bracket tolerance, dedup), the `UNSUPPORTED` downgrade, partial
citation coverage, conflict handling, status degradation, and the full pipeline
end to end. Dry runs confirm the pipeline builds real prompts from the frozen
1,745-chunk KB. Suite total: **74 passed** (32 pre-existing + 42 new).

**Not verified.** Nothing about answer quality. No live model has been run
against this prompt, so there is no measurement of answer correctness,
hallucination rate, abstention accuracy, or clarification accuracy. The
`--live` path exists and the JSON record it writes is labelled as a plumbing
check, not an evaluation.

---

## Next

1. `--live` smoke run on ~5 answerable questions once a key is available;
   inspect prompt-following and citation validity by hand before scaling up.
2. Extend to all 44 answerable questions; score answer correctness and citation
   correctness against `evaluation_v2` gold.
3. Add the 7 unanswerable questions; measure correct abstention and false-answer
   rate.
4. Add the 6 ambiguous questions; measure correct clarification.
5. Only then build the confidence gate, calibrating weights on those measured
   signals — and compare gated vs ungated RAG on the same question set.
