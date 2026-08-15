# Confidence Layer

**Status:** built and unit-tested. **Deliberately uncalibrated** — weights are
all 0.0 and thresholds are `None` until fitted on real data. The gate raises
rather than deciding in that state.

```
GeneratedAnswer + retrieval
        ↓
   extract_signals()          src/confidence/signals.py    8 signals, all [0,1]
        ↓
   ConfidenceModel.score()    src/confidence/gate.py       weighted sum
        ↓
   ConfidenceGate.decide()    ANSWER │ ABSTAIN │ CLARIFY
```

---

## Why the weights are empty

Constants like `0.4 / 0.2 / 0.1` chosen because they look balanced are, in a
write-up, **indistinguishable from constants fitted to data**. A reviewer
cannot tell which they are being shown, so neither can be trusted.

`ConfidenceModel()` therefore ships with every weight at zero, and `decide()`
raises `UncalibratedGateError` rather than falling back to something plausible.
An uncalibrated gate that refuses to run cannot be mistaken for a calibrated
one. A test asserts the shipped model is uncalibrated.

---

## The eight signals

All bounded to `[0, 1]` and oriented so **higher = more confident** — otherwise
a linear combination is impossible to reason about and one feature dominates
through scale alone.

| signal | what it captures |
|---|---|
| `retrieval_strength` | gap between the top hit and the rest of the pool |
| `evidence_concentration` | share of retrieved chunks from the dominant document |
| `citation_coverage` | fraction of claims carrying a valid citation |
| `citation_validity` | falls away fast with each invented evidence label |
| `evidence_utilisation` | fraction of supplied evidence the answer used |
| `top_rank_cited` | best retrieval rank the answer actually leaned on |
| `source_authority` | KB authority level of cited chunks |
| `answer_specificity` | does the answer commit to a figure, interval or unit |

Two worth explaining:

**`retrieval_strength` uses a gap, not a raw score.** BM25 scores are unbounded
and corpus-dependent — 30 means nothing on its own. The distance between the
top chunk and its neighbours is scale-free, and a flat profile (many chunks
scoring alike) is exactly the case where retrieval could not discriminate and a
confident answer is least warranted.

**`source_authority` is near-constant today.** KB_v1.1 records `HIGH`
throughout, so calibration should be expected to give it little or no weight.
It is extracted anyway: it costs nothing and becomes meaningful the moment a
lower-authority source enters the KB, by which time the gate already knows how
to use it. A signal that turns out not to matter is a finding, not a bug.

---

## What the gate may not override

The gate decides how far to trust an *attempted* answer. It never second-guesses
a refusal:

| model status | decision | always |
|---|---|---|
| `INSUFFICIENT_EVIDENCE` | ABSTAIN | yes |
| `NEEDS_CLARIFICATION` | CLARIFY | yes |
| `UNSUPPORTED` | ABSTAIN | yes (structural, decided pre-gate) |
| `PARSE_ERROR` / `LLM_ERROR` | ABSTAIN | yes |
| `ANSWER` | scored | — |

Only `ANSWER` reaches the score. Letting a confidence number talk the system
*into* answering a question the model declined would invert the safety property
the design exists to provide. Two tests pin this: with both thresholds
at 0.0 — where everything scores "confident enough" — a model refusal still
abstains.

An `LLM_ERROR` question is returned with `confidence = None`, never a number:
it was never asked, so it has no confidence, and it must not be counted as a
confident abstention.

### The middle band

Between the two thresholds sits a deliberate gap: evidence good enough to have
found something, not good enough to assert it. That routes to **CLARIFY**,
because "I found related material but cannot tell whether it answers your
question" is more useful to a maintenance engineer than either a confident
guess or a flat refusal.

---

## Calibration

Fitted on the **calibration split only** (40 questions). Labels come from
ground truth, never from the model's opinion:

```
answerable    should_answer = answered AND cited the gold location
unanswerable  should_answer = False    any assertion is unsafe
ambiguous     should_answer = False    the right behaviour is to ask
```

**Weights** are logistic-regression coefficients over the signal vector
(`C=1.0`, `liblinear`, fixed seed — 40 questions and 8 features overfit
trivially without regularisation, and the solver is deterministic at this size
so the fit reproduces).

**Thresholds are a policy choice and are treated as one.** Where to put the cut
lines is not a statistical question — it is a decision about acceptable risk.
`--max-unsafe-rate` has **no default** and must be stated on the command line;
a default would quietly become the project's safety policy without anyone
choosing it. Thresholds are then swept to maximise useful answers subject to
that stated ceiling.

Fit the weights to the data; set the operating point by policy. Doing both at
once produces numbers nobody can explain.

`calibrate()` refuses to fit on fewer than 20 examples or on single-class
labels — a model fitted on three questions is worse than no model, because it
looks like one.

### Running it

```powershell
# 1. produce a calibration run (never the holdout)
python experiments\run_generation_eval.py --live --split calibration

# 2. fit; --max-unsafe-rate is a stated policy decision, not a default
python experiments\calibrate_confidence.py ^
    --from experiments\generation_eval_<stamp>.json ^
    --max-unsafe-rate 0.05
```

`calibrate_confidence.py` **refuses** a run that was not on the calibration
split, and refuses a run marked invalid by unreached questions. Fitting on
holdout data destroys the only out-of-sample evidence the project has.

The calibration-set rates it prints are **in-sample** and are labelled as such.
They are not results.

---

## What is verified, and what is not

**Verified.** 24 offline tests: signal bounds and orientation, unreached
questions scoring nothing, invalid labels depressing validity, rank and
specificity ordering, the uncalibrated gate refusing to decide, every
non-override path, score bounds, model round-trip, and calibration refusing
insufficient or single-class data. Calibration verified end-to-end on synthetic
separable data. Suite total: **141 passed**.

**Not verified.** Nothing about real confidence behaviour. No weights have been
fitted, because that requires a live generation run that has not happened. No
claim is made anywhere that confidence gating improves anything — that is
Step 10, on the holdout, once.
