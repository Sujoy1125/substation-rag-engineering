# Evaluation Plan — Generation Layer

**Status:** harness implemented and tested offline. **No generation behaviour
has been measured yet** — no live model has run against the pipeline.
**Runner:** `experiments/run_generation_eval.py`
**Scoring:** `src/evaluation/generation_eval.py` · **Judge:** `src/evaluation/judge.py`

---

## Three classes, three correct behaviours, never pooled

| class | n | correct behaviour | headline failure |
|---|---|---|---|
| answerable | 44 | ANSWER, grounded and correctly cited | **false answer** — answered, wrong location cited |
| unanswerable | 7 | ABSTAIN | **hallucination** — answered anyway |
| ambiguous | 6 | ASK FOR CLARIFICATION | answered instead of asking |

These are scored separately and never combined into one accuracy figure. A
system that answers everything scores 1.000 on answerable and 0.000 on the
other two; a system that abstains from everything does the reverse. Only the
three read together mean anything — which is the entire argument for confidence
gating.

---

## The two numbers that carry the SIH claim

```
unsafe assertion rate   answered when it should not have, over all 57
answer coverage         answered at all, over all 57
```

**They must always be reported together.** Refusing to answer every question
drives unsafe to 0.000 — and coverage to 0.000 with it. A gate that improves
safety by destroying coverage has not improved anything, and a reviewer will
ask. `summarise_safety()` computes both plus `useful_answer_rate` (answered
*and* correctly grounded), and a test asserts the all-abstain case zeroes all
three.

An unsafe assertion is any of:

- answerable, answered, but no citation matching the gold location
- unanswerable, answered at all
- ambiguous, answered instead of asking

`NEEDS_CLARIFICATION` on an unanswerable question is deliberately neither
correct abstention nor a hallucination: nothing false was asserted, but it is
not the target behaviour.

---

## What is deterministic, and what is not

**Deterministic** — computed in code, reproducible, no model involved:
gold chunk retrieved and at what rank · gold chunk cited (exact id) · gold
location cited (document + page overlap) · citation precision · citation
coverage · invalid-label count · answered / abstained / clarified.

Citation correctness is scored at two levels. *Exact* requires the cited chunk
id to be the gold id. *Page-level* accepts any cited chunk from the gold
document whose page range overlaps the gold page — a neighbouring chunk on the
same page may genuinely carry the answer, and scoring exact-only would
understate correctness. Page-level is what the false-answer rate uses.

**Not deterministic** — whether the answer text is factually right. That needs
judgement, and it is the one place this harness cannot be self-sufficient.

---

## Answer correctness: judge, then spot-check

`src/evaluation/judge.py` grades attempted answers against the gold reference
as CORRECT / PARTIALLY_CORRECT / INCORRECT. Rubric decisions worth knowing:

- Facts are graded, not writing. Style preference is the classic LLM-judge bias
  and is irrelevant here.
- A specific figure the reference and evidence do not support makes an answer
  **INCORRECT, not partially correct**. In this domain an invented torque value
  alongside three correct facts is worse than an omission.
- Abstentions are not judged — there is no answer text, and the deterministic
  scorers already handle them.

**The judge is itself an uncalibrated LLM.** On a set where most answers are
correct, a judge that says CORRECT unconditionally scores ~0.90 raw agreement
and 0.00 kappa — a test asserts exactly this. So the workflow is:

```
--live --judge          judge all attempted answers
--review-sheet          xlsx: system answer next to gold, empty Human Verdict column
[you grade a sample]    CORRECT / PARTIALLY_CORRECT / INCORRECT
--agreement FILE        Cohen's kappa between judge and you
```

Report the kappa next to every judged number. A judge whose agreement with a
human is unknown is not a measurement. Grade at least 15 of the 44 — enough for
kappa to mean something — and prefer a spread across difficulties rather than
the first fifteen.

---

## Order of operations

```
STEP 6   --live                    deterministic scoring, all 57
   ↓
STEP 6b  --live --judge --review-sheet    add answer correctness
   ↓
STEP 6c  grade a sample → --agreement     establish judge trustworthiness
   ↓
STEP 7/8 (already covered — unanswerable and ambiguous run in the same pass)
   ↓
STEP 9   confidence gate, weights calibrated on the signals recorded above
   ↓
STEP 10  compare_reports(ungated, gated) on the same 57 questions
```

Steps 7 and 8 need no separate run: the runner evaluates all three classes in
one pass. Run `--live --limit 3` first — nine questions, a few cents — and read
the output before committing to the full 57.

---

## Measured ceiling (dry run, no model involved)

```
python experiments/run_generation_eval.py
```

Gold evidence reaches the top-5 context for **40 of 44** answerable questions
(0.909), mean rank 1.30 when present. Never reaches it: `V2-011`, `V2-013`,
`V2-015`, `V2-043`.

This is a hard ceiling on answer correctness — the model cannot ground an
answer in evidence it was never shown, so those four can only be answered
correctly by accident. It is also the cleanest test of the abstain path: the
correct behaviour on all four is to decline.

Unanswerable and ambiguous questions always retrieve something (BM25 always
returns its top matches — mean ~4.6k chars across ~2 documents). Abstention
therefore has to be a judgement about whether the evidence *answers the
question*, never about whether evidence exists.

---

## Calibration discipline

The confidence gate will be calibrated on `evaluation_v2`. That creates an
obvious hazard: weights tuned on the same 57 questions used to report the final
result will overstate the gain. Options, in order of preference:

1. Hold out a slice of the answerable set from calibration and report on it
   separately.
2. Calibrate on `evaluation_v2`, report on D09's 20 questions as an
   out-of-sample check.
3. Calibrate and report on the same set, and **say so explicitly** in the
   write-up.

Option 3 is acceptable only if stated. Silently doing it is the kind of thing
a sharp reviewer finds.

Note also that `KB_v1.1/rag_test_55.xlsx` was frozen as a retrieval benchmark
and is not to be used for confidence-threshold tuning.

---

## What the harness will not do

- Combine the three classes into a single accuracy number.
- Report a judged rate without the agreement figure beside it.
- Apply any threshold. Every rate is a count over a count; the gate lives in a
  later layer and gets its constants from these measurements, not from
  intuition.
