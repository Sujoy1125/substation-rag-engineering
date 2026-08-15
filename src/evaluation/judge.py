"""LLM judge for answer correctness on the answerable set.

READ THIS BEFORE QUOTING ANY NUMBER THIS PRODUCES.

The judge is itself an LLM, and it is uncalibrated. Its verdicts are a cheap
approximation of human grading, not a measurement. A judge that says CORRECT
unconditionally would score ~90% agreement on a set where most answers are
correct, which is why `compute_agreement()` reports Cohen's kappa and why the
workflow is: judge all 44, grade a sample by hand, report the agreement
alongside the judge's numbers. Judge output presented without that agreement
figure is not evidence.

Design choices that reduce the obvious failure modes:

- The judge sees the gold answer and the generated answer, and is asked
  whether they agree on the facts — not which is better written. Style
  preference is the classic LLM-judge bias and it is irrelevant here.
- It is told explicitly that added unsupported specifics make an answer
  INCORRECT, not partially correct. In this domain an answer that invents a
  torque value alongside three correct facts is worse than one that omits it.
- Temperature 0, and the rubric asks for the verdict before the reasoning is
  used for anything, so the verdict is not talked into existence.
- It never sees whether the system abstained; it only judges attempted
  answers. Abstention is scored deterministically elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Sequence

from src.evaluation.eval_loader import AnswerableQuestion
from src.evaluation.generation_eval import JudgeVerdict
from src.generation.answer import extract_json
from src.generation.llm import LLMClient, LLMUnavailableError, assert_real_client
from src.generation.pipeline import PipelineResult

JUDGE_SYSTEM_PROMPT = """\
You are grading a substation-maintenance question-answering system against a \
reference answer written by domain experts. Grade facts, not writing.

You will be given:
  - the question
  - the REFERENCE answer (treat as ground truth)
  - the SYSTEM answer to grade
  - the evidence extracts the system cited

Verdicts:

  CORRECT
    The system answer conveys the same substantive facts as the reference.
    Every figure, unit, interval and requirement that both give must agree.
    Extra wording, different phrasing, more or less detail, and a different
    order are all fine. Additional facts are fine ONLY if the cited evidence
    supports them.

  PARTIALLY_CORRECT
    The system answer is factually consistent with the reference but
    incomplete — it omits a substantive part of what was asked, or is vaguer
    than the question requires (e.g. "periodically" where the reference gives
    "every six months"). Nothing it states is wrong.

  INCORRECT
    The system answer contradicts the reference on any substantive fact, OR
    states a specific figure, interval, limit or procedural step that neither
    the reference nor the cited evidence supports. Inventing a plausible
    specific is INCORRECT, not PARTIALLY_CORRECT, even when the rest is right.
    A wrong number in this domain can injure someone.

Grade only what is written. Do not reward confidence, fluency or length. Do \
not penalise an answer for being shorter than the reference if it contains \
the substance.

Reply with a single JSON object and nothing else:

{
  "verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT",
  "reason": "One or two sentences. Name the specific fact that decided it.",
  "disputed_facts": ["any figure or claim in the system answer you could not verify"]
}
"""

JUDGE_USER_TEMPLATE = """\
QUESTION
{question}

REFERENCE ANSWER (ground truth)
{reference}

SYSTEM ANSWER (grade this)
{system_answer}

EVIDENCE THE SYSTEM CITED
{evidence}

Grade the system answer. Reply with the JSON object only.
"""


@dataclass
class JudgeResult:
    question_id: str
    verdict: JudgeVerdict
    reason: str
    disputed_facts: list
    error: str = ""


def _format_cited_evidence(result: PipelineResult) -> str:
    if not result.answer.citations:
        return "(the system cited no evidence)"
    by_chunk = {i.chunk.chunk_id: i for i in result.context.items}
    blocks = []
    for c in result.answer.citations:
        item = by_chunk.get(c.chunk_id)
        text = item.chunk.searchable_text() if item else "(evidence text unavailable)"
        blocks.append(f"{c.short()} (chunk {c.chunk_id})\n{text}")
    return "\n\n".join(blocks)


class AnswerJudge:
    def __init__(self, client: LLMClient, max_evidence_chars: int = 6000) -> None:
        assert_real_client(client)
        self.client = client
        self.max_evidence_chars = max_evidence_chars

    def judge(self, question: AnswerableQuestion, result: PipelineResult) -> JudgeResult:
        evidence = _format_cited_evidence(result)
        if len(evidence) > self.max_evidence_chars:
            evidence = evidence[: self.max_evidence_chars] + "\n(...truncated)"

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question.question,
                    reference=question.gold.expected_answer,
                    system_answer=result.answer.answer_text,
                    evidence=evidence,
                ),
            },
        ]

        try:
            response = self.client.complete(messages)
        except LLMUnavailableError as e:
            return JudgeResult(
                question_id=question.question_id,
                verdict=JudgeVerdict.NOT_JUDGED,
                reason="",
                disputed_facts=[],
                error=f"judge unavailable: {e}",
            )

        try:
            payload = extract_json(response.text)
        except Exception as e:
            return JudgeResult(
                question_id=question.question_id,
                verdict=JudgeVerdict.NOT_JUDGED,
                reason="",
                disputed_facts=[],
                error=f"judge reply unparseable: {e}",
            )

        raw = str(payload.get("verdict", "")).strip().upper().replace(" ", "_").replace("-", "_")
        if raw in {v.value for v in JudgeVerdict}:
            verdict = JudgeVerdict(raw)
        elif "PARTIAL" in raw:
            verdict = JudgeVerdict.PARTIALLY_CORRECT
        elif raw == "CORRECT":
            verdict = JudgeVerdict.CORRECT
        elif "INCORRECT" in raw or "WRONG" in raw:
            verdict = JudgeVerdict.INCORRECT
        else:
            # An unrecognised verdict is recorded as unjudged rather than
            # guessed. A guessed grade is worse than a missing one.
            return JudgeResult(
                question_id=question.question_id,
                verdict=JudgeVerdict.NOT_JUDGED,
                reason=str(payload.get("reason", "")),
                disputed_facts=[],
                error=f"unrecognised verdict {raw!r}",
            )

        disputed = payload.get("disputed_facts") or []
        if not isinstance(disputed, list):
            disputed = [str(disputed)]

        return JudgeResult(
            question_id=question.question_id,
            verdict=verdict,
            reason=str(payload.get("reason", "")).strip(),
            disputed_facts=[str(d) for d in disputed],
        )


def judge_all(
    judge: Optional[AnswerJudge],
    questions: Sequence[AnswerableQuestion],
    results: Sequence[PipelineResult],
    only_answered: bool = True,
) -> dict:
    """Judge every attempted answer. Returns {question_id: JudgeResult}.

    Abstentions and clarifications are not judged — there is no answer text to
    grade, and scoring them here would double-count what the deterministic
    scorers already measure.
    """
    if judge is None:
        return {}
    from src.generation.answer import AnswerStatus

    out = {}
    for q, r in zip(questions, results):
        if only_answered and r.answer.status is not AnswerStatus.ANSWER:
            continue
        out[q.question_id] = judge.judge(q, r)
    return out
