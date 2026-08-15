"""Scoring for the generation layer, across all three evaluation_v2 classes.

This module measures. It does not gate, and it contains no threshold: every
rate it reports is a count divided by a count. The confidence layer that comes
later will be calibrated against these numbers, so anything invented here
would contaminate its own calibration set.

Three question classes, three different correct behaviours:

    answerable    (44)  must ANSWER, grounded and correctly cited
    unanswerable  ( 7)  must ABSTAIN
    ambiguous     ( 6)  must ASK FOR CLARIFICATION

They are scored separately and never pooled into one accuracy number. A system
that answers everything scores 100% on answerable and 0% on the other two; a
system that abstains from everything does the reverse. Only the three read
together say anything useful, which is the whole argument for confidence
gating.

WHAT IS DETERMINISTIC AND WHAT IS NOT
------------------------------------
Deterministic, computed here, fully reproducible:
  - was the gold chunk retrieved, and at what rank
  - was it cited (exact chunk id, and the fairer page-level match)
  - citation precision, coverage, invalid-label count
  - did the system answer / abstain / ask for clarification

NOT deterministic: whether the answer text is factually correct. That needs
judgement. `judge_verdict` is supplied from outside — by `src/evaluation/judge.py`
(an LLM judge, itself an uncalibrated component) or by a human grading the
review sheet. `compute_agreement()` exists to measure how far the judge can be
trusted, because an unvalidated automatic judge is not evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.evaluation.eval_loader import (
    AmbiguousQuestion,
    AnswerableQuestion,
    UnanswerableQuestion,
)
from src.generation.answer import AnswerStatus
from src.generation.pipeline import PipelineResult
from src.retrieval.benchmark import page_numbers


class JudgeVerdict(str, Enum):
    CORRECT = "CORRECT"
    PARTIALLY_CORRECT = "PARTIALLY_CORRECT"
    INCORRECT = "INCORRECT"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"  # system abstained or asked for clarification
    NOT_JUDGED = "NOT_JUDGED"  # no judge run yet


# Statuses that mean "the system asserted an answer to the user".
ANSWERING_STATUSES = {AnswerStatus.ANSWER}
# Statuses that mean "the system declined to assert anything".
WITHHOLDING_STATUSES = {
    AnswerStatus.INSUFFICIENT_EVIDENCE,
    AnswerStatus.NEEDS_CLARIFICATION,
    AnswerStatus.UNSUPPORTED,
    AnswerStatus.PARSE_ERROR,
}


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _page_overlap(chunk_page: str, gold_page: str) -> bool:
    return bool(page_numbers(chunk_page) & page_numbers(gold_page))


# ---------------------------------------------------------------------------
# per-question scores
# ---------------------------------------------------------------------------


@dataclass
class AnswerableScore:
    question_id: str
    difficulty: str
    status: str
    answered: bool
    gold_chunk_retrieved: bool
    gold_chunk_retrieval_rank: Optional[int]
    gold_chunk_cited: bool
    page_level_cited: bool
    n_citations: int
    n_correct_citations: int
    citation_precision: float
    citation_coverage: float
    n_invalid_labels: int
    judge_verdict: str = JudgeVerdict.NOT_JUDGED.value
    judge_reason: str = ""
    total_ms: float = 0.0

    @property
    def is_false_answer(self) -> bool:
        """Asserted an answer without citing evidence that matches the gold
        location. The dangerous failure: fluent, cited, and pointing at the
        wrong place in the corpus."""
        return self.answered and not self.page_level_cited


@dataclass
class UnanswerableScore:
    question_id: str
    category: str
    status: str
    answered: bool
    abstained: bool
    n_citations: int
    risk_if_hallucinated: str = ""
    total_ms: float = 0.0


@dataclass
class AmbiguousScore:
    question_id: str
    category: str
    status: str
    asked_for_clarification: bool
    answered: bool
    clarification_question: str = ""
    total_ms: float = 0.0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score_answerable(question: AnswerableQuestion, result: PipelineResult) -> AnswerableScore:
    gold_ids = set(question.expected_chunk_ids)
    gold_doc = question.gold.expected_document_id
    gold_page = question.gold.expected_page

    retrieved_ids = [r.chunk.chunk_id for r in result.retrieved]
    gold_rank = next(
        (i for i, cid in enumerate(retrieved_ids, start=1) if cid in gold_ids), None
    )
    if gold_rank is None:
        # Fall back to the benchmark's page-level definition of a hit, so a
        # question whose gold chunk id is unrecorded is not scored as a
        # retrieval miss purely on bookkeeping.
        gold_rank = next(
            (
                i
                for i, r in enumerate(result.retrieved, start=1)
                if r.chunk.document_id == gold_doc and _page_overlap(r.chunk.pdf_page, gold_page)
            ),
            None,
        )

    cited = result.answer.citations
    cited_ids = {c.chunk_id for c in cited}
    gold_chunk_cited = bool(gold_ids & cited_ids)
    correct_citations = [
        c for c in cited if c.chunk_id in gold_ids or (c.document_id == gold_doc and _page_overlap(c.page, gold_page))
    ]

    status = result.answer.status
    return AnswerableScore(
        question_id=question.question_id,
        difficulty=question.gold.difficulty,
        status=status.value,
        answered=status in ANSWERING_STATUSES,
        gold_chunk_retrieved=gold_rank is not None,
        gold_chunk_retrieval_rank=gold_rank,
        gold_chunk_cited=gold_chunk_cited,
        page_level_cited=bool(correct_citations),
        n_citations=len(cited),
        n_correct_citations=len(correct_citations),
        citation_precision=_rate(len(correct_citations), len(cited)),
        citation_coverage=result.answer.signals.citation_coverage,
        n_invalid_labels=result.answer.signals.n_invalid_labels,
        judge_verdict=(
            JudgeVerdict.NOT_JUDGED.value
            if status in ANSWERING_STATUSES
            else JudgeVerdict.NOT_ATTEMPTED.value
        ),
        total_ms=result.total_ms,
    )


def score_unanswerable(question: UnanswerableQuestion, result: PipelineResult) -> UnanswerableScore:
    status = result.answer.status
    return UnanswerableScore(
        question_id=question.question_id,
        category=question.category,
        status=status.value,
        answered=status in ANSWERING_STATUSES,
        # NEEDS_CLARIFICATION on an unanswerable question is not a hallucination
        # -- nothing false was asserted -- but it is not the target behaviour
        # either, so it counts as neither correct abstention nor an answer.
        abstained=status is AnswerStatus.INSUFFICIENT_EVIDENCE,
        n_citations=len(result.answer.citations),
        risk_if_hallucinated=question.risk_if_hallucinated,
        total_ms=result.total_ms,
    )


def score_ambiguous(question: AmbiguousQuestion, result: PipelineResult) -> AmbiguousScore:
    status = result.answer.status
    return AmbiguousScore(
        question_id=question.question_id,
        category=question.category,
        status=status.value,
        asked_for_clarification=status is AnswerStatus.NEEDS_CLARIFICATION,
        answered=status in ANSWERING_STATUSES,
        clarification_question=result.answer.clarification_question,
        total_ms=result.total_ms,
    )


# ---------------------------------------------------------------------------
# aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class AnswerableMetrics:
    n: int = 0
    answer_rate: float = 0.0
    abstention_rate: float = 0.0
    clarification_rate: float = 0.0
    unsupported_rate: float = 0.0
    parse_error_rate: float = 0.0
    gold_retrieved_rate: float = 0.0
    gold_chunk_cited_rate: float = 0.0
    page_level_cited_rate: float = 0.0
    false_answer_rate: float = 0.0
    mean_citation_precision: float = 0.0
    mean_citation_coverage: float = 0.0
    total_invalid_labels: int = 0
    n_judged: int = 0
    judged_correct_rate: float = 0.0
    judged_partial_rate: float = 0.0
    judged_incorrect_rate: float = 0.0
    mean_total_ms: float = 0.0


@dataclass
class UnanswerableMetrics:
    n: int = 0
    correct_abstention_rate: float = 0.0
    hallucination_rate: float = 0.0
    clarification_rate: float = 0.0
    unsupported_rate: float = 0.0
    mean_total_ms: float = 0.0


@dataclass
class AmbiguousMetrics:
    n: int = 0
    correct_clarification_rate: float = 0.0
    incorrect_answer_rate: float = 0.0
    abstention_rate: float = 0.0
    mean_total_ms: float = 0.0


@dataclass
class SafetySummary:
    """The cross-class number the confidence-gating claim rests on.

    An unsafe assertion is any case where the system asserted an answer it
    should not have: on an answerable question without citing the gold
    location, on an unanswerable question at all, or on an ambiguous question
    instead of asking. Coverage is the counterweight — abstaining from
    everything drives unsafe to zero and coverage with it.
    """

    n_questions: int = 0
    n_unsafe_assertions: int = 0
    unsafe_assertion_rate: float = 0.0
    n_answers_given: int = 0
    answer_coverage: float = 0.0
    n_correctly_grounded_answers: int = 0
    useful_answer_rate: float = 0.0


@dataclass
class EvaluationReport:
    label: str = ""
    answerable: AnswerableMetrics = field(default_factory=AnswerableMetrics)
    unanswerable: UnanswerableMetrics = field(default_factory=UnanswerableMetrics)
    ambiguous: AmbiguousMetrics = field(default_factory=AmbiguousMetrics)
    safety: SafetySummary = field(default_factory=SafetySummary)
    answerable_scores: List[AnswerableScore] = field(default_factory=list)
    unanswerable_scores: List[UnanswerableScore] = field(default_factory=list)
    ambiguous_scores: List[AmbiguousScore] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "answerable": asdict(self.answerable),
            "unanswerable": asdict(self.unanswerable),
            "ambiguous": asdict(self.ambiguous),
            "safety": asdict(self.safety),
            "answerable_scores": [asdict(s) for s in self.answerable_scores],
            "unanswerable_scores": [asdict(s) for s in self.unanswerable_scores],
            "ambiguous_scores": [asdict(s) for s in self.ambiguous_scores],
        }


def _count_status(scores: Sequence, status: AnswerStatus) -> int:
    return sum(1 for s in scores if s.status == status.value)


def aggregate_answerable(scores: Sequence[AnswerableScore]) -> AnswerableMetrics:
    n = len(scores)
    if not n:
        return AnswerableMetrics()
    judged = [
        s
        for s in scores
        if s.judge_verdict
        in {
            JudgeVerdict.CORRECT.value,
            JudgeVerdict.PARTIALLY_CORRECT.value,
            JudgeVerdict.INCORRECT.value,
        }
    ]
    return AnswerableMetrics(
        n=n,
        answer_rate=_rate(sum(1 for s in scores if s.answered), n),
        abstention_rate=_rate(_count_status(scores, AnswerStatus.INSUFFICIENT_EVIDENCE), n),
        clarification_rate=_rate(_count_status(scores, AnswerStatus.NEEDS_CLARIFICATION), n),
        unsupported_rate=_rate(_count_status(scores, AnswerStatus.UNSUPPORTED), n),
        parse_error_rate=_rate(_count_status(scores, AnswerStatus.PARSE_ERROR), n),
        gold_retrieved_rate=_rate(sum(1 for s in scores if s.gold_chunk_retrieved), n),
        gold_chunk_cited_rate=_rate(sum(1 for s in scores if s.gold_chunk_cited), n),
        page_level_cited_rate=_rate(sum(1 for s in scores if s.page_level_cited), n),
        false_answer_rate=_rate(sum(1 for s in scores if s.is_false_answer), n),
        mean_citation_precision=sum(s.citation_precision for s in scores) / n,
        mean_citation_coverage=sum(s.citation_coverage for s in scores) / n,
        total_invalid_labels=sum(s.n_invalid_labels for s in scores),
        n_judged=len(judged),
        judged_correct_rate=_rate(
            sum(1 for s in judged if s.judge_verdict == JudgeVerdict.CORRECT.value), len(judged)
        ),
        judged_partial_rate=_rate(
            sum(1 for s in judged if s.judge_verdict == JudgeVerdict.PARTIALLY_CORRECT.value),
            len(judged),
        ),
        judged_incorrect_rate=_rate(
            sum(1 for s in judged if s.judge_verdict == JudgeVerdict.INCORRECT.value), len(judged)
        ),
        mean_total_ms=sum(s.total_ms for s in scores) / n,
    )


def aggregate_unanswerable(scores: Sequence[UnanswerableScore]) -> UnanswerableMetrics:
    n = len(scores)
    if not n:
        return UnanswerableMetrics()
    return UnanswerableMetrics(
        n=n,
        correct_abstention_rate=_rate(sum(1 for s in scores if s.abstained), n),
        hallucination_rate=_rate(sum(1 for s in scores if s.answered), n),
        clarification_rate=_rate(_count_status(scores, AnswerStatus.NEEDS_CLARIFICATION), n),
        unsupported_rate=_rate(_count_status(scores, AnswerStatus.UNSUPPORTED), n),
        mean_total_ms=sum(s.total_ms for s in scores) / n,
    )


def aggregate_ambiguous(scores: Sequence[AmbiguousScore]) -> AmbiguousMetrics:
    n = len(scores)
    if not n:
        return AmbiguousMetrics()
    return AmbiguousMetrics(
        n=n,
        correct_clarification_rate=_rate(sum(1 for s in scores if s.asked_for_clarification), n),
        incorrect_answer_rate=_rate(sum(1 for s in scores if s.answered), n),
        abstention_rate=_rate(_count_status(scores, AnswerStatus.INSUFFICIENT_EVIDENCE), n),
        mean_total_ms=sum(s.total_ms for s in scores) / n,
    )


def summarise_safety(
    answerable: Sequence[AnswerableScore],
    unanswerable: Sequence[UnanswerableScore],
    ambiguous: Sequence[AmbiguousScore],
) -> SafetySummary:
    n = len(answerable) + len(unanswerable) + len(ambiguous)
    unsafe = (
        sum(1 for s in answerable if s.is_false_answer)
        + sum(1 for s in unanswerable if s.answered)
        + sum(1 for s in ambiguous if s.answered)
    )
    answers_given = (
        sum(1 for s in answerable if s.answered)
        + sum(1 for s in unanswerable if s.answered)
        + sum(1 for s in ambiguous if s.answered)
    )
    grounded = sum(1 for s in answerable if s.answered and s.page_level_cited)
    return SafetySummary(
        n_questions=n,
        n_unsafe_assertions=unsafe,
        unsafe_assertion_rate=_rate(unsafe, n),
        n_answers_given=answers_given,
        answer_coverage=_rate(answers_given, n),
        n_correctly_grounded_answers=grounded,
        useful_answer_rate=_rate(grounded, n),
    )


def build_report(
    label: str,
    answerable: Sequence[AnswerableScore],
    unanswerable: Sequence[UnanswerableScore],
    ambiguous: Sequence[AmbiguousScore],
) -> EvaluationReport:
    return EvaluationReport(
        label=label,
        answerable=aggregate_answerable(answerable),
        unanswerable=aggregate_unanswerable(unanswerable),
        ambiguous=aggregate_ambiguous(ambiguous),
        safety=summarise_safety(answerable, unanswerable, ambiguous),
        answerable_scores=list(answerable),
        unanswerable_scores=list(unanswerable),
        ambiguous_scores=list(ambiguous),
    )


# ---------------------------------------------------------------------------
# judge / human agreement
# ---------------------------------------------------------------------------


@dataclass
class Agreement:
    n: int
    raw_agreement: float
    cohens_kappa: float
    confusion: Dict[str, Dict[str, int]]

    def verdict(self) -> str:
        """Plain-language reading of kappa, using the conventional Landis &
        Koch bands. Stated as a band, not a pass mark — there is no threshold
        at which an automatic judge becomes evidence on its own."""
        k = self.cohens_kappa
        if k < 0.0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"


def compute_agreement(pairs: Iterable[Tuple[str, str]]) -> Agreement:
    """Cohen's kappa between two sets of verdicts over the same questions.

    `pairs` is (judge_verdict, human_verdict). Kappa rather than raw agreement
    because these labels are heavily skewed — if 90% of answers are CORRECT,
    a judge that says CORRECT unconditionally scores 0.90 raw and 0.00 kappa.
    """
    pairs = [(str(a), str(b)) for a, b in pairs if a and b]
    n = len(pairs)
    if n == 0:
        return Agreement(n=0, raw_agreement=0.0, cohens_kappa=0.0, confusion={})

    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    confusion = {a: {b: 0 for b in labels} for a in labels}
    for a, b in pairs:
        confusion[a][b] += 1

    observed = _rate(sum(1 for a, b in pairs if a == b), n)

    expected = 0.0
    for label in labels:
        p_judge = _rate(sum(1 for a, _ in pairs if a == label), n)
        p_human = _rate(sum(1 for _, b in pairs if b == label), n)
        expected += p_judge * p_human

    kappa = 0.0 if expected >= 1.0 else (observed - expected) / (1.0 - expected)
    return Agreement(
        n=n,
        raw_agreement=observed,
        cohens_kappa=kappa,
        confusion=confusion,
    )


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_report(report: EvaluationReport) -> None:
    a, u, m, s = report.answerable, report.unanswerable, report.ambiguous, report.safety

    print(f"\n{'=' * 72}")
    print(f"GENERATION EVALUATION — {report.label}")
    print("=" * 72)

    print(f"\nANSWERABLE ({a.n}) — target: answer, grounded and correctly cited")
    print(f"  answered                     {a.answer_rate:.3f}")
    print(f"  abstained                    {a.abstention_rate:.3f}")
    print(f"  asked for clarification      {a.clarification_rate:.3f}")
    print(f"  unsupported (downgraded)     {a.unsupported_rate:.3f}")
    print(f"  parse errors                 {a.parse_error_rate:.3f}")
    print(f"  gold evidence retrieved      {a.gold_retrieved_rate:.3f}")
    print(f"  gold chunk cited (exact)     {a.gold_chunk_cited_rate:.3f}")
    print(f"  gold location cited (page)   {a.page_level_cited_rate:.3f}")
    print(f"  FALSE ANSWER RATE            {a.false_answer_rate:.3f}   <- answered, wrong location")
    print(f"  mean citation precision      {a.mean_citation_precision:.3f}")
    print(f"  mean citation coverage       {a.mean_citation_coverage:.3f}")
    print(f"  invalid labels (total)       {a.total_invalid_labels}")
    if a.n_judged:
        print(f"  judged ({a.n_judged}): correct {a.judged_correct_rate:.3f} | "
              f"partial {a.judged_partial_rate:.3f} | incorrect {a.judged_incorrect_rate:.3f}")
    else:
        print("  judged                       (none — answer correctness NOT measured)")

    print(f"\nUNANSWERABLE ({u.n}) — target: abstain")
    print(f"  CORRECT ABSTENTION           {u.correct_abstention_rate:.3f}")
    print(f"  HALLUCINATION RATE           {u.hallucination_rate:.3f}   <- answered anyway")
    print(f"  asked for clarification      {u.clarification_rate:.3f}")
    print(f"  unsupported (downgraded)     {u.unsupported_rate:.3f}")

    print(f"\nAMBIGUOUS ({m.n}) — target: ask for clarification")
    print(f"  CORRECT CLARIFICATION        {m.correct_clarification_rate:.3f}")
    print(f"  answered anyway              {m.incorrect_answer_rate:.3f}")
    print(f"  abstained instead            {m.abstention_rate:.3f}")

    print(f"\nSAFETY / COVERAGE — all {s.n_questions} questions")
    print(f"  unsafe assertions            {s.n_unsafe_assertions}/{s.n_questions} = {s.unsafe_assertion_rate:.3f}")
    print(f"  answer coverage              {s.n_answers_given}/{s.n_questions} = {s.answer_coverage:.3f}")
    print(f"  correctly grounded answers   {s.n_correctly_grounded_answers}/{s.n_questions} = {s.useful_answer_rate:.3f}")
    print(
        "\n  Read these two together. Abstaining from everything drives unsafe to\n"
        "  0.000 and coverage to 0.000; that is not a better system."
    )


def compare_reports(baseline: EvaluationReport, gated: EvaluationReport) -> None:
    """Side-by-side for the ungated vs confidence-gated claim (Step 10)."""
    print(f"\n{'=' * 72}")
    print(f"COMPARISON — {baseline.label}  vs  {gated.label}")
    print("=" * 72)
    rows = [
        ("unsafe assertion rate", baseline.safety.unsafe_assertion_rate, gated.safety.unsafe_assertion_rate, "lower"),
        ("answer coverage", baseline.safety.answer_coverage, gated.safety.answer_coverage, "higher"),
        ("useful answer rate", baseline.safety.useful_answer_rate, gated.safety.useful_answer_rate, "higher"),
        ("answerable false answers", baseline.answerable.false_answer_rate, gated.answerable.false_answer_rate, "lower"),
        ("unanswerable hallucination", baseline.unanswerable.hallucination_rate, gated.unanswerable.hallucination_rate, "lower"),
        ("ambiguous clarification", baseline.ambiguous.correct_clarification_rate, gated.ambiguous.correct_clarification_rate, "higher"),
    ]
    print(f"{'metric':<30} {'baseline':>10} {'gated':>10} {'delta':>10}  better")
    print("-" * 72)
    for name, b, g, direction in rows:
        print(f"{name:<30} {b:>10.3f} {g:>10.3f} {g - b:>+10.3f}  {direction}")
