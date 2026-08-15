"""Tests for the generation evaluation harness.

The scorers decide what gets reported to SIH judges, so the cases that matter
most are the ones where a naive implementation would flatter the system: an
answer that cites confidently but points at the wrong document, an abstention
on a question that was answerable, a judge that agrees by accident.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.eval_loader import (
    AmbiguousQuestion,
    AnswerableQuestion,
    UnanswerableQuestion,
)
from src.evaluation.generation_eval import (
    JudgeVerdict,
    aggregate_answerable,
    build_report,
    compute_agreement,
    score_ambiguous,
    score_answerable,
    score_unanswerable,
    summarise_safety,
)
from src.generation.answer import AnswerStatus
from src.generation.llm import ScriptedClient
from src.generation.pipeline import RAGPipeline
from src.retrieval.gold_questions import GoldQuestion
from tests.test_generation import make_chunk


def answerable_q(qid="V2-001", chunk_ids=("D09-C0001",), doc="D09", page="p.123"):
    return AnswerableQuestion(
        gold=GoldQuestion(
            question_id=qid,
            question="How often is transformer oil BDV tested?",
            expected_answer="Annually.",
            expected_document_ref=f"KB Document {doc}",
            expected_document_id=doc,
            expected_page=page,
            expected_section="Section 7.2",
            difficulty="Easy",
        ),
        expected_chunk_ids=list(chunk_ids),
        evidence_basis="Frequency: Annually",
    )


def unanswerable_q(qid="V2-U01"):
    return UnanswerableQuestion(
        question_id=qid,
        question="What is the routine weekly checklist?",
        why_unanswerable="D01 holds design guidance only.",
        category="Maintenance schedule",
        risk_if_hallucinated="Fabricated schedule could cause missed maintenance.",
    )


def ambiguous_q(qid="V2-A01"):
    return AmbiguousQuestion(
        question_id=qid,
        question="What is the maximum operating temperature for the transformer?",
        why_ambiguous="Several different temperature limits exist.",
        ideal_system_behavior="Ask which temperature quantity is meant.",
        category="Transformer / thermal limits",
    )


# Two fixture chunks with deliberately DISJOINT vocabularies. `make_chunk`'s
# defaults are all oil/BDV, so overriding only one field leaves the other chunk
# still full of oil terms and retrieval order becomes unpredictable — which is
# exactly what these scorer tests must not depend on.
def oil_chunk(chunk_id="D09-C0001", document_id="D09", pdf_page="p.123"):
    return make_chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        pdf_page=pdf_page,
        equipment="Transformer",
        equipment_subtype="Power Transformer",
        topic="Insulating oil",
        subtopic="Dielectric strength",
        verified_information="Insulating oil shall be tested for BDV annually.",
        procedure="Draw the sample from the bottom drain valve.",
        frequency="Annually",
        technical_limit_value="BDV >= 50 kV",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT VERIFIED",
        applicability="All power transformers",
    )


def breaker_chunk(chunk_id="D01-C0007", document_id="D01", pdf_page="p.9"):
    return make_chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        pdf_page=pdf_page,
        document_title="Switchgear Manual",
        equipment="Circuit Breaker",
        equipment_subtype="SF6 Breaker",
        topic="Contact erosion",
        subtopic="Arcing contacts",
        verified_information="Arcing contact erosion shall be measured every five years.",
        procedure="Withdraw the interrupter and gauge the arcing tip.",
        frequency="Every five years",
        technical_limit_value="Erosion <= 3 mm",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT VERIFIED",
        applicability="SF6 circuit breakers",
    )


def _filler(i: int, topic: str, text: str):
    """BM25's IDF term is log((N-n+0.5)/(n+0.5)); with a two-document corpus
    every term scores 0 and ranking collapses to insertion order. These
    fillers give the index enough documents for scoring to be real, so the
    fixtures below discriminate for the reason they claim to."""
    return make_chunk(
        chunk_id=f"D0{i}-F000{i}",
        document_id=f"D0{i}",
        pdf_page=f"p.{200 + i}",
        document_title=f"Filler Manual {i}",
        equipment=topic,
        equipment_subtype="",
        topic=topic,
        subtopic="",
        verified_information=text,
        procedure="NOT COVERED",
        frequency="NOT COVERED",
        technical_limit_value="NOT COVERED",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT VERIFIED",
        applicability="",
    )


FILLERS = [
    _filler(2, "Busbar", "Busbar clamps shall be thermographically surveyed each summer."),
    _filler(3, "Battery", "Station battery specific gravity is recorded during monthly rounds."),
    _filler(4, "Earthing", "Earth mat resistance is measured before the monsoon season."),
    _filler(5, "Relay", "Numerical relay settings are verified after firmware upgrades."),
    _filler(6, "Cable", "Cable termination stress cones are examined for tracking."),
    _filler(7, "Fencing", "Perimeter fencing continuity is checked during quarterly patrols."),
]

OIL_QUERY = "insulating oil dielectric strength BDV"
BREAKER_QUERY = "arcing contact erosion interrupter gauge"


def default_corpus():
    return [oil_chunk(), breaker_chunk(), *FILLERS]


def pipeline_with(reply, chunks=None):
    chunks = chunks if chunks is not None else default_corpus()
    return RAGPipeline(chunks, llm=ScriptedClient([reply]), top_k=5).index()


def test_fixture_corpus_discriminates_between_the_two_target_chunks():
    """Sanity check on the fixtures themselves: if this fails, every scorer
    test below is asserting against an accidental ordering."""
    pipe = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    assert pipe.retrieve(OIL_QUERY)[0].chunk.chunk_id == "D09-C0001"
    assert pipe.retrieve(BREAKER_QUERY)[0].chunk.chunk_id == "D01-C0007"


def reply_citing(labels, answer="Annually.", status="ANSWER"):
    return json.dumps(
        {
            "status": status,
            "answer": answer,
            "claims": [{"text": answer, "evidence_labels": list(labels)}] if labels else [],
        }
    )


# --------------------------------------------------------------------------
# answerable scoring
# --------------------------------------------------------------------------


def test_correct_citation_scores_as_grounded():
    pipe = pipeline_with(reply_citing(["E1"]))
    result = pipe.answer(OIL_QUERY)
    assert result.answer.citations[0].chunk_id == "D09-C0001"
    s = score_answerable(answerable_q(), result)
    assert s.answered
    assert s.gold_chunk_cited
    assert s.page_level_cited
    assert s.citation_precision == 1.0
    assert not s.is_false_answer


def test_answer_citing_the_wrong_document_is_a_false_answer():
    """The failure mode that matters: confident, cited, and pointing at the
    wrong place in the corpus. It must not score as grounded."""
    pipe = pipeline_with(reply_citing(["E1"], answer="Every five years."))
    result = pipe.answer(BREAKER_QUERY)
    assert result.answer.citations[0].chunk_id == "D01-C0007"
    # Gold lives in a different document entirely.
    s = score_answerable(answerable_q(chunk_ids=("D09-C0001",), doc="D09", page="p.123"), result)
    assert s.answered
    assert not s.gold_chunk_cited
    assert not s.page_level_cited
    assert s.citation_precision == 0.0
    assert s.is_false_answer


def test_page_level_match_accepts_a_neighbouring_chunk_on_the_gold_page():
    """A different chunk on the gold page may genuinely carry the answer;
    scoring only exact chunk ids would understate correctness."""
    chunks = [make_chunk(chunk_id="D09-C0002", document_id="D09", pdf_page="p.123")]
    pipe = pipeline_with(reply_citing(["E1"]), chunks=chunks)
    result = pipe.answer("transformer oil BDV")
    s = score_answerable(answerable_q(chunk_ids=("D09-C0001",)), result)
    assert not s.gold_chunk_cited  # exact id differs
    assert s.page_level_cited  # same document and page
    assert not s.is_false_answer


def test_page_range_overlap_counts_as_a_match():
    chunks = [make_chunk(chunk_id="D09-C0009", document_id="D09", pdf_page="PDF p. 26-30")]
    pipe = pipeline_with(reply_citing(["E1"]), chunks=chunks)
    result = pipe.answer("oil")
    s = score_answerable(answerable_q(chunk_ids=("X",), page="p.27-28"), result)
    assert s.page_level_cited


def test_abstention_on_an_answerable_question_is_not_an_answer():
    pipe = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    result = pipe.answer("transformer oil BDV")
    s = score_answerable(answerable_q(), result)
    assert not s.answered
    assert not s.is_false_answer  # withheld, so nothing false was asserted
    assert s.judge_verdict == JudgeVerdict.NOT_ATTEMPTED.value


def test_unsupported_downgrade_is_not_counted_as_an_answer():
    pipe = pipeline_with(reply_citing([], answer="Annually."))
    result = pipe.answer("transformer oil BDV")
    assert result.answer.status is AnswerStatus.UNSUPPORTED
    s = score_answerable(answerable_q(), result)
    assert not s.answered
    assert not s.is_false_answer


def test_gold_retrieval_is_scored_independently_of_citation():
    """Retrieval reaching the gold chunk and the model citing it are separate
    failures and must be distinguishable in the report."""
    pipe = pipeline_with(reply_citing(["E2"]))
    result = pipe.answer(OIL_QUERY)
    s = score_answerable(answerable_q(), result)
    assert s.gold_chunk_retrieval_rank == 1  # retrieval did its job
    assert s.gold_chunk_retrieved
    assert not s.gold_chunk_cited  # the model cited something else
    assert s.is_false_answer


# --------------------------------------------------------------------------
# unanswerable / ambiguous scoring
# --------------------------------------------------------------------------


def test_abstaining_on_unanswerable_is_correct():
    pipe = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    result = pipe.answer("weekly checklist")
    s = score_unanswerable(unanswerable_q(), result)
    assert s.abstained
    assert not s.answered


def test_answering_an_unanswerable_question_is_a_hallucination():
    pipe = pipeline_with(reply_citing(["E1"], answer="Check it weekly."))
    result = pipe.answer("weekly checklist")
    s = score_unanswerable(unanswerable_q(), result)
    assert s.answered
    assert not s.abstained


def test_clarification_on_unanswerable_counts_as_neither():
    """Asking a question asserts nothing false, so it is not a hallucination —
    but it is not the target behaviour either."""
    pipe = pipeline_with(
        json.dumps({"status": "NEEDS_CLARIFICATION", "clarification_question": "Which?", "claims": []})
    )
    result = pipe.answer("weekly checklist")
    s = score_unanswerable(unanswerable_q(), result)
    assert not s.answered
    assert not s.abstained


def test_clarifying_an_ambiguous_question_is_correct():
    pipe = pipeline_with(
        json.dumps(
            {
                "status": "NEEDS_CLARIFICATION",
                "clarification_question": "Top-oil or winding hot-spot?",
                "claims": [],
            }
        )
    )
    result = pipe.answer("maximum operating temperature")
    s = score_ambiguous(ambiguous_q(), result)
    assert s.asked_for_clarification
    assert not s.answered


def test_answering_an_ambiguous_question_is_scored_as_incorrect():
    pipe = pipeline_with(reply_citing(["E1"], answer="105 C."))
    result = pipe.answer("maximum operating temperature")
    s = score_ambiguous(ambiguous_q(), result)
    assert s.answered
    assert not s.asked_for_clarification


# --------------------------------------------------------------------------
# aggregation and safety
# --------------------------------------------------------------------------


def test_answerable_rates_are_plain_counts():
    pipe_ok = pipeline_with(reply_citing(["E1"]))
    ok = score_answerable(answerable_q(), pipe_ok.answer("transformer oil BDV test frequency"))
    pipe_no = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    no = score_answerable(answerable_q(qid="V2-002"), pipe_no.answer("transformer oil BDV"))

    m = aggregate_answerable([ok, no])
    assert m.n == 2
    assert m.answer_rate == 0.5
    assert m.abstention_rate == 0.5


def test_abstaining_from_everything_zeroes_unsafe_and_coverage_together():
    """The claim 'confidence gating is safer' must not be satisfiable by
    refusing to answer. Both halves have to be reported."""
    pipe = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    a = score_answerable(answerable_q(), pipe.answer("q"))
    pipe2 = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    u = score_unanswerable(unanswerable_q(), pipe2.answer("q"))
    pipe3 = pipeline_with(json.dumps({"status": "INSUFFICIENT_EVIDENCE", "claims": []}))
    m = score_ambiguous(ambiguous_q(), pipe3.answer("q"))

    s = summarise_safety([a], [u], [m])
    assert s.n_unsafe_assertions == 0
    assert s.answer_coverage == 0.0
    assert s.useful_answer_rate == 0.0


def test_safety_counts_every_class_of_unsafe_assertion():
    pipe = pipeline_with(reply_citing(["E1"], answer="Every five years."))
    wrong_doc = score_answerable(
        answerable_q(chunk_ids=("D77-C0001",), doc="D77", page="p.500"),
        pipe.answer(BREAKER_QUERY),
    )
    pipe2 = pipeline_with(reply_citing(["E1"], answer="Weekly."))
    hallucinated = score_unanswerable(unanswerable_q(), pipe2.answer("q"))
    pipe3 = pipeline_with(reply_citing(["E1"], answer="105 C."))
    guessed = score_ambiguous(ambiguous_q(), pipe3.answer("q"))

    s = summarise_safety([wrong_doc], [hallucinated], [guessed])
    assert s.n_questions == 3
    assert s.n_unsafe_assertions == 3
    assert s.unsafe_assertion_rate == 1.0


class DeadClient(ScriptedClient):
    """A client that cannot reach the model, like a firewalled network."""

    is_real = True  # must pass assert_real_client; the failure is the network

    def complete(self, messages):
        from src.generation.llm import LLMUnavailableError

        raise LLMUnavailableError("OpenAI request failed: Connection error.")


def dead_pipeline():
    return RAGPipeline(default_corpus(), llm=DeadClient([]), top_k=5).index()


def test_unreachable_model_is_llm_error_not_parse_error():
    """A dead connection is a statement about the plumbing; PARSE_ERROR is a
    statement about the model's output. Conflating them sends you debugging
    the wrong layer."""
    result = dead_pipeline().answer(OIL_QUERY)
    assert result.answer.status is AnswerStatus.LLM_ERROR
    assert "Connection error" in result.error
    assert "could not be reached" in result.rendered()


def test_run_with_unreachable_model_is_marked_invalid():
    """The failure that motivated this: 9/9 calls failed and the report still
    printed 'unsafe assertions 0.000', which reads as a perfect safety score."""
    pipe = dead_pipeline()
    a = score_answerable(answerable_q(), pipe.answer(OIL_QUERY))
    u = score_unanswerable(unanswerable_q(), dead_pipeline().answer(OIL_QUERY))
    m = score_ambiguous(ambiguous_q(), dead_pipeline().answer(OIL_QUERY))

    report = build_report("dead", [a], [u], [m])
    assert report.n_llm_errors == 3
    assert report.n_questions == 3
    assert report.is_valid is False
    assert report.to_dict()["results_are_valid"] is False


def test_unreachable_questions_are_not_counted_as_safe_abstentions():
    """An unreached question must not be scored as the system correctly
    declining to answer."""
    u = score_unanswerable(unanswerable_q(), dead_pipeline().answer(OIL_QUERY))
    assert not u.answered
    assert not u.abstained  # never asked, so it did not abstain

    a = score_answerable(answerable_q(), dead_pipeline().answer(OIL_QUERY))
    assert not a.answered
    assert not a.is_false_answer


def test_successful_run_is_valid():
    pipe = pipeline_with(reply_citing(["E1"]))
    a = score_answerable(answerable_q(), pipe.answer(OIL_QUERY))
    report = build_report("live", [a], [], [])
    assert report.n_llm_errors == 0
    assert report.is_valid is True


def test_report_serialises():
    pipe = pipeline_with(reply_citing(["E1"]))
    a = score_answerable(answerable_q(), pipe.answer("transformer oil BDV test frequency"))
    report = build_report("test", [a], [], [])
    json.dumps(report.to_dict())


# --------------------------------------------------------------------------
# judge / human agreement
# --------------------------------------------------------------------------


def test_kappa_is_zero_for_a_judge_that_always_says_correct():
    """Raw agreement flatters a constant judge on a skewed set; kappa is the
    reason the harness reports kappa."""
    pairs = [("CORRECT", "CORRECT")] * 9 + [("CORRECT", "INCORRECT")]
    ag = compute_agreement(pairs)
    assert ag.raw_agreement == pytest.approx(0.9)
    assert ag.cohens_kappa == pytest.approx(0.0)


def test_kappa_is_one_for_perfect_agreement():
    pairs = [("CORRECT", "CORRECT")] * 5 + [("INCORRECT", "INCORRECT")] * 5
    ag = compute_agreement(pairs)
    assert ag.cohens_kappa == pytest.approx(1.0)
    assert ag.verdict() == "almost perfect"


def test_kappa_is_negative_when_judge_systematically_disagrees():
    pairs = [("CORRECT", "INCORRECT")] * 5 + [("INCORRECT", "CORRECT")] * 5
    ag = compute_agreement(pairs)
    assert ag.cohens_kappa < 0
    assert ag.verdict() == "worse than chance"


def test_agreement_with_no_graded_rows_is_empty_not_a_crash():
    ag = compute_agreement([])
    assert ag.n == 0
    assert ag.cohens_kappa == 0.0
