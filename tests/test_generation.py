"""Tests for the generation layer.

Every test here runs offline against `ScriptedClient`, so the suite stays
deterministic and costs nothing. The point is not to test whether an LLM
answers well — that requires a real model and belongs in the end-to-end
evaluation — but to test that the code around it cannot be fooled: invented
citations are caught, an uncited answer is not returned as grounded, and
malformed replies degrade instead of crashing.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citation.citations import Citation, render_reference_list, resolve_labels
from src.common.chunk import Chunk
from src.generation.answer import (
    AnswerStatus,
    MalformedAnswerError,
    build_answer,
    extract_json,
)
from src.generation.context import build_context
from src.generation.llm import (
    LLMClient,
    MockClientInEvaluationError,
    ScriptedClient,
    assert_real_client,
)
from src.generation.pipeline import RAGPipeline
from src.generation.prompt import SYSTEM_PROMPT, build_messages
from src.retrieval.retrievers import RetrievedResult


def make_chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="D09-C0001",
        document_id="D09",
        original_filename="Transformer_Manual__Amendment_01.pdf",
        document_title="Transformer Manual",
        organization="CEA",
        authority_level="Regulatory",
        equipment="Transformer",
        equipment_subtype="Power Transformer",
        topic="Maintenance",
        subtopic="Oil testing",
        knowledge_type="Procedure",
        verified_information="Insulating oil shall be tested for BDV annually.",
        procedure="Draw the sample from the bottom drain valve.",
        frequency="Annually",
        technical_limit_value="BDV >= 50 kV",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT VERIFIED",
        applicability="All power transformers",
        pdf_page="p.123",
        source_section="Section 7.2",
        notes="",
    )
    base.update(overrides)
    return Chunk(**base)


def make_results(n: int = 3):
    return [
        RetrievedResult(
            chunk=make_chunk(chunk_id=f"D0{i}-C000{i}", document_id=f"D0{i}", pdf_page=f"p.{100 + i}"),
            score=10.0 - i,
            rank=i,
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------


def test_context_labels_follow_retrieval_rank():
    ctx = build_context(make_results(3))
    assert ctx.labels == ["E1", "E2", "E3"]
    assert ctx.items[0].rank == 1
    assert "[E1]" in ctx.text and "[E3]" in ctx.text


def test_context_omits_sentinel_fields():
    """'NOT COVERED' / 'NOT VERIFIED' mean 'checked, nothing there'. Rendering
    them would invite the model to treat them as content."""
    ctx = build_context(make_results(1))
    assert "NOT COVERED" not in ctx.text
    assert "NOT VERIFIED" not in ctx.text
    assert "BDV" in ctx.text  # real content still present


def test_context_marks_chunk_with_no_content_fields():
    empty = make_chunk(
        verified_information="NOT VERIFIED",
        procedure="N/A",
        frequency="",
        technical_limit_value="NOT APPLICABLE",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT VERIFIED",
        applicability="",
    )
    ctx = build_context([RetrievedResult(chunk=empty, score=1.0, rank=1)])
    assert "no content fields populated" in ctx.text


def test_context_shows_duplicated_field_text_once_under_merged_labels():
    """A KB paragraph often sits in both 'Verified Information' and
    'Troubleshooting / Failure Information'. Printing it twice wastes prompt
    budget and makes one source look like two corroborating statements."""
    dup = "Thirty six transformer failure cases were reported."
    chunk = make_chunk(
        verified_information=dup,
        troubleshooting_failure_information=dup,
        procedure="NOT COVERED",
        frequency="",
        technical_limit_value="",
        safety_information="",
        applicability="",
    )
    ctx = build_context([RetrievedResult(chunk=chunk, score=1.0, rank=1)])
    assert ctx.text.count(dup) == 1
    assert "Verified information / Troubleshooting / failure information:" in ctx.text


def test_context_truncation_keeps_top_ranks():
    ctx = build_context(make_results(5), max_items=2)
    assert ctx.labels == ["E1", "E2"]
    assert ctx.items[-1].rank == 2


def test_empty_context_is_flagged():
    ctx = build_context([])
    assert ctx.is_empty()
    assert ctx.labels == []


def test_context_reports_distinct_documents():
    ctx = build_context(make_results(3))
    assert ctx.document_ids() == ["D01", "D02", "D03"]


# --------------------------------------------------------------------------
# prompt
# --------------------------------------------------------------------------


def test_prompt_carries_evidence_and_question():
    ctx = build_context(make_results(2))
    messages = build_messages("What is the BDV limit?", ctx)
    assert messages[0]["role"] == "system"
    assert "ANSWER ONLY FROM THE PROVIDED EVIDENCE" in messages[0]["content"]
    assert "What is the BDV limit?" in messages[1]["content"]
    assert "[E1]" in messages[1]["content"]


def test_prompt_forbids_the_specific_fabrications_that_matter():
    for forbidden in ("frequencies", "page numbers", "procedure steps"):
        assert forbidden in SYSTEM_PROMPT


def test_prompt_does_not_ask_the_model_for_a_confidence_score():
    """A self-reported confidence would be an uncalibrated number that later
    stages would be tempted to trust. Confidence is computed downstream."""
    lowered = SYSTEM_PROMPT.lower()
    assert '"confidence"' not in lowered
    assert "confidence score" not in lowered


# --------------------------------------------------------------------------
# JSON extraction
# --------------------------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"status": "ANSWER"}')["status"] == "ANSWER"


def test_extract_json_from_markdown_fence():
    text = 'Sure!\n```json\n{"status": "ANSWER", "answer": "x"}\n```\n'
    assert extract_json(text)["answer"] == "x"


def test_extract_json_from_surrounding_prose():
    text = 'Here is the result: {"status": "ANSWER", "answer": "y"} Hope that helps.'
    assert extract_json(text)["answer"] == "y"


def test_extract_json_handles_braces_inside_strings():
    text = '{"status": "ANSWER", "answer": "use {this} form"}'
    assert extract_json(text)["answer"] == "use {this} form"


@pytest.mark.parametrize("bad", ["", "   ", "no json at all", "{unclosed: "])
def test_extract_json_rejects_garbage(bad):
    with pytest.raises(MalformedAnswerError):
        extract_json(bad)


# --------------------------------------------------------------------------
# citations
# --------------------------------------------------------------------------


def test_citation_is_built_from_chunk_metadata_only():
    c = Citation.from_chunk("E1", make_chunk(), retrieval_rank=1, retrieval_score=9.5)
    assert c.short() == "[D09, p.123]"
    assert "Section 7.2" in c.full()
    assert "D09-C0001" in c.full()


def test_resolve_labels_rejects_labels_not_in_context():
    """The core anti-hallucination check: the model cannot cite evidence it
    was never given."""
    ctx = build_context(make_results(2))
    cites, invalid = resolve_labels(["E1", "E9"], ctx)
    assert [c.label for c in cites] == ["E1"]
    assert invalid == ["E9"]


def test_resolve_labels_is_case_and_bracket_tolerant():
    ctx = build_context(make_results(2))
    cites, invalid = resolve_labels([" e2 ", "[E1]"], ctx)
    assert {c.label for c in cites} == {"E1", "E2"}
    assert invalid == []


def test_resolve_labels_deduplicates():
    ctx = build_context(make_results(2))
    cites, _ = resolve_labels(["E1", "E1", "e1"], ctx)
    assert len(cites) == 1


def test_reference_list_deduplicates_by_chunk():
    ctx = build_context(make_results(2))
    cites, _ = resolve_labels(["E1", "E2"], ctx)
    rendered = render_reference_list(cites + cites)
    assert rendered.count("D01-C0001") == 1
    assert rendered.startswith("1. ")


# --------------------------------------------------------------------------
# answer validation
# --------------------------------------------------------------------------


def good_reply(labels=("E1",)) -> str:
    return json.dumps(
        {
            "status": "ANSWER",
            "answer": "Insulating oil shall be tested for BDV annually.",
            "claims": [
                {
                    "text": "Insulating oil shall be tested for BDV annually.",
                    "evidence_labels": list(labels),
                }
            ],
            "conflict": {"present": False, "description": "", "evidence_labels": []},
            "clarification_question": "",
            "missing_information": "",
        }
    )


def test_build_answer_happy_path():
    ctx = build_context(make_results(3))
    a = build_answer("How often is BDV tested?", good_reply(), ctx)
    assert a.status is AnswerStatus.ANSWER
    assert a.signals.citation_coverage == 1.0
    assert a.signals.n_invalid_labels == 0
    assert a.citations[0].document_id == "D01"


def test_answer_claiming_evidence_it_was_not_given_is_flagged():
    ctx = build_context(make_results(2))
    a = build_answer("q", good_reply(labels=("E7",)), ctx)
    assert a.signals.invalid_labels == ["E7"]
    assert a.status is AnswerStatus.UNSUPPORTED
    assert "cited no evidence label" in a.downgrade_reason


def test_answer_with_no_citations_is_downgraded_to_unsupported():
    """Structural rule, not a tuned threshold: an answer citing nothing that
    was retrieved is by definition not evidence-grounded."""
    ctx = build_context(make_results(2))
    reply = json.dumps(
        {"status": "ANSWER", "answer": "Test the oil yearly.", "claims": []}
    )
    a = build_answer("q", reply, ctx)
    assert a.status is AnswerStatus.UNSUPPORTED


def test_partially_cited_answer_keeps_answer_status_but_records_coverage():
    ctx = build_context(make_results(3))
    reply = json.dumps(
        {
            "status": "ANSWER",
            "answer": "Two things.",
            "claims": [
                {"text": "Cited claim.", "evidence_labels": ["E1"]},
                {"text": "Uncited claim.", "evidence_labels": []},
            ],
        }
    )
    a = build_answer("q", reply, ctx)
    assert a.status is AnswerStatus.ANSWER
    assert a.signals.n_claims == 2
    assert a.signals.n_supported_claims == 1
    assert a.signals.citation_coverage == 0.5


def test_insufficient_evidence_is_preserved():
    ctx = build_context(make_results(2))
    reply = json.dumps(
        {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": "",
            "claims": [],
            "missing_information": "no maintenance interval stated in D01",
        }
    )
    a = build_answer("q", reply, ctx)
    assert a.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert "no maintenance interval" in a.missing_information


def test_clarification_is_preserved_and_not_collapsed_into_abstention():
    ctx = build_context(make_results(2))
    reply = json.dumps(
        {
            "status": "NEEDS_CLARIFICATION",
            "answer": "",
            "claims": [],
            "clarification_question": "Which temperature limit: top-oil or winding hot-spot?",
        }
    )
    a = build_answer("q", reply, ctx)
    assert a.status is AnswerStatus.NEEDS_CLARIFICATION
    assert "top-oil" in a.clarification_question


def test_unrecognised_status_defaults_to_abstention_not_answer():
    ctx = build_context(make_results(2))
    reply = json.dumps({"status": "MAYBE", "answer": "something", "claims": []})
    a = build_answer("q", reply, ctx)
    assert a.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_parse_error_is_recorded_not_raised():
    ctx = build_context(make_results(2))
    a = build_answer("q", "I'm afraid I can't do that.", ctx)
    assert a.status is AnswerStatus.PARSE_ERROR
    assert a.parse_error


def test_conflict_without_description_is_not_reported_as_a_conflict():
    """Guards against a checkbox conflict — the handoff is explicit that
    conflicts must not be manufactured."""
    ctx = build_context(make_results(2))
    reply = json.dumps(
        {
            "status": "ANSWER",
            "answer": "x",
            "claims": [{"text": "x", "evidence_labels": ["E1"]}],
            "conflict": {"present": True, "description": "", "evidence_labels": ["E1"]},
        }
    )
    a = build_answer("q", reply, ctx)
    assert a.conflict_present is False
    assert a.signals.conflict_reported is False


def test_real_conflict_is_preserved_with_its_citations():
    ctx = build_context(make_results(3))
    reply = json.dumps(
        {
            "status": "ANSWER",
            "answer": "Sources disagree.",
            "claims": [{"text": "Sources disagree.", "evidence_labels": ["E1", "E2"]}],
            "conflict": {
                "present": True,
                "description": "E1 states 50 kV, E2 states 60 kV.",
                "evidence_labels": ["E1", "E2"],
            },
        }
    )
    a = build_answer("q", reply, ctx)
    assert a.conflict_present is True
    assert len(a.conflict_citations) == 2


def test_answer_serialises_to_json_safe_dict():
    ctx = build_context(make_results(2))
    a = build_answer("q", good_reply(), ctx)
    json.dumps(a.to_dict())  # must not raise


# --------------------------------------------------------------------------
# llm client guards
# --------------------------------------------------------------------------


def test_quota_error_is_not_reported_as_a_network_problem():
    """A 429 means the key works, the network works, and the account has no
    credit. Reporting it as 'request failed' sends the reader to the firewall,
    which is exactly what happened during the first live runs."""
    from src.generation.llm import explain_api_error

    msg = explain_api_error(
        Exception("Error code: 429 - {'error': {'message': 'You exceeded your current quota'}}")
    )
    assert "no credit" in msg or "quota" in msg
    assert "billing" in msg
    assert "listing models is free" in msg.lower()


def test_auth_error_points_at_the_key_not_the_network():
    from src.generation.llm import explain_api_error

    msg = explain_api_error(Exception("Error code: 401 - invalid_api_key"))
    assert "401" in msg and "api-keys" in msg


def test_connection_error_points_at_the_diagnostic():
    from src.generation.llm import explain_api_error

    msg = explain_api_error(Exception("Connection error."))
    assert "diagnose_network" in msg


def test_openai_client_supplies_its_own_http_client(monkeypatch):
    """The SDK's default client construction recurses until the stack blows on
    Python 3.14, surfacing as a bare 'Connection error'. Passing an explicit
    httpx client is measured to work (scripts/probe_sdk.py)."""
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    from src.generation.llm import OpenAIClient

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    OpenAIClient(api_key="sk-test")._ensure_client()
    assert "http_client" in captured, "SDK was left to build its own client"


# --------------------------------------------------------------------------
# the two kinds of 429
# --------------------------------------------------------------------------


class FakeHeaders(dict):
    """httpx-style headers: case-insensitive .get()."""

    def get(self, key, default=None):
        return super().get(key.lower(), default)


def api_error(message: str, status=None, code=None, headers=None):
    """An exception shaped like the ones the SDKs actually raise."""
    e = Exception(message)
    if status is not None:
        e.status_code = status
    if code is not None:
        e.body = {"error": {"code": code, "message": message}}
    if headers is not None:
        e.response = type("R", (), {"status_code": status, "headers": FakeHeaders(headers)})()
    return e


def test_billing_429_and_rate_limit_429_are_told_apart():
    """Both are 429. One is permanent until someone pays; the other clears in
    seconds. Conflating them is how a run gets abandoned for a billing problem
    that does not exist."""
    from src.generation.llm import classify_api_error

    empty_account = api_error(
        "Error code: 429 - You exceeded your current quota, please check your "
        "plan and billing details",
        status=429,
        code="insufficient_quota",
    )
    too_fast = api_error(
        "Error code: 429 - Rate limit reached for gpt-4o-mini, please try again in 6s",
        status=429,
        code="rate_limit_exceeded",
    )

    assert classify_api_error(empty_account) == "quota"
    assert classify_api_error(too_fast) == "rate_limit"


def test_azure_token_rate_limit_is_a_rate_limit():
    """Azure phrases it differently from OpenAI and never says 'rate_limit_exceeded'."""
    from src.generation.llm import classify_api_error

    e = api_error(
        "Requests to the ChatCompletions_Create Operation under Azure OpenAI API "
        "have exceeded token rate limit of your current OpenAI S0 pricing tier. "
        "Please retry after 34 seconds.",
        status=429,
    )
    assert classify_api_error(e) == "rate_limit"


def test_unrecognised_429_is_treated_as_a_rate_limit_not_a_billing_failure():
    """Deliberate asymmetry: guessing 'rate limit' costs a minute of backoff,
    guessing 'billing' tells someone their account is empty when it is not."""
    from src.generation.llm import classify_api_error

    assert classify_api_error(api_error("Error code: 429", status=429)) == "rate_limit"


def test_rate_limit_message_does_not_tell_the_user_to_pay():
    from src.generation.llm import explain_api_error

    msg = explain_api_error(api_error("429 rate limit reached", status=429))
    assert "billing" not in msg.lower()
    assert "no credit" not in msg.lower()
    assert "rate limit" in msg.lower()


def test_retry_after_header_is_preferred_over_our_own_backoff():
    from src.generation.llm import RetryPolicy, retry_after_seconds

    e = api_error("429", status=429, headers={"retry-after": "7"})
    assert retry_after_seconds(e) == 7.0
    assert RetryPolicy().delay_for(1, retry_after_seconds(e)) == 7.0


@pytest.mark.parametrize(
    "raw,expected",
    [("6m0s", 360.0), ("1.5s", 1.5), ("20ms", 0.02), ("2", 2.0), ("soon", None)],
)
def test_duration_headers_parse(raw, expected):
    from src.generation.llm import retry_after_seconds

    assert retry_after_seconds(api_error("429", 429, headers={"retry-after": raw})) == expected


def test_rate_limits_are_retried_and_the_call_eventually_succeeds():
    from src.generation.llm import RetryPolicy, call_with_retry

    attempts, slept = [], []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise api_error("429 rate limit reached", status=429, code="rate_limit_exceeded")
        return "ok"

    out = call_with_retry(flaky, RetryPolicy(max_attempts=5, base_delay=2.0), sleep=slept.append)
    assert out == "ok"
    assert len(attempts) == 3
    assert slept == [2.0, 4.0]  # exponential, deterministic


def test_a_quota_failure_is_never_retried():
    """Retrying an empty account just multiplies the wait before the person
    sees the one message that would have helped them."""
    from src.generation.llm import RetryPolicy, call_with_retry

    attempts, slept = [], []

    def broke():
        attempts.append(1)
        raise api_error("429 You exceeded your current quota", 429, code="insufficient_quota")

    with pytest.raises(Exception):
        call_with_retry(broke, RetryPolicy(max_attempts=5), sleep=slept.append)
    assert len(attempts) == 1
    assert slept == []


def test_backoff_is_capped():
    from src.generation.llm import RetryPolicy

    policy = RetryPolicy(base_delay=2.0, max_delay=60.0)
    assert policy.delay_for(10) == 60.0
    assert policy.delay_for(1, suggested=900.0) == 60.0


def test_exhausted_retries_surface_the_rate_limit_not_a_billing_message():
    from src.generation.llm import RetryPolicy, call_with_retry, explain_api_error

    def always():
        raise api_error("429 rate limit reached", 429, code="rate_limit_exceeded")

    with pytest.raises(Exception) as excinfo:
        call_with_retry(always, RetryPolicy(max_attempts=2), sleep=lambda _: None)
    assert "rate limit" in explain_api_error(excinfo.value).lower()


def test_model_404_mentions_the_azure_deployment_name_trap():
    from src.generation.llm import explain_api_error

    msg = explain_api_error(api_error("Error code: 404 - The model does not exist", status=404))
    assert "OPENAI_MODEL" in msg and "deployment" in msg.lower()


def test_scripted_client_is_rejected_on_reported_result_paths():
    with pytest.raises(MockClientInEvaluationError):
        assert_real_client(ScriptedClient(["{}"]))


def test_real_client_passes_the_guard():
    class Fake(LLMClient):
        provider = "fake"
        model = "fake-1"

        def complete(self, messages):  # pragma: no cover - not called
            raise NotImplementedError

    assert_real_client(Fake())  # must not raise


def test_dotenv_is_loaded_into_environment(tmp_path, monkeypatch):
    """Every entry point must pick up .env. The evaluation runner previously
    did not, and reported 'API key unset' for a key that was already set."""
    from src.generation.llm import load_dotenv

    env = tmp_path / ".env"
    env.write_text('OPENAI_API_KEY="sk-from-file"\n# comment\nOPENAI_MODEL=gpt-x\n')
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    assert load_dotenv(env) is True
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"
    assert os.environ["OPENAI_MODEL"] == "gpt-x"


def test_existing_environment_wins_over_dotenv(tmp_path, monkeypatch):
    from src.generation.llm import load_dotenv

    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-from-file\n")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-already-exported")

    load_dotenv(env)
    assert os.environ["OPENAI_API_KEY"] == "sk-already-exported"


def test_empty_values_are_skipped_not_exported_as_empty_strings(tmp_path, monkeypatch):
    """`KEY=` means "not configured". Exporting it as "" means "configured, to
    nothing" — and to the OpenAI SDK, an empty OPENAI_BASE_URL is a base URL,
    producing a schemeless request and a bare "Connection error" that looks
    exactly like a firewall. Copying .env.example was enough to trigger it."""
    from src.generation.llm import load_dotenv

    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=sk-real\nOPENAI_BASE_URL=\nEMBEDDING_PROVIDER=\n")
    for var in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "EMBEDDING_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    load_dotenv(env)
    assert os.environ["OPENAI_API_KEY"] == "sk-real"
    assert "OPENAI_BASE_URL" not in os.environ
    assert "EMBEDDING_PROVIDER" not in os.environ


def test_shipped_env_example_has_no_blank_optional_keys():
    """The template must not be able to poison a .env just by being copied."""
    from src.generation.llm import REPO_ROOT

    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        # OPENAI_API_KEY is intentionally blank: the user must fill it in, and
        # an empty key is caught with a clear message rather than a URL fault.
        if key.strip() == "OPENAI_API_KEY":
            continue
        assert value.strip(), f"{key.strip()} is blank in .env.example; comment it out instead"


def test_missing_dotenv_is_not_an_error(tmp_path):
    from src.generation.llm import load_dotenv

    assert load_dotenv(tmp_path / "nope.env") is False


def test_unavailable_reason_names_the_missing_key_not_both_causes(monkeypatch):
    """'package missing or key unset' sends you checking two things when only
    one is wrong."""
    from src.generation.llm import OpenAIClient

    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reason = OpenAIClient(api_key="").availability_error()
    assert reason and "OPENAI_API_KEY is empty" in reason
    assert ".env.example" in reason  # points at the right file


def test_scripted_client_raises_when_exhausted():
    c = ScriptedClient(["{}"])
    c.complete([{"role": "user", "content": "x"}])
    with pytest.raises(AssertionError):
        c.complete([{"role": "user", "content": "x"}])


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------


def build_test_pipeline(responses, top_k=3):
    chunks = [
        make_chunk(
            chunk_id="D09-C0001",
            document_id="D09",
            verified_information="Insulating oil shall be tested for BDV annually.",
        ),
        make_chunk(
            chunk_id="D01-C0001",
            document_id="D01",
            equipment="Circuit Breaker",
            verified_information="Circuit breaker contacts shall be inspected every five years.",
            pdf_page="p.42",
        ),
    ]
    return RAGPipeline(chunks, llm=ScriptedClient(responses), top_k=top_k).index()


def test_pipeline_end_to_end_produces_grounded_answer_with_citations():
    pipe = build_test_pipeline([good_reply()])
    result = pipe.answer("How often should insulating oil BDV be tested?")
    assert result.status is AnswerStatus.ANSWER
    assert result.answer.citations
    assert "Sources:" in result.rendered()
    assert result.retrieval_ms >= 0


def test_pipeline_passes_real_evidence_text_to_the_model():
    pipe = build_test_pipeline([good_reply()])
    pipe.answer("insulating oil BDV test frequency")
    sent = pipe.llm.calls[0][1]["content"]
    assert "Insulating oil shall be tested for BDV annually." in sent


def test_pipeline_renders_abstention_without_inventing_content():
    reply = json.dumps(
        {
            "status": "INSUFFICIENT_EVIDENCE",
            "answer": "",
            "claims": [],
            "missing_information": "no such interval in the KB",
        }
    )
    pipe = build_test_pipeline([reply])
    result = pipe.answer("What is the SF6 top-up interval?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    rendered = result.rendered()
    assert "does not contain sufficient evidence" in rendered
    assert "Sources:" not in rendered


def test_pipeline_abstains_without_calling_the_model_when_nothing_retrieved():
    """No evidence must never reach the model — an empty evidence block is an
    invitation to answer from parametric knowledge."""
    pipe = RAGPipeline([], llm=ScriptedClient([]), top_k=3).index()
    result = pipe.answer("anything at all")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert pipe.llm.calls == []


def test_pipeline_result_serialises():
    pipe = build_test_pipeline([good_reply()])
    result = pipe.answer("oil BDV")
    payload = result.to_dict()
    json.dumps(payload)
    assert payload["retrieved_chunk_ids"]
    assert payload["timing_ms"]["total"] >= 0
