"""HTTP surface tests.

Every one of these runs offline against `ScriptedClient` — no API key, no cost,
no network. That is the point: the service layer must be verifiable without the
provider, or it only gets tested when someone has credit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.service import RAGService, create_app  # noqa: E402
from src.common.chunk import Chunk  # noqa: E402
from src.confidence.gate import ConfidenceGate, ConfidenceModel, Decision  # noqa: E402
from src.generation.llm import LLMUnavailableError, ScriptedClient  # noqa: E402
from src.generation.pipeline import RAGPipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_HTML = (REPO_ROOT / "src" / "api" / "static" / "index.html").read_text(encoding="utf-8")


def make_chunk(**overrides) -> Chunk:
    base = dict(
        chunk_id="D02-C0007",
        document_id="D02",
        original_filename="cbip_manual.pdf",
        document_title="CBIP Transformer Maintenance Manual",
        organization="CBIP",
        authority_level="National",
        equipment="Power Transformer",
        equipment_subtype="Oil-immersed",
        topic="Insulating Oil",
        subtopic="BDV",
        knowledge_type="Frequency",
        verified_information="Insulating oil shall be tested for BDV annually.",
        procedure="NOT COVERED",
        frequency="Annually",
        technical_limit_value="60 kV minimum",
        safety_information="NOT COVERED",
        troubleshooting_failure_information="NOT COVERED",
        applicability="NOT APPLICABLE",
        pdf_page="p.118",
        source_section="Section 7.3",
        notes="NOT VERIFIED",
    )
    base.update(overrides)
    return Chunk(**base)


CHUNKS = [
    make_chunk(),
    make_chunk(
        chunk_id="D01-C0001",
        document_id="D01",
        document_title="CEA Substation Safety Code",
        equipment="Circuit Breaker",
        topic="Contacts",
        verified_information="Circuit breaker contacts shall be inspected every five years.",
        frequency="Every five years",
        pdf_page="p.42",
    ),
    make_chunk(
        chunk_id="D03-C0002",
        document_id="D03",
        document_title="IEEE C57 Guide",
        equipment="Power Transformer",
        topic="Bushings",
        verified_information="Bushing tan delta shall be measured every two years.",
        frequency="Every two years",
        pdf_page="p.9",
    ),
]

GOOD_REPLY = json.dumps(
    {
        "status": "ANSWER",
        "answer": "Insulating oil should be tested for BDV annually [E1].",
        "claims": [{"text": "BDV tested annually", "evidence_labels": ["E1"]}],
    }
)
ABSTAIN_REPLY = json.dumps(
    {
        "status": "INSUFFICIENT_EVIDENCE",
        "answer": "",
        "claims": [],
        "missing_information": "no SF6 top-up interval in the KB",
    }
)


def build_service(replies, gate=None, gate_warning="") -> RAGService:
    pipeline = RAGPipeline(CHUNKS, llm=ScriptedClient(list(replies)), top_k=3).index()
    return RAGService(pipeline, gate=gate, gate_warning=gate_warning)


def client_for(service: RAGService) -> TestClient:
    return TestClient(create_app(service), raise_server_exceptions=False)


# --------------------------------------------------------------------------
# /ask
# --------------------------------------------------------------------------


def test_ask_returns_answer_with_resolvable_citations():
    c = client_for(build_service([GOOD_REPLY]))
    body = c.post("/ask", json={"question": "How often is insulating oil BDV tested?"}).json()

    assert body["status"] == "ANSWER"
    assert body["citations"], "an ANSWER must carry citations"
    for cite in body["citations"]:
        assert c.get(f"/evidence/{cite['chunk_id']}").status_code == 200, (
            "every citation must resolve to a chunk actually in the KB"
        )


def test_ask_abstention_carries_no_answer_text_and_no_citations():
    c = client_for(build_service([ABSTAIN_REPLY]))
    body = c.post("/ask", json={"question": "What is the SF6 top-up interval?"}).json()

    assert body["status"] == "INSUFFICIENT_EVIDENCE"
    assert body["answer"] == ""
    assert body["citations"] == []
    assert body["missing_information"]


def test_blank_and_oversized_questions_are_rejected_before_the_model_is_called():
    service = build_service([GOOD_REPLY])
    c = client_for(service)

    assert c.post("/ask", json={"question": "   "}).status_code == 422
    assert c.post("/ask", json={"question": "x" * 5000}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422
    assert service.pipeline.llm.calls == [], "a rejected request must not reach the model"


def test_provider_failure_is_503_and_names_the_cause_not_a_generic_error():
    """A dead provider is not a broken service. The caller needs to know which,
    because 'out of credit' and 'this code is broken' have different fixes."""

    class DeadClient(ScriptedClient):
        is_real = True

        def complete(self, messages):
            raise LLMUnavailableError(
                "the API returned 429 with a quota/billing code — the account has no credit"
            )

    pipeline = RAGPipeline(CHUNKS, llm=DeadClient([]), top_k=3).index()
    c = client_for(RAGService(pipeline))
    r = c.post("/ask", json={"question": "oil BDV frequency"})

    assert r.status_code == 503
    assert "credit" in r.json()["detail"]


# --------------------------------------------------------------------------
# the gate must never be silently absent
# --------------------------------------------------------------------------


def test_ungated_service_says_so_on_every_answer():
    """A demo that serves ungated answers while claiming to be confidence-gated
    is the same failure as reporting metrics for a run where every call died."""
    c = client_for(build_service([GOOD_REPLY], gate_warning="no calibrated model"))
    body = c.post("/ask", json={"question": "oil BDV frequency"}).json()

    assert body["gated"] is False
    assert body["gate"] is None
    assert "warning" in body


def test_gated_service_reports_the_decision_and_its_signals():
    model = ConfidenceModel(
        weights={
            "retrieval_strength": 1.0,
            "evidence_concentration": 1.0,
            "citation_coverage": 1.0,
            "citation_validity": 1.0,
            "evidence_utilisation": 1.0,
            "top_rank_cited": 1.0,
            "source_authority": 1.0,
            "answer_specificity": 1.0,
        },
        answer_threshold=0.5,
        clarify_threshold=0.3,
        fitted_on="test",
        fitted_n_questions=1,
    )
    c = client_for(build_service([GOOD_REPLY], gate=ConfidenceGate(model)))
    body = c.post("/ask", json={"question": "oil BDV frequency"}).json()

    assert body["gated"] is True
    assert body["gate"]["decision"] in {d.value for d in Decision}
    assert body["gate"]["signals"], "the decision must be inspectable, not a bare verdict"


def test_gate_that_withholds_an_answer_leaves_nothing_to_display():
    """An overruled answer must lose its text and citations, or the gated system
    takes credit for output the user never saw."""
    model = ConfidenceModel(
        weights={k: -50.0 for k in [
            "retrieval_strength", "evidence_concentration", "citation_coverage",
            "citation_validity", "evidence_utilisation", "top_rank_cited",
            "source_authority", "answer_specificity",
        ]},
        answer_threshold=0.9,
        clarify_threshold=0.8,
        fitted_on="test",
        fitted_n_questions=1,
    )
    c = client_for(build_service([GOOD_REPLY], gate=ConfidenceGate(model)))
    body = c.post("/ask", json={"question": "oil BDV frequency"}).json()

    assert body["status"] != "ANSWER"
    assert body["answer"] == ""
    assert body["citations"] == []
    assert "withheld by confidence gate" in body["downgrade_reason"]


# --------------------------------------------------------------------------
# /health and /evidence
# --------------------------------------------------------------------------


def test_health_reports_readiness_without_calling_the_model():
    service = build_service([GOOD_REPLY])
    c = client_for(service)
    body = c.get("/health").json()

    assert body["kb_chunks"] == len(CHUNKS)
    assert body["llm"]["provider"]
    assert "gated" in body
    assert service.pipeline.llm.calls == [], "/health must never spend a request"


def test_health_is_503_when_the_provider_is_not_configured():
    class Unconfigured(ScriptedClient):
        def availability_error(self):
            return "OPENAI_API_KEY is empty."

    pipeline = RAGPipeline(CHUNKS, llm=Unconfigured([]), top_k=3).index()
    r = client_for(RAGService(pipeline)).get("/health")

    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert "OPENAI_API_KEY" in r.json()["llm"]["problem"]


def test_evidence_returns_the_full_record_including_sentinels():
    """Someone auditing a citation must see 'NOT VERIFIED' if that is what the
    KB says. Hiding sentinels would make coverage look better than it is."""
    c = client_for(build_service([GOOD_REPLY]))
    body = c.get("/evidence/D02-C0007").json()

    assert body["verified_information"].startswith("Insulating oil")
    assert body["notes"] == "NOT VERIFIED"


def test_evidence_schema_covers_every_kb_field():
    """FastAPI filters a response to its declared model, so a field missing from
    the schema disappears silently from the endpoint people use to audit
    citations. The schema is derived from Chunk for exactly this reason; this
    test fails loudly if that ever stops being true."""
    from dataclasses import fields

    from src.api.service import ChunkOut

    assert set(ChunkOut.model_fields) == {f.name for f in fields(Chunk)}


def test_documented_schema_does_not_drop_fields_the_service_emits():
    """A response_model that omits a key would quietly truncate live responses."""
    from src.api.service import AskResponse

    service = build_service([GOOD_REPLY], gate_warning="uncalibrated")
    emitted = set(service.ask("oil BDV frequency").keys())
    assert emitted <= set(AskResponse.model_fields), (
        f"service emits keys the schema drops: {emitted - set(AskResponse.model_fields)}"
    )


def test_openapi_documents_the_answer_shape_not_a_bare_string():
    c = client_for(build_service([GOOD_REPLY]))
    spec = c.get("/openapi.json").json()

    ask = spec["paths"]["/ask"]["post"]["responses"]
    assert "$ref" in json.dumps(ask["200"]), "/ask 200 must reference a schema"
    assert "503" in ask, "an unreachable provider is a documented outcome, not a surprise"
    assert "AskResponse" in spec["components"]["schemas"]


def test_root_serves_the_ui():
    """Opening the root URL is the first thing anyone does. A 404 there reads
    as a broken service."""
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/", follow_redirects=True)
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text


def test_root_falls_back_to_docs_when_the_ui_file_is_absent(monkeypatch):
    """A missing UI asset must not take the API down with it."""
    from src.api import service as service_module

    monkeypatch.setattr(service_module, "UI_PATH", Path("/nonexistent/index.html"))
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307)
    assert r.headers["location"] == "/docs"


def test_ui_uses_no_external_resources():
    """A demo venue's wifi is not something to bet a presentation on, and every
    external request is a chance to show a judge a blank page."""
    html = UI_HTML
    for pattern in ("http://", "https://", "//cdn", "src=\"//"):
        assert pattern not in html, f"UI reaches outside itself: {pattern!r}"


def test_ui_calls_the_endpoints_the_service_actually_exposes():
    html = UI_HTML
    for path in ('"/ask"', '"/health"', '"/facets"', "/evidence/"):
        assert path in html, f"UI calls {path} which the service must expose"

    c = client_for(build_service([GOOD_REPLY]))
    spec = c.get("/openapi.json").json()["paths"]
    for path in ("/ask", "/health", "/facets", "/evidence/{chunk_id}"):
        assert path in spec, f"UI calls {path} but the API does not serve it"


def test_ui_never_reads_domain_values_out_of_generated_text():
    """Intervals, limits and safety text must come from the citation payload —
    i.e. from KB metadata — not from parsing the model's prose. The only thing
    scraped from answer text is the [E1] label, which is validated upstream."""
    html = UI_HTML

    assert "c.frequency" in html and "c.technical_limit_value" in html
    assert "safety_notes" in html
    # exactly one regex over the answer body, and it only finds evidence labels
    assert html.count(".replace(/\\[(E\\d+)\\]/g") == 1


def test_ui_renders_every_answer_status():
    """A status the UI does not know about would render as a blank card, which
    at a demo looks like a crash."""
    html = UI_HTML
    from src.generation.answer import AnswerStatus

    for status in AnswerStatus:
        if status is AnswerStatus.LLM_ERROR:
            continue  # surfaced as HTTP 503, never as an answer body
        assert status.value in html, f"UI has no rendering for {status.value}"


def test_citations_carry_the_operational_fields_from_the_kb():
    """A technician needs the interval and the limit, and they must come from
    the KB record rather than from whatever the model wrote."""
    c = client_for(build_service([GOOD_REPLY]))
    body = c.post("/ask", json={"question": "insulating oil BDV frequency"}).json()

    cite = body["citations"][0]
    assert cite["frequency"] == "Annually"
    assert cite["technical_limit_value"] == "60 kV minimum"
    assert cite["knowledge_type"] == "Frequency"


def test_sentinel_fields_are_null_not_the_word_NOT_COVERED():
    """"NOT COVERED" is an absence. Displaying it to a technician as though it
    were content is worse than showing nothing."""
    c = client_for(build_service([GOOD_REPLY]))
    cite = c.post("/ask", json={"question": "insulating oil BDV"}).json()["citations"][0]

    assert cite["safety_information"] is None  # the fixture chunk has NOT COVERED


def test_safety_notes_appear_only_when_a_cited_chunk_carries_them():
    safety_chunk = make_chunk(
        chunk_id="D05-C0100",
        document_title="CEA Safety Regulations",
        knowledge_type="SAFETY",
        equipment="Circuit Breaker",
        topic="Isolation",
        verified_information="Circuit breakers shall be isolated and earthed before any work.",
        safety_information="Confirm isolation and apply earths before approaching the equipment.",
    )
    pipeline = RAGPipeline(CHUNKS + [safety_chunk], llm=ScriptedClient([GOOD_REPLY]), top_k=3).index()
    body = client_for(RAGService(pipeline)).post(
        "/ask", json={"question": "insulating oil BDV frequency"}
    ).json()

    # The fixture reply cites E1 only; safety notes must follow the citations,
    # not the retrieved set — evidence the model did not rely on says nothing.
    for note in body["safety_notes"]:
        assert note["chunk_id"] in {c["chunk_id"] for c in body["citations"]}


def test_facets_report_real_coverage():
    c = client_for(build_service([GOOD_REPLY]))
    f = c.get("/facets").json()

    assert f["total_chunks"] == len(CHUNKS)
    assert {e["name"] for e in f["equipment"]} >= {"Power Transformer", "Circuit Breaker"}
    assert sum(d["chunks"] for d in f["documents"]) == len(CHUNKS)


def test_unknown_chunk_id_is_404():
    c = client_for(build_service([GOOD_REPLY]))
    assert c.get("/evidence/NOPE-C9999").status_code == 404
