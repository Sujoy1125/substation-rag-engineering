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


def test_root_sends_a_browser_to_the_docs_instead_of_404ing():
    """Opening the root URL is the first thing anyone does. A 404 there reads
    as a broken service."""
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307)
    assert r.headers["location"] == "/docs"
    assert c.get("/", follow_redirects=True).status_code == 200


def test_unknown_chunk_id_is_404():
    c = client_for(build_service([GOOD_REPLY]))
    assert c.get("/evidence/NOPE-C9999").status_code == 404
