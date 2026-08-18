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
PANEL_HTML = (REPO_ROOT / "src" / "api" / "static" / "panel.html").read_text(encoding="utf-8")


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


def test_service_starts_with_no_api_key_configured(monkeypatch, tmp_path):
    """The README promises retrieval, /evidence and /facets work without a key.
    Booting the whole service on a missing key would make that false, and a new
    contributor could not see the system work at all."""
    from src.generation.llm import UnavailableClient, client_from_env

    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    client = client_from_env(env_path=tmp_path / "absent.env", strict=False)
    assert isinstance(client, UnavailableClient)
    assert "OPENAI_API_KEY" in (client.availability_error() or "")

    pipeline = RAGPipeline(CHUNKS, llm=client, top_k=3).index()
    c = client_for(RAGService(pipeline))

    # The endpoints that never touch a model must work.
    assert c.get("/", follow_redirects=True).status_code == 200
    assert c.get("/facets").status_code == 200
    assert c.get("/evidence/D02-C0007").status_code == 200
    # Readiness is honest about it.
    health = c.get("/health")
    assert health.status_code == 503
    assert health.json()["llm"]["ready"] is False
    # And asking fails with the cause named, not a crash.
    ask = c.post("/ask", json={"question": "oil BDV frequency"})
    assert ask.status_code == 503
    assert "OPENAI_API_KEY" in ask.json()["detail"]


def test_strict_client_still_raises_so_evaluation_runs_fail_early(tmp_path, monkeypatch):
    """Only the HTTP service tolerates a missing key. An evaluation run must
    still fail at startup rather than halfway through 44 questions."""
    from src.generation.llm import LLMUnavailableError, client_from_env

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    with pytest.raises(LLMUnavailableError):
        client_from_env(env_path=tmp_path / "absent.env")


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


# --------------------------------------------------------------------------
# the interface must not assert what the data does not say
# --------------------------------------------------------------------------


def test_ui_only_promotes_values_that_read_as_values():
    """A headline card asserts "THIS is the interval". Most frequency fields in
    KB_v1.1 hold the table row an interval was extracted from — row number, sub
    item marker and all. Rendering that under the word INTERVAL is the
    interface making a claim the data does not support."""
    import re

    js = re.search(r"function readsAsValue\(v\) \{(.*?)\n\}", UI_HTML, re.S)
    assert js, "readsAsValue must exist in the UI"

    # Mirror of the JS rule, kept in step by the assertions below.
    def reads_as_value(t):
        t = t.strip()
        if not t or len(t) > 40: return False
        if len(t.split()) > 6: return False
        if re.search(r"\(?[a-z]\)|\b\d+\)", t): return False
        if re.search(r"\b(shall|should|must|will|during|when|if)\b", t, re.I): return False
        if re.search(r"\s\d+$", t): return False
        return True

    # Real values from KB_v1.1 that MUST still get a card.
    for good in ["Annually", "Half yearly", "2500 KVA; 33 KV", "60 kV minimum",
                 "0.5 ppm", "Monthly, Half-Yearly, Yearly", "415 V; 80°C"]:
        assert reads_as_value(good), f"{good!r} should be promoted"

    # Real values from KB_v1.1 that MUST NOT.
    for bad in [
        "12 Electrical Oil BDV Yearly a) BDV of transformer oil",
        "(b) Fire hydrant pumps shall be tested weekly and jockey pumps daily",
        "Boiler shall be externally inspected every time after annual maintenance",
        "Checking of space heater Yearly 18",
    ]:
        assert not reads_as_value(bad), f"{bad!r} should be demoted"


def test_demoted_values_are_still_shown_not_hidden():
    """Demotion is about the claim, not the content. A messy frequency is still
    a fact about the source and must remain visible and attributed — hiding it
    would make coverage look better than it is."""
    assert "Recorded in the source" in UI_HTML
    assert "Quoted verbatim from the knowledge base" in UI_HTML
    assert "quoted" in UI_HTML


def test_ungated_warning_is_for_a_user_not_a_developer(tmp_path, capsys):
    """The user needs one fact: these answers were not confidence-gated. Paths
    and commands belong in the console the developer is watching — an absolute
    home-directory path on screen reads as unfinished software to a reviewer,
    and tells a technician nothing they can act on."""
    service = RAGService.from_env(
        ScriptedClient([GOOD_REPLY]),
        chunks=CHUNKS,
        config=__import__("src.api.service", fromlist=["ServiceConfig"]).ServiceConfig(
            confidence_model_path=tmp_path / "absent.json"
        ),
    )
    body = client_for(service).post("/ask", json={"question": "oil BDV"}).json()

    warning = body["warning"]
    assert "not confidence-gated" in warning
    for leak in ("/Users/", "python experiments", "--from", ".json", "\\"):
        assert leak not in warning, f"user-facing warning leaks {leak!r}"
    assert len(warning) < 120, "a user-facing warning should be one sentence"

    # The detail must still exist — for the developer, on the console.
    printed = capsys.readouterr().out
    assert "calibrate_confidence.py" in printed


# --------------------------------------------------------------------------
# statement-by-statement attribution
# --------------------------------------------------------------------------


def test_claims_carry_their_evidence_through_the_api():
    """The link between a statement and its source existed on the server and
    never reached the interface. With three sentences and three sources a
    reader could not tell which came from which."""
    c = client_for(build_service([GOOD_REPLY]))
    body = c.post("/ask", json={"question": "oil BDV frequency"}).json()

    assert body["claims"], "an ANSWER must expose its claims"
    claim = body["claims"][0]
    assert claim["text"]
    assert claim["labels"] == ["E1"]
    assert claim["supported"] is True
    # Every chunk_id must resolve, or the attribution is decorative.
    for cid in claim["chunk_ids"]:
        assert c.get(f"/evidence/{cid}").status_code == 200


def test_an_uncited_claim_is_shown_not_filtered_out():
    """A claim the model asserted without valid evidence is exactly what a
    reader needs to see. Hiding it would present a partly-grounded answer as a
    fully-grounded one."""
    reply = json.dumps({
        "status": "ANSWER",
        "answer": "Oil is tested yearly and bushings every two years [E1].",
        "claims": [
            {"text": "oil tested yearly", "evidence_labels": ["E1"]},
            {"text": "bushings every two years", "evidence_labels": ["E9"]},
        ],
    })
    body = client_for(build_service([reply])).post(
        "/ask", json={"question": "intervals"}
    ).json()

    by_text = {c["text"]: c for c in body["claims"]}
    assert by_text["oil tested yearly"]["supported"] is True
    bad = by_text["bushings every two years"]
    assert bad["supported"] is False, "an uncited claim must still appear"
    assert bad["labels"] == []
    assert "E9" in bad["invalid_labels"], "an invented label is counted, not dropped"


def test_ui_renders_attribution_and_flags_uncited_claims():
    for marker in ["function grounding(", "Grounded in", "claim-bad", "not cited", "invented "]:
        assert marker in UI_HTML, f"UI missing {marker!r}"
    # It must read claims from the payload, never re-parse the prose.
    assert "b.claims" in UI_HTML


# --------------------------------------------------------------------------
# the control-panel skin
#
# /panel is a second face on one service, not a second service. The risk a
# decorative skin carries is that the decoration quietly drops something the
# plain view was careful about — the gate state, the wording that separates a
# recorded safety line from advice, the difference between "we declined" and
# "we broke". These tests hold both surfaces to the same properties, so the
# skin cannot be chosen and later found to have cost honesty.
# --------------------------------------------------------------------------


SURFACES = [("index.html", UI_HTML), ("panel.html", PANEL_HTML)]


def test_panel_is_served():
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/panel", follow_redirects=True)
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text


def test_panel_falls_back_to_the_plain_ui_when_absent(monkeypatch):
    """A missing skin is a cosmetic loss, not an outage — and mid-demo the
    right behaviour is to show the working interface, not an error."""
    from src.api import service as service_module

    monkeypatch.setattr(service_module, "PANEL_PATH", Path("/nonexistent/panel.html"))
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/panel", follow_redirects=False)
    assert r.status_code in (301, 302, 307)
    assert r.headers["location"] == "/"


def test_both_surfaces_reach_only_this_service():
    """Demo wifi is not worth betting a presentation on, and a skin is where
    a webfont or an icon CDN usually sneaks in."""
    for name, html in SURFACES:
        for pattern in ("http://", "https://", "//cdn", 'src="//'):
            assert pattern not in html, f"{name} reaches outside itself: {pattern!r}"


def test_both_surfaces_call_endpoints_the_service_exposes():
    c = client_for(build_service([GOOD_REPLY]))
    spec = c.get("/openapi.json").json()["paths"]

    for name, html in SURFACES:
        for path in ('"/ask"', '"/health"', '"/facets"', "/evidence/"):
            assert path in html, f"{name} calls {path}"
    for path in ("/ask", "/health", "/facets", "/evidence/{chunk_id}"):
        assert path in spec, f"a surface calls {path} but the API does not serve it"


def test_both_surfaces_render_every_answer_status():
    from src.generation.answer import AnswerStatus

    for name, html in SURFACES:
        for status in AnswerStatus:
            if status is AnswerStatus.LLM_ERROR:
                continue  # surfaced as HTTP 503, never as an answer body
            assert status.value in html, f"{name} has no rendering for {status.value}"


def test_both_surfaces_show_whether_the_answer_was_gated():
    """An ungated answer that looks identical to a gated one is the single
    most misleading thing this interface could do."""
    for name, html in SURFACES:
        assert "b.gated" in html, f"{name} does not read the gate state"
        assert "ungated" in html, f"{name} never says an answer was ungated"
        assert "b.warning" in html, f"{name} drops the ungated warning"


def test_both_surfaces_label_safety_text_as_recorded_not_as_advice():
    """The KB's safety line is a quotation from a document, not an instruction
    this system is issuing. A panel that renders it as a warning lamp would be
    making a claim no one has authorised."""
    for name, html in SURFACES:
        assert "recorded in the cited source" in html.lower(), f"{name} presents safety text as its own"
        assert "Quoted verbatim from the knowledge base" in html, f"{name} drops the provenance caveat"
        assert "permit-to-work" in html, f"{name} drops the deferral to site procedure"


def test_both_surfaces_separate_a_service_fault_from_an_abstention():
    """"We could not reach the model" and "the knowledge base does not support
    an answer" are opposite facts. Conflated, a broken key reads as a gap in
    the documents."""
    for name, html in SURFACES:
        assert "renderFailure" in html, f"{name} has no distinct failure path"
        assert "not a finding about the knowledge base" in html, f"{name} may read a fault as a gap"


def test_both_surfaces_take_domain_values_from_metadata_not_from_prose():
    """Intervals and limits come from the citation payload — KB metadata —
    never from parsing what the model wrote. The one regex over the answer
    body finds evidence labels, which are validated upstream."""
    for name, html in SURFACES:
        assert "c.frequency" in html and "c.technical_limit_value" in html, f"{name}"
        assert "safety_notes" in html, f"{name}"
        assert html.count(".replace(/\\[(E\\d+)\\]/g") == 1, f"{name} parses the prose more than once"


def test_both_surfaces_apply_the_same_promotion_rule():
    """A card headed INTERVAL asserts "this IS the interval". The rule that
    decides what earns one must not drift between the two skins."""
    import re

    rules = []
    for name, html in SURFACES:
        m = re.search(r"function readsAsValue\(v\) \{(.*?)\n\}", html, re.S)
        assert m, f"{name} has no readsAsValue"
        rules.append(re.sub(r"\s+", " ", m.group(1)).strip())
    assert rules[0] == rules[1], "the two surfaces would promote different values"


def test_both_surfaces_keep_demoted_values_visible():
    for name, html in SURFACES:
        assert "Recorded in the source" in html, f"{name} hides messy values instead of quoting them"


def test_both_surfaces_show_attribution_and_flag_uncited_claims():
    for name, html in SURFACES:
        for marker in ["function grounding(", "Grounded in", "not cited", "invented ", "b.claims"]:
            assert marker in html, f"{name} missing {marker!r}"


def test_both_surfaces_link_inline_evidence_labels_to_their_source():
    """[E4] in the answer text is the reader's shortest path to the document.
    Left as plain text it is noise; linked, it is the check."""
    for name, html in SURFACES:
        assert "function linkLabels(" in html, f"{name} does not link inline labels"
        assert "linkLabels(b.answer" in html, f"{name} defines linkLabels but never uses it"


def test_no_surface_uses_browser_storage():
    """Nothing here is worth persisting, and a stale cached answer shown
    beside a fresh question is a correctness bug wearing a UI costume."""
    for name, html in SURFACES:
        for api in ("localStorage", "sessionStorage", "indexedDB"):
            assert api not in html, f"{name} uses {api}"


# --------------------------------------------------------------------------
# /box — the standard interface behind a door
#
# The risk of a decorative enclosure is not that it looks bad. It is that it
# becomes a second copy of the interface that quietly diverges from the one
# under test, or that it fails shut and takes the demo with it. These tests
# hold it to being an overlay on the real page and nothing else.
# --------------------------------------------------------------------------


def test_box_serves_the_standard_ui_plus_the_enclosure():
    c = client_for(build_service([GOOD_REPLY]))

    plain = c.get("/", follow_redirects=True).text
    boxed = c.get("/box", follow_redirects=True).text

    assert "encl" in boxed, "the enclosure was not injected"
    # Everything the plain page has, the boxed page still has.
    body = plain.split("<body>", 1)[1].rsplit("</body>", 1)[0]
    assert body in boxed, "/box is not the same interface underneath"
    assert len(boxed) > len(plain)


def test_box_falls_back_to_the_plain_ui_when_the_enclosure_is_missing(monkeypatch):
    """A missing decoration is a cosmetic loss. It must not cost the demo the
    interface."""
    from src.api import service as service_module

    monkeypatch.setattr(service_module, "ENCLOSURE_PATH", Path("/nonexistent.html"))
    c = client_for(build_service([GOOD_REPLY]))

    r = c.get("/box", follow_redirects=True)
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.text
    assert "encl-leaf" not in r.text


def test_the_door_is_built_by_script_not_left_shut_in_the_html():
    """A door drawn in HTML stays shut when its opener fails, turning a
    decoration into a total failure. Built by script, a broken script means no
    door at all — the interface is simply there."""
    html = (REPO_ROOT / "src" / "api" / "static" / "enclosure.html").read_text(
        encoding="utf-8"
    )

    before_script = html.split("<script>")[0]  # comment + <style> only
    assert "<div" not in before_script, "the door is in the HTML and would stay shut"
    assert 'createElement("div")' in html
    assert "el.remove()" in html, "the overlay must leave the document, not just hide"


def test_the_door_cannot_trap_the_user():
    """Three independent ways out: it opens itself, any click or key opens it,
    and a final timeout removes it regardless."""
    html = (REPO_ROOT / "src" / "api" / "static" / "enclosure.html").read_text(
        encoding="utf-8"
    )

    assert "setTimeout(open, 900)" in html, "it must open on its own"
    assert 'addEventListener("click", open)' in html
    assert "3000" in html, "a last-resort removal must exist"
    assert "prefers-reduced-motion" in html
    assert 'p.get("box") === "0"' in html, "there must be a way to skip it"


def test_box_reaches_only_this_service():
    html = (REPO_ROOT / "src" / "api" / "static" / "enclosure.html").read_text(
        encoding="utf-8"
    )
    for pattern in ("http://", "https://", "//cdn"):
        assert pattern not in html, f"the enclosure reaches outside itself: {pattern!r}"
