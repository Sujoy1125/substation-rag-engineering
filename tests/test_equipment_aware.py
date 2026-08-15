import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.chunk import Chunk
from src.retrieval.equipment_aware import EquipmentAwareRetriever, extract_equipment
from src.retrieval.retrievers import BM25Retriever


def _fake_chunk(chunk_id: str, text: str, equipment: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id="D09", original_filename="x.pdf",
        document_title="Test Doc", organization="Test Org", authority_level="HIGH",
        equipment=equipment, equipment_subtype="NOT VERIFIED", topic="O&M",
        subtopic="", knowledge_type="PROCEDURE", verified_information=text,
        procedure="", frequency="", technical_limit_value="", safety_information="",
        troubleshooting_failure_information="", applicability="", pdf_page="p.1",
        source_section="", notes="",
    )


def test_extract_equipment_finds_transformer():
    assert "Transformer" in extract_equipment("What is the OTI limit for a power transformer?")


def test_extract_equipment_finds_circuit_breaker_via_alias():
    assert "Circuit Breaker" in extract_equipment("How should a VCB be inspected?")


def test_extract_equipment_returns_empty_for_generic_query():
    assert extract_equipment("What maintenance is required?") == set()


def test_extract_equipment_can_return_multiple_types():
    found = extract_equipment("current transformer connected to the power transformer")
    assert "CT" in found
    assert "Transformer" in found


def test_equipment_aware_boosts_matching_chunk_above_nonmatching_equal_score():
    # BM25 IDF is degenerate/unstable on a 2-document corpus (shared terms
    # can get negative IDF), so a handful of unrelated filler chunks are
    # included to give BM25 a realistic-sized corpus and make the two
    # candidate chunks' base scores genuinely tied on the query terms.
    filler = [
        _fake_chunk(f"F{i}", f"unrelated filler content number {i} about isolators and earthing", equipment="Isolator / Disconnector")
        for i in range(8)
    ]
    c1 = _fake_chunk("C1", "circuit breaker maintenance schedule", equipment="Circuit Breaker")
    c2 = _fake_chunk("C2", "circuit breaker maintenance schedule", equipment="Circuit Breaker")
    # give C2 a different equipment field so only C1 gets the boost
    c2_no_match = _fake_chunk("C2", "circuit breaker maintenance schedule", equipment="NOT VERIFIED")
    chunks = filler + [c1, c2_no_match]

    base = BM25Retriever()
    eq_aware = EquipmentAwareRetriever(base, boost=0.5)
    eq_aware.index(chunks)
    results = eq_aware.retrieve("How is the circuit breaker maintained?", top_k=2)
    assert results[0].chunk.chunk_id == "C1"


def test_equipment_aware_does_not_hard_filter_out_nonmatching_chunks():
    chunks = [
        _fake_chunk("C1", "totally unrelated content about something else", equipment="Busbar"),
    ]
    base = BM25Retriever()
    eq_aware = EquipmentAwareRetriever(base, boost=0.5)
    eq_aware.index(chunks)
    results = eq_aware.retrieve("circuit breaker inspection", top_k=5)
    # even with zero equipment match, the only chunk must still be returned
    # (soft boost, not a hard filter that could create false negatives)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "C1"


def test_equipment_aware_name_reflects_base_retriever():
    eq_aware = EquipmentAwareRetriever(BM25Retriever())
    assert "bm25" in eq_aware.name
    assert "equipment_aware" in eq_aware.name
