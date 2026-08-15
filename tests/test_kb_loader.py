import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.kb_loader import load_chunks, EXPECTED_COLUMNS

KB_PATH = Path(__file__).resolve().parents[1] / "KB_v1" / "knowledge_chunks.xlsx"


def _load():
    return load_chunks(str(KB_PATH))


def test_loads_all_1744_chunks():
    chunks, report = _load()
    assert report.loaded == 1744
    assert len(chunks) == 1744


def test_schema_matches_expected_21_columns():
    _, report = _load()
    assert report.header_matches_expected
    assert len(EXPECTED_COLUMNS) == 21


def test_no_malformed_rows():
    _, report = _load()
    assert report.malformed_row_numbers == []


def test_no_duplicate_chunk_ids():
    _, report = _load()
    assert report.duplicate_chunk_ids == []


def test_chunk_ids_unique_across_loaded_objects():
    chunks, _ = _load()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_report_ok_is_true():
    _, report = _load()
    assert report.ok()


def test_sentinel_fields_preserved_not_dropped():
    chunks, _ = _load()
    # D01 chunks are known (from manual inspection) to carry
    # "NOT VERIFIED" equipment subtype and "NOT COVERED" procedure —
    # the loader must preserve these literal sentinel strings, not blank them.
    d01 = [c for c in chunks if c.document_id == "D01"]
    assert d01, "expected at least one D01 chunk"
    assert any(c.equipment_subtype == "NOT VERIFIED" for c in d01)


def test_searchable_text_excludes_sentinels():
    chunks, _ = _load()
    c = next(c for c in chunks if c.equipment_subtype == "NOT VERIFIED")
    assert "NOT VERIFIED" not in c.searchable_text()
    assert "NOT COVERED" not in c.searchable_text()


def test_citation_is_deterministic_from_metadata_only():
    chunks, _ = _load()
    c = chunks[0]
    cit = c.citation()
    assert c.chunk_id in cit
    assert c.document_title in cit
    assert c.pdf_page in cit


def test_deterministic_across_repeated_loads():
    chunks1, _ = _load()
    chunks2, _ = _load()
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_known_chunk_d01_c0001_content():
    chunks, _ = _load()
    c = next(c for c in chunks if c.chunk_id == "D01-C0001")
    assert c.document_id == "D01"
    assert c.pdf_page == "PDF p. 9"
    assert c.authority_level == "HIGH"
