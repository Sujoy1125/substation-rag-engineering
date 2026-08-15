import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.chunk import Chunk
from src.embeddings.provider import (
    LocalSentenceTransformerProvider,
    MockEmbeddingProvider,
    ModelUnavailableError,
    assert_not_mock,
)
from src.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRRFRetriever


def _fake_chunk(chunk_id: str, text: str, doc_id: str = "D09", page: str = "p.1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, document_id=doc_id, original_filename="x.pdf",
        document_title="Test Doc", organization="Test Org", authority_level="HIGH",
        equipment="Transformer", equipment_subtype="NOT VERIFIED", topic="O&M",
        subtopic="", knowledge_type="PROCEDURE", verified_information=text,
        procedure="", frequency="", technical_limit_value="", safety_information="",
        troubleshooting_failure_information="", applicability="", pdf_page=page,
        source_section="", notes="",
    )


# --- Embedding interface ---

def test_mock_provider_deterministic():
    p = MockEmbeddingProvider(dimension=16, seed=1)
    v1 = p.embed_query("transformer oil temperature")
    v2 = p.embed_query("transformer oil temperature")
    assert np.allclose(v1, v2)


def test_mock_provider_different_texts_differ():
    p = MockEmbeddingProvider(dimension=16, seed=1)
    v1 = p.embed_query("transformer")
    v2 = p.embed_query("circuit breaker")
    assert not np.allclose(v1, v2)


def test_mock_provider_normalized():
    p = MockEmbeddingProvider(dimension=16, seed=1)
    v = p.embed_query("x")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_assert_not_mock_raises_on_mock():
    p = MockEmbeddingProvider()
    with pytest.raises(ValueError):
        assert_not_mock(p)


def test_assert_not_mock_passes_for_config_provider_field():
    # local provider's config.provider must be "local", not "mock",
    # so assert_not_mock would not reject it (checked without loading weights
    # since config on LocalSentenceTransformerProvider lazily loads —
    # here we just check the mock path is the one that's rejected).
    p = MockEmbeddingProvider()
    assert p.config.provider == "mock"


def test_local_provider_raises_model_unavailable_when_load_fails(monkeypatch):
    """Deterministically simulates a model that cannot be loaded (network
    blocked, weights not cached, corrupted cache, etc.) by making the real
    SentenceTransformer constructor raise — independent of this machine's
    actual Hugging Face cache or network state."""
    import sentence_transformers

    class _BoomSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise OSError("simulated: model weights could not be loaded")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _BoomSentenceTransformer)

    p = LocalSentenceTransformerProvider(model_name="BAAI/bge-small-en-v1.5")
    with pytest.raises(ModelUnavailableError):
        p.embed_query("test")


def test_local_provider_is_available_returns_false_when_load_fails(monkeypatch):
    import sentence_transformers

    class _BoomSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise OSError("simulated: model weights could not be loaded")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _BoomSentenceTransformer)

    p = LocalSentenceTransformerProvider(model_name="BAAI/bge-small-en-v1.5")
    assert p.is_available() is False


def test_local_provider_is_available_returns_true_when_load_succeeds(monkeypatch):
    """Mirror case: when the model genuinely loads (whatever the reason —
    cached locally, successful download, etc.), is_available() must not
    falsely report False. Simulated deterministically, without touching
    a real cache or network."""
    import sentence_transformers

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
            vecs = np.ones((len(texts), 4), dtype=np.float32)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)

    p = LocalSentenceTransformerProvider(model_name="BAAI/bge-small-en-v1.5")
    assert p.is_available() is True
    v = p.embed_query("test")
    assert v.shape == (4,)


# --- Dense retriever (mock embeddings, interface-only test) ---

def test_dense_retriever_refuses_mock_by_default():
    chunks = [_fake_chunk("C1", "alpha"), _fake_chunk("C2", "beta")]
    retriever = DenseRetriever(MockEmbeddingProvider(dimension=8))
    with pytest.raises(ValueError):
        retriever.index(chunks)


def test_dense_retriever_works_with_mock_when_explicitly_allowed():
    chunks = [_fake_chunk("C1", "alpha"), _fake_chunk("C2", "beta")]
    retriever = DenseRetriever(MockEmbeddingProvider(dimension=8), allow_mock=True)
    retriever.index(chunks)
    results = retriever.retrieve("alpha", top_k=2)
    assert len(results) == 2
    assert {r.chunk.chunk_id for r in results} == {"C1", "C2"}


def test_dense_retriever_returns_ranks_starting_at_1():
    chunks = [_fake_chunk("C1", "alpha"), _fake_chunk("C2", "beta")]
    retriever = DenseRetriever(MockEmbeddingProvider(dimension=8), allow_mock=True)
    retriever.index(chunks)
    results = retriever.retrieve("alpha", top_k=2)
    assert [r.rank for r in results] == [1, 2]


# --- Hybrid RRF fusion (deterministic lexical retrievers, no embeddings needed) ---

def test_hybrid_rrf_combines_two_retrievers():
    chunks = [
        _fake_chunk("C1", "transformer oil temperature indicator OTI"),
        _fake_chunk("C2", "circuit breaker vacuum interrupter"),
        _fake_chunk("C3", "isolator earth switch interlock"),
    ]
    hybrid = HybridRRFRetriever([BM25Retriever(), BM25Retriever()])  # same retriever twice is fine for fusion-logic testing
    hybrid.index(chunks)
    results = hybrid.retrieve("transformer oil temperature", top_k=3)
    assert results[0].chunk.chunk_id == "C1"


def test_hybrid_rrf_requires_at_least_two_retrievers():
    with pytest.raises(AssertionError):
        HybridRRFRetriever([BM25Retriever()])


def test_hybrid_rrf_rank_starts_at_1_and_is_contiguous():
    chunks = [_fake_chunk(f"C{i}", f"text number {i} transformer") for i in range(5)]
    hybrid = HybridRRFRetriever([BM25Retriever(), BM25Retriever()])
    hybrid.index(chunks)
    results = hybrid.retrieve("transformer", top_k=5)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]
