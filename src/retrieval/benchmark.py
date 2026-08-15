"""Reproducible retrieval benchmark against the frozen 20 answerable
questions in rag_test_55.xlsx.

A retrieval "hit" for a gold question = a returned chunk whose
document_id matches the gold Document ID AND whose PDF page overlaps the
gold expected page (page strings on both sides may be single pages or
ranges, e.g. "PDF p. 26-30" vs "p.26-27" — overlap, not exact string
equality, is checked).
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.chunk import Chunk
from src.embeddings.provider import LocalSentenceTransformerProvider, ModelUnavailableError
from src.ingestion.kb_loader import load_chunks
from src.retrieval.document_diversity import DocumentDiversityReranker
from src.retrieval.equipment_aware import EquipmentAwareRetriever
from src.retrieval.gold_questions import GoldQuestion, load_gold_answerable
from src.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRRFRetriever,
    Retriever,
    RetrievedResult,
    TfidfRetriever,
)

_NUM_RE = re.compile(r"\d+")


def page_numbers(page_str: str) -> set[int]:
    """Extract the set of integer page numbers referenced by a page field,
    handling both single pages ('p.18', 'PDF p. 9') and ranges
    ('p.26-27', 'PDF p. 26-30')."""
    nums = [int(n) for n in _NUM_RE.findall(page_str)]
    if len(nums) >= 2:
        return set(range(min(nums), max(nums) + 1))
    return set(nums)


def is_hit(chunk: Chunk, gold: GoldQuestion) -> bool:
    if chunk.document_id != gold.expected_document_id:
        return False
    return bool(page_numbers(chunk.pdf_page) & page_numbers(gold.expected_page))


@dataclass
class QuestionResult:
    question_id: str
    question: str
    hit_rank: int | None  # 1-indexed rank of first hit, None if no hit in top_k
    retrieved_chunk_ids: List[str]
    retrieved_scores: List[float]
    expected_document_id: str
    expected_page: str


@dataclass
class BenchmarkResult:
    retriever_name: str
    top_k: int
    n_questions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_latency_ms: float
    per_question: List[QuestionResult]
    failures: List[str]  # question_ids with no hit in top_k


def _recall_at(per_question: List[QuestionResult], k: int) -> float:
    n = len(per_question)
    hits = sum(1 for q in per_question if q.hit_rank is not None and q.hit_rank <= k)
    return hits / n if n else 0.0


def run_benchmark(
    retriever: Retriever,
    chunks: List[Chunk],
    gold_questions: List[GoldQuestion],
    top_k: int = 10,
) -> BenchmarkResult:
    retriever.index(chunks)

    per_question: List[QuestionResult] = []
    latencies_ms: List[float] = []

    for gold in gold_questions:
        t0 = time.perf_counter()
        results: List[RetrievedResult] = retriever.retrieve(gold.question, top_k=top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        hit_rank = None
        for r in results:
            if is_hit(r.chunk, gold):
                hit_rank = r.rank
                break

        per_question.append(
            QuestionResult(
                question_id=gold.question_id,
                question=gold.question,
                hit_rank=hit_rank,
                retrieved_chunk_ids=[r.chunk.chunk_id for r in results],
                retrieved_scores=[round(r.score, 4) for r in results],
                expected_document_id=gold.expected_document_id,
                expected_page=gold.expected_page,
            )
        )

    mrr = sum((1.0 / q.hit_rank) if q.hit_rank else 0.0 for q in per_question) / len(per_question)
    failures = [q.question_id for q in per_question if q.hit_rank is None]

    return BenchmarkResult(
        retriever_name=retriever.name,
        top_k=top_k,
        n_questions=len(gold_questions),
        recall_at_1=_recall_at(per_question, 1),
        recall_at_3=_recall_at(per_question, 3),
        recall_at_5=_recall_at(per_question, 5),
        recall_at_10=_recall_at(per_question, 10),
        mrr=mrr,
        mean_latency_ms=sum(latencies_ms) / len(latencies_ms),
        per_question=per_question,
        failures=failures,
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"\n=== {result.retriever_name} ===")
    print(f"Recall@1:  {result.recall_at_1:.2f}")
    print(f"Recall@3:  {result.recall_at_3:.2f}")
    print(f"Recall@5:  {result.recall_at_5:.2f}")
    print(f"Recall@10: {result.recall_at_10:.2f}")
    print(f"MRR:       {result.mrr:.3f}")
    print(f"Mean latency: {result.mean_latency_ms:.2f} ms")
    print(f"Failures ({len(result.failures)}): {result.failures}")


def print_failure_analysis(result: BenchmarkResult, gold_by_id: dict[str, GoldQuestion]) -> None:
    """Recall@5 failure detail: gold vs top retrieved chunks, for manual
    classification (lexical mismatch / semantic mismatch / wrong doc / etc).
    """
    fails5 = [q for q in result.per_question if q.hit_rank is None or q.hit_rank > 5]
    if not fails5:
        print(f"\n{result.retriever_name}: no Recall@5 failures.")
        return
    print(f"\n--- {result.retriever_name}: Recall@5 failure detail ({len(fails5)}) ---")
    for q in fails5:
        gold = gold_by_id[q.question_id]
        top5 = q.retrieved_chunk_ids[:5]
        print(f"{q.question_id}: {q.question}")
        print(f"  expected: {gold.expected_document_id} / {gold.expected_page}")
        print(f"  top5 retrieved: {top5}")
        print(f"  hit_rank: {q.hit_rank}")


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    chunks, load_report = load_chunks(str(base / "KB_v1" / "knowledge_chunks.xlsx"))
    assert load_report.ok(), f"KB loader did not pass validation: {load_report}"

    gold = load_gold_answerable(str(base / "KB_v1" / "rag_test_55.xlsx"))
    gold_by_id = {g.question_id: g for g in gold}

    all_results: list[BenchmarkResult] = []
    blocked_note = None

    bm25 = BM25Retriever()
    bm25_result = run_benchmark(bm25, chunks, gold, top_k=10)
    all_results.append(bm25_result)
    print_result(bm25_result)

    tfidf = TfidfRetriever()
    tfidf_result = run_benchmark(tfidf, chunks, gold, top_k=10)
    all_results.append(tfidf_result)
    print_result(tfidf_result)

    bm25_equipment_aware = EquipmentAwareRetriever(BM25Retriever())
    bm25_equipment_aware_result = run_benchmark(bm25_equipment_aware, chunks, gold, top_k=10)
    all_results.append(bm25_equipment_aware_result)
    print_result(bm25_equipment_aware_result)

    diversity = DocumentDiversityReranker(
        EquipmentAwareRetriever(BM25Retriever()),
        cap=2,
        pool_k=30,
    )
    diversity_result = run_benchmark(diversity, chunks, gold, top_k=10)
    all_results.append(diversity_result)
    print_result(diversity_result)

    # --- Dense: attempt real model load; never fabricate on failure ---
    dense_result = None
    hybrid_result = None
    try:
        provider = LocalSentenceTransformerProvider(model_name="BAAI/bge-small-en-v1.5")
        dense = DenseRetriever(provider)
        dense_result = run_benchmark(dense, chunks, gold, top_k=10)
        all_results.append(dense_result)
        print_result(dense_result)

        hybrid = HybridRRFRetriever([BM25Retriever(), DenseRetriever(provider)])
        hybrid_result = run_benchmark(hybrid, chunks, gold, top_k=10)
        all_results.append(hybrid_result)
        print_result(hybrid_result)
        print_failure_analysis(hybrid_result, gold_by_id)
    except ModelUnavailableError as e:
        blocked_note = str(e)
        print("\n=== dense ===\nBLOCKED — " + blocked_note)
        print("\n=== hybrid_rrf ===\nBLOCKED — depends on dense, not run.")

    exp_dir = base / "experiments"
    exp_dir.mkdir(exist_ok=True)
    out_path = exp_dir / f"retrieval_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    payload = {
        "experiment": "retrieval_benchmark_bm25_tfidf_dense_hybrid",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kb_chunks_loaded": len(chunks),
        "gold_questions": len(gold),
        "dense_hybrid_blocked": blocked_note is not None,
        "blocked_reason": blocked_note,
        "results": [
            {
                **{k: v for k, v in asdict(r).items() if k != "per_question"},
                "per_question": [asdict(q) for q in r.per_question],
            }
            for r in all_results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()