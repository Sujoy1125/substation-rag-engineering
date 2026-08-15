"""Experiment A: D09 regression check, baseline vs EquipmentAwareRetrieverV2.

Does not modify retrievers.py, equipment_aware.py, document_diversity.py,
benchmark.py, gold_questions.py, or either frozen KB. Reads KB_v1.1 only.
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.kb_loader import load_chunks
from src.retrieval.gold_questions import load_gold_answerable
from src.retrieval.retrievers import BM25Retriever
from src.retrieval.equipment_aware import EquipmentAwareRetriever
from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2
from src.retrieval.document_diversity import DocumentDiversityReranker
from src.retrieval.benchmark import run_benchmark, print_result

KB = Path(__file__).resolve().parents[1] / "KB_v1.1_extracted" / "KB_v1.1_final"

chunks, load_report = load_chunks(str(KB / "knowledge_chunks.xlsx"))
assert load_report.ok(), f"KB loader did not pass validation: {load_report}"
gold = load_gold_answerable(str(KB / "rag_test_55.xlsx"))
gold_by_id = {g.question_id: g for g in gold}

baseline_pipeline = DocumentDiversityReranker(
    EquipmentAwareRetriever(BM25Retriever()), cap=2, pool_k=30,
    name="bm25_equipment_aware_diversity_cap2",
)
v2_pipeline = DocumentDiversityReranker(
    EquipmentAwareRetrieverV2(BM25Retriever()), cap=2, pool_k=30,
    name="bm25_equipment_aware_v2_diversity_cap2",
)

baseline_result = run_benchmark(baseline_pipeline, chunks, gold, top_k=10)
v2_result = run_benchmark(v2_pipeline, chunks, gold, top_k=10)
print_result(baseline_result)
print_result(v2_result)

base_by_id = {q.question_id: q for q in baseline_result.per_question}
v2_by_id = {q.question_id: q for q in v2_result.per_question}

hit_to_miss, miss_to_hit, improved, worsened, unchanged = [], [], [], [], []
for qid in base_by_id:
    b, v = base_by_id[qid].hit_rank, v2_by_id[qid].hit_rank
    if b is not None and v is None:
        hit_to_miss.append(qid)
    elif b is None and v is not None:
        miss_to_hit.append(qid)
    elif b is not None and v is not None and v < b:
        improved.append(qid)
    elif b is not None and v is not None and v > b:
        worsened.append(qid)
    else:
        unchanged.append(qid)

print("\n--- D09 regression summary ---")
print("hit -> miss:", hit_to_miss)
print("miss -> hit:", miss_to_hit)
print("rank improved:", improved)
print("rank worsened:", worsened)

exp_dir = Path(__file__).resolve().parent
out_path = exp_dir / f"d09_regression_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
payload = {
    "experiment": "d09_regression_equipment_aware_v2",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "kb_chunks_loaded": len(chunks),
    "gold_questions": len(gold),
    "baseline": {k: v for k, v in asdict(baseline_result).items() if k != "per_question"},
    "v2": {k: v for k, v in asdict(v2_result).items() if k != "per_question"},
    "per_question_comparison": [
        {
            "question_id": qid,
            "baseline_rank": base_by_id[qid].hit_rank,
            "v2_rank": v2_by_id[qid].hit_rank,
            "baseline_top10": base_by_id[qid].retrieved_chunk_ids,
            "v2_top10": v2_by_id[qid].retrieved_chunk_ids,
        }
        for qid in base_by_id
    ],
    "hit_to_miss": hit_to_miss,
    "miss_to_hit": miss_to_hit,
    "rank_improved": improved,
    "rank_worsened": worsened,
    "unchanged": unchanged,
}
out_path.write_text(json.dumps(payload, indent=2))
print(f"\nSaved: {out_path}")
