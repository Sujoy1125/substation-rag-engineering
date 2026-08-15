
"""Experiment B: evaluation_v2 (44 answerable, D01-D08) comparison,
baseline EquipmentAwareRetriever vs EquipmentAwareRetrieverV2.
"""
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.kb_loader import load_chunks
from src.retrieval.gold_questions import GoldQuestion
from src.retrieval.retrievers import BM25Retriever
from src.retrieval.equipment_aware import EquipmentAwareRetriever, extract_equipment, _chunk_mentions_equipment
from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2, extract_equipment_v2, _chunk_mentions_equipment_v2
from src.retrieval.document_diversity import DocumentDiversityReranker
from src.retrieval.benchmark import run_benchmark, print_result, is_hit

# KB = Path(__file__).resolve().parents[1] / "KB_v1.1_extracted" / "KB_v1.1_final"
KB = Path(__file__).resolve().parents[1] / "KB_v1.1" / "KB_v1.1_final"
EVAL = Path(__file__).resolve().parents[1] / "evaluation_v2"


def load_eval_v2_answerable(xlsx_path: str):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    out = []
    for row in rows:
        if row is None or row[0] is None:
            continue
        qid, question, ans, expdoc, docid, page, section, diff, avail, chunkid, evidence = row
        out.append(GoldQuestion(
            question_id=str(qid).strip(),
            question=str(question).strip(),
            expected_answer=str(ans).strip(),
            expected_document_ref=str(expdoc).strip(),
            expected_document_id=str(docid).strip(),
            expected_page=str(page).strip(),
            expected_section=str(section).strip(),
            difficulty=str(diff).strip(),
        ))
    return out


chunks, load_report = load_chunks(str(KB / "knowledge_chunks.xlsx"))
assert load_report.ok(), f"KB loader did not pass validation: {load_report}"
gold = load_eval_v2_answerable(str(EVAL / "answerable.xlsx"))
gold_by_id = {g.question_id: g for g in gold}
chunk_by_id = {c.chunk_id: c for c in chunks}

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

# Root-cause verification for every recovered (miss->hit) or improved question:
# was the change actually driven by Bug A (hyphen) or Bug B (alias), per the
# extraction/matching functions themselves -- not just "rank got better".
root_cause = {}
for qid in miss_to_hit + improved:
    g = gold_by_id[qid]
    q = g.question
    eq_v1 = extract_equipment(q)
    eq_v2 = extract_equipment_v2(q)
    bug_a = eq_v1 != eq_v2  # query-side extraction itself changed (hyphen normalization mattered)
    # Did the specific chunk that now hits benefit from alias-aware chunk matching (Bug B)?
    hit_cid = v2_by_id[qid].retrieved_chunk_ids[v2_by_id[qid].hit_rank - 1]
    chunk = chunk_by_id.get(hit_cid)
    bug_b = False
    if chunk is not None:
        matched_v1 = _chunk_mentions_equipment(chunk, eq_v2)
        matched_v2 = _chunk_mentions_equipment_v2(chunk, eq_v2)
        bug_b = matched_v2 and not matched_v1
    if bug_a and bug_b:
        cause = "Bug A + Bug B"
    elif bug_a:
        cause = "Bug A (hyphen normalization)"
    elif bug_b:
        cause = "Bug B (alias-aware chunk matching)"
    else:
        cause = "UNATTRIBUTED - not explained by Bug A or Bug B, investigate separately"
    root_cause[qid] = {
        "cause": cause,
        "query_equipment_v1": sorted(eq_v1),
        "query_equipment_v2": sorted(eq_v2),
        "hit_chunk_id": hit_cid,
        "hit_chunk_equipment_field": chunk.equipment if chunk else None,
    }

print("\n--- evaluation_v2 regression summary ---")
print("hit -> miss:", hit_to_miss)
print("miss -> hit:", miss_to_hit)
print("rank improved:", improved)
print("rank worsened:", worsened)
print("\n--- root cause attribution ---")
for qid, info in root_cause.items():
    print(qid, info["cause"], "| chunk", info["hit_chunk_id"], "equip field:", info["hit_chunk_equipment_field"])

exp_dir = Path(__file__).resolve().parent
out_path = exp_dir / f"eval_v2_regression_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
payload = {
    "experiment": "evaluation_v2_equipment_aware_v2_comparison",
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
            "baseline_hit@1": base_by_id[qid].hit_rank == 1,
            "v2_hit@1": v2_by_id[qid].hit_rank == 1,
            "baseline_hit@3": bool(base_by_id[qid].hit_rank and base_by_id[qid].hit_rank <= 3),
            "v2_hit@3": bool(v2_by_id[qid].hit_rank and v2_by_id[qid].hit_rank <= 3),
            "baseline_hit@5": bool(base_by_id[qid].hit_rank and base_by_id[qid].hit_rank <= 5),
            "v2_hit@5": bool(v2_by_id[qid].hit_rank and v2_by_id[qid].hit_rank <= 5),
            "baseline_hit@10": bool(base_by_id[qid].hit_rank and base_by_id[qid].hit_rank <= 10),
            "v2_hit@10": bool(v2_by_id[qid].hit_rank and v2_by_id[qid].hit_rank <= 10),
        }
        for qid in base_by_id
    ],
    "hit_to_miss": hit_to_miss,
    "miss_to_hit": miss_to_hit,
    "rank_improved": improved,
    "rank_worsened": worsened,
    "unchanged": unchanged,
    "root_cause_attribution": root_cause,
}
out_path.write_text(json.dumps(payload, indent=2))
print(f"\nSaved: {out_path}")
