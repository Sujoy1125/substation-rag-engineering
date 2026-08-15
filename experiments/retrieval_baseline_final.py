"""Retrieval baseline decision experiment — the reproducible artifact behind
docs/RETRIEVAL_BASELINE_V2.md.

Purpose: measure every candidate retrieval configuration on BOTH evaluation
sets in one run, so the selected baseline rests on a single reproducible
command rather than on numbers transcribed between documents.

Configurations compared (all lexical; dense/hybrid remain unmeasured, see
docs/BLOCKERS.md):

    1. bm25                              BM25 only
    2. bm25_equipment_aware              BM25 -> EquipmentAwareRetriever
    3. bm25_equipment_aware_v2           BM25 -> EquipmentAwareRetrieverV2
    4. bm25_equipment_aware_diversity    (2) -> DocumentDiversityReranker(cap=2, pool_k=30)
    5. bm25_equipment_aware_v2_diversity (3) -> DocumentDiversityReranker(cap=2, pool_k=30)

Datasets:

    D09            20 answerable questions, single document
                   KB_v1.1/KB_v1.1_final/rag_test_55.xlsx, sheet 20_Answerable
    evaluation_v2  44 answerable questions, D01-D08
                   evaluation_v2/answerable.xlsx

Why both: D09 is single-document and was the original benchmark; the
diversity reranker was accepted on D09 alone. evaluation_v2 spans eight
documents and is the harder, more representative set. A configuration that
wins on D09 but loses on evaluation_v2 must not be selected on the strength
of D09 (handoff RULE 7).

Reads only. Modifies no KB, no evaluation workbook, and no retrieval module.

Run from the repository root:

    python experiments/retrieval_baseline_final.py

Writes experiments/retrieval_baseline_final_<UTC timestamp>.json
(experiments/*.json is gitignored by design; the committed artifact is the
script plus docs/RETRIEVAL_BASELINE_V2.md).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.eval_loader import load_gold_for_retrieval
from src.ingestion.kb_loader import load_chunks
from src.retrieval.benchmark import BenchmarkResult, run_benchmark
from src.retrieval.document_diversity import DocumentDiversityReranker
from src.retrieval.equipment_aware import EquipmentAwareRetriever
from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2
from src.retrieval.gold_questions import load_gold_answerable
from src.retrieval.retrievers import BM25Retriever

KB_DIR = REPO_ROOT / "KB_v1.1" / "KB_v1.1_final"
EVAL_DIR = REPO_ROOT / "evaluation_v2"

TOP_K = 10
DIVERSITY_CAP = 2
DIVERSITY_POOL_K = 30

# The configuration this experiment exists to justify. Asserted at the end
# so the script fails loudly if a future change silently invalidates the
# selection rather than quietly reporting different numbers.
SELECTED = "bm25_equipment_aware_v2"

# Rank a configuration by the dataset that spans eight documents, not the
# single-document one. MRR breaks ties on Recall@3.
PRIMARY_DATASET = "evaluation_v2"


def build_configs():
    """Fresh retriever objects per call: run_benchmark() calls index(),
    so reusing one instance across datasets would carry state over."""
    return [
        ("bm25", BM25Retriever()),
        (
            "bm25_equipment_aware",
            EquipmentAwareRetriever(BM25Retriever(), name="bm25_equipment_aware"),
        ),
        (
            "bm25_equipment_aware_v2",
            EquipmentAwareRetrieverV2(BM25Retriever(), name="bm25_equipment_aware_v2"),
        ),
        (
            "bm25_equipment_aware_diversity",
            DocumentDiversityReranker(
                EquipmentAwareRetriever(BM25Retriever()),
                cap=DIVERSITY_CAP,
                pool_k=DIVERSITY_POOL_K,
                name="bm25_equipment_aware_diversity",
            ),
        ),
        (
            "bm25_equipment_aware_v2_diversity",
            DocumentDiversityReranker(
                EquipmentAwareRetrieverV2(BM25Retriever()),
                cap=DIVERSITY_CAP,
                pool_k=DIVERSITY_POOL_K,
                name="bm25_equipment_aware_v2_diversity",
            ),
        ),
    ]


def print_table(dataset_name: str, results: dict[str, BenchmarkResult]) -> None:
    first = next(iter(results.values()))
    print(f"\n=== {dataset_name} ({first.n_questions} answerable questions) ===")
    print(f"{'configuration':<38} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'R@10':>6} {'MRR':>7} {'ms':>8}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<38} {r.recall_at_1:>6.2f} {r.recall_at_3:>6.2f} "
            f"{r.recall_at_5:>6.2f} {r.recall_at_10:>6.2f} {r.mrr:>7.3f} "
            f"{r.mean_latency_ms:>8.1f}"
        )


def main() -> int:
    chunks, load_report = load_chunks(str(KB_DIR / "knowledge_chunks.xlsx"))
    assert load_report.ok(), f"KB loader did not pass validation: {load_report}"

    datasets = {
        "D09": load_gold_answerable(str(KB_DIR / "rag_test_55.xlsx")),
        "evaluation_v2": load_gold_for_retrieval(str(EVAL_DIR / "answerable.xlsx")),
    }

    print(f"KB: {KB_DIR.relative_to(REPO_ROOT)}  ({len(chunks)} chunks)")
    for name, gold in datasets.items():
        print(f"dataset {name}: {len(gold)} answerable questions")

    all_results: dict[str, dict[str, BenchmarkResult]] = {}
    for dataset_name, gold in datasets.items():
        results: dict[str, BenchmarkResult] = {}
        for config_name, retriever in build_configs():
            results[config_name] = run_benchmark(retriever, chunks, gold, top_k=TOP_K)
        all_results[dataset_name] = results
        print_table(dataset_name, results)

    # --- selection check -------------------------------------------------
    primary = all_results[PRIMARY_DATASET]
    ranked = sorted(
        primary.items(),
        key=lambda kv: (kv[1].recall_at_3, kv[1].mrr),
        reverse=True,
    )
    best_name, best = ranked[0]
    selected = primary[SELECTED]

    print(f"\n--- selection check (ranked on {PRIMARY_DATASET}: R@3, then MRR) ---")
    for i, (name, r) in enumerate(ranked, start=1):
        marker = "  <-- SELECTED" if name == SELECTED else ""
        print(f"{i}. {name:<38} R@3={r.recall_at_3:.2f} MRR={r.mrr:.3f}{marker}")

    selection_holds = (best.recall_at_3, round(best.mrr, 6)) == (
        selected.recall_at_3,
        round(selected.mrr, 6),
    )
    print(
        f"\nSelected configuration: {SELECTED}\n"
        f"Top-ranked on {PRIMARY_DATASET}: {best_name}\n"
        f"Selection holds: {selection_holds}"
    )

    # Diversity is not in the default pipeline. Record, from this run, on
    # which datasets that call is supported rather than asserting it.
    diversity_verdict = {}
    for dataset_name, results in all_results.items():
        with_div = results["bm25_equipment_aware_v2_diversity"]
        without_div = results["bm25_equipment_aware_v2"]
        diversity_verdict[dataset_name] = {
            "recall_at_3_delta": round(with_div.recall_at_3 - without_div.recall_at_3, 4),
            "recall_at_5_delta": round(with_div.recall_at_5 - without_div.recall_at_5, 4),
            "mrr_delta": round(with_div.mrr - without_div.mrr, 4),
        }
    print("\n--- diversity reranker effect (v2 + diversity minus v2) ---")
    for dataset_name, d in diversity_verdict.items():
        print(
            f"{dataset_name:<16} dR@3={d['recall_at_3_delta']:+.2f} "
            f"dR@5={d['recall_at_5_delta']:+.2f} dMRR={d['mrr_delta']:+.3f}"
        )

    out_path = (
        REPO_ROOT
        / "experiments"
        / f"retrieval_baseline_final_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    payload = {
        "experiment": "retrieval_baseline_final",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "kb": str(KB_DIR.relative_to(REPO_ROOT)),
        "kb_chunks_loaded": len(chunks),
        "top_k": TOP_K,
        "diversity_cap": DIVERSITY_CAP,
        "diversity_pool_k": DIVERSITY_POOL_K,
        "selected_configuration": SELECTED,
        "primary_dataset": PRIMARY_DATASET,
        "selection_holds": selection_holds,
        "top_ranked_on_primary": best_name,
        "diversity_effect": diversity_verdict,
        "datasets": {
            dataset_name: {
                "n_questions": len(datasets[dataset_name]),
                "results": {
                    config_name: {
                        **{k: v for k, v in asdict(r).items() if k != "per_question"},
                        "per_question": [asdict(q) for q in r.per_question],
                    }
                    for config_name, r in results.items()
                },
            }
            for dataset_name, results in all_results.items()
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {out_path.relative_to(REPO_ROOT)}")

    if not selection_holds:
        print(
            f"\nWARNING: '{SELECTED}' is no longer top-ranked on {PRIMARY_DATASET} "
            f"('{best_name}' is). Re-open the baseline decision and update "
            f"docs/RETRIEVAL_BASELINE_V2.md before changing the default pipeline."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
