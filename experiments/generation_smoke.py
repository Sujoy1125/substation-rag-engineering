"""Generation smoke test: retrieve -> context -> prompt -> LLM -> citations,
on a small subset of evaluation_v2 answerable questions.

STEP 5 of the engineering plan. This is a plumbing check, not an evaluation.
It reports what the pipeline produced; it does not score answer correctness,
and no number it prints should be quoted as a generation metric. Full scoring
over all 44 answerable questions comes later, with the unanswerable and
ambiguous sets alongside it.

Two modes:

    --dry-run   (default)  No LLM call. Builds the real evidence context and
                           the real prompt from the frozen KB and prints them.
                           Verifies everything up to the model boundary with
                           no API key and no cost.

    --live                 Calls the configured provider (LLM_PROVIDER, default
                           openai). Requires a key. Writes a JSON record of
                           every question: status, citations, grounding
                           signals, timings, tokens.

Usage:

    python experiments/generation_smoke.py                    # dry run, 5 questions
    python experiments/generation_smoke.py -n 3 --show-prompt
    python experiments/generation_smoke.py --live -n 5
    python experiments/generation_smoke.py --live --ids V2-001 V2-011
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.eval_loader import load_answerable
from src.generation.context import build_context
from src.generation.llm import LLMUnavailableError, assert_real_client, client_from_env
from src.generation.pipeline import DEFAULT_TOP_K, RAGPipeline, load_kb
from src.generation.prompt import build_user_prompt

EVAL_DIR = REPO_ROOT / "evaluation_v2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--num", type=int, default=5, help="how many questions (default 5)")
    p.add_argument("--ids", nargs="*", default=None, help="specific question IDs, e.g. V2-001")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--live", action="store_true", help="call the configured LLM")
    p.add_argument("--dry-run", action="store_true", help="explicit; this is the default")
    p.add_argument("--show-prompt", action="store_true", help="print the full prompt in dry-run mode")
    return p.parse_args()


def select_questions(args):
    questions = load_answerable(str(EVAL_DIR / "answerable.xlsx"))
    if args.ids:
        wanted = {i.strip().upper() for i in args.ids}
        picked = [q for q in questions if q.question_id.upper() in wanted]
        missing = wanted - {q.question_id.upper() for q in picked}
        if missing:
            print(f"WARNING: unknown question ids ignored: {sorted(missing)}")
        return picked
    return questions[: args.num]


def run_dry(questions, chunks, top_k, show_prompt):
    from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2
    from src.retrieval.retrievers import BM25Retriever

    retriever = EquipmentAwareRetrieverV2(BM25Retriever())
    retriever.index(chunks)

    print("MODE: dry run — no LLM call, no API key required.\n")
    for q in questions:
        results = retriever.retrieve(q.question, top_k=top_k)
        ctx = build_context(results)
        gold_ids = set(q.expected_chunk_ids)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        gold_rank = next(
            (i for i, cid in enumerate(retrieved_ids, start=1) if cid in gold_ids), None
        )

        print("=" * 78)
        print(f"{q.question_id}  [{q.gold.difficulty}]  expects {q.gold.expected_document_id} {q.gold.expected_page}")
        print(f"Q: {q.question}")
        print(f"gold chunk id(s): {sorted(gold_ids) or '(none recorded)'}")
        print(f"retrieved ({len(results)}): {retrieved_ids}")
        print(f"gold chunk rank in context: {gold_rank if gold_rank else 'NOT IN TOP-K'}")
        print(f"documents in context: {ctx.document_ids()}")
        print(f"evidence block: {len(ctx.text)} chars")
        if show_prompt:
            print("-" * 78)
            print(build_user_prompt(q.question, ctx))
        print()

    print(
        "Dry run complete. This confirms retrieval, context assembly and prompt\n"
        "construction against the frozen KB. It says nothing about answer quality."
    )


def run_live(questions, chunks, top_k):
    try:
        client = client_from_env()
    except LLMUnavailableError as e:
        print(f"Cannot run live: {e}")
        print("Set OPENAI_API_KEY (see .env.example), or run without --live for a dry run.")
        return 2
    assert_real_client(client)

    print(f"MODE: live — provider={client.provider} model={client.model}\n")
    pipeline = RAGPipeline(chunks, llm=client, top_k=top_k).index()

    records = []
    for q in questions:
        result = pipeline.answer(q.question)
        a = result.answer
        gold_ids = set(q.expected_chunk_ids)
        cited_ids = {c.chunk_id for c in a.citations}

        print("=" * 78)
        print(f"{q.question_id}  ->  {a.status.value}")
        print(f"Q: {q.question}")
        print(f"expected (gold): {q.gold.expected_answer[:160]}")
        print("-" * 78)
        print(result.rendered())
        print("-" * 78)
        print(
            f"claims={a.signals.n_claims} supported={a.signals.n_supported_claims} "
            f"coverage={a.signals.citation_coverage:.2f} invalid_labels={a.signals.invalid_labels} "
            f"cited_gold_chunk={bool(gold_ids & cited_ids)} "
            f"total={result.total_ms:.0f}ms"
        )
        if result.error:
            print(f"ERROR: {result.error}")
        print()

        records.append(
            {
                "question_id": q.question_id,
                "difficulty": q.gold.difficulty,
                "expected_document_id": q.gold.expected_document_id,
                "expected_page": q.gold.expected_page,
                "expected_chunk_ids": sorted(gold_ids),
                "gold_chunk_cited": bool(gold_ids & cited_ids),
                **result.to_dict(),
            }
        )

    out = (
        REPO_ROOT
        / "experiments"
        / f"generation_smoke_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.write_text(
        json.dumps(
            {
                "experiment": "generation_smoke",
                "note": (
                    "Plumbing check on a subset. NOT an evaluation — answer correctness "
                    "is not scored here and these counts must not be reported as metrics."
                ),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "provider": client.provider,
                "model": client.model,
                "top_k": top_k,
                "n_questions": len(questions),
                "records": records,
            },
            indent=2,
        )
    )
    print(f"Saved: {out.relative_to(REPO_ROOT)}")

    statuses: dict[str, int] = {}
    for r in records:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    print(f"\nStatus counts over {len(records)} questions (descriptive only): {statuses}")
    invalid = sum(r["signals"]["n_invalid_labels"] for r in records)
    print(f"Invalid citation labels across the run: {invalid}")
    return 0


def main() -> int:
    args = parse_args()
    questions = select_questions(args)
    if not questions:
        print("No questions selected.")
        return 1

    chunks = load_kb()
    print(f"KB: {len(chunks)} chunks | questions: {len(questions)} | top_k={args.top_k}\n")

    if args.live:
        return run_live(questions, chunks, args.top_k)
    run_dry(questions, chunks, args.top_k, args.show_prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
