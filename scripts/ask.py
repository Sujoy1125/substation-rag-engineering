"""Ask the RAG pipeline a single question from the command line.

    python scripts/ask.py "How often should transformer oil BDV be tested?"
    python scripts/ask.py --evidence "SF6 gas pressure alarm setting"   # no LLM call
    python scripts/ask.py --json "..." > answer.json

Requires OPENAI_API_KEY (or LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY) unless
--evidence is used. See .env.example.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.generation.context import build_context
from src.generation.llm import LLMUnavailableError, client_from_env
from src.generation.pipeline import DEFAULT_TOP_K, RAGPipeline, load_kb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--evidence", action="store_true", help="show retrieved evidence only; no LLM call")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args()

    chunks = load_kb()  # client_from_env() loads .env when the LLM is needed

    if args.evidence:
        from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2
        from src.retrieval.retrievers import BM25Retriever

        retriever = EquipmentAwareRetrieverV2(BM25Retriever())
        retriever.index(chunks)
        ctx = build_context(retriever.retrieve(args.question, top_k=args.top_k))
        print(ctx.text)
        return 0

    try:
        client = client_from_env()
    except LLMUnavailableError as e:
        print(f"LLM not configured: {e}", file=sys.stderr)
        print("Run with --evidence to inspect retrieval without an API key.", file=sys.stderr)
        return 2

    pipeline = RAGPipeline(chunks, llm=client, top_k=args.top_k).index()
    result = pipeline.answer(args.question)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.rendered())
        print(
            f"\n[status={result.status.value} "
            f"coverage={result.answer.signals.citation_coverage:.2f} "
            f"{result.total_ms:.0f}ms]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
