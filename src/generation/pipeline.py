"""End-to-end RAG pipeline: retrieve -> context -> generate -> cite.

    question
       |
       v
  EquipmentAwareRetrieverV2(BM25Retriever)     <- docs/RETRIEVAL_BASELINE_V2.md
       |
       v  top-K RetrievedResult
  build_context()                              <- labelled evidence, sentinels dropped
       |
       v  EvidenceContext
  build_messages()                             <- evidence-only system prompt
       |
       v
  LLMClient.complete()
       |
       v  raw JSON reply
  build_answer()                               <- validated, labels checked
       |
       v
  GeneratedAnswer   (ANSWER | INSUFFICIENT_EVIDENCE | NEEDS_CLARIFICATION
                     | UNSUPPORTED | PARSE_ERROR)

The retriever default is the selected baseline; it is passed in rather than
hard-wired, so a retrieval change is a one-line substitution at the call site
and nothing here needs editing.

There is no confidence gate in this pipeline. Status is what the model
reported, plus the two structural downgrades in `answer.py`. The gate is a
later layer that will consume `GeneratedAnswer.signals`, with weights
calibrated against evaluation_v2 — not invented here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.citation.citations import render_reference_list
from src.common.chunk import Chunk
from src.generation.answer import AnswerStatus, GeneratedAnswer, GroundingSignals, build_answer
from src.generation.context import EvidenceContext, build_context
from src.generation.llm import LLMClient, LLMResponse, LLMUnavailableError
from src.generation.prompt import build_messages
from src.ingestion.kb_loader import load_chunks
from src.retrieval.equipment_aware_v2 import EquipmentAwareRetrieverV2
from src.retrieval.retrievers import BM25Retriever, RetrievedResult, Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KB_PATH = REPO_ROOT / "KB_v1.1" / "KB_v1.1_final" / "knowledge_chunks.xlsx"

# Retrieval top-K measures 0.909 Recall@3 and 0.909 Recall@5 on evaluation_v2:
# nothing between rank 4 and 5 is ever the sole gold hit, and rank 6-10 adds
# 0.023. Five is the smallest K that captures everything ranks 1-5 can, and it
# keeps the prompt short enough that the model cannot hide a weak answer in a
# wall of marginally-related extracts.
DEFAULT_TOP_K = 5


@dataclass
class PipelineResult:
    question: str
    answer: GeneratedAnswer
    retrieved: List[RetrievedResult]
    context: EvidenceContext
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    llm_provider: str = ""
    llm_model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str = ""

    @property
    def status(self) -> AnswerStatus:
        return self.answer.status

    def rendered(self) -> str:
        """Human-readable form: answer, then the reference list."""
        a = self.answer
        if a.status is AnswerStatus.ANSWER:
            body = a.answer_text
            refs = render_reference_list(a.citations)
            return f"{body}\n\nSources:\n{refs}" if refs else body
        if a.status is AnswerStatus.NEEDS_CLARIFICATION:
            q = a.clarification_question or "Could you clarify the question?"
            return f"Clarification needed: {q}"
        if a.status is AnswerStatus.INSUFFICIENT_EVIDENCE:
            detail = f" Missing: {a.missing_information}" if a.missing_information else ""
            return (
                "The knowledge base does not contain sufficient evidence to answer this."
                + detail
            )
        if a.status is AnswerStatus.UNSUPPORTED:
            return (
                "An answer was produced but could not be tied to retrieved evidence, "
                "so it is withheld. " + a.downgrade_reason
            )
        return f"The answer could not be processed ({a.status.value}). {a.parse_error}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.answer.to_dict(),
            "retrieved_chunk_ids": [r.chunk.chunk_id for r in self.retrieved],
            "retrieved_document_ids": [r.chunk.document_id for r in self.retrieved],
            "retrieved_scores": [round(r.score, 4) for r in self.retrieved],
            "timing_ms": {
                "retrieval": round(self.retrieval_ms, 2),
                "generation": round(self.generation_ms, 2),
                "total": round(self.total_ms, 2),
            },
            "llm": {
                "provider": self.llm_provider,
                "model": self.llm_model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            },
            "error": self.error,
        }


class RAGPipeline:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        llm: LLMClient,
        retriever: Retriever | None = None,
        top_k: int = DEFAULT_TOP_K,
        max_context_items: int | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.llm = llm
        self.retriever = retriever or EquipmentAwareRetrieverV2(BM25Retriever())
        self.top_k = top_k
        self.max_context_items = max_context_items
        self._indexed = False

    def index(self) -> "RAGPipeline":
        # BM25Okapi divides by the corpus size, so indexing an empty chunk
        # list raises ZeroDivisionError deep inside rank_bm25. An empty KB is
        # a configuration problem, not a crash: retrieval simply returns
        # nothing and the pipeline abstains.
        if self.chunks:
            self.retriever.index(self.chunks)
        self._indexed = True
        return self

    def _ensure_indexed(self) -> None:
        if not self._indexed:
            self.index()

    def retrieve(self, question: str, top_k: int | None = None) -> List[RetrievedResult]:
        self._ensure_indexed()
        if not self.chunks:
            return []
        return self.retriever.retrieve(question, top_k=top_k or self.top_k)

    def answer(self, question: str, top_k: int | None = None) -> PipelineResult:
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        retrieved = self.retrieve(question, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context = build_context(retrieved, max_items=self.max_context_items)

        # Nothing retrieved: abstain without spending a request. Calling the
        # model with an empty evidence block only invites it to answer from
        # its own knowledge, which is exactly what must not happen.
        if context.is_empty():
            empty = GeneratedAnswer(
                question=question,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                answer_text="",
                claims=[],
                citations=[],
                clarification_question="",
                missing_information="retrieval returned no candidate evidence for this question",
                conflict_present=False,
                conflict_description="",
                conflict_citations=[],
                signals=GroundingSignals(),
            )
            total = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                question=question,
                answer=empty,
                retrieved=[],
                context=context,
                retrieval_ms=retrieval_ms,
                total_ms=total,
                llm_provider=self.llm.provider,
                llm_model=self.llm.model,
            )

        messages = build_messages(question, context)

        t1 = time.perf_counter()
        try:
            response: LLMResponse = self.llm.complete(messages)
        except LLMUnavailableError as e:
            generation_ms = (time.perf_counter() - t1) * 1000
            failed = build_answer(question, "", context)
            failed.parse_error = f"LLM unavailable: {e}"
            total = (time.perf_counter() - t_start) * 1000
            return PipelineResult(
                question=question,
                answer=failed,
                retrieved=retrieved,
                context=context,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total,
                llm_provider=self.llm.provider,
                llm_model=self.llm.model,
                error=f"LLM unavailable: {e}",
            )
        generation_ms = (time.perf_counter() - t1) * 1000

        answer = build_answer(question, response.text, context)
        total = (time.perf_counter() - t_start) * 1000

        return PipelineResult(
            question=question,
            answer=answer,
            retrieved=retrieved,
            context=context,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total,
            llm_provider=response.provider,
            llm_model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )


def load_kb(kb_path: str | Path = DEFAULT_KB_PATH) -> List[Chunk]:
    chunks, report = load_chunks(str(kb_path))
    assert report.ok(), f"KB loader did not pass validation: {report}"
    return chunks


def build_default_pipeline(
    llm: LLMClient,
    kb_path: str | Path = DEFAULT_KB_PATH,
    top_k: int = DEFAULT_TOP_K,
) -> RAGPipeline:
    """The selected baseline, indexed and ready."""
    return RAGPipeline(load_kb(kb_path), llm=llm, top_k=top_k).index()
