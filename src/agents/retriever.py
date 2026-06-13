"""
src/agents/retriever.py
=======================
RAG Retriever Agent
-------------------
Second node in the LangGraph graph. Responsible for:
  1. Reading ticker / doc_type_filter from state (set by fetcher_node)
  2. Running hybrid search (dense + sparse + RRF) via RAGPipeline
  3. Converting results to LangChain Documents and storing in state

Node signature (LangGraph):
    def retriever_node(state: AgentState) -> dict   ← partial state update
"""

from __future__ import annotations

import time

from langchain_core.documents import Document
from loguru import logger

from src.config import get_settings
from src.observability import trace_node
from src.rag_pipeline import RAGPipeline
from src.state import AgentState, WorkflowStatus

# Module-level singleton — avoid re-loading embedding models on every call
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ---------------------------------------------------------------------------
# Retriever Node
# ---------------------------------------------------------------------------

@trace_node("retriever")
def retriever_node(state: AgentState) -> dict:
    """
    LangGraph node: run hybrid RAG search → populate retrieved_docs.

    Reads from state:
        user_query, query_intent, ticker, doc_type_filter

    Writes to state:
        retrieved_docs, retrieval_metadata, status, node_timings
    """
    t0         = time.perf_counter()
    session_id = state.get("session_id", "unknown")

    # Build the search query — prefer the refined query_intent over raw query
    search_query = state.get("query_intent") or state["user_query"]
    tickers      = state.get("ticker", [])
    doc_filter   = state.get("doc_type_filter")

    logger.info(
        f"[retriever] START | session={session_id} | "
        f"query='{search_query[:60]}' | tickers={tickers} | doc_type={doc_filter}"
    )

    pipeline = _get_pipeline()
    settings = get_settings()
    top_n    = settings.qdrant_config["retrieval"]["rerank_top_n"]

    # Build Qdrant metadata filter
    # If multiple tickers, run one search per ticker and merge
    all_chunks = []

    if tickers:
        for ticker in tickers:
            filters = {"ticker": ticker}
            if doc_filter:
                filters["doc_type"] = doc_filter
            chunks = pipeline.retrieve(query=search_query, top_n=top_n, filters=filters)
            all_chunks.extend(chunks)
            logger.debug(f"[retriever] ticker={ticker} → {len(chunks)} chunks")
    else:
        # No ticker filter — search all documents
        filters = {"doc_type": doc_filter} if doc_filter else None
        all_chunks = pipeline.retrieve(query=search_query, top_n=top_n, filters=filters)

    # De-duplicate by chunk_id (same chunk can appear across ticker searches)
    seen_ids    = set()
    unique_chunks = []
    for chunk in all_chunks:
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            unique_chunks.append(chunk)

    # Re-sort by RRF score descending after merging
    unique_chunks.sort(key=lambda c: c.score, reverse=True)
    unique_chunks = unique_chunks[:top_n]

    # Convert to LangChain Documents (compatible with LLM context building)
    lc_docs: list[Document] = [chunk.to_langchain_doc() for chunk in unique_chunks]

    retrieval_metadata = {
        "chunks_retrieved":  len(lc_docs),
        "tickers_searched":  tickers,
        "doc_type_filter":   doc_filter,
        "top_rrf_score":     unique_chunks[0].score if unique_chunks else 0.0,
        "search_query_used": search_query,
    }

    elapsed = time.perf_counter() - t0
    logger.success(
        f"[retriever] DONE | {len(lc_docs)} unique chunks | "
        f"top_score={retrieval_metadata['top_rrf_score']:.4f} | {elapsed:.2f}s"
    )

    return {
        "retrieved_docs":     lc_docs,
        "retrieval_metadata": retrieval_metadata,
        "status":             WorkflowStatus.RETRIEVING,
        "node_timings":       {**state.get("node_timings", {}), "retriever": round(elapsed, 3)},
    }