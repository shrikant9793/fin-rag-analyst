"""
tests/test_rag_pipeline.py
==========================
Unit + integration tests for RAGPipeline (Day 1).

Run:
    pytest tests/test_rag_pipeline.py -v

Integration tests (require running Qdrant) are marked @pytest.mark.integration
and are skipped by default in CI unless QDRANT_URL is set.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag_pipeline import RAGPipeline, RetrievedChunk

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a small sample financial document for testing."""
    content = """
    Apple Inc. (AAPL) — Q4 FY2024 Earnings Summary

    Revenue: $94.9 billion, up 6% year over year
    Net income: $14.7 billion
    Earnings per share: $0.97 diluted

    Products revenue: $69.6 billion (+4% YoY)
    Services revenue: $25.3 billion (+12% YoY)

    Gross margin: 46.2% overall
    Products gross margin: 36.3%
    Services gross margin: 74.0%

    CEO Tim Cook: "We are thrilled to report another record Services revenue..."
    CFO Luca Maestri: "Our strong cash generation allowed us to return $29B to shareholders."

    Geographic Revenue:
    Americas: $40.3B
    Europe: $24.3B
    Greater China: $15.0B
    Japan: $6.7B
    Rest of Asia Pacific: $8.6B
    """
    file = tmp_path / "AAPL_Q4_2024_earnings.txt"
    file.write_text(content)
    return file


@pytest.fixture
def mock_pipeline() -> RAGPipeline:
    """RAGPipeline with mocked Qdrant client and embedding models."""
    pipeline = RAGPipeline.__new__(RAGPipeline)

    # Minimal settings mock
    settings_mock = MagicMock()
    settings_mock.qdrant_config = {
        "url": "http://localhost:6333",
        "collection_name": "test_fin_docs",
        "vector_size": 1024,
        "distance": "Cosine",
        "retrieval": {"top_k": 10, "rerank_top_n": 5, "rrf_k": 60, "score_threshold": 0.35},
    }
    settings_mock.ingestion_config = {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "supported_extensions": [".pdf", ".txt", ".docx"],
        "metadata_fields": ["ticker", "doc_type", "filing_date", "company_name", "source_file"],
    }
    settings_mock.embedding_dense  = {"model": "BAAI/bge-m3", "device": "cpu", "batch_size": 32, "normalize": True}
    settings_mock.embedding_sparse = {"model": "Qdrant/bm42-all-minilm-l6-v2-attentions"}

    pipeline.settings   = settings_mock
    pipeline.qdrant_cfg = settings_mock.qdrant_config
    pipeline.ing_cfg    = settings_mock.ingestion_config
    pipeline.emb_dense  = settings_mock.embedding_dense
    pipeline.emb_sparse = settings_mock.embedding_sparse

    pipeline._client       = MagicMock()
    pipeline._dense_model  = MagicMock()
    pipeline._sparse_model = MagicMock()

    return pipeline


# ---------------------------------------------------------------------------
# Unit Tests — no Qdrant required
# ---------------------------------------------------------------------------

class TestRRFFusion:
    """Test Reciprocal Rank Fusion logic in isolation."""

    def test_rrf_both_lists_same_doc(self) -> None:
        """A doc appearing in both lists should score higher than doc in one."""
        dense_results = [MagicMock(id="doc_a"), MagicMock(id="doc_b")]
        sparse_results = [MagicMock(id="doc_a"), MagicMock(id="doc_c")]

        fused = RAGPipeline._rrf_fuse(dense_results, sparse_results, k=60)
        ids   = [item[0] for item in fused]

        assert ids[0] == "doc_a", "doc_a should rank first (appears in both lists)"

    def test_rrf_scores_descending(self) -> None:
        """RRF scores must be sorted descending."""
        dense  = [MagicMock(id=f"d_{i}") for i in range(5)]
        sparse = [MagicMock(id=f"s_{i}") for i in range(5)]
        fused  = RAGPipeline._rrf_fuse(dense, sparse, k=60)

        scores = [item[1] for item in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_empty_lists(self) -> None:
        """Empty inputs should return empty list."""
        assert RAGPipeline._rrf_fuse([], []) == []

    def test_rrf_single_list(self) -> None:
        """One empty list — result equals the other list in rank order."""
        dense  = [MagicMock(id="doc_x"), MagicMock(id="doc_y")]
        fused  = RAGPipeline._rrf_fuse(dense, [], k=60)
        assert fused[0][0] == "doc_x"
        assert fused[1][0] == "doc_y"


class TestChunking:
    """Test text chunking produces sensible output."""

    def test_chunk_splits_long_text(self, mock_pipeline: RAGPipeline) -> None:
        long_text = "This is a sentence about financial markets. " * 100
        chunks    = mock_pipeline._chunk_text(long_text)
        assert len(chunks) > 1, "Long text should produce multiple chunks"

    def test_chunk_short_text_single_chunk(self, mock_pipeline: RAGPipeline) -> None:
        short_text = "Apple revenue was $94.9 billion."
        chunks     = mock_pipeline._chunk_text(short_text)
        assert len(chunks) >= 1

    def test_chunk_size_respected(self, mock_pipeline: RAGPipeline) -> None:
        text   = "word " * 1000
        chunks = mock_pipeline._chunk_text(text)
        for chunk in chunks:
            assert len(chunk) <= mock_pipeline.ing_cfg["chunk_size"] * 2


class TestFileLoading:
    """Test file loading for supported formats."""

    def test_load_txt_file(self, mock_pipeline: RAGPipeline, sample_txt_file: Path) -> None:
        text = mock_pipeline._load_file(sample_txt_file)
        assert "Apple" in text
        assert "94.9 billion" in text

    def test_load_unsupported_raises(self, mock_pipeline: RAGPipeline, tmp_path: Path) -> None:
        bad_file = tmp_path / "data.csv"
        bad_file.write_text("col1,col2\n1,2")
        with pytest.raises(ValueError, match="Unsupported file type"):
            mock_pipeline._load_file(bad_file)

    def test_load_nonexistent_file(self, mock_pipeline: RAGPipeline) -> None:
        with pytest.raises(Exception):
            mock_pipeline._load_file(Path("/nonexistent/file.txt"))


class TestRetrievedChunk:
    """Test RetrievedChunk dataclass helpers."""

    def test_to_langchain_doc(self) -> None:
        chunk = RetrievedChunk(
            content="Apple Q4 revenue was $94.9B",
            score=0.87,
            metadata={"ticker": "AAPL", "doc_type": "earnings_transcript"},
            chunk_id="abc123",
        )
        doc = chunk.to_langchain_doc()
        assert doc.page_content == chunk.content
        assert doc.metadata["ticker"] == "AAPL"
        assert doc.metadata["score"] == 0.87


# ---------------------------------------------------------------------------
# Integration Tests — require live Qdrant
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("QDRANT_URL"),
    reason="Integration test — requires QDRANT_URL env var and live Qdrant instance",
)
class TestRAGIntegration:
    """Full integration tests against a running Qdrant instance."""

    @pytest.fixture(autouse=True)
    def pipeline(self) -> RAGPipeline:
        p = RAGPipeline()
        # Use an isolated test collection
        p.qdrant_cfg = {**p.qdrant_cfg, "collection_name": "test_integration_col"}
        p.ensure_collection(recreate=True)
        yield p
        # Cleanup
        p.delete_collection()

    def test_ingest_and_retrieve(self, pipeline: RAGPipeline, sample_txt_file: Path) -> None:
        total = pipeline.ingest_documents(
            file_paths=[sample_txt_file],
            doc_metadata={
                "ticker": "AAPL",
                "doc_type": "earnings_transcript",
                "filing_date": "2024-11-01",
                "company_name": "Apple Inc.",
            },
        )
        assert total > 0, "Should ingest at least one chunk"

        results = pipeline.retrieve("What is Apple's gross margin?", top_n=3)
        assert len(results) > 0, "Should retrieve at least one result"
        assert any("margin" in r.content.lower() for r in results)

    def test_retrieve_with_ticker_filter(self, pipeline: RAGPipeline, sample_txt_file: Path) -> None:
        pipeline.ingest_documents(
            file_paths=[sample_txt_file],
            doc_metadata={"ticker": "AAPL", "doc_type": "sec_filing", "filing_date": "2024-11-01"},
        )
        results = pipeline.retrieve("revenue", filters={"ticker": "AAPL"})
        assert all(r.metadata.get("ticker") == "AAPL" for r in results)

    def test_collection_stats(self, pipeline: RAGPipeline, sample_txt_file: Path) -> None:
        pipeline.ingest_documents(
            file_paths=[sample_txt_file],
            doc_metadata={"ticker": "TEST", "doc_type": "sec_filing", "filing_date": "2024-01-01"},
        )
        stats = pipeline.collection_stats()
        assert stats["points_count"] > 0
        assert stats["status"] == "green"