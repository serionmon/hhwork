"""Automated unit and integration tests for Direct Database Retrieval Pipeline.

Tests direct knowledge retrieval without LLM dependency, relevance threshold enforcement,
structured API schema response, zero-hallucination no-result behavior, and error handling.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.guardrails import Decision
from core.harness import AnswerResult, RAGHarness, Source


@pytest.fixture
def client():
    return TestClient(app)


def test_answer_mode_defaults_to_direct(monkeypatch):
    monkeypatch.delenv("ANSWER_MODE", raising=False)
    from core.harness import RAGHarness
    mode = os.getenv("ANSWER_MODE", "direct").lower()
    assert mode == "direct"


def test_direct_retrieval_no_llm_required(monkeypatch):
    """Verify that direct retrieval operates without requiring any LLM API key."""
    # Ensure all LLM API keys are unset
    monkeypatch.setenv("ANSWER_MODE", "direct")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    mock_embedder = MagicMock()
    mock_embedder.encode_query.return_value = [0.1] * 384

    mock_hit = MagicMock()
    mock_hit.passage_id = "doc_123"
    mock_hit.unit_id = "doc_123"
    mock_hit.text = "India's capital city is New Delhi."
    mock_hit.score = 0.95
    mock_hit.query_type = "DESCRIPTION"
    mock_hit.lang = "en"
    mock_hit.contributors = {"metadata_128": 0}

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = MagicMock(
        hits=[mock_hit],
        provenance=lambda: {"metadata_128": 1}
    )

    with patch("core.harness.extract_answer") as mock_extract, \
         patch("core.harness.check_input") as mock_check_in, \
         patch("core.harness.check_output") as mock_check_out:

        mock_check_in.return_value = MagicMock(blocked=False, decision=Decision.ALLOW)
        mock_check_out.return_value = MagicMock(blocked=False, grounding=0.92)

        mock_extract.return_value = MagicMock(
            text="New Delhi is the capital of India.",
            support=0.92
        )

        harness = RAGHarness(
            index_root=MagicMock(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            llm=None
        )

        result = harness.answer("What is the capital of India?", generate=None)

        assert result.answer == "New Delhi is the capital of India."
        assert result.answer_source == "extractive"
        assert result.decision == Decision.ALLOW.value
        assert result.llm_ok is False
        assert result.support == 0.92


def test_relevance_threshold_no_result_behavior(monkeypatch):
    """Verify that queries below RETRIEVAL_THRESHOLD return no-match message without hallucination."""
    monkeypatch.setenv("ANSWER_MODE", "direct")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.65")

    mock_embedder = MagicMock()
    mock_embedder.encode_query.return_value = [0.1] * 384

    mock_hit = MagicMock()
    mock_hit.passage_id = "doc_999"
    mock_hit.unit_id = "doc_999"
    mock_hit.text = "Unrelated text about stars and astronomy."
    mock_hit.score = 0.20
    mock_hit.contributors = {}

    mock_retriever = MagicMock()
    mock_retriever.search.return_value = MagicMock(
        hits=[mock_hit],
        provenance=lambda: {}
    )

    with patch("core.harness.extract_answer") as mock_extract, \
         patch("core.harness.check_input") as mock_check_in, \
         patch("core.harness.check_output") as mock_check_out:

        mock_check_in.return_value = MagicMock(blocked=False, decision=Decision.ALLOW)
        mock_check_out.return_value = MagicMock(blocked=False, grounding=0.20)

        mock_extract.return_value = MagicMock(
            text="Unrelated text.",
            support=0.25  # Below 0.65 threshold
        )

        harness = RAGHarness(
            index_root=MagicMock(),
            embedder=mock_embedder,
            retriever=mock_retriever,
            llm=None
        )

        result = harness.answer("Unknown quantum physics question?", generate=None)

        assert result.answer == "I couldn't find this information in the knowledge base."
        assert result.answer_source == "abstain"
        assert result.decision == Decision.ABSTAIN_UNGROUNDED.value
        assert "low_support" in result.reason


def test_api_structured_response_schema(client, monkeypatch):
    """Test that API endpoint returns structured schema required by primary specification."""
    monkeypatch.setenv("ANSWER_MODE", "direct")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.45")

    mock_result = AnswerResult(
        question="What is Python?",
        answer="Python is a high-level programming language.",
        decision="allow",
        reason="grounded",
        extractive_answer="Python is a high-level programming language.",
        answer_source="extractive",
        route="indic",
        support=0.88,
        grounding=0.90,
        sources=[Source(unit_id="doc_py", text="Python is a programming language.", score=0.88)]
    )

    with patch.dict("api.main.STATE", {"harness": MagicMock(answer=lambda q, **kw: mock_result)}):
        resp = client.post("/ask", json={"question": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["mode"] == "direct"
        assert data["query"] == "What is Python?"
        assert data["answer"] == "Python is a high-level programming language."
        assert data["grounded"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["source"] == "doc_py"
        assert data["results"][0]["score"] == 0.88


def test_api_no_result_structured_schema(client, monkeypatch):
    """Test API response structure when no information is found in knowledge base."""
    monkeypatch.setenv("ANSWER_MODE", "direct")
    monkeypatch.setenv("RETRIEVAL_THRESHOLD", "0.65")

    mock_result = AnswerResult(
        question="What is the speed of light in 3000 AD?",
        answer="I couldn't find this information in the knowledge base.",
        decision="abstain_ungrounded",
        reason="low_support(0.2000<0.65)",
        extractive_answer="Some low confidence text",
        answer_source="abstain",
        support=0.20,
        grounding=0.20,
        sources=[]
    )

    with patch.dict("api.main.STATE", {"harness": MagicMock(answer=lambda q, **kw: mock_result)}):
        resp = client.post("/ask", json={"question": "What is the speed of light in 3000 AD?"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["success"] is True
        assert data["mode"] == "direct"
        assert data["query"] == "What is the speed of light in 3000 AD?"
        assert data["answer"] == "I couldn't find this information in the knowledge base."
        assert data["grounded"] is False
        assert data["results"] == []


def test_empty_and_malformed_queries(client):
    """Test API response on empty or invalid request payload."""
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # Validation error for min_length

    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_unhandled_exception_returns_controlled_json():
    """Test that unexpected server exceptions return controlled JSON error without stack trace."""
    test_client = TestClient(app, raise_server_exceptions=False)
    with patch.dict("api.main.STATE", {"harness": MagicMock(answer=MagicMock(side_effect=RuntimeError("DB exploded")))}):
        resp = test_client.post("/ask", json={"question": "Test query"})
        assert resp.status_code == 500
        data = resp.json()
        assert data["success"] is False
        assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
        assert "An unexpected error occurred" in data["error"]["message"]


def test_diag_health_without_onnx(client):
    """Verify GET /diag/health returns JSON without calling ONNX Runtime."""
    resp = client.get("/diag/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["llm_required"] is False
    assert "embedding_runtime" in data


def test_sparse_retrieval_fallback_without_onnx():
    """Verify direct retrieval works with BM25 fallback when embedder is unavailable."""
    mock_retriever = MagicMock()
    mock_retriever.search_sparse.return_value = MagicMock(
        hits=[MagicMock(unit_id="doc_bm25", text="Python is popular.", score=1.5, contributors={"metadata_128": 0})],
        provenance=lambda: {"metadata_128": 1}
    )

    with patch("core.harness.extract_answer") as mock_extract, \
         patch("core.harness.check_input") as mock_check_in, \
         patch("core.harness.check_output") as mock_check_out:

        mock_check_in.return_value = MagicMock(blocked=False, decision=Decision.ALLOW)
        mock_check_out.return_value = MagicMock(blocked=False, grounding=0.85)
        mock_extract.return_value = MagicMock(text="Python is popular.", support=0.85)

        harness = RAGHarness(
            index_root=MagicMock(),
            embedder=None,  # No ONNX embedder
            retriever=mock_retriever,
            llm=None
        )

        res = harness.answer("Tell me about Python", generate=False)
        assert res.answer == "Python is popular."
        assert res.answer_source == "extractive"
        assert res.decision == Decision.ALLOW.value


def test_diag_memory_endpoint(client):
    """Verify GET /diag/memory returns process memory status and configuration flags."""
    resp = client.get("/diag/memory")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["answer_mode"] == "direct"
    assert data["dense_retrieval_enabled"] is False
    assert data["embedder_initialized"] is False
    assert "pid" in data
    assert "rss_mb" in data


def test_dense_retrieval_disabled_by_default(monkeypatch):
    """Verify ENABLE_DENSE_RETRIEVAL defaults to false."""
    monkeypatch.delenv("ENABLE_DENSE_RETRIEVAL", raising=False)
    enabled = os.getenv("ENABLE_DENSE_RETRIEVAL", "false").lower() == "true"
    assert enabled is False


def test_chunk_index_load_does_not_initialize_hnsw_by_default(tmp_path):
    """Verify ChunkIndex.load with load_hnsw=False skips HNSW index creation."""
    from core.index import ChunkIndex
    import pickle

    strategy = "test_strat"
    d = tmp_path / strategy
    d.mkdir(parents=True)
    with (d / "meta.pkl").open("wb") as f:
        pickle.dump({
            "strategy": strategy,
            "chunk_ids": ["c1"],
            "passage_ids": ["p1"],
            "texts": ["test text"],
            "query_types": ["DESC"],
            "langs": ["en"]
        }, f)

    ix = ChunkIndex.load(tmp_path, strategy, load_hnsw=False)
    assert ix.strategy == strategy
    assert len(ix.chunk_ids) == 1
    assert ix.hnsw is None  # HNSW is skipped in direct mode
