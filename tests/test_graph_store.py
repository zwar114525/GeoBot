"""
Unit tests for GraphStore.
"""
import pytest
import os
import tempfile
from src.retrieval.graph_store import GraphStore


@pytest.fixture
def temp_graph_store():
    """Create temporary GraphStore for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    store = GraphStore(storage_path=temp_path)
    yield store
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_add_edge(temp_graph_store):
    """Test adding a single edge."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    assert "chunk_1" in temp_graph_store.edges
    assert "chunk_2" in temp_graph_store.edges["chunk_1"]


def test_add_duplicate_edge(temp_graph_store):
    """Test adding duplicate edge doesn't create duplicates."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    assert len(temp_graph_store.edges["chunk_1"]) == 1


def test_get_references(temp_graph_store):
    """Test getting outgoing references."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_1", "chunk_3")
    refs = temp_graph_store.get_references("chunk_1")
    assert len(refs) == 2
    assert "chunk_2" in refs
    assert "chunk_3" in refs


def test_get_referenced_by(temp_graph_store):
    """Test getting incoming references."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_3", "chunk_2")
    refs = temp_graph_store.get_referenced_by("chunk_2")
    assert len(refs) == 2
    assert "chunk_1" in refs
    assert "chunk_3" in refs


def test_has_references(temp_graph_store):
    """Test checking if chunk has references."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    assert temp_graph_store.has_references("chunk_1") is True
    assert temp_graph_store.has_references("nonexistent") is False


def test_get_stats(temp_graph_store):
    """Test getting graph statistics."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_1", "chunk_3")
    stats = temp_graph_store.get_stats()
    assert stats["total_nodes"] == 1
    assert stats["total_edges"] == 2


def test_clear(temp_graph_store):
    """Test clearing all edges."""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.clear()
    stats = temp_graph_store.get_stats()
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0


def test_add_edges_from_chunk(temp_graph_store):
    """Test batch adding edges from chunk references."""
    chunk = {
        "chunk_id": "beam_flexure",
        "references": ["table_6.1", "eq_6.1", "clause_6.2"]
    }
    temp_graph_store.rebuild_from_chunks([chunk])
    refs = temp_graph_store.get_references("beam_flexure")
    assert len(refs) == 3
