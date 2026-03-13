import pytest
from unittest.mock import Mock, MagicMock
from src.retrieval.graph_expander import GraphExpander, GraphExpansionResult
from src.schemas.design_chunk_schemas import (
    DesignChunk, ChunkMetadata, CanonicalSource, ContentType, RetrievalResult
)


def create_mock_chunk(chunk_id: str, clause_id: str, page: int, references: list = None):
    """Helper to create mock chunks."""
    source = CanonicalSource(
        clause_id=clause_id,
        clause_title=f"Clause {clause_id}",
        page_number=page
    )
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        content_type=ContentType.DESIGN_RULE,
        canonical_source=source,
        references=references or []
    )
    return DesignChunk(
        id=chunk_id,
        text=f"Content for {chunk_id}",
        metadata=metadata
    )


def test_expand_fetches_referenced_chunks():
    """Test that expansion fetches referenced chunks."""
    # Setup
    expander = GraphExpander()
    
    # Primary chunk (Clause 6.1.2) references Table 6.1
    primary = create_mock_chunk(
        chunk_id="rule_6.1.2",
        clause_id="6.1.2",
        page=38,
        references=["table_6.1", "eq_6.1"]
    )
    
    # Mock vector store to return referenced chunks
    table_chunk = create_mock_chunk(
        chunk_id="table_6.1",
        clause_id="6.1.1",
        page=37,
        references=[]
    )
    eq_chunk = create_mock_chunk(
        chunk_id="eq_6.1",
        clause_id="6.1.2",
        page=38,
        references=[]
    )
    
    def mock_get_by_id(chunk_id):
        if chunk_id == "table_6.1":
            return table_chunk
        if chunk_id == "eq_6.1":
            return eq_chunk
        return None
    
    expander.get_chunk_by_id = mock_get_by_id
    
    # Execute
    results = expander.expand([primary], max_depth=1)
    
    # Verify - should have primary + 2 references
    assert len(results.results) == 3
    
    primary_result = [r for r in results.results if r.role == "primary"][0]
    assert primary_result.chunk.id == "rule_6.1.2"
    
    ref_results = [r for r in results.results if r.role == "reference"]
    assert len(ref_results) == 2
    ref_ids = [r.chunk.id for r in ref_results]
    assert "table_6.1" in ref_ids
    assert "eq_6.1" in ref_ids


def test_canonical_source_preserved_in_references():
    """KEY TEST: Verify canonical_source is NOT overwritten."""
    expander = GraphExpander()
    
    # Clause 6.1.2 references Table 6.1 (which is actually in 6.1.1)
    primary = create_mock_chunk(
        chunk_id="rule_6.1.2",
        clause_id="6.1.2",
        page=38,
        references=["table_6.1"]
    )
    
    # Table 6.1 is actually in clause 6.1.1, page 37
    table_chunk = create_mock_chunk(
        chunk_id="table_6.1",
        clause_id="6.1.1",  # Different from primary!
        page=37,  # Different from primary!
        references=[]
    )
    
    expander.get_chunk_by_id = lambda x: table_chunk if x == "table_6.1" else None
    
    results = expander.expand([primary], max_depth=1)
    
    # KEY ASSERTION: Table should cite 6.1.1, NOT 6.1.2
    ref_result = [r for r in results.results if r.role == "reference"][0]
    assert ref_result.chunk.metadata.canonical_source.clause_id == "6.1.1"
    assert ref_result.chunk.metadata.canonical_source.page_number == 37
    assert ref_result.referenced_from == "rule_6.1.2"


def test_deduplication_prevents_duplicates():
    """Test that same chunk is not returned twice."""
    expander = GraphExpander()
    
    # Two primary chunks reference the same table
    primary1 = create_mock_chunk("rule_6.1.2", "6.1.2", 38, ["table_6.1"])
    primary2 = create_mock_chunk("rule_6.2.1", "6.2.1", 42, ["table_6.1"])
    
    table_chunk = create_mock_chunk("table_6.1", "6.1.1", 37, [])
    
    expander.get_chunk_by_id = lambda x: table_chunk if x == "table_6.1" else None
    
    results = expander.expand([primary1, primary2], max_depth=1)
    
    # Should have 3 unique results, not 4
    assert len(results.results) == 3
    
    # Table should appear only once
    table_results = [r for r in results.results if r.chunk.id == "table_6.1"]
    assert len(table_results) == 1
