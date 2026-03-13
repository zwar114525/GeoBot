"""Tests for HybridRetriever - integration component for hybrid hierarchical graph retrieval."""
import pytest
from unittest.mock import Mock, MagicMock, patch
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_router import QueryRouter, QueryType
from src.schemas.design_chunk_schemas import (
    DesignChunk, ChunkMetadata, CanonicalSource, ContentType, RetrievalResult
)


def create_mock_chunk_dict(chunk_id: str, clause_id: str, page: int, references: list = None):
    """Helper to create mock chunk as dict (as returned by GeoVectorStore)."""
    return {
        "id": chunk_id,
        "text": f"Content for {chunk_id}",
        "metadata": {
            "chunk_id": chunk_id,
            "content_type": "design_rule",
            "clause_id": clause_id,
            "canonical_source": {
                "clause_id": clause_id,
                "clause_title": f"Clause {clause_id}",
                "page_number": page,
                "document_id": "FoundationCode2017"
            },
            "references": references or [],
            "referenced_by": []
        }
    }


def create_design_chunk(chunk_id: str, clause_id: str, page: int, references: list = None):
    """Helper to create DesignChunk object."""
    source = CanonicalSource(
        clause_id=clause_id,
        clause_title=f"Clause {clause_id}",
        page_number=page,
        document_id="FoundationCode2017"
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


class TestHybridRetrieverInit:
    """Test HybridRetriever initialization."""

    def test_init_with_vector_store(self):
        """Test that HybridRetriever initializes with vector store."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store)
        
        assert retriever.vector_store is mock_store
        assert retriever.router is not None
        assert retriever.expander is not None
        assert retriever.assembler is not None

    def test_init_sets_up_graph_expander(self):
        """Test that init connects expander to chunk lookup."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store)
        
        # The expander should have get_chunk_by_id set
        assert retriever.expander.get_chunk_by_id is not None


class TestHybridRetrieverRetrieve:
    """Test retrieve method and routing logic."""

    def test_retrieve_routes_design_queries(self):
        """Test that design queries use hierarchical search + graph expansion."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("rule_6.1.2", "6.1.2", 38, ["table_6.1"])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        # Mock the _get_chunk_by_id to return DesignChunk for graph expansion
        with patch.object(retriever, '_get_chunk_by_id') as mock_get_chunk:
            mock_get_chunk.return_value = create_design_chunk("table_6.1", "6.1.1", 37, [])
            
            results = retriever.retrieve("How to design a concrete beam?", top_k=5)
            
        # Should have results
        assert len(results) > 0
        # Router should classify as comprehensive_design
        assert retriever.router.route("How to design a concrete beam?").type == QueryType.COMPREHENSIVE_DESIGN.value

    def test_retrieve_routes_definition_queries(self):
        """Test that definition queries use semantic search without graph expansion."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("def_1", "6.1.1", 35, [])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever.retrieve("What is effective depth?", top_k=3)
        
        # Should get results from semantic search
        assert mock_store.search.called

    def test_retrieve_routes_data_lookup_queries(self):
        """Test that data lookup queries filter by content type."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("table_6.1", "6.1.1", 37, [])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever.retrieve("What are the φ values?", top_k=3)
        
        # Router should classify as data_lookup
        route = retriever.router.route("What are the φ values?")
        assert route.type == QueryType.DATA_LOOKUP.value

    def test_retrieve_respects_use_graph_expansion_flag(self):
        """Test that graph expansion can be disabled."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("rule_6.1.2", "6.1.2", 38, ["table_6.1"])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        # Without graph expansion
        results = retriever.retrieve(
            "How to design a concrete beam?",
            top_k=5,
            use_graph_expansion=False
        )
        
        # Should return primary results only (no references fetched)
        assert all(r.role == "primary" for r in results)

    def test_retrieve_returns_empty_for_no_results(self):
        """Test that empty results return empty list."""
        mock_store = Mock()
        mock_store.search.return_value = []
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever.retrieve("nonexistent query xyz", top_k=5)
        
        assert results == []


class TestHybridRetrieverSearchMethods:
    """Test the individual search methods."""

    def test_semantic_search_calls_vector_store(self):
        """Test _semantic_search delegates to vector store."""
        mock_store = Mock()
        mock_store.search.return_value = [create_mock_chunk_dict("chunk_1", "6.1", 10, [])]
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever._semantic_search("test query", top_k=5)
        
        mock_store.search.assert_called_once()
        assert len(results) == 1

    def test_search_with_filter_passes_content_type(self):
        """Test _search_with_filter passes content type to vector store."""
        mock_store = Mock()
        mock_store.search.return_value = [create_mock_chunk_dict("table_1", "6.1", 10, [])]
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever._search_with_filter("test query", top_k=5, content_type="table")
        
        # Verify content_type was passed
        call_kwargs = mock_store.search.call_args[1]
        assert "content_type" in call_kwargs or len(mock_store.search.call_args[0]) >= 4

    def test_hierarchical_search_uses_semantic_fallback(self):
        """Test _hierarchical_search falls back to semantic search."""
        mock_store = Mock()
        mock_store.search.return_value = [create_mock_chunk_dict("rule_1", "6.1", 10, [])]
        
        retriever = HybridRetriever(mock_store)
        
        results = retriever._hierarchical_search("test query", top_k=5)
        
        # Should call search (hierarchical uses semantic as fallback)
        assert mock_store.search.called


class TestHybridRetrieverChunkLookup:
    """Test _get_chunk_by_id method for graph expansion."""

    def test_get_chunk_by_id_finds_chunk(self):
        """Test that _get_chunk_by_id finds chunk by metadata chunk_id."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("table_6.1", "6.1.1", 37, [])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        result = retriever._get_chunk_by_id("table_6.1")
        
        assert result is not None
        # Returns a DesignChunk, not a dict
        assert result.id == "table_6.1"
        assert result.metadata.chunk_id == "table_6.1"

    def test_get_chunk_by_id_returns_none_for_missing(self):
        """Test that _get_chunk_by_id returns None for non-existent chunk."""
        mock_store = Mock()
        mock_store.search.return_value = [
            create_mock_chunk_dict("other_chunk", "6.1.1", 37, [])
        ]
        
        retriever = HybridRetriever(mock_store)
        
        result = retriever._get_chunk_by_id("nonexistent")
        
        # Should return None when chunk not found
        assert result is None


class TestHybridRetrieverContextAssembly:
    """Test assemble_context method."""

    def test_assemble_context_returns_string(self):
        """Test that assemble_context returns formatted string."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store)
        
        # Create test results
        chunk = create_design_chunk("rule_6.1.2", "6.1.2", 38)
        results = [RetrievalResult(chunk=chunk, role="primary", relevance_score=0.95)]
        
        context = retriever.assemble_context(results, "How to design a beam?")
        
        assert isinstance(context, str)
        assert "How to design a beam?" in context

    def test_assemble_context_includes_citation_rules(self):
        """Test that context includes citation instructions."""
        mock_store = Mock()
        retriever = HybridRetriever(mock_store)
        
        chunk = create_design_chunk("rule_6.1.2", "6.1.2", 38)
        results = [RetrievalResult(chunk=chunk, role="primary", relevance_score=0.95)]
        
        context = retriever.assemble_context(results, "test query")
        
        assert "Citation" in context or "citation" in context


class TestHybridRetrieverIntegration:
    """Integration tests for the full retrieval flow."""

    def test_full_design_query_flow(self):
        """Test complete flow for a design query."""
        mock_store = Mock()
        # Primary result with reference
        primary = create_mock_chunk_dict("rule_6.1.2", "6.1.2", 38, ["table_6.1"])
        mock_store.search.return_value = [primary]
        
        retriever = HybridRetriever(mock_store)
        
        # Mock the chunk lookup to return design chunks
        def mock_lookup(chunk_id):
            if chunk_id == "table_6.1":
                return create_design_chunk("table_6.1", "6.1.1", 37, [])
            return None
        
        with patch.object(retriever, '_get_chunk_by_id', side_effect=mock_lookup):
            results = retriever.retrieve(
                "How to design a concrete beam for flexure?",
                top_k=5,
                use_graph_expansion=True
            )
        
        # Should get results
        assert len(results) >= 1
        # Primary result should have role "primary"
        primary_results = [r for r in results if r.role == "primary"]
        assert len(primary_results) >= 1

    def test_query_routing_integration(self):
        """Test that query routing works correctly."""
        mock_store = Mock()
        mock_store.search.return_value = []
        
        retriever = HybridRetriever(mock_store)
        
        # Test different query types
        test_cases = [
            ("How to design a foundation?", QueryType.COMPREHENSIVE_DESIGN),
            ("What is bearing capacity?", QueryType.DEFINITION),
            ("What are the φ factors?", QueryType.DATA_LOOKUP),
            ("Explain soil mechanics", QueryType.GENERAL),
        ]
        
        for query, expected_type in test_cases:
            route = retriever.router.route(query)
            assert route.type == expected_type.value, f"Query '{query}' should route to {expected_type}"
