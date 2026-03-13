# tests/test_query_router.py
import pytest
from src.retrieval.query_router import QueryRouter, QueryType


def test_detect_comprehensive_design_query():
    """Test detection of 'how to design' queries."""
    router = QueryRouter()
    
    # Comprehensive design queries
    assert router.route("How to design a concrete beam").type == QueryType.COMPREHENSIVE_DESIGN
    assert router.route("design procedure for foundations").type == QueryType.COMPREHENSIVE_DESIGN
    assert router.route("steps for calculating bearing capacity").type == QueryType.COMPREHENSIVE_DESIGN
    assert router.route("What is the design process for piles?").type == QueryType.COMPREHENSIVE_DESIGN


def test_detect_definition_query():
    """Test detection of definition queries."""
    router = QueryRouter()
    
    assert router.route("What is effective depth?").type == QueryType.DEFINITION
    assert router.route("Define bearing capacity").type == QueryType.DEFINITION
    assert router.route("meaning of consolidation").type == QueryType.DEFINITION


def test_detect_data_lookup_query():
    """Test detection of table/value lookup queries."""
    router = QueryRouter()
    
    assert router.route("What are the φ values?").type == QueryType.DATA_LOOKUP
    assert router.route("table of soil properties").type == QueryType.DATA_LOOKUP
    assert router.route("coefficient for settlement calculation").type == QueryType.DATA_LOOKUP


def test_extract_elements():
    """Test extraction of engineering elements from query."""
    router = QueryRouter()
    
    elements = router.extract_elements("How to design a concrete beam for flexure")
    assert "beam" in elements or "concrete" in elements
    
    elements = router.extract_elements("pile foundation design")
    assert "pile" in elements or "foundation" in elements


def test_route_returns_full_route_info():
    """Test that route returns full routing decision."""
    router = QueryRouter()
    
    result = router.route("How to design a retaining wall")
    
    assert result.type == QueryType.COMPREHENSIVE_DESIGN
    assert result.strategy == "hierarchical_graph"
    assert "wall" in result.target_elements or "retaining" in result.target_elements
