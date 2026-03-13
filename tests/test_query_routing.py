"""
Unit tests for Query Router.
"""
import pytest
from src.retrieval.query_router import QueryRouter, QueryType, RetrievalStrategy


@pytest.fixture
def router():
    return QueryRouter()


def test_comprehensive_design_query(router):
    """Test design query classification."""
    result = router.route("How to design a concrete beam?")
    assert result.type == QueryType.COMPREHENSIVE_DESIGN.value
    assert result.strategy == RetrievalStrategy.HIERARCHICAL_GRAPH.value
    assert "beam" in result.target_elements


def test_definition_query(router):
    """Test definition query classification."""
    result = router.route("What is effective depth?")
    assert result.type == QueryType.DEFINITION.value
    assert result.strategy == RetrievalStrategy.ATOMIC_LOOKUP.value


def test_data_lookup_query(router):
    """Test data lookup query classification."""
    result = router.route("What is the φ value for tension-controlled?")
    assert result.type == QueryType.DATA_LOOKUP.value
    assert result.strategy == RetrievalStrategy.TABLE_FIRST.value


def test_table_lookup_query(router):
    """Test table lookup query."""
    result = router.route("Show me Table 6.1")
    assert result.type == QueryType.DATA_LOOKUP.value
    assert result.strategy == RetrievalStrategy.TABLE_FIRST.value


def test_extract_beam_element(router):
    """Test beam element extraction."""
    result = router.route("How to design a concrete beam for flexure?")
    assert "beam" in result.target_elements


def test_extract_multiple_elements(router):
    """Test multiple element extraction."""
    result = router.route("How to design a pile foundation?")
    assert "pile" in result.target_elements
    assert "foundation" in result.target_elements


def test_extract_shear_element(router):
    """Test shear element extraction."""
    result = router.route("Calculate shear reinforcement")
    assert "shear" in result.target_elements


def test_chinese_design_query(router):
    """Test Chinese design query classification."""
    result = router.route("如何設計鋼筋混凝土梁?")
    assert result.type == QueryType.COMPREHENSIVE_DESIGN.value


def test_chinese_definition_query(router):
    """Test Chinese definition query classification."""
    result = router.route("什麼是有效深度?")
    assert result.type == QueryType.DEFINITION.value


def test_extract_actions(router):
    """Test action extraction."""
    result = router.route("How to calculate the required reinforcement?")
    assert "calculate" in result.target_actions


def test_general_query(router):
    """Test general query classification."""
    result = router.route("Tell me about concrete")
    assert result.type == QueryType.GENERAL.value
    assert result.strategy == RetrievalStrategy.SEMANTIC_SEARCH.value
