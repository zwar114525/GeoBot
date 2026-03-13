# tests/test_context_assembler.py
import pytest
from src.retrieval.context_assembler import ContextAssembler
from src.schemas.design_chunk_schemas import (
    DesignChunk, ChunkMetadata, CanonicalSource, ContentType, RetrievalResult
)


def create_result(chunk_id: str, clause_id: str, page: int, role: str, text: str):
    """Helper to create retrieval results."""
    source = CanonicalSource(
        clause_id=clause_id,
        clause_title=f"Clause {clause_id}",
        page_number=page
    )
    metadata = ChunkMetadata(
        chunk_id=chunk_id,
        content_type=ContentType.DESIGN_RULE,
        canonical_source=source
    )
    chunk = DesignChunk(id=chunk_id, text=text, metadata=metadata)
    return RetrievalResult(chunk=chunk, role=role, relevance_score=0.9)


def test_assemble_context_formats_sources():
    """Test that context is assembled with clear source attribution."""
    assembler = ContextAssembler()
    
    results = [
        create_result("rule_6.1.2", "6.1.2", 38, "primary", 
                     "Beams shall be designed such that M_u ≤ φ * M_n"),
        create_result("table_6.1", "6.1.1", 37, "reference",
                     "| Condition | φ value |\n|---|---|\n| Tension | 0.90 |"),
    ]
    
    context = assembler.assemble(results, "How to design a concrete beam?")
    
    # Should include query
    assert "How to design a concrete beam?" in context
    
    # Should include primary with proper tagging
    assert "[6.1.2]" in context
    assert "PRIMARY" in context
    
    # Should include reference with its TRUE source (6.1.1, not 6.1.2!)
    assert "[6.1.1]" in context
    assert "REFERENCED" in context


def test_citation_instructions_included():
    """Test that citation rules are included in context."""
    assembler = ContextAssembler()
    
    results = [create_result("rule_6.1.2", "6.1.2", 38, "primary", "Test content")]
    
    context = assembler.assemble(results, "test query")
    
    # Should include citation instructions
    assert "Citation Rules" in context or "CITATION" in context
