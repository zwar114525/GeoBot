# tests/test_design_chunk_schemas.py
import pytest
from src.schemas.design_chunk_schemas import (
    CanonicalSource,
    ChunkMetadata,
    DesignChunk,
    ContentType,
    RetrievalResult,
)
from pydantic import ValidationError


def test_canonical_source_creation():
    """Test CanonicalSource with required fields."""
    source = CanonicalSource(
        clause_id="6.1.2",
        clause_title="Flexural Design of Beams",
        page_number=38,
        document_id="FoundationCode2017"
    )
    assert source.clause_id == "6.1.2"
    assert source.page_number == 38


def test_chunk_metadata_with_references():
    """Test ChunkMetadata with graph references."""
    source = CanonicalSource(
        clause_id="6.1.2",
        clause_title="Flexural Design",
        page_number=38
    )
    metadata = ChunkMetadata(
        chunk_id="rule_6.1.2_v1",
        content_type=ContentType.DESIGN_RULE,
        canonical_source=source,
        references=["table_6.1", "eq_6.1", "clause_6.2.1"],
        referenced_by=["clause_6.1.3"]
    )
    assert "table_6.1" in metadata.references
    assert metadata.content_type == ContentType.DESIGN_RULE


def test_design_chunk_complete():
    """Test complete DesignChunk with all fields."""
    source = CanonicalSource(
        clause_id="6.1.2",
        clause_title="Flexural Design",
        page_number=38
    )
    chunk = DesignChunk(
        id="rule_6.1.2_v1",
        text="Beams shall be designed such that M_u ≤ φ * M_n",
        metadata=ChunkMetadata(
            chunk_id="rule_6.1.2_v1",
            content_type=ContentType.DESIGN_RULE,
            canonical_source=source,
            regulatory_strength="mandatory"
        )
    )
    assert chunk.id == "rule_6.1.2_v1"
    assert chunk.metadata.canonical_source.clause_id == "6.1.2"


def test_retrieval_result_with_role():
    """Test RetrievalResult distinguishes primary vs reference."""
    source = CanonicalSource(
        clause_id="6.1.2",
        clause_title="Flexural Design",
        page_number=38
    )
    chunk = DesignChunk(
        id="rule_6.1.2_v1",
        text="Beams shall be designed...",
        metadata=ChunkMetadata(
            chunk_id="rule_6.1.2_v1",
            content_type=ContentType.DESIGN_RULE,
            canonical_source=source
        )
    )
    result = RetrievalResult(
        chunk=chunk,
        role="primary",
        relevance_score=0.95
    )
    assert result.role == "primary"
    
    # Reference result
    ref_source = CanonicalSource(
        clause_id="6.1.1",
        clause_title="Strength Reduction Factors",
        page_number=37
    )
    ref_chunk = DesignChunk(
        id="table_6.1",
        text="Table 6.1 content...",
        metadata=ChunkMetadata(
            chunk_id="table_6.1",
            content_type=ContentType.TABLE,
            canonical_source=ref_source
        )
    )
    ref_result = RetrievalResult(
        chunk=ref_chunk,
        role="reference",
        relevance_score=0.85,
        referenced_from="clause_6.1.2"
    )
    assert ref_result.role == "reference"
    assert ref_result.referenced_from == "clause_6.1.2"
