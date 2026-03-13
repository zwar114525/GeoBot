"""
Data models for Hybrid Hierarchical Graph Retrieval.
Defines strict schemas for chunks with canonical source metadata.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ContentType(str, Enum):
    """Types of content in engineering documents."""
    SECTION_OVERVIEW = "section_overview"
    DESIGN_RULE = "design_rule"
    TABLE = "table"
    EQUATION = "equation"
    DEFINITION = "definition"
    FIGURE = "figure"


class CanonicalSource(BaseModel):
    """
    Immutable source information for citation integrity.
    This is the KEY innovation - every chunk knows its TRUE source.
    """
    clause_id: str
    clause_title: str
    page_number: int
    document_id: str = "default"
    pdf_anchor: Optional[str] = None
    # Enhanced fields for hierarchical retrieval
    section_number: Optional[str] = None  # e.g., "6"
    subsection_number: Optional[str] = None  # e.g., "6.1"

    def to_citation_string(self) -> str:
        """Generate a citation string."""
        citation = f"[{self.clause_id}] {self.clause_title}"
        if self.page_number:
            citation += f", p. {self.page_number}"
        return citation

    def to_markdown_citation(self) -> str:
        """Generate Markdown format citation."""
        parts = [f"**{self.clause_id}**"]
        if self.clause_title:
            parts.append(self.clause_title)
        if self.page_number:
            parts.append(f"(p. {self.page_number})")
        return " ".join(parts)


class ChunkMetadata(BaseModel):
    """Metadata for design concept chunks."""
    chunk_id: str
    content_type: ContentType
    
    # Canonical source - NEVER modified during retrieval
    canonical_source: CanonicalSource
    
    # Graph links - stored as chunk IDs, not text
    references: List[str] = Field(default_factory=list)
    referenced_by: List[str] = Field(default_factory=list)
    
    # Engineering specifics
    regulatory_strength: Optional[str] = None
    table_caption: Optional[str] = None
    equation_id: Optional[str] = None
    
    # Hierarchy
    clause_id: str = "unknown"
    hierarchy_level: int = 0


class DesignChunk(BaseModel):
    """The fundamental unit of knowledge in the engineering RAG."""
    id: str
    text: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class QueryRoute(BaseModel):
    """Query routing decision."""
    type: str  # "comprehensive_design", "definition", "data_lookup", "general"
    strategy: str  # "hierarchical_graph", "atomic_lookup", "table_first", "semantic_search"
    target_elements: List[str] = Field(default_factory=list)
    target_actions: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result from retrieval with role information."""
    chunk: DesignChunk
    role: str = "primary"  # "primary" or "reference"
    relevance_score: float
    referenced_from: Optional[str] = None  # Which chunk referenced this
