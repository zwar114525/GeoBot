"""
Graph Expander for Hybrid Hierarchical Graph Retrieval.

KEY RESPONSIBILITY: Fetch referenced chunks dynamically while
preserving canonical_source metadata. This is the critical
component that ensures citation accuracy.
"""
from typing import List, Set, Callable, Optional
from dataclasses import dataclass
from loguru import logger

from src.schemas.design_chunk_schemas import (
    DesignChunk, RetrievalResult
)
from src.retrieval.graph_store import GraphStore


@dataclass
class GraphExpansionResult:
    """Result of graph expansion operation."""
    results: List[RetrievalResult]
    primary_count: int
    reference_count: int
    fetched_ids: Set[str]


class GraphExpander:
    """
    Expands retrieval results by fetching referenced chunks.

    KEY BEHAVIOR: Never modifies canonical_source when fetching references.
    This ensures Table 6.1 (in Clause 6.1.1) is always cited as 6.1.1,
    even when referenced from Clause 6.1.2.
    """

    def __init__(self):
        # Persistent graph store for cross-references
        self.graph_store = GraphStore()
        # Will be set by the caller
        self.get_chunk_by_id: Optional[Callable[[str], DesignChunk]] = None
    
    def expand(
        self,
        primary_chunks: List[DesignChunk],
        max_depth: int = 1,
    ) -> GraphExpansionResult:
        """
        Expand primary chunks by fetching their dependencies.

        Args:
            primary_chunks: Initially retrieved chunks
            max_depth: How many levels of references to fetch

        Returns:
            GraphExpansionResult with all results including references
        """
        if not primary_chunks:
            return GraphExpansionResult([], 0, 0, set())

        results: List[RetrievalResult] = []
        fetched_ids: Set[str] = set()

        # Process each primary chunk
        for chunk in primary_chunks:
            # Skip if already fetched
            if chunk.id in fetched_ids:
                continue

            # Add primary chunk
            results.append(RetrievalResult(
                chunk=chunk,
                role="primary",
                relevance_score=1.0
            ))
            fetched_ids.add(chunk.id)

            # Fetch references if enabled
            if max_depth > 0:
                # Get references from graph store (persistent)
                references = self.graph_store.get_references(chunk.id)
                # Also check chunk metadata for references
                metadata_refs = chunk.metadata.references or []
                # Combine unique references
                all_refs = list(set(references + metadata_refs))

                for ref_id in all_refs:
                    if ref_id in fetched_ids:
                        continue

                    ref_result = self._fetch_reference(
                        ref_id,
                        primary_chunk_id=chunk.id,
                        max_depth=max_depth - 1,
                        fetched_ids=fetched_ids
                    )
                    if ref_result:
                        results.append(ref_result)
                        fetched_ids.add(ref_id)

        primary_count = sum(1 for r in results if r.role == "primary")
        reference_count = sum(1 for r in results if r.role == "reference")

        logger.info(
            f"Graph expansion complete: {primary_count} primary, "
            f"{reference_count} references, {len(fetched_ids)} total"
        )

        return GraphExpansionResult(
            results=results,
            primary_count=primary_count,
            reference_count=reference_count,
            fetched_ids=fetched_ids
        )
    
    def _fetch_reference(
        self,
        ref_id: str,
        primary_chunk_id: str,
        max_depth: int,
        fetched_ids: Set[str],
    ) -> Optional[RetrievalResult]:
        """
        Fetch a single referenced chunk.
        
        CRITICAL: This preserves the chunk's canonical_source.
        """
        if not self.get_chunk_by_id:
            logger.warning("GraphExpander: get_chunk_by_id not set")
            return None
        
        try:
            ref_chunk = self.get_chunk_by_id(ref_id)
            if not ref_chunk:
                logger.debug(f"Reference chunk not found: {ref_id}")
                return None
            
            # KEY: Role is "reference", but canonical_source stays intact!
            return RetrievalResult(
                chunk=ref_chunk,
                role="reference",
                relevance_score=0.8,  # Slightly lower than primary
                referenced_from=primary_chunk_id
            )
        except Exception as e:
            logger.error(f"Failed to fetch reference {ref_id}: {e}")
            return None
