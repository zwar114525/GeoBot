"""
Hybrid Hierarchical Graph Retriever.

Main entry point that combines:
1. Query routing
2. Hierarchical search (section + rule)
3. Graph expansion
4. Citation-aware context assembly
"""
from typing import List, Dict, Optional, Union
from loguru import logger

from src.retrieval.query_router import QueryRouter, QueryType
from src.retrieval.graph_expander import GraphExpander
from src.retrieval.context_assembler import ContextAssembler
from src.schemas.design_chunk_schemas import RetrievalResult, DesignChunk, ChunkMetadata, CanonicalSource, ContentType
from src.vectordb.qdrant_store import GeoVectorStore


class HybridRetriever:
    """
    Complete retrieval system for engineering documents.
    
    Usage:
        retriever = HybridRetriever(vector_store)
        
        # For comprehensive design queries
        results = retriever.retrieve("How to design a concrete beam?")
        
        # Get formatted context for LLM
        context = retriever.assemble_context(results, query)
    """
    
    def __init__(self, vector_store: GeoVectorStore):
        self.vector_store = vector_store
        self.router = QueryRouter()
        self.expander = GraphExpander()
        self.assembler = ContextAssembler()
        
        # Set up graph expansion with chunk lookup
        self.expander.get_chunk_by_id = self._get_chunk_by_id
    
    def _dict_to_design_chunk(self, chunk_dict: Dict) -> DesignChunk:
        """Convert a dict from vector store to DesignChunk."""
        metadata_dict = chunk_dict.get("metadata", {})
        
        # Handle canonical_source - could be dict or CanonicalSource
        canonical = metadata_dict.get("canonical_source")
        if isinstance(canonical, dict):
            canonical_source = CanonicalSource(**canonical)
        elif canonical is None:
            # Create default if not present
            canonical_source = CanonicalSource(
                clause_id=metadata_dict.get("clause_id", "unknown"),
                clause_title=metadata_dict.get("clause_title", "Unknown"),
                page_number=metadata_dict.get("page_number", 0),
                document_id=metadata_dict.get("document_id", "default")
            )
        else:
            canonical_source = canonical
        
        # Handle content_type
        content_type_str = metadata_dict.get("content_type", "design_rule")
        if isinstance(content_type_str, str):
            try:
                content_type = ContentType(content_type_str)
            except ValueError:
                content_type = ContentType.DESIGN_RULE
        else:
            content_type = content_type_str
        
        # Build ChunkMetadata
        chunk_metadata = ChunkMetadata(
            chunk_id=metadata_dict.get("chunk_id", chunk_dict.get("id", "unknown")),
            content_type=content_type,
            canonical_source=canonical_source,
            references=metadata_dict.get("references", []),
            referenced_by=metadata_dict.get("referenced_by", []),
            regulatory_strength=metadata_dict.get("regulatory_strength"),
            clause_id=metadata_dict.get("clause_id", "unknown"),
            hierarchy_level=metadata_dict.get("hierarchy_level", 0),
        )
        
        return DesignChunk(
            id=chunk_dict.get("id", metadata_dict.get("chunk_id", "unknown")),
            text=chunk_dict.get("text", ""),
            metadata=chunk_metadata,
        )
    
    def _get_chunk_by_id(self, chunk_id: str) -> Optional[DesignChunk]:
        """
        Get chunk by ID from vector store.
        
        This is used by GraphExpander to fetch referenced chunks.
        """
        # Search with a broad query to get chunks, then filter by chunk_id
        # In production, you might want to add a direct lookup method to GeoVectorStore
        try:
            results = self.vector_store.search(
                query="",  # Empty query - we just want to get chunks
                top_k=100,
                score_threshold=0.0,  # Accept all results
            )
            
            for r in results:
                metadata = r.get("metadata", {})
                if metadata.get("chunk_id") == chunk_id:
                    return self._dict_to_design_chunk(r)
            
            logger.debug(f"Chunk not found: {chunk_id}")
            return None
        except Exception as e:
            logger.error(f"Error fetching chunk {chunk_id}: {e}")
            return None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_graph_expansion: bool = True,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid hierarchical approach.
        
        Args:
            query: User query
            top_k: Number of primary results
            use_graph_expansion: Whether to fetch dependencies
            
        Returns:
            List of RetrievalResult with primary and reference chunks
        """
        # Step 1: Route query
        route = self.router.route(query)
        logger.info(f"Query routed: {route.type} -> {route.strategy}")
        
        # Step 2: Determine retrieval strategy
        if route.type == QueryType.COMPREHENSIVE_DESIGN.value:
            # Use hierarchical search + graph expansion
            primary_chunks = self._hierarchical_search(query, top_k)
        elif route.type == QueryType.DATA_LOOKUP.value:
            # Prioritize tables
            primary_chunks = self._search_with_filter(
                query, top_k, content_type="table"
            )
        else:
            # General semantic search (definition, general)
            primary_chunks = self._semantic_search(query, top_k)
        
        if not primary_chunks:
            return []
        
        # Step 3: Graph expansion (if enabled and appropriate)
        if use_graph_expansion and route.type == QueryType.COMPREHENSIVE_DESIGN.value:
            # Convert dicts to DesignChunks for graph expansion
            design_chunks = []
            for chunk_dict in primary_chunks:
                if isinstance(chunk_dict, dict):
                    design_chunks.append(self._dict_to_design_chunk(chunk_dict))
                else:
                    design_chunks.append(chunk_dict)
            
            expansion_result = self.expander.expand(design_chunks, max_depth=1)
            return expansion_result.results
        else:
            # Just return primary results - convert to RetrievalResult
            results = []
            for chunk_dict in primary_chunks:
                if isinstance(chunk_dict, dict):
                    chunk = self._dict_to_design_chunk(chunk_dict)
                else:
                    chunk = chunk_dict
                
                results.append(RetrievalResult(
                    chunk=chunk,
                    role="primary",
                    relevance_score=1.0
                ))
            return results
    
    def _hierarchical_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Two-stage hierarchical search:
        1. Find relevant sections (using Section Index)
        2. Find rules within those sections (using Rule Index)

        Falls back to semantic search if dual index not available.
        """
        try:
            # Stage 1: Try to find relevant sections
            section_store = self.vector_store.get_section_store()
            section_results = section_store.search(
                query=query,
                top_k=3,
                content_type="section_overview"
            )

            # Extract relevant clause prefixes
            relevant_clauses = []
            for sr in section_results:
                clause_id = sr.get("metadata", {}).get("clause_id", "")
                if clause_id:
                    # 6.1.2 -> prefixes: ["6", "6.1", "6.1.2"]
                    parts = clause_id.split(".")
                    for i in range(len(parts)):
                        prefix = ".".join(parts[:i+1])
                        if prefix not in relevant_clauses:
                            relevant_clauses.append(prefix)

            # Stage 2: Search rules within relevant sections
            if relevant_clauses:
                rule_store = self.vector_store.get_rule_store()
                rule_results = rule_store.search(
                    query=query,
                    top_k=top_k,
                    clause_id=None  # We would need to implement $in filter
                )

                if rule_results:
                    logger.info(f"Hierarchical search: found {len(relevant_clauses)} sections, {len(rule_results)} rules")
                    return rule_results

            # Fallback to semantic search
            logger.info("Hierarchical search falling back to semantic search")
            return self._semantic_search(query, top_k)

        except Exception as e:
            logger.warning(f"Dual index not available, using semantic search: {e}")
            return self._semantic_search(query, top_k)
    
    def _search_with_filter(
        self, 
        query: str, 
        top_k: int, 
        content_type: str
    ) -> List[Dict]:
        """Search with specific content type filter."""
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            content_type=content_type
        )
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict]:
        """Standard semantic search."""
        results = self.vector_store.search(
            query=query,
            top_k=top_k
        )
        return results
    
    def assemble_context(
        self, 
        results: List[RetrievalResult], 
        query: str
    ) -> str:
        """
        Assemble formatted context for LLM.
        
        Args:
            results: Retrieval results
            query: Original query
            
        Returns:
            Formatted context string
        """
        return self.assembler.assemble(results, query)
