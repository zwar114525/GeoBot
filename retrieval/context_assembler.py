"""
Citation-Aware Context Assembler.

Formats retrieved chunks for LLM consumption with explicit
source attribution and citation instructions.
"""
from typing import List
from src.schemas.design_chunk_schemas import RetrievalResult


class ContextAssembler:
    """
    Assembles LLM context with clear source attribution.
    
    KEY BEHAVIOR: Each chunk is tagged with its canonical source,
    and explicit citation instructions tell the LLM how to cite.
    """
    
    def assemble(self, results: List[RetrievalResult], query: str) -> str:
        """
        Assemble context from retrieval results.
        
        Args:
            results: List of RetrievalResult from graph expansion
            query: Original user query
            
        Returns:
            Formatted context string for LLM
        """
        context = f"Query: {query}\n\n"
        context += "## Retrieved Engineering Knowledge\n\n"
        
        # Group by canonical source for organized output
        by_source = {}
        for result in results:
            source = result.chunk.metadata.canonical_source
            key = f"{source.clause_id}"
            
            if key not in by_source:
                by_source[key] = {
                    "title": source.clause_title,
                    "page": source.page_number,
                    "items": []
                }
            
            by_source[key]["items"].append(result)
        
        # Format context
        for clause_id, group in sorted(by_source.items()):
            context += f"### [{clause_id}] {group['title']} (Page {group['page']})\n"
            
            for item in group["items"]:
                role_tag = "[PRIMARY]" if item.role == "primary" else "[REFERENCED]"
                context += f"{role_tag}:\n{item.chunk.text}\n\n"
        
        # Add citation instructions
        context += self._get_citation_instructions()
        
        return context
    
    def _get_citation_instructions(self) -> str:
        """Get citation instruction text for the LLM."""
        return """
## Citation Rules (STRICT)
1. **Primary Source:** If information comes from a [PRIMARY] block, cite: [Clause X.Y.Z, p. XX]
2. **Referenced Source:** If information comes from a [REFERENCED] block, cite the **Source Clause** listed in the header, NOT the clause that referenced it.
   - Example: If Table 6.1 is in Clause 6.1.1 but referenced by 6.1.2, cite [Table 6.1, Clause 6.1.1, p. 37].
3. **Equations:** Include Equation ID if available.
4. **Verification:** All claims must be traceable to the provided Clause IDs and Page Numbers.
"""
