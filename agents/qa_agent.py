import sys
from pathlib import Path
import re
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.vectordb.qdrant_store import GeoVectorStore
from src.utils.llm_client import call_llm_with_context
from src.utils.citation_verifier import CitationVerifier, CitationAwareQAAgent as CitationWrapper
from src.utils.enhanced_search import create_enhanced_search, QueryExpander
from src.utils.analytics import get_analytics
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.context_assembler import ContextAssembler
from config.settings import PRIMARY_MODEL


class QAAgent:
    def __init__(self, use_local_db: bool = False, use_hybrid_graph: bool = False):
        self.store = GeoVectorStore(use_local=use_local_db)
        self.conversation_history: list[dict] = []
        # Initialize enhanced search with hybrid + reranking
        self.enhanced_search = create_enhanced_search(self.store, use_reranking=True)
        self.query_expander = QueryExpander()
        # Toggle for hybrid search (can be enabled via UI)
        self.use_enhanced_search = True
        # Toggle for hybrid hierarchical graph retrieval
        self.use_hybrid_graph = use_hybrid_graph
        # Initialize HybridRetriever for graph-based retrieval
        if self.use_hybrid_graph:
            self.hybrid_retriever = HybridRetriever(self.store)
            self.context_assembler = ContextAssembler()

    def ask(
        self,
        question: str,
        document_type: str | None = None,
        document_id: str | None = None,
        clause_id: str | None = None,
        content_type: str | None = None,
        regulatory_strength: str | None = None,
        top_k: int = 8,
        model: str = PRIMARY_MODEL,
        equation_mode: bool = False,
        use_hybrid: bool = True,
        use_hybrid_graph: bool = None,
    ) -> dict:
        # Use instance setting unless explicitly overridden
        if use_hybrid_graph is None:
            use_hybrid_graph = self.use_hybrid_graph

        # Build filter kwargs for enhanced search
        filter_kwargs = {
            "document_type": document_type,
            "document_id": document_id,
            "clause_id": clause_id,
            "content_type": content_type,
            "regulatory_strength": regulatory_strength,
        }

        # Use hybrid hierarchical graph retrieval if enabled
        if use_hybrid_graph and hasattr(self, 'hybrid_retriever'):
            # Use HybridRetriever for citation-aware retrieval
            retrieval_results = self.hybrid_retriever.retrieve(
                query=question,
                top_k=top_k,
                use_graph_expansion=True,
            )

            if not retrieval_results:
                return {
                    "answer": "No relevant information found in the knowledge base.",
                    "sources": [],
                    "query": question,
                }

            # Convert RetrievalResult objects to dict format for call_llm_with_context
            # while preserving citation metadata
            chunks = []
            for result in retrieval_results:
                chunk_dict = {
                    "text": result.chunk.text,
                    "metadata": {
                        "document_name": result.chunk.metadata.canonical_source.document_id,
                        "section_title": result.chunk.metadata.canonical_source.clause_title,
                        "clause_id": result.chunk.metadata.canonical_source.clause_id,
                        "page_number": result.chunk.metadata.canonical_source.page_number,
                        "content_type": result.chunk.metadata.content_type.value if hasattr(result.chunk.metadata.content_type, 'value') else result.chunk.metadata.content_type,
                        "role": result.role,
                        "relevance_score": result.relevance_score,
                        "referenced_from": result.referenced_from,
                    },
                    "score": result.relevance_score,
                }
                chunks.append(chunk_dict)

            # Get formatted context with citations for logging/debugging
            formatted_context = self.context_assembler.assemble(retrieval_results, question)
        else:
            # Use enhanced search with hybrid + reranking
            if self.use_enhanced_search and use_hybrid:
                chunks = self.enhanced_search.search(
                    query=question,
                    top_k=top_k,
                    use_hybrid=True,
                    use_reranking=True,
                    semantic_weight=0.7,
                    keyword_weight=0.3,
                    **filter_kwargs,
                )
            else:
                # Fallback to basic search
                chunks = self.store.search(
                    query=question,
                    top_k=top_k,
                    **filter_kwargs,
                )

        if equation_mode:
            equation_query = (
                f"{question} equation formula design check criterion "
                "Mpl Rd Ft Rd Fb Rd V Ed V Rd resistance capacity"
            )
            if use_hybrid_graph and hasattr(self, 'hybrid_retriever'):
                # Use hybrid retriever for equation search
                eq_results = self.hybrid_retriever.retrieve(
                    query=equation_query,
                    top_k=max(top_k, 12),
                    use_graph_expansion=False,  # Don't expand graph for equations
                )
                eq_chunks = []
                for result in eq_results:
                    chunk_dict = {
                        "text": result.chunk.text,
                        "metadata": {
                            "document_name": result.chunk.metadata.canonical_source.document_id,
                            "section_title": result.chunk.metadata.canonical_source.clause_title,
                            "clause_id": result.chunk.metadata.canonical_source.clause_id,
                            "page_number": result.chunk.metadata.canonical_source.page_number,
                            "content_type": result.chunk.metadata.content_type.value if hasattr(result.chunk.metadata.content_type, 'value') else result.chunk.metadata.content_type,
                            "role": result.role,
                            "relevance_score": result.relevance_score,
                        },
                        "score": result.relevance_score,
                    }
                    eq_chunks.append(chunk_dict)
            elif self.use_enhanced_search and use_hybrid:
                eq_chunks = self.enhanced_search.search(
                    query=equation_query,
                    top_k=max(top_k, 12),
                    use_hybrid=True,
                    use_reranking=False,
                    **filter_kwargs,
                )
            else:
                eq_chunks = self.store.search(
                    query=equation_query,
                    top_k=max(top_k, 12),
                    **filter_kwargs,
                    score_threshold=0.0,
                )
            chunks = self._merge_chunks(chunks, eq_chunks, limit=max(top_k, 12))
        if not chunks:
            return {
                "answer": "No relevant information found in the knowledge base.",
                "sources": [],
                "query": question,
            }

        # Track search performance
        search_start = time.time()

        try:
            answer_start = time.time()
            answer = call_llm_with_context(
                question=question,
                context_chunks=chunks,
                model=model,
                equation_mode=equation_mode,
            )
            answer_duration_ms = int((time.time() - answer_start) * 1000)

            if equation_mode:
                answer = self._normalize_latex_blocks(answer)

            # Log to analytics
            try:
                analytics = get_analytics()
                analytics.log_query(
                    query=question,
                    results=chunks,
                    duration_ms=int((time.time() - search_start) * 1000),
                )
                analytics.log_answer(
                    query=question,
                    has_sources=len(chunks) > 0,
                    duration_ms=answer_duration_ms,
                )
            except Exception as e:
                logger.debug(f"Analytics logging failed: {e}")

        except Exception as e:
            preview = "\n\n".join([c["text"][:350] for c in chunks[:3]])
            answer = (
                "Retrieved relevant context, but LLM response generation failed. "
                f"Error: {e}\n\n"
                "Top retrieved excerpts:\n"
                f"{preview}"
            )
        sources = []
        seen = set()
        for chunk in chunks:
            source_key = (chunk["metadata"].get("document_name", ""), chunk["metadata"].get("section_title", ""))
            if source_key not in seen:
                seen.add(source_key)
                # Support both store format (score) and enhanced_search format (rerank_score / combined_score)
                relevance = chunk.get("score") or chunk.get("rerank_score") or chunk.get("combined_score") or 0.0
                source = {
                    "document": chunk["metadata"].get("document_name", "Unknown"),
                    "section": chunk["metadata"].get("section_title", "N/A"),
                    "page": chunk["metadata"].get("page_number", "N/A"),
                    "relevance_score": round(relevance, 3),
                }
                # Include clause_id if available (from hybrid graph retrieval)
                if chunk["metadata"].get("clause_id"):
                    source["clause_id"] = chunk["metadata"].get("clause_id")
                # Include role if available (from hybrid graph retrieval)
                if chunk["metadata"].get("role"):
                    source["role"] = chunk["metadata"].get("role")
                sources.append(source)
        self.conversation_history.append({"question": question, "answer": answer, "sources": sources})
        return {"answer": answer, "sources": sources, "query": question}

    def _normalize_latex_blocks(self, text: str) -> str:
        if not text:
            return text
        text = text.replace("<br>", "\n")
        lines = text.splitlines()
        normalized = []
        inline_pattern = re.compile(r"^\s*\[(.+)\]\s*$")
        step_pattern = re.compile(r"^\s*(###\s*Step|\d+\)\s*Step)")
        skip_var_section_until_next_heading = False
        no_equation_in_step = False
        for line in lines:
            stripped = line.strip()
            if step_pattern.match(stripped):
                no_equation_in_step = False
                skip_var_section_until_next_heading = False
                normalized.append(line)
                continue
            if skip_var_section_until_next_heading:
                if (
                    stripped.lower().startswith("**acceptance")
                    or stripped.lower().startswith("**citation")
                    or step_pattern.match(stripped)
                ):
                    skip_var_section_until_next_heading = False
                else:
                    continue
            if stripped.startswith("[Source:"):
                normalized.append(line)
                continue
            if "Equation not found in retrieved context" in stripped:
                no_equation_in_step = True
                normalized.append(line)
                continue
            if stripped.lower().startswith("**variable definitions") and no_equation_in_step:
                normalized.append(line)
                normalized.append("N/A (no equation in retrieved context).")
                skip_var_section_until_next_heading = True
                continue
            match = inline_pattern.match(stripped)
            if match:
                inner = match.group(1).strip()
                if any(token in inner for token in ("\\frac", "\\sum", "\\min", "_", "^", "=")):
                    normalized.append(f"$${inner}$$")
                    continue
            bullet_var = re.match(r"^(\s*[-*]\s*)\(([^)]+)\)\s*[-:]\s*(.*)$", line)
            if bullet_var:
                prefix, var_name, desc = bullet_var.groups()
                normalized.append(f"{prefix}${var_name.strip()}$: {desc.strip()}")
                continue
            normalized.append(line)
        normalized_text = "\n".join(normalized)
        normalized_text = normalized_text.replace("\\[", "$$").replace("\\]", "$$")
        return normalized_text

    def _merge_chunks(self, base_chunks: list[dict], extra_chunks: list[dict], limit: int) -> list[dict]:
        merged = []
        seen = set()
        for chunk in base_chunks + extra_chunks:
            key = (
                chunk.get("metadata", {}).get("document_name", ""),
                chunk.get("metadata", {}).get("section_title", ""),
                chunk.get("metadata", {}).get("page_number", ""),
                chunk.get("text", "")[:160],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= limit:
                break
        return merged

    def ask_followup(self, question: str, **kwargs) -> dict:
        if self.conversation_history:
            last = self.conversation_history[-1]
            enhanced_question = (
                f"Context from previous question: {last['question']}\n"
                f"Previous answer summary: {last['answer'][:500]}\n\n"
                f"Follow-up question: {question}"
            )
            return self.ask(enhanced_question, **kwargs)
        return self.ask(question, **kwargs)

    def list_knowledge_base(self) -> list[dict]:
        return self.store.list_documents()

    def ask_with_verification(self, question: str, **kwargs) -> dict:
        """
        Ask a question with automatic citation verification.
        This method verifies that all citations in the answer exist in retrieved chunks.
        
        Args:
            question: User question
            **kwargs: Additional arguments passed to ask()
            
        Returns:
            Answer dict with verification info added
        """
        result = self.ask(question, **kwargs)
        
        # Verify citations if we have an answer and sources
        if result.get("answer") and result.get("sources"):
            verifier = CitationVerifier()
            
            # Reconstruct retrieved chunks for verification
            retrieved_chunks = [
                {
                    "id": f"{s['document']}_{s['section']}_{s.get('page', '')}",
                    "metadata": {
                        "document_name": s["document"],
                        "section_title": s["section"],
                        "clause_id": s.get("clause_id", ""),
                        "page_number": s.get("page"),
                    },
                    "score": s.get("relevance_score", 0),
                }
                for s in result["sources"]
            ]
            
            verification = verifier.verify(
                answer=result["answer"],
                retrieved_chunks=retrieved_chunks,
                strict_mode=True,
            )
            
            # Add verification info to result
            result["verification"] = {
                "total_citations": verification.total_citations,
                "verified_count": verification.verified_count,
                "unverified_count": verification.unverified_count,
                "hallucinated_count": verification.hallucinated_count,
                "overall_confidence": verification.overall_confidence,
                "warnings": verification.warnings,
            }
            
            # Add warning if hallucinations detected
            if verification.hallucinated_count > 0:
                result["warning"] = (
                    f"⚠️ {verification.hallucinated_count} citation(s) could not be verified. "
                    "Please verify against source documents."
                )
        
        return result


def interactive_qa():
    agent = QAAgent(use_local_db=True)
    print("\nGEOTECH AI Q&A AGENT\nType your question, or 'quit' to exit\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "docs":
            docs = agent.list_knowledge_base()
            for doc in docs:
                print(f"{doc['document_name']} ({doc['document_type']}) - {doc['chunk_count']} chunks")
            continue
        if not question:
            continue
        result = agent.ask(question)
        print(f"\nAnswer:\n{result['answer']}\n")
        for s in result["sources"][:5]:
            print(f"- {s['document']} | {s['section']} ({s['relevance_score']})")


if __name__ == "__main__":
    interactive_qa()
