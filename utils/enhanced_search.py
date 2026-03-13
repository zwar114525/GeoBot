"""
Enhanced search with hybrid (keyword + semantic) search and re-ranking.
"""
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import numpy as np
    from scipy.spatial.distance import cosine
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy/scipy not available. Some re-ranking features disabled.")


@dataclass
class SearchResult:
    """Search result with scores."""
    text: str
    metadata: dict
    semantic_score: float
    keyword_score: float
    combined_score: float
    rank: int = 0


class HybridSearch:
    """
    Hybrid search combining semantic (vector) and keyword (BM25-style) search.
    """
    
    def __init__(self, vector_store):
        """
        Initialize hybrid search.
        
        Args:
            vector_store: GeoVectorStore instance
        """
        self.vector_store = vector_store
        self.keyword_index: Dict[str, Dict] = {}
        self._build_keyword_index()
    
    def _build_keyword_index(self):
        """Build in-memory keyword index from vector store."""
        try:
            all_points, _ = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                limit=50000,
                with_payload=["text", "document_id", "document_name", "clause_id", "section_title"],
            )
            
            for point in all_points:
                payload = point.payload or {}
                text = payload.get("text", "").lower()
                doc_id = str(point.id)
                
                # Tokenize and index
                tokens = self._tokenize(text)
                for token in tokens:
                    if token not in self.keyword_index:
                        self.keyword_index[token] = {}
                    self.keyword_index[token][doc_id] = self.keyword_index[token].get(doc_id, 0) + 1
                    
            logger.info(f"Built keyword index with {len(self.keyword_index)} terms")
        except Exception as e:
            logger.warning(f"Failed to build keyword index: {e}")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        # Remove punctuation, lowercase, split
        tokens = re.findall(r'\b[a-z]{2,}\b', text.lower())
        # Remove stopwords
        stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'shall', 'should', 'must', 'may'}
        return [t for t in tokens if t not in stopwords]
    
    def _keyword_search(self, query: str, top_k: int = 20) -> Dict[str, float]:
        """
        Perform keyword search.
        
        Returns:
            Dict mapping doc_id to keyword score
        """
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}
        
        for token in query_tokens:
            if token in self.keyword_index:
                for doc_id, count in self.keyword_index[token].items():
                    scores[doc_id] = scores.get(doc_id, 0) + count
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs,
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
            semantic_weight: Weight for semantic score (0-1)
            keyword_weight: Weight for keyword score (0-1)
            **kwargs: Additional args passed to vector store
            
        Returns:
            List of result dicts with combined scores
        """
        # Get semantic search results
        semantic_results = self.vector_store.search(
            query=query,
            top_k=top_k * 2,  # Get more for re-ranking
            **kwargs,
        )
        
        # Get keyword scores
        keyword_scores = self._keyword_search(query, top_k=top_k * 2)
        
        # Combine scores
        combined_results = []
        seen_ids = set()
        
        for result in semantic_results:
            doc_id = result.get("id", str(hash(result["text"])))
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            
            semantic_score = result.get("score", 0)
            keyword_score = keyword_scores.get(doc_id, 0)
            
            # Reciprocal Rank Fusion for keyword results not in semantic
            if keyword_score > 0 and semantic_score == 0:
                rank = list(keyword_scores.keys()).index(doc_id) + 1 if doc_id in keyword_scores else 100
                semantic_score = 1.0 / (rank + 60)  # RRF formula
            
            combined_score = (semantic_weight * semantic_score) + (keyword_weight * keyword_score)
            
            combined_results.append({
                "text": result["text"],
                "metadata": result["metadata"],
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "combined_score": combined_score,
            })
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:top_k]


class CrossEncoderReranker:
    """
    Re-rank search results using a cross-encoder model.
    More accurate but slower than bi-encoder.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-6-v2"):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model from HuggingFace
        """
        self.model = None
        self.model_name = model_name
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available. Re-ranking disabled.")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder: {e}")
    
    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Re-rank results using cross-encoder.
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of results to return
            
        Returns:
            Re-ranked results
        """
        if not self.model or not results:
            return results[:top_k]
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, r["text"]] for r in results]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)
            
            # Sort by rerank score
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return results[:top_k]


class EnhancedSearch:
    """
    Enhanced search with hybrid search and optional re-ranking.
    This is the main entry point for improved retrieval.
    """
    
    def __init__(self, vector_store, use_reranking: bool = True):
        """
        Initialize enhanced search.
        
        Args:
            vector_store: GeoVectorStore instance
            use_reranking: Whether to use cross-encoder re-ranking
        """
        self.vector_store = vector_store
        self.hybrid_search = HybridSearch(vector_store)
        self.reranker = CrossEncoderReranker() if use_reranking else None
        self.use_reranking = use_reranking
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        use_hybrid: bool = True,
        use_reranking: bool = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **kwargs,
    ) -> List[Dict]:
        """
        Enhanced search with hybrid and re-ranking options.
        
        Args:
            query: Search query
            top_k: Number of results
            use_hybrid: Use hybrid search (default True)
            use_reranking: Override default re-ranking setting
            semantic_weight: Weight for semantic score
            keyword_weight: Weight for keyword score
            **kwargs: Additional args
            
        Returns:
            List of search results
        """
        if use_hybrid:
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k * 2 if self.reranker else top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
                **kwargs,
            )
        else:
            results = self.vector_store.search(query=query, top_k=top_k * 2, **kwargs)
        
        # Apply re-ranking if enabled
        if self.reranker and (use_reranking if use_reranking is not None else self.use_reranking):
            results = self.reranker.rerank(query=query, results=results, top_k=top_k)
        else:
            results = results[:top_k]
        
        # Add rank and normalize "score" so consumers (e.g. qa_agent) always have chunk["score"]
        for i, result in enumerate(results):
            result["rank"] = i + 1
            if "score" not in result:
                result["score"] = result.get("rerank_score") or result.get("combined_score") or 0.0
        
        return results


class QueryExpander:
    """
    Expand queries with synonyms and related terms for better retrieval.
    """
    
    # Geotechnical engineering synonyms
    SYNONYMS = {
        "bearing": ["capacity", "resistance", "foundation"],
        "capacity": ["bearing", "resistance", "load"],
        "shear": ["strength", "resistance", "failure"],
        "settlement": ["displacement", "deflection", "subsidence"],
        "slope": ["incline", "gradient", "embankment"],
        "stability": ["factor", "safety", "FoS"],
        "cohesion": ["c", "undrained", "clay"],
        "friction": ["angle", "phi", "φ", "sand"],
        "groundwater": ["water", "table", "GWL", "phreatic"],
        "retaining": ["wall", "structure", "support"],
        "excavation": ["digging", "cut", "trench"],
        "pile": ["deep", "foundation", "shaft"],
        "shallow": ["footing", "pad", "raft", "mat"],
    }
    
    def __init__(self, llm_client=None):
        """
        Initialize query expander.
        
        Args:
            llm_client: Optional LLM client for intelligent expansion
        """
        self.llm_client = llm_client
    
    def expand(self, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand query with synonyms.
        
        Args:
            query: Original query
            max_expansions: Maximum number of expanded queries
            
        Returns:
            List of expanded queries
        """
        expanded = [query]
        query_lower = query.lower()
        
        # Add synonym expansions
        for term, synonyms in self.SYNONYMS.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit synonyms per term
                    new_query = query_lower.replace(term, synonym)
                    if new_query != query_lower and new_query not in expanded:
                        expanded.append(new_query)
                        if len(expanded) >= max_expansions + 1:
                            break
        
        return expanded[:max_expansions + 1]
    
    def expand_with_llm(self, query: str, top_k: int = 5) -> List[str]:
        """
        Use LLM to generate query variations.
        
        Args:
            query: Original query
            top_k: Number of variations
            
        Returns:
            List of expanded queries
        """
        if not self.llm_client:
            return self.expand(query)
        
        try:
            from src.utils.llm_client import call_llm
            
            prompt = f"""Generate {top_k} variations of this geotechnical engineering query for better document retrieval.
Each variation should use different terminology but mean the same thing.

Original query: {query}

Return ONLY a JSON array of query variations."""
            
            response = call_llm(prompt, temperature=0.3, max_tokens=500)
            
            # Parse response
            import json
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                variations = json.loads(json_match.group(0))
                return [query] + variations[:top_k]
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
        
        return self.expand(query)


def create_enhanced_search(vector_store, use_reranking: bool = True) -> EnhancedSearch:
    """
    Factory function to create enhanced search instance.
    
    Args:
        vector_store: GeoVectorStore instance
        use_reranking: Whether to use cross-encoder re-ranking
        
    Returns:
        EnhancedSearch instance
    """
    return EnhancedSearch(vector_store, use_reranking=use_reranking)
