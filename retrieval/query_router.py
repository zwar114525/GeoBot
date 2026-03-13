# src/retrieval/query_router.py
"""
Query Router for Hybrid Hierarchical Graph Retrieval.
Classifies user queries and determines retrieval strategy.
"""
from enum import Enum
from typing import List, Dict
import re


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    COMPREHENSIVE_DESIGN = "comprehensive_design"  # "how to design X"
    DEFINITION = "definition"  # "what is X"
    DATA_LOOKUP = "data_lookup"  # "table/value of X"
    GENERAL = "general"  # Fallback


class RetrievalStrategy(str, Enum):
    """Retrieval strategies based on query type."""
    HIERARCHICAL_GRAPH = "hierarchical_graph"  # For design queries
    TABLE_FIRST = "table_first"  # For data lookup
    ATOMIC_LOOKUP = "atomic_lookup"  # For definitions
    SEMANTIC_SEARCH = "semantic_search"  # Fallback


class QueryRoute:
    """Simple DTO for query routing results."""
    def __init__(self, type: str, strategy: str, target_elements: List[str], target_actions: List[str]):
        self.type = type
        self.strategy = strategy
        self.target_elements = target_elements
        self.target_actions = target_actions


class QueryRouter:
    """
    Routes queries to appropriate retrieval strategy.
    
    Key insight: Different query types need different retrieval approaches.
    """
    
    # Query type detection keywords
    DESIGN_TERMS = [
        "how to design", "design procedure", "steps for",
        "design process", "design guide", "calculate", "procedure for",
        "如何設計", "設計方法", "設計步驟", "設計程序", "計算"
    ]
    DEFINITION_TERMS = [
        "what is", "define", "meaning of", "definition of", "what are",
        "什麼是", "定義", "含義", "說明"
    ]
    DATA_TERMS = [
        "table", "value", "coefficient", "parameter", "factor",
        "list of", "range of", "typical",
        "表格", "數值", "係數", "參數", "值"
    ]
    
    # Engineering elements to extract
    ELEMENT_PATTERNS = {
        "beam": ["beam", "梁", "flexure", "bending", "鋼筋混凝土梁"],
        "column": ["column", "柱", "axial", "compression", "鋼筋混凝土柱"],
        "pile": ["pile", "樁", "deep foundation", "shaft", "基樁", "打樁"],
        "foundation": ["foundation", "footing", "基礎", "基底", "shallow"],
        "retaining": ["retaining", "wall", "earth pressure", "擋土牆", "擋土結構"],
        "slope": ["slope", "stability", "inclination", "邊坡", "山坡"],
        "shear": ["shear", "v_u", "v_c", "剪切", "剪力"],
        "settlement": ["settlement", "consolidation", "deflection", "沉降", "變形"],
    }
    
    def route(self, query: str) -> QueryRoute:
        """
        Determine routing for a query.

        Args:
            query: User query string

        Returns:
            QueryRoute with type, strategy, and extracted elements
        """
        query_lower = query.lower()

        # Detect query type
        if self._is_design_query(query_lower):
            query_type = QueryType.COMPREHENSIVE_DESIGN
            strategy = RetrievalStrategy.HIERARCHICAL_GRAPH
        elif self._is_data_lookup_query(query_lower):
            query_type = QueryType.DATA_LOOKUP
            strategy = RetrievalStrategy.TABLE_FIRST
        elif self._is_definition_query(query_lower):
            query_type = QueryType.DEFINITION
            strategy = RetrievalStrategy.ATOMIC_LOOKUP
        else:
            query_type = QueryType.GENERAL
            strategy = RetrievalStrategy.SEMANTIC_SEARCH

        # Extract target elements
        elements = self.extract_elements(query_lower)
        actions = self.extract_actions(query_lower)

        return QueryRoute(
            type=query_type.value,
            strategy=strategy.value,
            target_elements=elements,
            target_actions=actions
        )
    
    def _is_design_query(self, query: str) -> bool:
        """Check if query is a comprehensive design query."""
        return any(term in query for term in self.DESIGN_TERMS)
    
    def _is_definition_query(self, query: str) -> bool:
        """Check if query is a definition/understanding query."""
        return any(term in query for term in self.DEFINITION_TERMS)
    
    def _is_data_lookup_query(self, query: str) -> bool:
        """Check if query is looking for tables/values."""
        return any(term in query for term in self.DATA_TERMS)
    
    def extract_elements(self, query: str) -> List[str]:
        """Extract engineering elements from query."""
        elements = []
        for element, keywords in self.ELEMENT_PATTERNS.items():
            if any(kw in query for kw in keywords):
                elements.append(element)
        return elements
    
    def extract_actions(self, query: str) -> List[str]:
        """Extract design/calculation actions from query."""
        actions = []
        action_keywords = {
            "design": ["design", "sizing", "select"],
            "calculate": ["calculate", "compute", "determine", "find"],
            "check": ["check", "verify", "validate"],
            "select": ["choose", "select", "pick"],
        }
        for action, keywords in action_keywords.items():
            if any(kw in query for kw in keywords):
                actions.append(action)
        return actions

