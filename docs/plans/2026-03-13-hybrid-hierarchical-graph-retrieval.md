# Hybrid Hierarchical Graph Retrieval 實施計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**目標：** 為 GeoBot 實現完整的 Hybrid Hierarchical Graph Retrieval 系統，增強工程設計查詢的檢索準確性和引用完整性

**架構：** 將單一 Qdrant 索引拆分為 Section Index + Rule Index 雙索引系統，實現兩階段階層式檢索，增強圖形擴展和引用來源保留

**技術棧：** Qdrant, Python, Pydantic

---

## 任務 1: 建立雙索引系統（Section Index + Rule Index）

### Files:
- Modify: `src/vectordb/qdrant_store.py`
- Modify: `src/ingestion/ingest.py`
- Modify: `src/ingestion/pdf_processor_enhanced.py`

### Step 1: 在 qdrant_store.py 中新增雙索引支援

在 `GeoVectorStore` 類中新增以下方法：

```python
def create_section_index(self) -> str:
    """建立 Section Index"""
    section_collection = f"{self.collection_name}_sections"
    if not self.client.collection_exists(section_collection):
        self.client.create_collection(
            collection_name=section_collection,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    return section_collection

def create_rule_index(self) -> str:
    """建立 Rule Index"""
    rule_collection = f"{self.collection_name}_rules"
    if not self.client.collection_exists(rule_collection):
        self.client.create_collection(
            collection_name=rule_collection,
            vectors_config=VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        )
    return rule_collection

def get_by_chunk_id(self, collection: str, chunk_id: str) -> Optional[Dict]:
    """通過 chunk_id 直接獲取區塊"""
    results = self.client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))]
        ),
        limit=1
    )
    return results[0][0] if results[0] else None

def get_section_store(self) -> 'GeoVectorStore':
    """獲取 Section Index 存儲器"""
    section_store = GeoVectorStore(
        collection_name=f"{self.collection_name}_sections",
        embed_model=self.embed_model,
        client=self.client
    )
    return section_store

def get_rule_store(self) -> 'GeoVectorStore':
    """獲取 Rule Index 存儲器"""
    rule_store = GeoVectorStore(
        collection_name=f"{self.collection_name}_rules",
        embed_model=self.embed_model,
        client=self.client
    )
    return rule_store
```

### Step 2: 在 pdf_processor_enhanced.py 中新增分類方法

在 `EnhancedPDFProcessor` 類中新增：

```python
def classify_for_dual_index(self, chunk: Dict) -> str:
    """分類區塊應該進入哪個索引"""
    content_type = chunk.get("content_type", "")
    hierarchy_level = chunk.get("hierarchy_level", 0)
    
    # Section Index: 章節標題和概述
    if content_type == "section_header" or hierarchy_level <= 1:
        return "section"
    
    # Rule Index: 具體條款、表格、方程式
    return "rule"
```

### Step 3: 驗證更改

Run: `python -c "from src.vectordb.qdrant_store import GeoVectorStore; print('Import OK')"`

Expected: Import successful

---

## 任務 2: 增強查詢路由器

### Files:
- Modify: `src/retrieval/query_router.py`

### Step 1: 增強查詢分類規則

在 `QueryRouter` 類中新增/修改 `route_query` 方法：

```python
def route_query(self, query: str) -> Dict:
    """增強的查詢路由"""
    query_lower = query.lower()
    
    # 提取工程元素
    engineering_elements = self._extract_engineering_elements(query_lower)
    
    # 提取動作
    actions = self._extract_actions(query_lower)
    
    # 根據模式分類
    if any(term in query_lower for term in [
        "how to design", "design procedure", "steps for", 
        "design method", "design approach", "設計"
    ]):
        query_type = QueryType.COMPREHENSIVE_DESIGN
        strategy = RetrievalStrategy.HIERARCHICAL_GRAPH
    elif any(term in query_lower for term in [
        "what is", "define", "meaning of", "definition", "什麼是", "定義"
    ]):
        query_type = QueryType.DEFINITION
        strategy = RetrievalStrategy.ATOMIC_LOOKUP
    elif any(term in query_lower for term in [
        "table", "value", "coefficient", "parameter", "表格", "數值", "係數"
    ]):
        query_type = QueryType.DATA_LOOKUP
        strategy = RetrievalStrategy.TABLE_FIRST
    else:
        query_type = QueryType.GENERAL
        strategy = RetrievalStrategy.HYBRID_SEARCH
    
    return {
        "type": query_type,
        "strategy": strategy,
        "elements": engineering_elements,
        "actions": actions
    }

def _extract_engineering_elements(self, query: str) -> List[str]:
    """提取工程元素"""
    elements = []
    element_patterns = {
        "beam": ["beam", "梁", "flexure", "flexural"],
        "column": ["column", "柱", "compression"],
        "pile": ["pile", "樁", "foundation"],
        "foundation": ["foundation", "基礎", "footing"],
        "slab": ["slab", "板"],
        "retaining_wall": ["retaining wall", "擋土牆", "earth pressure"],
        "shear": ["shear", "剪切", "shear wall"],
    }
    
    for element, patterns in element_patterns.items():
        if any(p in query for p in patterns):
            elements.append(element)
    
    return elements
```

### Step 2: 驗證更改

Run: `python -c "from src.retrieval.query_router import QueryRouter; r = QueryRouter(); print(r.route_query('How to design a concrete beam?'))"`

Expected: 返回包含 type, strategy, elements, actions 的字典

---

## 任務 3: 實現圖形邊持久化存儲

### Files:
- Create: `src/retrieval/graph_store.py`
- Modify: `src/retrieval/graph_expander.py`

### Step 1: 創建 GraphStore 類

創建新文件 `src/retrieval/graph_store.py`：

```python
import json
import os
from typing import Dict, List, Optional
from pathlib import Path


class GraphStore:
    """圖形邊存儲 - 持久化跨引用關係"""
    
    def __init__(self, storage_path: str = "data/graph_edges.json"):
        self.storage_path = storage_path
        # 確保目錄存在
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        self.edges = self._load_edges()
    
    def _load_edges(self) -> Dict[str, List[str]]:
        """從磁盤加載圖形邊"""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_edges(self):
        """保存圖形邊到磁盤"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.edges, f, indent=2, ensure_ascii=False)
    
    def add_edge(self, source_id: str, target_id: str):
        """添加圖形邊 (source -> target)"""
        if source_id not in self.edges:
            self.edges[source_id] = []
        if target_id not in self.edges[source_id]:
            self.edges[source_id].append(target_id)
        self._save_edges()
    
    def add_edges_from_chunk(self, chunk_id: str, references: List[str]):
        """從區塊的 references 字段批量添加邊"""
        for ref_id in references:
            self.add_edge(chunk_id, ref_id)
    
    def get_references(self, chunk_id: str) -> List[str]:
        """獲取區塊的所有引用 (chunk_id -> [])"""
        return self.edges.get(chunk_id, [])
    
    def get_referenced_by(self, chunk_id: str) -> List[str]:
        """獲取引用此區塊的所有區塊 ([] -> chunk_id)"""
        return [
            source_id 
            for source_id, targets in self.edges.items() 
            if chunk_id in targets
        ]
    
    def has_references(self, chunk_id: str) -> bool:
        """檢查區塊是否有引用"""
        return len(self.get_references(chunk_id)) > 0
    
    def clear(self):
        """清除所有邊"""
        self.edges = {}
        self._save_edges()
    
    def get_stats(self) -> Dict:
        """獲取圖形統計"""
        return {
            "total_nodes": len(self.edges),
            "total_edges": sum(len(v) for v in self.edges.values())
        }
```

### Step 2: 修改 graph_expander.py 使用 GraphStore

在 `GraphExpander` 類中：

```python
from src.retrieval.graph_store import GraphStore


class GraphExpander:
    """圖形擴展器 - 動態獲取引用區塊"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.graph_store = GraphStore()
    
    async def expand(
        self,
        chunks: List[Chunk],
        max_depth: int = 1
    ) -> List[RetrievalResult]:
        """擴展檢索結果"""
        expanded = []
        seen_ids = set()
        
        for chunk in chunks:
            if chunk.chunk_id in seen_ids:
                continue
            
            # 添加主區塊
            expanded.append(RetrievalResult(
                chunk=chunk,
                role="primary",
                relevance_score=1.0
            ))
            seen_ids.add(chunk.chunk_id)
            
            # 獲取引用
            if max_depth >= 1:
                ref_ids = self.graph_store.get_references(chunk.chunk_id)
                for ref_id in ref_ids:
                    if ref_id not in seen_ids:
                        ref_chunk = await self._fetch_chunk_by_id(ref_id)
                        if ref_chunk:
                            expanded.append(RetrievalResult(
                                chunk=ref_chunk,
                                role="reference",
                                relevance_score=0.8
                            ))
                            seen_ids.add(ref_id)
        
        return expanded
    
    async def _fetch_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """通過 ID 獲取區塊"""
        # 使用 vector_store 的 get_by_chunk_id 方法
        result = self.vector_store.get_by_chunk_id(chunk_id)
        if result:
            return self._convert_to_chunk(result)
        return None
    
    def _convert_to_chunk(self, data: Dict) -> Chunk:
        """將 Qdrant 記錄轉換為 Chunk 對象"""
        return Chunk(
            chunk_id=data.get("chunk_id", ""),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            embedding=data.get("vector")
        )
```

### Step 3: 驗證更改

Run: `python -c "from src.retrieval.graph_store import GraphStore; g = GraphStore(); print(g.get_stats())"`

Expected: {"total_nodes": 0, "total_edges": 0}

---

## 任務 4: 增強引用來源元數據

### Files:
- Modify: `src/schemas/design_chunk_schemas.py`

### Step 1: 擴展 CanonicalSource 結構

在 `design_chunk_schemas.py` 中修改或新增：

```python
class CanonicalSource(BaseModel):
    """不可變的來源信息 - 確保引用完整性"""
    clause_id: str
    clause_title: str
    page_number: int
    document_id: str = "CoP_SUC2013e"
    pdf_anchor: Optional[str] = None  # 例如 "#page=38&zoom=100,120,450"
    section_number: Optional[str] = None  # 例如 "6"
    subsection_number: Optional[str] = None  # 例如 "6.1"
    
    def to_citation_string(self) -> str:
        """生成引用字符串"""
        citation = f"[{self.clause_id}] {self.clause_title}"
        if self.page_number:
            citation += f", p. {self.page_number}"
        return citation
    
    def to_markdown_citation(self) -> str:
        """生成 Markdown 格式的引用"""
        parts = [f"**{self.clause_id}**"]
        if self.clause_title:
            parts.append(self.clause_title)
        if self.page_number:
            parts.append(f"(p. {self.page_number})")
        return " ".join(parts)
```

### Step 2: 驗證更改

Run: `python -c "from src.schemas.design_chunk_schemas import CanonicalSource; c = CanonicalSource(clause_id='6.1.2', clause_title='Flexural Design', page_number=38); print(c.to_citation_string())"`

Expected: [6.1.2] Flexural Design, p. 38

---

## 任務 5: 實現階層式檢索

### Files:
- Modify: `src/retrieval/hybrid_retriever.py`

### Step 1: 實現兩階段檢索

在 `HybridRetriever` 類中新增方法：

```python
async def hierarchical_retrieve(
    self, 
    query: str, 
    route: Dict,
    k: int = 5
) -> List[RetrievalResult]:
    """兩階段階層式檢索"""
    results = []
    
    if route.get("strategy") == RetrievalStrategy.HIERARCHICAL_GRAPH:
        # Stage 1: 在 Section Index 中查找相關章節
        section_results = await self._search_sections(query, k=3)
        
        # 提取相關條款前綴
        relevant_clauses = []
        for sr in section_results:
            clause_id = sr.chunk.metadata.get("clause_id", "")
            if clause_id:
                parts = clause_id.split(".")
                for i in range(len(parts)):
                    prefix = ".".join(parts[:i+1])
                    if prefix not in relevant_clauses:
                        relevant_clauses.append(prefix)
        
        # Stage 2: 在 Rule Index 中查找，過濾相關條款
        if relevant_clauses:
            rule_results = await self._search_rules_with_filter(
                query, 
                clause_filter=relevant_clauses,
                k=k
            )
            results.extend(rule_results)
        
        # Fallback: 如果結果不夠，添加直接語義搜索
        if len(results) < k:
            fallback = await self.semantic_search(query, k=k - len(results))
            results.extend(fallback)
    
    return self._deduplicate_by_clause(results)

async def _search_sections(self, query: str, k: int) -> List[RetrievalResult]:
    """搜索章節索引"""
    section_store = self.vector_store.get_section_store()
    results = await section_store.hybrid_search(query, k=k)
    return results

async def _search_rules_with_filter(
    self, 
    query: str, 
    clause_filter: List[str],
    k: int
) -> List[RetrievalResult]:
    """使用條款過濾器搜索規則索引"""
    rule_store = self.vector_store.get_rule_store()
    results = await rule_store.hybrid_search(
        query, 
        k=k,
        filter={"clause_id": {"$in": clause_filter}}
    )
    return results

def _deduplicate_by_clause(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
    """根據條款 ID 去重"""
    seen_clauses = set()
    unique_results = []
    
    for result in results:
        clause_id = result.chunk.metadata.get("clause_id", "")
        if clause_id not in seen_clauses:
            seen_clauses.add(clause_id)
            unique_results.append(result)
    
    return unique_results
```

### Step 2: 驗證更改

Run: `python -c "from src.retrieval.hybrid_retriever import HybridRetriever; print('Import OK')"`

Expected: Import successful

---

## 任務 6: 編寫單元測試

### Files:
- Create: `tests/test_dual_index.py`
- Create: `tests/test_graph_store.py`

### Step 1: 測試 GraphStore

創建 `tests/test_graph_store.py`：

```python
import pytest
import os
import tempfile
from src.retrieval.graph_store import GraphStore


@pytest.fixture
def temp_graph_store():
    """創建臨時 GraphStore"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    store = GraphStore(storage_path=temp_path)
    yield store
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_add_edge(temp_graph_store):
    """測試添加邊"""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    assert "chunk_1" in temp_graph_store.edges
    assert "chunk_2" in temp_graph_store.edges["chunk_1"]


def test_get_references(temp_graph_store):
    """測試獲取引用"""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_1", "chunk_3")
    refs = temp_graph_store.get_references("chunk_1")
    assert len(refs) == 2
    assert "chunk_2" in refs
    assert "chunk_3" in refs


def test_get_referenced_by(temp_graph_store):
    """測試獲取被引用"""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_3", "chunk_2")
    refs = temp_graph_store.get_referenced_by("chunk_2")
    assert len(refs) == 2
    assert "chunk_1" in refs
    assert "chunk_3" in refs


def test_get_stats(temp_graph_store):
    """測試統計"""
    temp_graph_store.add_edge("chunk_1", "chunk_2")
    temp_graph_store.add_edge("chunk_1", "chunk_3")
    stats = temp_graph_store.get_stats()
    assert stats["total_nodes"] == 1
    assert stats["total_edges"] == 2
```

### Step 2: 測試查詢路由

創建 `tests/test_query_routing.py`：

```python
import pytest
from src.retrieval.query_router import QueryRouter, QueryType, RetrievalStrategy


@pytest.fixture
def router():
    return QueryRouter()


def test_comprehensive_design_query(router):
    """測試設計查詢分類"""
    result = router.route_query("How to design a concrete beam?")
    assert result["type"] == QueryType.COMPREHENSIVE_DESIGN
    assert result["strategy"] == RetrievalStrategy.HIERARCHICAL_GRAPH
    assert "beam" in result["elements"]


def test_definition_query(router):
    """測試定義查詢分類"""
    result = router.route_query("What is effective depth?")
    assert result["type"] == QueryType.DEFINITION
    assert result["strategy"] == RetrievalStrategy.ATOMIC_LOOKUP


def test_data_lookup_query(router):
    """測試數據查詢分類"""
    result = router.route_query("What is the φ value for tension-controlled?")
    assert result["type"] == QueryType.DATA_LOOKUP
    assert result["strategy"] == RetrievalStrategy.TABLE_FIRST


def test_extract_elements(router):
    """測試工程元素提取"""
    result = router.route_query("How to design a pile foundation?")
    assert "pile" in result["elements"]
    assert "foundation" in result["elements"]
```

### Step 3: 運行測試

Run: `pytest tests/test_graph_store.py tests/test_query_routing.py -v`

Expected: All tests pass

---

## 完成標準

所有任務完成後，系統應支持：
1. 雙索引檢索（Section + Rule）
2. 階層式查詢路由
3. 圖形邊持久化存儲
4. 引用來源完整保留
5. 單元測試覆蓋核心功能
