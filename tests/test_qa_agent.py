"""
Tests for RAG retrieval and QA Agent functionality.
Tests semantic search, filtering, and answer generation.
"""
import pytest
from unittest.mock import patch, MagicMock

from src.agents.qa_agent import QAAgent
from src.vectordb.qdrant_store import GeoVectorStore


class TestQAAgentInitialization:
    """Tests for QA Agent initialization."""
    
    def test_agent_initialization(self):
        """Test agent initializes with vector store."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            
            agent = QAAgent(use_local_db=False)
            
            assert agent.store is not None
            assert agent.conversation_history == []
    
    def test_agent_with_local_db(self):
        """Test agent initialization with local DB flag."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            
            agent = QAAgent(use_local_db=True)
            
            MockStore.assert_called_once_with(use_local=True)


class TestSemanticSearch:
    """Tests for semantic search functionality."""
    
    def test_search_returns_results(self, sample_retrieved_chunks):
        """Test search returns relevant chunks."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="bearing capacity requirements",
                top_k=5,
                document_type="code",
            )
            
            assert len(results) > 0
            assert "text" in results[0]
            assert "metadata" in results[0]
            assert "score" in results[0]
    
    def test_search_with_document_filter(self):
        """Test search with document ID filter."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {
                    "text": "Filtered result",
                    "metadata": {"document_id": "hk_cop_2017"},
                    "score": 0.9,
                }
            ]
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="foundation depth",
                document_id="hk_cop_2017",
                top_k=3,
            )
            
            assert len(results) > 0
            assert results[0]["metadata"]["document_id"] == "hk_cop_2017"
    
    def test_search_with_clause_filter(self):
        """Test search with clause ID filter."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {
                    "text": "Clause 6.1.5 content",
                    "metadata": {"clause_id": "6.1.5"},
                    "score": 0.85,
                }
            ]
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="bearing capacity",
                clause_id="6.1.5",
                top_k=3,
            )
            
            assert len(results) > 0
    
    def test_search_with_content_type_filter(self):
        """Test search with content type filter."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {
                    "text": "Table content",
                    "metadata": {"content_type": "table"},
                    "score": 0.75,
                }
            ]
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="concrete strength table",
                content_type="table",
                top_k=3,
            )
            
            assert len(results) > 0
            assert results[0]["metadata"]["content_type"] == "table"
    
    def test_search_with_regulatory_strength_filter(self):
        """Test search with regulatory strength filter."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {
                    "text": "Shall be designed...",
                    "metadata": {"regulatory_strength": "mandatory"},
                    "score": 0.88,
                }
            ]
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="mandatory requirements",
                regulatory_strength="mandatory",
                top_k=3,
            )
            
            assert len(results) > 0
    
    def test_search_with_score_threshold(self):
        """Test search respects score threshold."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            # Return only high-scoring results
            mock_store.search.return_value = [
                {"text": "High score result", "metadata": {}, "score": 0.85},
                {"text": "Another high score", "metadata": {}, "score": 0.72},
            ]
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="test query",
                top_k=10,
                score_threshold=0.7,
            )
            
            # All results should meet threshold
            assert all(r["score"] >= 0.7 for r in results)
    
    def test_search_no_results(self):
        """Test search handles no results."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = []
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            results = agent.store.search(
                query="nonexistent topic",
                top_k=5,
            )
            
            assert results == []


class TestQAAgentAsk:
    """Tests for QA agent question answering."""
    
    def test_ask_returns_answer_with_sources(self, sample_retrieved_chunks):
        """Test ask returns answer with source citations."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = """Based on the HK Code of Practice:

### Step 1: Bearing Capacity Check
**Critical equation**
$$
q_{ult} = cN_c + qN_q + 0.5\gamma B N_\gamma
$$

[Source: HK Code of Practice for Foundations 2017, Section 6.1.5, Page 45]
"""
                
                agent = QAAgent()
                result = agent.ask("What is the bearing capacity requirement?")
                
                assert "answer" in result
                assert "sources" in result
                assert len(result["sources"]) > 0
                assert "document" in result["sources"][0]
                assert "section" in result["sources"][0]
    
    def test_ask_handles_no_results(self):
        """Test ask handles no search results."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = []
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            result = agent.ask("Question about nonexistent topic")
            
            assert "answer" in result
            assert "No relevant information found" in result["answer"]
            assert result["sources"] == []
    
    def test_ask_handles_llm_error(self, sample_retrieved_chunks):
        """Test ask handles LLM errors gracefully."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.side_effect = Exception("LLM API error")
                
                agent = QAAgent()
                result = agent.ask("Test question")
                
                # Should return retrieved context instead of crashing
                assert "answer" in result
                assert "Retrieved relevant context" in result["answer"] or "error" in result["answer"].lower()
    
    def test_ask_equation_mode(self, sample_retrieved_chunks):
        """Test ask with equation mode enabled."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = """
### Step 1: Design Check
**Critical equation**
$$
M_{Rd} = f_y \\cdot Z
$$

**Variable definitions (units)**
- $f_y$: Yield strength (MPa)
- $Z$: Section modulus (mm³)

[Source: Test Doc, Section 1.0, Page 1]
"""
                
                agent = QAAgent()
                result = agent.ask(
                    "What is the design equation?",
                    equation_mode=True,
                )
                
                # Equation mode should normalize LaTeX
                assert "$$" in result["answer"] or "$" in result["answer"]
    
    def test_ask_stores_conversation_history(self, sample_retrieved_chunks):
        """Test ask stores conversation in history."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = "Test answer"
                
                agent = QAAgent()
                result = agent.ask("First question")
                
                assert len(agent.conversation_history) == 1
                assert agent.conversation_history[0]["question"] == "First question"
    
    def test_ask_followup_uses_context(self, sample_retrieved_chunks):
        """Test follow-up question uses previous context."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = "Answer to follow-up"
                
                agent = QAAgent()
                # First question
                agent.ask("Initial question")
                
                # Follow-up
                result = agent.ask_followup("What about the second point?")
                
                assert "answer" in result
                # LLM should have been called twice
                assert mock_llm.call_count == 2


class TestKnowledgeBaseManagement:
    """Tests for knowledge base management."""
    
    def test_list_knowledge_base(self):
        """Test listing documents in knowledge base."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = [
                {
                    "document_id": "hk_cop_2017",
                    "document_name": "HK Code of Practice 2017",
                    "document_type": "code",
                    "chunk_count": 150,
                },
                {
                    "document_id": "geoguide_1",
                    "document_name": "Geoguide 1",
                    "document_type": "manual",
                    "chunk_count": 80,
                },
            ]
            
            agent = QAAgent()
            docs = agent.list_knowledge_base()
            
            assert len(docs) == 2
            assert docs[0]["document_id"] == "hk_cop_2017"
            assert docs[1]["document_type"] == "manual"
    
    def test_list_empty_knowledge_base(self):
        """Test listing when knowledge base is empty."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            
            agent = QAAgent()
            docs = agent.list_knowledge_base()
            
            assert docs == []


class TestChunkMetadata:
    """Tests for chunk metadata handling."""
    
    def test_source_extraction_from_chunks(self, sample_retrieved_chunks):
        """Test source extraction from retrieved chunks."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = sample_retrieved_chunks
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = "Test answer"
                
                agent = QAAgent()
                result = agent.ask("Test question")
                
                sources = result["sources"]
                assert len(sources) > 0
                assert sources[0]["document"] == "HK Code of Practice for Foundations 2017"
                assert sources[0]["section"] == "Bearing Capacity"
                assert sources[0]["page"] == 45
    
    def test_duplicate_source_handling(self):
        """Test duplicate sources are deduplicated."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            # Return chunks from same source
            mock_store.search.return_value = [
                {
                    "text": "Chunk 1",
                    "metadata": {
                        "document_name": "Same Doc",
                        "section_title": "Same Section",
                        "page_no": 1,
                    },
                    "score": 0.9,
                },
                {
                    "text": "Chunk 2",
                    "metadata": {
                        "document_name": "Same Doc",
                        "section_title": "Same Section",
                        "page_no": 1,
                    },
                    "score": 0.8,
                },
            ]
            mock_store.list_documents.return_value = []
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = "Test answer"
                
                agent = QAAgent()
                result = agent.ask("Test question")
                
                # Should deduplicate to single source
                source_keys = [(s["document"], s["section"]) for s in result["sources"]]
                assert len(source_keys) == len(set(source_keys))
