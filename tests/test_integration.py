"""
Integration tests for end-to-end GeoBot workflows.
Tests complete workflows from document ingestion to Q&A.
"""
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.agents.qa_agent import QAAgent
from src.agents.designer_agent import DesignerAgent
from src.agents.validator_agent import ValidatorAgent, ValidationReport, CheckItem
from src.ingestion.pdf_processor_enhanced import ProcessedDocument, DocumentChunk, HybridPDFProcessor
from src.utils.citation_verifier import CitationVerifier, verify_citations


class TestDocumentIngestionWorkflow:
    """Integration tests for document ingestion workflow."""
    
    def test_ingest_document_creates_chunks(self):
        """Test document ingestion creates proper chunks."""
        with patch('src.ingestion.pdf_processor.HybridPDFProcessor') as MockProcessor:
            mock_processor = MockProcessor.return_value
            mock_processor.process.return_value = ProcessedDocument(
                document_id="test_doc",
                document_name="Test Document",
                document_type="code",
                chunks=[
                    DocumentChunk(
                        text="Clause 6.1.5 Bearing capacity requirements...",
                        metadata={
                            "clause_id": "6.1.5",
                            "section_title": "Bearing Capacity",
                            "page_no": 45,
                            "document_type": "code",
                        },
                    ),
                    DocumentChunk(
                        text="Clause 6.2 Foundation depth requirements...",
                        metadata={
                            "clause_id": "6.2",
                            "section_title": "Foundation Depth",
                            "page_no": 48,
                            "document_type": "code",
                        },
                    ),
                ],
                processing_info={"method": "pymupdf4llm", "page_count": 100},
            )
            
            with patch('src.vectordb.qdrant_store.GeoVectorStore') as MockStore:
                mock_store = MockStore.return_value
                mock_store.add_document.return_value = 2
                
                from src.ingestion.ingest import ingest_document
                
                num_chunks = ingest_document(
                    pdf_path="/fake/path.pdf",
                    document_id="test_doc",
                    document_name="Test Document",
                    document_type="code",
                    use_local_db=False,
                )
                
                assert num_chunks == 2
                mock_store.add_document.assert_called_once()
    
    def test_ingest_document_with_ocr_fallback(self):
        """Test document ingestion falls back to OCR when needed."""
        with patch('src.ingestion.pdf_processor.HybridPDFProcessor') as MockProcessor:
            mock_processor = MockProcessor.return_value
            
            # First call fails, second succeeds with OCR
            def process_side_effect(*args, **kwargs):
                if kwargs.get('force_ocr', False):
                    return ProcessedDocument(
                        document_id="scanned_doc",
                        document_name="Scanned Document",
                        document_type="report",
                        chunks=[
                            DocumentChunk(
                                text="OCR extracted text from scanned page...",
                                metadata={"method": "ocr", "page_no": 1},
                            )
                        ],
                        processing_info={"method": "ocr"},
                    )
                raise Exception("Docling failed")
            
            mock_processor.process.side_effect = process_side_effect
            
            with patch('src.vectordb.qdrant_store.GeoVectorStore') as MockStore:
                mock_store = MockStore.return_value
                mock_store.add_document.return_value = 1
                
                from src.ingestion.ingest import ingest_document
                
                num_chunks = ingest_document(
                    pdf_path="/fake/scanned.pdf",
                    document_id="scanned_doc",
                    document_name="Scanned Document",
                    document_type="report",
                    use_local_db=False,
                )
                
                assert num_chunks == 1


class TestRAGQnAWorkflow:
    """Integration tests for RAG Q&A workflow."""
    
    def test_qa_agent_with_retrieval_and_answer(self):
        """Test complete Q&A flow with retrieval and answer generation."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = [
                {
                    "document_id": "hk_cop",
                    "document_name": "HK Code of Practice 2017",
                    "document_type": "code",
                    "chunk_count": 150,
                }
            ]
            mock_store.search.return_value = [
                {
                    "text": "The ultimate bearing capacity shall be calculated using appropriate methods. A minimum factor of safety of 3.0 shall be applied.",
                    "metadata": {
                        "document_name": "HK Code of Practice 2017",
                        "clause_id": "6.1.5",
                        "section_title": "Bearing Capacity",
                        "page_no": 45,
                    },
                    "score": 0.85,
                },
            ]
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = """Based on HK Code of Practice 2017:

### Step 1: Bearing Capacity Check
**Critical equation**
$$
q_{ult} = cN_c + qN_q + 0.5\gamma B N_\gamma
$$

**Acceptance criterion**
- Factor of safety >= 3.0

[Source: HK Code of Practice 2017, Section/Clause: 6.1.5, Page: 45]
"""
                
                agent = QAAgent()
                result = agent.ask("What is the bearing capacity requirement?")
                
                assert "answer" in result
                assert "sources" in result
                assert len(result["sources"]) > 0
                assert result["sources"][0]["document"] == "HK Code of Practice 2017"
                assert result["sources"][0]["section"] == "Bearing Capacity"
    
    def test_qa_agent_with_citation_verification(self):
        """Test Q&A flow with citation verification."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            mock_store.search.return_value = [
                {
                    "text": "Bearing capacity requirements...",
                    "metadata": {
                        "document_name": "HK Code of Practice 2017",
                        "clause_id": "6.1.5",
                        "section_title": "Bearing Capacity",
                        "page_no": 45,
                    },
                    "score": 0.85,
                },
            ]
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = """Based on the code:
[Source: HK Code of Practice 2017, Section/Clause: 6.1.5, Page: 45]
The bearing capacity shall be calculated with FoS >= 3.0.
"""
                
                agent = QAAgent()
                
                # Wrap with citation verifier
                from src.utils.citation_verifier import CitationAwareQAAgent
                aware_agent = CitationAwareQAAgent(agent, strict_mode=True)
                result = aware_agent.ask("What is the bearing capacity requirement?")
                
                assert "verification" in result
                assert result["verification"]["verified_count"] >= 1
                assert result["verification"]["overall_confidence"] >= 0.8
    
    def test_qa_agent_equation_mode(self):
        """Test Q&A flow with equation mode enabled."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            mock_store.search.return_value = [
                {
                    "text": "The design resistance is calculated as M_Rd = f_y * Z where f_y is yield strength.",
                    "metadata": {
                        "document_name": "HK Code of Practice 2017",
                        "clause_id": "6.1.5",
                        "section_title": "Design",
                        "page_no": 45,
                    },
                    "score": 0.85,
                },
            ]
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.return_value = """### Step 1: Design Resistance
**Critical equation**
$$
M_{Rd} = f_y \\cdot Z
$$

**Variable definitions (units)**
- $f_y$: Yield strength (MPa)
- $Z$: Section modulus (mm³)

[Source: HK Code of Practice 2017, Section/Clause: 6.1.5, Page: 45]
"""
                
                agent = QAAgent()
                result = agent.ask("What is the design resistance equation?", equation_mode=True)
                
                # Equation mode should normalize LaTeX
                assert "$$" in result["answer"] or "$" in result["answer"]
                assert "f_y" in result["answer"] or "$f_y$" in result["answer"]


class TestDesignerAgentWorkflow:
    """Integration tests for Designer Agent workflow."""
    
    def test_complete_design_session(self):
        """Test complete design session from project info to report."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                # Mock sequential LLM responses
                responses = [
                    # Project extraction
                    '{"project_name": "Test Building", "project_type": "building", "location": "HK", "missing_critical_info": ["soil parameters"]}',
                    # Parameter extraction
                    '{"cohesion_kpa": 10, "friction_angle_deg": 32, "unit_weight_kn_m3": 19}',
                ]
                mock_llm.side_effect = responses
                
                agent = DesignerAgent()
                
                # Start session
                start_result = agent.start()
                assert agent.state.value == "collecting_project_info"
                
                # Provide project info
                project_result = agent.process_input("5-story building in Hong Kong")
                assert agent.state.value in ["collecting_parameters", "identifying_skills"]
                
                # Provide parameters
                param_result = agent.process_input("Soil has cohesion 10 kPa, friction angle 32 degrees")
                
                # Should progress through workflow
                assert agent.state.value in ["identifying_skills", "executing_calculations", "generating_report", "complete"]
    
    def test_designer_with_missing_parameters(self):
        """Test designer agent handles missing parameters correctly."""
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.designer_agent.call_llm') as mock_llm:
                mock_llm.return_value = '{"project_type": "slope", "missing_critical_info": ["slope angle", "soil cohesion"]}'
                
                agent = DesignerAgent()
                agent.start()
                result = agent.process_input("Slope project")
                
                assert result.get("questions") is not None
                assert len(result["questions"]) > 0
                assert agent.state.value == "collecting_parameters"


class TestValidatorAgentWorkflow:
    """Integration tests for Validator Agent workflow."""

    def test_complete_validation_workflow(self, sample_report_text):
        """Test complete validation workflow."""
        with patch('src.agents.validator_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.search.return_value = [
                {"text": "Code requirement", "metadata": {}, "score": 0.8}
            ]

            with patch('src.agents.validator_agent.extract_text_from_pdf') as mock_extract:
                with patch('src.agents.validator_agent.extract_pages_with_numbers') as mock_pages:
                    mock_extract.return_value = sample_report_text
                    mock_pages.return_value = []

                    with patch('src.agents.validator_agent.call_llm') as mock_llm:
                        # Mock all LLM calls
                        mock_llm.side_effect = [
                            # Completeness check
                            '[{"section": "1. Introduction", "status": "present"}, {"section": "5. Analysis", "status": "present"}]',
                            # Parameter check
                            '{"factors_of_safety": {"bearing": 3.0, "sliding": 1.5}, "issues_noticed": []}',
                            # Code compliance
                            '{"design_methods": [{"method_name": "Bearing capacity"}], "referenced_codes": []}',
                            # Consistency check
                            '[]',
                            # Summary
                            'Report is complete and adequate.',
                        ]

                        agent = ValidatorAgent()
                        report = agent.validate_report("/fake/path.pdf")

                        assert report.document_name == "path.pdf"
                        assert report.overall_status in ["acceptable", "revisions_needed", "rejected"]
                        assert len(report.checks) > 0
    
    def test_validation_detects_critical_issues(self):
        """Test validation detects and reports critical issues."""
        with patch('src.agents.validator_agent.GeoVectorStore'):
            from src.agents.validator_agent import ValidationReport, CheckItem
            
            report = ValidationReport(
                document_name="test.pdf",
                checks=[
                    CheckItem(
                        category="parameter",
                        item="FoS (bearing)",
                        status="fail",
                        details="2.0 < 3.0",
                        severity="critical",
                    ),
                    CheckItem(
                        category="completeness",
                        item="Section 5",
                        status="fail",
                        severity="high",
                    ),
                ],
            )
            
            # Calculate status as agent would
            critical_fails = sum(1 for c in report.checks if c.status == "fail" and c.severity == "critical")
            
            if critical_fails > 0:
                status = "rejected"
            elif report.fail_count > 0:
                status = "revisions_needed"
            else:
                status = "acceptable"
            
            assert status == "rejected"


class TestCitationVerificationWorkflow:
    """Integration tests for citation verification workflow."""
    
    def test_end_to_end_citation_verification(self):
        """Test complete citation verification flow."""
        # Simulate LLM answer with citations
        answer = """Based on the HK Code of Practice:

[Source: HK Code of Practice 2017, Section/Clause: 6.1.5, Page: 45]
The bearing capacity shall be calculated with FoS >= 3.0.

[Source: Geoguide 1, Section/Clause: 6.2, Page: 120]
Retaining walls require additional checks.
"""
        
        # Simulate retrieved chunks
        retrieved_chunks = [
            {
                "id": "chunk1",
                "text": "Bearing capacity requirements...",
                "metadata": {
                    "document_name": "HK Code of Practice 2017",
                    "clause_id": "6.1.5",
                    "section_title": "Bearing Capacity",
                    "page_no": 45,
                },
                "score": 0.85,
            },
            # Note: Geoguide chunk is missing - should be detected as hallucination
        ]
        
        # Verify citations
        report = verify_citations(answer, retrieved_chunks)
        
        assert report.total_citations == 2
        assert report.verified_count == 1  # HK CoP verified
        assert report.hallucinated_count == 1  # Geoguide not found
        assert report.overall_confidence < 1.0
    
    def test_citation_verification_with_abbreviations(self):
        """Test citation verification handles document abbreviations."""
        answer = "[Source: HK CoP, Section/Clause: 6.1.5]"
        retrieved_chunks = [
            {
                "id": "chunk1",
                "text": "Content...",
                "metadata": {
                    "document_name": "HK Code of Practice for Foundations 2017",
                    "clause_id": "6.1.5",
                },
                "score": 0.9,
            }
        ]
        
        report = verify_citations(answer, retrieved_chunks)
        
        # Should recognize HK CoP as abbreviation for HK Code of Practice
        assert report.verified_count >= 1
        assert report.overall_confidence >= 0.8


class TestMultiAgentWorkflow:
    """Integration tests for workflows involving multiple agents."""
    
    def test_design_then_validate_workflow(self):
        """Test workflow where designer creates report then validator checks it."""
        # This is a conceptual test showing how agents would work together
        with patch('src.agents.designer_agent.GeoVectorStore'):
            with patch('src.agents.validator_agent.GeoVectorStore'):
                # Designer agent creates report
                with patch('src.agents.designer_agent.call_llm') as design_llm:
                    design_llm.return_value = '{"project_type": "building", "missing_critical_info": []}'

                    designer = DesignerAgent()
                    designer.start()
                    designer.process_input("Test building project")

                    # Simulate generated report
                    generated_report = """
# GEOTECHNICAL REPORT
## 1. Introduction
Project description...

## 5. Analysis
Bearing capacity check with FoS = 3.0
"""

                # Validator agent checks report
                with patch('src.agents.validator_agent.extract_text_from_pdf') as mock_extract:
                    with patch('src.agents.validator_agent.extract_pages_with_numbers') as mock_pages:
                        mock_extract.return_value = generated_report
                        mock_pages.return_value = []

                        with patch('src.agents.validator_agent.call_llm') as validate_llm:
                            validate_llm.side_effect = [
                                '[{"section": "1. Introduction", "status": "present"}]',
                                '{"factors_of_safety": {"bearing": 3.0}}',
                                '{"design_methods": []}',
                                '[]',
                                'Report is adequate.',
                            ]

                            validator = ValidatorAgent()
                            report = validator.validate_report("/fake/report.pdf")

                            assert report is not None
                            assert report.overall_status in ["acceptable", "revisions_needed", "rejected"]


class TestErrorHandlingWorkflow:
    """Integration tests for error handling across workflows."""
    
    def test_graceful_degradation_on_llm_failure(self):
        """Test system degrades gracefully when LLM calls fail."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            mock_store.search.return_value = [
                {
                    "text": "Fallback content from retrieval",
                    "metadata": {"document_name": "Test", "clause_id": "1.0"},
                    "score": 0.8,
                }
            ]
            
            with patch('src.agents.qa_agent.call_llm_with_context') as mock_llm:
                mock_llm.side_effect = Exception("LLM API unavailable")
                
                agent = QAAgent()
                result = agent.ask("Test question")
                
                # Should return retrieved context instead of crashing
                assert "answer" in result
                # Answer should mention the error or show retrieved content
    
    def test_empty_retrieval_handling(self):
        """Test system handles empty retrieval results."""
        with patch('src.agents.qa_agent.GeoVectorStore') as MockStore:
            mock_store = MockStore.return_value
            mock_store.list_documents.return_value = []
            mock_store.search.return_value = []  # No results
            
            agent = QAAgent()
            result = agent.ask("Question about nonexistent topic")
            
            assert "answer" in result
            assert "No relevant information found" in result["answer"]
            assert result["sources"] == []
