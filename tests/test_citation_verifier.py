"""
Tests for citation verification system.
Tests citation extraction, verification, and hallucination detection.
"""
import pytest
from src.utils.citation_verifier import (
    CitationExtractor,
    CitationVerifier,
    Citation,
    VerificationReport,
    extract_citations,
    verify_citations,
)


class TestCitationExtractor:
    """Tests for citation extraction."""
    
    def test_extract_full_format_citation(self):
        """Test extraction of full format citations."""
        extractor = CitationExtractor()
        text = """Based on the code:
[Source: HK Code of Practice for Foundations 2017, Section/Clause: 6.1.5, Page: 45]
The bearing capacity shall be calculated."""
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        assert citations[0].document == "HK Code of Practice for Foundations 2017"
        assert citations[0].clause_id == "6.1.5"
        assert citations[0].page == 45
    
    def test_extract_short_format_citation(self):
        """Test extraction of short format citations."""
        extractor = CitationExtractor()
        text = """As specified in:
[Source: Geoguide 1, Clause: 6.2]
Retaining walls should be designed."""
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        assert citations[0].document == "Geoguide 1"
        assert citations[0].clause_id == "6.2"
        assert citations[0].page is None
    
    def test_extract_parenthetical_citation(self):
        """Test extraction of parenthetical citations."""
        extractor = CitationExtractor()
        text = """The factor of safety shall be 3.0 (HK CoP 2017, Clause 6.1.5)."""
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        assert citations[0].document == "HK CoP 2017"
        assert citations[0].clause_id == "6.1.5"
    
    def test_extract_according_to_citation(self):
        """Test extraction of 'According to' format citations."""
        extractor = CitationExtractor()
        text = """According to Clause 6.1.5 of HK Code of Practice,
the bearing capacity shall be calculated."""
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        assert citations[0].clause_id == "6.1.5"
        assert "HK Code" in citations[0].document
    
    def test_extract_standalone_clause(self):
        """Test extraction of standalone clause references."""
        extractor = CitationExtractor()
        text = """As required by Clause 6.1.5.2, the design shall consider..."""
        
        citations = extractor.extract(text)
        
        assert len(citations) >= 1
        assert any(c.clause_id == "6.1.5.2" for c in citations)
    
    def test_extract_multiple_citations(self):
        """Test extraction of multiple citations."""
        extractor = CitationExtractor()
        text = """
[Source: HK Code of Practice, Section/Clause: 6.1.5, Page: 45]
[Source: Geoguide 1, Section/Clause: 6.2, Page: 120]
"""
        
        citations = extractor.extract(text)
        
        assert len(citations) == 2
        documents = [c.document for c in citations]
        assert any("HK Code" in d for d in documents)
        assert any("Geoguide" in d for d in documents)
    
    def test_extract_no_citations(self):
        """Test extraction when no citations present."""
        extractor = CitationExtractor()
        text = """This is a plain answer without any citations."""
        
        citations = extractor.extract(text)
        
        assert citations == []
    
    def test_extract_complex_text(self):
        """Test extraction from complex technical text."""
        extractor = CitationExtractor()
        text = """
Based on the HK Code of Practice for Foundations 2017:

### Step 1: Bearing Capacity
[Source: HK Code of Practice for Foundations 2017, Section/Clause: 6.1.5, Page: 45]
The ultimate bearing capacity is calculated as...

### Step 2: Settlement
According to Clause 6.3.2 of the same code, settlement should not exceed 25mm.

Additionally, Geoguide 1 (Clause 6.5) provides guidance on retaining walls.
"""
        
        citations = extractor.extract(text)
        
        assert len(citations) >= 2
        clause_ids = [c.clause_id for c in citations]
        assert "6.1.5" in clause_ids
        assert "6.3.2" in clause_ids or "6.5" in clause_ids


class TestCitationVerifier:
    """Tests for citation verification."""
    
    def test_verify_valid_citation(self):
        """Test verification of valid citation."""
        verifier = CitationVerifier()
        answer = "[Source: Test Document, Section/Clause: 6.1.5, Page: 45]"
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "Test Document",
                    "clause_id": "6.1.5",
                    "section_title": "Bearing Capacity",
                    "page_number": 45,
                },
                "text": "Bearing capacity requirements...",
                "score": 0.9,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        assert report.total_citations == 1
        assert report.verified_count == 1
        assert report.hallucinated_count == 0
        assert report.overall_confidence >= 0.9
    
    def test_verify_hallucinated_citation(self):
        """Test detection of hallucinated citation."""
        verifier = CitationVerifier()
        answer = "[Source: Nonexistent Document, Section/Clause: 9.9.9, Page: 999]"
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "Test Document",
                    "clause_id": "6.1.5",
                    "section_title": "Bearing Capacity",
                    "page_number": 45,
                },
                "text": "Different content...",
                "score": 0.9,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        assert report.total_citations == 1
        assert report.hallucinated_count >= 1
        assert report.overall_confidence < 0.5
    
    def test_verify_partial_match(self):
        """Test verification with partial document match."""
        verifier = CitationVerifier()
        answer = "[Source: HK CoP, Section/Clause: 6.1.5, Page: 45]"
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "HK Code of Practice for Foundations 2017",
                    "clause_id": "6.1.5",
                    "section_title": "Bearing Capacity",
                    "page_number": 45,
                },
                "text": "Bearing capacity...",
                "score": 0.9,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        # Should recognize HK CoP as abbreviation
        assert report.verified_count >= 1
        assert report.overall_confidence >= 0.8
    
    def test_verify_parent_clause_match(self):
        """Test verification with parent clause match."""
        verifier = CitationVerifier()
        answer = "[Source: Test Doc, Section/Clause: 6.1, Page: 40]"
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "Test Doc",
                    "clause_id": "6.1.5",  # Child clause
                    "section_title": "Design",
                    "page_number": 40,
                },
                "text": "Design requirements...",
                "score": 0.85,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        # Parent clause 6.1 should match child 6.1.5
        assert report.verified_count >= 1
        assert report.overall_confidence >= 0.5
    
    def test_verify_empty_chunks(self):
        """Test verification with no retrieved chunks."""
        verifier = CitationVerifier()
        answer = "[Source: Test Doc, Section/Clause: 6.1.5, Page: 45]"
        
        report = verifier.verify(answer, [])
        
        assert report.total_citations == 1
        assert report.hallucinated_count == 1
        assert report.overall_confidence <= 0.3
    
    def test_verify_no_citations(self):
        """Test verification when answer has no citations."""
        verifier = CitationVerifier()
        answer = "This is a general answer without specific citations."
        chunks = [{"id": "chunk1", "metadata": {}, "text": "Some text", "score": 0.8}]
        
        report = verifier.verify(answer, chunks)
        
        assert report.total_citations == 0
        assert report.verified_count == 0
        assert report.overall_confidence == 1.0  # No citations to verify
    
    def test_verify_multiple_citations_mixed(self):
        """Test verification of multiple citations with mixed results."""
        verifier = CitationVerifier()
        answer = """
[Source: Valid Doc, Section/Clause: 6.1.5, Page: 45]
[Source: Fake Doc, Section/Clause: 9.9.9, Page: 999]
"""
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "Valid Doc",
                    "clause_id": "6.1.5",
                    "section_title": "Design",
                    "page_number": 45,
                },
                "text": "Valid content...",
                "score": 0.9,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        assert report.total_citations == 2
        assert report.verified_count == 1
        assert report.hallucinated_count == 1


class TestVerificationReport:
    """Tests for verification report structure."""
    
    def test_report_structure(self):
        """Test verification report has correct structure."""
        verifier = CitationVerifier()
        answer = "[Source: Test, Section/Clause: 1.0, Page: 1]"
        chunks = [
            {
                "id": "chunk1",
                "metadata": {
                    "document_name": "Test",
                    "clause_id": "1.0",
                    "section_title": "Intro",
                    "page_number": 1,
                },
                "text": "Content",
                "score": 0.9,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        assert hasattr(report, 'total_citations')
        assert hasattr(report, 'verified_count')
        assert hasattr(report, 'unverified_count')
        assert hasattr(report, 'hallucinated_count')
        assert hasattr(report, 'results')
        assert hasattr(report, 'overall_confidence')
        assert hasattr(report, 'warnings')
    
    def test_report_warnings(self):
        """Test report generates warnings for unverified citations."""
        verifier = CitationVerifier()
        answer = "[Source: Unknown, Section/Clause: 9.9.9, Page: 999]"
        
        report = verifier.verify(answer, [], strict_mode=True)
        
        assert len(report.warnings) > 0
        assert any("9.9.9" in w for w in report.warnings)


class TestStandaloneFunctions:
    """Tests for standalone helper functions."""
    
    def test_extract_citations_function(self):
        """Test standalone extract_citations function."""
        text = "[Source: Test Doc, Section/Clause: 1.0, Page: 1]"
        citations = extract_citations(text)
        
        assert len(citations) == 1
        assert citations[0].clause_id == "1.0"
    
    def test_verify_citations_function(self):
        """Test standalone verify_citations function."""
        answer = "[Source: Test, Section/Clause: 1.0]"
        chunks = [
            {
                "id": "c1",
                "metadata": {"document_name": "Test", "clause_id": "1.0"},
                "text": "Content",
                "score": 0.8,
            }
        ]
        
        report = verify_citations(answer, chunks)
        
        assert isinstance(report, VerificationReport)
        assert report.verified_count >= 1


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_malformed_citation(self):
        """Test handling of malformed citations."""
        extractor = CitationExtractor()
        text = "[Source: Incomplete citation without proper format"
        
        citations = extractor.extract(text)
        # Should not crash, may return empty or partial results
        assert isinstance(citations, list)
    
    def test_unicode_in_citation(self):
        """Test handling of unicode characters."""
        extractor = CitationExtractor()
        text = "[Source: 香港規範，Section/Clause: 6.1.5, Page: 45]"
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        assert "香港" in citations[0].document
    
    def test_special_characters_in_clause(self):
        """Test handling of special characters in clause IDs."""
        extractor = CitationExtractor()
        text = "[Source: Test, Section/Clause: 6.1.5(a), Page: 45]"
        
        citations = extractor.extract(text)
        
        assert len(citations) == 1
        # Should extract the clause ID even with suffix
        assert "6.1.5" in citations[0].clause_id
    
    def test_case_insensitive_matching(self):
        """Test case-insensitive document matching."""
        verifier = CitationVerifier()
        answer = "[Source: test document, Section/Clause: 1.0]"
        chunks = [
            {
                "id": "c1",
                "metadata": {
                    "document_name": "TEST DOCUMENT",
                    "clause_id": "1.0",
                },
                "text": "Content",
                "score": 0.8,
            }
        ]
        
        report = verifier.verify(answer, chunks)
        
        assert report.verified_count >= 1
