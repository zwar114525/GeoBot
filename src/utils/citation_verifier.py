"""
Citation verification system to prevent LLM hallucination of clause numbers and references.
Validates that cited clauses exist in retrieved context before returning answers.
"""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from loguru import logger


@dataclass
class Citation:
    """Represents a citation extracted from text."""
    document: str
    clause_id: Optional[str]
    section: Optional[str]
    page: Optional[int]
    raw_text: str
    start_pos: int
    end_pos: int


@dataclass
class VerificationResult:
    """Result of citation verification."""
    citation: Citation
    verified: bool
    confidence: float
    matched_chunk_ids: List[str]
    notes: str = ""


@dataclass
class VerificationReport:
    """Complete verification report for an answer."""
    total_citations: int
    verified_count: int
    unverified_count: int
    hallucinated_count: int
    results: List[VerificationResult]
    overall_confidence: float
    warnings: List[str]


class CitationExtractor:
    """Extracts citations from LLM-generated text."""
    
    # Patterns for common citation formats
    PATTERNS = [
        # [Source: document name, Section/Clause: X.X.X, Page: N]
        re.compile(
            r'\[Source:\s*([^,\]]+),\s*(?:Section|Clause|Section/Clause):\s*([^\],]+),\s*Page:\s*(\d+)\]',
            re.IGNORECASE
        ),
        # [Source: document name, Section X.X.X]
        re.compile(
            r'\[Source:\s*([^,\]]+),\s*(?:Section|Clause):\s*([^\]]+)\]',
            re.IGNORECASE
        ),
        # (HK CoP 2017, Clause 6.1.5)
        re.compile(
            r'\(([^,]+),\s*(?:Clause|Section)\s*(\d+(?:\.\d+)*)\)',
            re.IGNORECASE
        ),
        # According to Clause 6.1.5 of HK CoP
        re.compile(
            r'(?:According to|See|Refer to)\s+(?:Clause|Section)\s*(\d+(?:\.\d+)*)\s+(?:of\s+)?([^,.]+)',
            re.IGNORECASE
        ),
        # Standalone clause references like "Clause 6.1.5"
        re.compile(
            r'Clause\s+(\d+(?:\.\d+)+)',
            re.IGNORECASE
        ),
    ]
    
    def extract(self, text: str) -> List[Citation]:
        """
        Extract all citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for i, pattern in enumerate(self.PATTERNS):
            for match in pattern.finditer(text):
                try:
                    if i == 0:  # Full format with page
                        document = match.group(1).strip()
                        clause_id = match.group(2).strip()
                        page = int(match.group(3))
                        section = None
                    elif i == 1:  # Document + clause
                        document = match.group(1).strip()
                        clause_id = match.group(2).strip()
                        page = None
                        section = None
                    elif i == 2:  # Parenthetical format
                        document = match.group(1).strip()
                        clause_id = match.group(2).strip()
                        page = None
                        section = None
                    elif i == 3:  # "According to" format
                        clause_id = match.group(1).strip()
                        document = match.group(2).strip()
                        page = None
                        section = None
                    else:  # Standalone clause
                        clause_id = match.group(1).strip()
                        document = "Unknown"
                        page = None
                        section = None
                    
                    citations.append(Citation(
                        document=document,
                        clause_id=clause_id,
                        section=section,
                        page=page,
                        raw_text=match.group(0),
                        start_pos=match.start(),
                        end_pos=match.end(),
                    ))
                except (IndexError, ValueError) as e:
                    logger.debug(f"Failed to parse citation match: {e}")
                    continue
        
        return citations


class CitationVerifier:
    """Verifies citations against retrieved context chunks."""
    
    def __init__(self):
        self.extractor = CitationExtractor()
        self.clause_pattern = re.compile(r'^(\d+(?:\.\d+)*)')
    
    def verify(
        self,
        answer: str,
        retrieved_chunks: List[Dict],
        strict_mode: bool = True,
    ) -> VerificationReport:
        """
        Verify all citations in an answer against retrieved chunks.
        
        Args:
            answer: LLM-generated answer text
            retrieved_chunks: List of chunks that were provided to LLM
            strict_mode: If True, mark unverified citations as hallucinated
            
        Returns:
            VerificationReport with verification results
        """
        citations = self.extractor.extract(answer)
        results = []
        warnings = []
        
        for citation in citations:
            result = self._verify_citation(citation, retrieved_chunks)
            results.append(result)
            
            if not result.verified and strict_mode:
                warnings.append(
                    f"Unverified citation: {citation.raw_text} "
                    f"(document: {citation.document}, clause: {citation.clause_id})"
                )
        
        verified_count = sum(1 for r in results if r.verified)
        unverified_count = sum(1 for r in results if not r.verified and r.confidence > 0.3)
        hallucinated_count = sum(1 for r in results if r.confidence <= 0.3)
        
        overall_confidence = (
            sum(r.confidence for r in results) / len(results)
            if results else 1.0
        )
        
        return VerificationReport(
            total_citations=len(citations),
            verified_count=verified_count,
            unverified_count=unverified_count,
            hallucinated_count=hallucinated_count,
            results=results,
            overall_confidence=overall_confidence,
            warnings=warnings,
        )
    
    def _verify_citation(
        self,
        citation: Citation,
        retrieved_chunks: List[Dict],
    ) -> VerificationResult:
        """
        Verify a single citation against retrieved chunks.
        
        Args:
            citation: Citation to verify
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            VerificationResult for the citation
        """
        matched_chunks = []
        confidence = 0.0
        
        # Normalize clause ID for matching
        citation_clause = self._normalize_clause_id(citation.clause_id)
        
        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            chunk_clause = self._normalize_clause_id(metadata.get("clause_id", ""))
            chunk_doc = metadata.get("document_name", "")
            chunk_section = metadata.get("section_title", "")
            
            # Check document match
            doc_match = self._documents_match(citation.document, chunk_doc)
            
            # Check clause match
            clause_match = False
            if citation_clause and chunk_clause:
                clause_match = self._clauses_match(citation_clause, chunk_clause)
            
            # Check section match
            section_match = False
            if citation.section:
                section_match = self._sections_match(citation.section, chunk_section)
            
            # Calculate confidence
            if doc_match and clause_match:
                confidence = max(confidence, 0.95)
                matched_chunks.append(str(chunk.get("id", "")))
            elif doc_match and section_match:
                confidence = max(confidence, 0.8)
                matched_chunks.append(str(chunk.get("id", "")))
            elif doc_match:
                confidence = max(confidence, 0.5)
                matched_chunks.append(str(chunk.get("id", "")))
            elif clause_match:
                confidence = max(confidence, 0.4)
        
        verified = confidence >= 0.8
        notes = self._generate_notes(confidence, verified, citation, matched_chunks)
        
        return VerificationResult(
            citation=citation,
            verified=verified,
            confidence=confidence,
            matched_chunk_ids=matched_chunks,
            notes=notes,
        )
    
    def _normalize_clause_id(self, clause_id: Optional[str]) -> Optional[str]:
        """Normalize clause ID for comparison."""
        if not clause_id:
            return None
        # Remove common prefixes
        clause_id = re.sub(r'^(Clause|Section|Clauses?|Sections?)\s*', '', clause_id, flags=re.IGNORECASE)
        # Extract numeric portion
        match = self.clause_pattern.match(clause_id.strip())
        return match.group(1) if match else clause_id.strip()
    
    def _documents_match(self, citation_doc: str, chunk_doc: str) -> bool:
        """Check if document names match."""
        if not citation_doc or not chunk_doc:
            return False
        
        # Normalize for comparison
        citation_norm = citation_doc.lower().strip()
        chunk_norm = chunk_doc.lower().strip()
        
        # Exact match
        if citation_norm == chunk_norm:
            return True
        
        # Abbreviation match
        abbreviations = {
            "hk cop": "hk code of practice",
            "cop": "code of practice",
            "ec7": "eurocode 7",
            "bs en 1997": "eurocode 7",
        }
        
        for abbrev, full in abbreviations.items():
            if abbrev in citation_norm and full in chunk_norm:
                return True
            if full in citation_norm and abbrev in chunk_norm:
                return True
        
        # Partial match (one contains the other)
        if citation_norm in chunk_norm or chunk_norm in citation_norm:
            return True
        
        return False
    
    def _clauses_match(self, citation_clause: str, chunk_clause: str) -> bool:
        """Check if clause IDs match."""
        if not citation_clause or not chunk_clause:
            return False
        
        # Exact match
        if citation_clause == chunk_clause:
            return True
        
        # Parent clause match (e.g., 6.1 matches 6.1.5)
        if chunk_clause.startswith(citation_clause + "."):
            return True
        if citation_clause.startswith(chunk_clause + "."):
            return True
        
        return False
    
    def _sections_match(self, citation_section: str, chunk_section: str) -> bool:
        """Check if section titles match."""
        if not citation_section or not chunk_section:
            return False
        
        citation_norm = citation_section.lower().strip()
        chunk_norm = chunk_section.lower().strip()
        
        # Exact or contains match
        if citation_norm == chunk_norm:
            return True
        if citation_norm in chunk_norm or chunk_norm in citation_norm:
            return True
        
        return False
    
    def _generate_notes(
        self,
        confidence: float,
        verified: bool,
        citation: Citation,
        matched_chunks: List[str],
    ) -> str:
        """Generate verification notes."""
        if verified:
            return f"Citation verified with {confidence:.0%} confidence"
        elif confidence > 0.5:
            return f"Partial match found ({confidence:.0%} confidence) - clause may be referenced indirectly"
        elif confidence > 0.3:
            return f"Weak match ({confidence:.0%} confidence) - document match but clause not found"
        else:
            return "No matching chunk found - possible hallucination"


class CitationAwareQAAgent:
    """
    QA Agent wrapper that adds citation verification.
    Use this instead of QAAgent for production to prevent hallucination.
    """
    
    def __init__(self, qa_agent, strict_mode: bool = True):
        """
        Initialize citation-aware QA agent.
        
        Args:
            qa_agent: Underlying QAAgent instance
            strict_mode: If True, warn about unverified citations
        """
        self.qa_agent = qa_agent
        self.verifier = CitationVerifier()
        self.strict_mode = strict_mode
    
    def ask(self, question: str, **kwargs) -> dict:
        """
        Ask a question with citation verification.
        
        Args:
            question: User question
            **kwargs: Additional arguments passed to underlying agent
            
        Returns:
            Answer dict with verification info
        """
        # Get answer from underlying agent
        result = self.qa_agent.ask(question, **kwargs)
        
        # Verify citations if we have an answer and sources
        if result.get("answer") and result.get("sources"):
            # Reconstruct retrieved chunks for verification
            retrieved_chunks = [
                {
                    "id": f"{s['document']}_{s['section']}",
                    "metadata": {
                        "document_name": s["document"],
                        "section_title": s["section"],
                        "page_number": s.get("page"),
                    },
                    "score": s.get("relevance_score", 0),
                }
                for s in result["sources"]
            ]
            
            verification = self.verifier.verify(
                answer=result["answer"],
                retrieved_chunks=retrieved_chunks,
                strict_mode=self.strict_mode,
            )
            
            # Add verification info to result
            result["verification"] = {
                "total_citations": verification.total_citations,
                "verified_count": verification.verified_count,
                "unverified_count": verification.unverified_count,
                "hallucinated_count": verification.hallucinated_count,
                "overall_confidence": verification.overall_confidence,
                "warnings": verification.warnings if self.strict_mode else [],
            }
            
            # Add warning if hallucinations detected
            if verification.hallucinated_count > 0 and self.strict_mode:
                result["warning"] = (
                    f"⚠️ {verification.hallucinated_count} citation(s) could not be verified. "
                    "Please verify against source documents."
                )
        
        return result


def verify_citations(answer: str, retrieved_chunks: List[Dict]) -> VerificationReport:
    """
    Standalone function to verify citations in an answer.
    
    Args:
        answer: LLM-generated answer text
        retrieved_chunks: List of retrieved context chunks
        
    Returns:
        VerificationReport with verification results
    """
    verifier = CitationVerifier()
    return verifier.verify(answer, retrieved_chunks)


def extract_citations(text: str) -> List[Citation]:
    """
    Standalone function to extract citations from text.
    
    Args:
        text: Text to extract citations from
        
    Returns:
        List of Citation objects
    """
    extractor = CitationExtractor()
    return extractor.extract(text)
