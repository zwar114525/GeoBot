"""
Enhanced PDF processor with OCR support and hybrid chunking strategies.
Handles scanned documents, large files, and preserves document structure.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

import fitz
import pymupdf4llm
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MAX_CHARS,
    USE_STRUCTURE_ASSEMBLY,
    DOCLING_MAX_PAGES,
)

try:
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("pytesseract or PIL not available. OCR features disabled.")

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning("Docling not available. Using fallback PDF processing.")


@dataclass
class DocumentChunk:
    """A single chunk of document text with metadata."""
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    """A fully processed document ready for embedding."""
    document_id: str
    document_name: str
    document_type: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    processing_info: dict = field(default_factory=dict)


class OCRProcessor:
    """OCR processor for scanned documents."""
    
    def __init__(self, lang: str = None):
        """
        Initialize OCR processor.
        
        Args:
            lang: OCR languages (English + Simplified Chinese for HK documents)
        """
        from config.settings import OCR_LANGUAGE
        self.lang = lang or OCR_LANGUAGE  # Default: eng+chi_sim
        if not OCR_AVAILABLE:
            logger.warning("OCR not available - install pytesseract and PIL")
    
    def available(self) -> bool:
        """Check if OCR is available."""
        return OCR_AVAILABLE
    
    def extract_text_from_page(self, page: fitz.Page) -> str:
        """
        Extract text from a PDF page using OCR.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text string
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = None) -> str:
        """
        Extract text from entire PDF using OCR.
        
        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to process
            
        Returns:
            Combined text from all pages
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            texts = []
            
            page_limit = max_pages if max_pages else len(doc)
            
            for i, page in enumerate(doc):
                if i >= page_limit:
                    break
                logger.info(f"OCR processing page {i + 1}/{page_limit}")
                text = self.extract_text_from_page(page)
                if text:
                    texts.append(f"\n[Page {i + 1}]\n{text}")
            
            doc.close()
            return "\n".join(texts)
            
        except Exception as e:
            logger.error(f"OCR PDF extraction failed: {e}")
            return ""


class CoPChunker:
    """Structure-aware chunker for Code of Practice documents using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter() if DOCLING_AVAILABLE else None
        self.clause_pattern = re.compile(r"^(\d+(?:\.\d+)*)")
        self.cross_ref_pattern = re.compile(r"Clause\s+(\d+(?:\.\d+)+)", re.IGNORECASE)
    
    def available(self) -> bool:
        """Check if Docling is available."""
        return self.converter is not None
    
    def parse_and_chunk(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        document_type: str,
        max_chunk_chars: int = 1500,
    ) -> ProcessedDocument:
        """
        Parse PDF using Docling's structure analysis.
        
        Args:
            pdf_path: Path to PDF file
            document_id: Document identifier
            document_name: Document display name
            document_type: Type (code, manual, report, etc.)
            max_chunk_chars: Maximum characters per chunk
            
        Returns:
            ProcessedDocument with structured chunks
        """
        if self.converter is None:
            raise RuntimeError("Docling is not available")
        
        result = self.converter.convert(pdf_path)
        doc = result.document
        processed = ProcessedDocument(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            processing_info={"method": "docling", "page_count": doc.page_count},
        )
        
        current_chunk_lines: list[str] = []
        current_metadata = self._base_metadata(document_id, document_name, document_type)
        
        for item, level in doc.iterate_items():
            item_type = self._label_name(item)
            
            if item_type == "SECTION_HEADER":
                self._flush_text_chunk(processed, current_chunk_lines, current_metadata, max_chunk_chars)
                header_text = self._item_text(item)
                clause_id = self._extract_clause_id(header_text)
                current_metadata = self._base_metadata(document_id, document_name, document_type)
                current_metadata.update(
                    {
                        "clause_id": clause_id,
                        "clause_title": header_text,
                        "hierarchy_level": level,
                        "content_type": "clause_text",
                        "page_no": self._item_page(item),
                        "section_title": header_text,
                    }
                )
                if header_text:
                    current_chunk_lines.append(f"# {header_text}")
                continue
            
            if item_type == "TABLE":
                self._flush_text_chunk(processed, current_chunk_lines, current_metadata, max_chunk_chars)
                table_text = self._table_to_markdown(item)
                table_meta = dict(current_metadata)
                table_meta.update(
                    {
                        "content_type": "table",
                        "page_no": self._item_page(item),
                        "table_caption": self._table_caption(item),
                    }
                )
                table_struct = self._table_to_dict(item)
                if table_struct is not None:
                    table_meta["table_struct"] = table_struct
                table_meta["regulatory_strength"] = self._detect_regulatory_strength(table_text)
                table_meta["cross_references"] = self._extract_cross_references(table_text)
                table_meta["char_count"] = len(table_text)
                processed.chunks.append(DocumentChunk(text=table_text, metadata=table_meta))
                continue
            
            if item_type in {"PARAGRAPH", "FORMULA", "LIST_ITEM"}:
                text = self._item_text(item)
                if text:
                    if item_type == "FORMULA":
                        current_metadata["formula_latex"] = text
                    current_chunk_lines.append(text)
                    current_metadata["page_no"] = self._item_page(item) or current_metadata.get("page_no")
                    if len("\n".join(current_chunk_lines)) >= max_chunk_chars:
                        self._flush_text_chunk(processed, current_chunk_lines, current_metadata, max_chunk_chars)
        
        self._flush_text_chunk(processed, current_chunk_lines, current_metadata, max_chunk_chars)
        return processed
    
    def _base_metadata(self, document_id: str, document_name: str, document_type: str) -> dict:
        return {
            "document_id": document_id,
            "document_name": document_name,
            "document_type": document_type,
            "clause_id": "unknown",
            "clause_title": "",
            "hierarchy_level": 0,
            "content_type": "clause_text",
            "page_no": None,
            "section_title": "",
        }
    
    def _label_name(self, item) -> str:
        label = getattr(item, "label", None)
        name = getattr(label, "name", None)
        return str(name or "").upper()
    
    def _item_text(self, item) -> str:
        text = getattr(item, "text", "")
        if text is None:
            return ""
        return str(text).strip()
    
    def _item_page(self, item):
        page_no = getattr(item, "page_no", None)
        if page_no is not None:
            return page_no
        prov = getattr(item, "prov", None)
        if prov:
            pages = getattr(prov, "pages", None)
            if pages:
                first = pages[0]
                return getattr(first, "page_no", None)
        return None
    
    def _extract_clause_id(self, header_text: str) -> str:
        match = self.clause_pattern.match((header_text or "").strip())
        return match.group(1) if match else "unknown"
    
    def _extract_cross_references(self, text: str) -> list[str]:
        refs = self.cross_ref_pattern.findall(text or "")
        return list(dict.fromkeys(refs))
    
    def _table_to_markdown(self, item) -> str:
        try:
            text = item.export_to_markdown()
            if text:
                return text
        except Exception:
            pass
        return self._item_text(item)
    
    def _table_to_dict(self, item):
        try:
            return item.export_to_dict()
        except Exception:
            return None
    
    def _table_caption(self, item) -> str:
        caption = getattr(item, "caption", None)
        if caption is None:
            return "Untitled Table"
        caption_text = getattr(caption, "text", "")
        return str(caption_text).strip() or "Untitled Table"
    
    def _flush_text_chunk(
        self,
        processed: ProcessedDocument,
        current_chunk_lines: list[str],
        current_metadata: dict,
        max_chunk_chars: int,
    ):
        if not current_chunk_lines:
            return
        text = "\n".join(current_chunk_lines).strip()
        if not text:
            current_chunk_lines.clear()
            return
        if len(text) <= max_chunk_chars:
            metadata = dict(current_metadata)
            metadata["regulatory_strength"] = self._detect_regulatory_strength(text)
            metadata["cross_references"] = self._extract_cross_references(text)
            metadata["char_count"] = len(text)
            processed.chunks.append(DocumentChunk(text=text, metadata=metadata))
        else:
            for idx, sub_text in enumerate(self._split_large_text_by_paragraph(text, max_chunk_chars)):
                metadata = dict(current_metadata)
                metadata["sub_chunk_index"] = idx
                metadata["regulatory_strength"] = self._detect_regulatory_strength(sub_text)
                metadata["cross_references"] = self._extract_cross_references(sub_text)
                metadata["char_count"] = len(sub_text)
                processed.chunks.append(DocumentChunk(text=sub_text, metadata=metadata))
        current_chunk_lines.clear()
    
    def _split_large_text_by_paragraph(self, text: str, max_chunk_chars: int) -> list[str]:
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para) + 2
            if current and current_len + para_len > max_chunk_chars:
                chunks.append("\n\n".join(current).strip())
                current = [para]
                current_len = len(para)
            else:
                current.append(para)
                current_len += para_len
        if current:
            chunks.append("\n\n".join(current).strip())
        return chunks
    
    def _detect_regulatory_strength(self, text: str) -> str:
        text_lower = text.lower()
        if "shall" in text_lower or "must" in text_lower:
            return "mandatory"
        if "should" in text_lower:
            return "advisory"
        return "informative"


class HybridPDFProcessor:
    """
    Hybrid PDF processor that combines multiple extraction strategies.
    Uses Docling when available, falls back to pymupdf4llm, then OCR for scanned docs.
    """
    
    def __init__(self, ocr_lang: str = None):
        """
        Initialize hybrid processor.
        
        Args:
            ocr_lang: OCR language configuration (default from settings)
        """
        from config.settings import OCR_LANGUAGE, OCR_ENABLED
        self.ocr_enabled = OCR_ENABLED
        self.ocr_lang = ocr_lang or OCR_LANGUAGE
        self.docling_chunker = CoPChunker() if DOCLING_AVAILABLE else None
        self.ocr_processor = OCRProcessor(lang=self.ocr_lang) if OCR_AVAILABLE and self.ocr_enabled else None
    
    def process(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        document_type: str,
        max_chunk_chars: int = CHUNK_MAX_CHARS,
        force_ocr: bool = False,
    ) -> ProcessedDocument:
        """
        Process PDF using best available method.
        
        Strategy:
        1. If force_ocr=True, use OCR directly
        2. If Docling available and page count <= limit, use Docling
        3. Otherwise use pymupdf4llm with structure-aware chunking
        4. If text extraction yields poor results, fall back to OCR
        
        Args:
            pdf_path: Path to PDF
            document_id: Document identifier
            document_name: Document display name
            document_type: Document type
            max_chunk_chars: Maximum characters per chunk
            force_ocr: Force OCR processing
            
        Returns:
            ProcessedDocument with chunks
        """
        page_count = self._get_page_count(pdf_path)
        logger.info(f"Processing {document_name}: {page_count} pages")
        
        # Strategy 1: Force OCR
        if force_ocr and self.ocr_processor and self.ocr_processor.available():
            logger.info("Using OCR extraction (forced)")
            return self._process_with_ocr(
                pdf_path, document_id, document_name, document_type, max_chunk_chars
            )
        
        # Strategy 2: Docling (if available and within page limit)
        if self.docling_chunker and self.docling_chunker.available():
            if page_count <= DOCLING_MAX_PAGES:
                try:
                    logger.info(f"Using Docling structure-aware chunking")
                    doc = self.docling_chunker.parse_and_chunk(
                        pdf_path=pdf_path,
                        document_id=document_id,
                        document_name=document_name,
                        document_type=document_type,
                        max_chunk_chars=max_chunk_chars,
                    )
                    if doc.chunks:
                        doc.processing_info["method"] = "docling"
                        doc.processing_info["page_count"] = page_count
                        return doc
                except Exception as e:
                    logger.warning(f"Docling failed: {e}. Falling back to pymupdf4llm")
            else:
                logger.warning(
                    f"Skipping Docling: {page_count} pages exceeds limit {DOCLING_MAX_PAGES}"
                )
        
        # Strategy 3: pymupdf4llm with hybrid chunking
        logger.info("Using pymupdf4llm with hybrid chunking")
        return self._process_with_pymupdf(
            pdf_path, document_id, document_name, document_type, max_chunk_chars
        )
    
    def _process_with_ocr(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        document_type: str,
        max_chunk_chars: int,
    ) -> ProcessedDocument:
        """Process PDF using OCR."""
        full_text = self.ocr_processor.extract_text_from_pdf(pdf_path)
        
        if not full_text or len(full_text.strip()) < 100:
            logger.warning("OCR extraction yielded poor results")
        
        doc = chunk_by_sections(
            full_text=full_text,
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            max_chunk_tokens=CHUNK_SIZE,
            overlap_tokens=CHUNK_OVERLAP,
        )
        doc.processing_info["method"] = "ocr"
        return doc
    
    def _process_with_pymupdf(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        document_type: str,
        max_chunk_chars: int,
    ) -> ProcessedDocument:
        """Process PDF using pymupdf4llm with hybrid chunking."""
        try:
            full_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
            
            # Check if extraction was successful
            if not full_text or len(full_text.strip()) < 200:
                logger.warning("pymupdf4llm extraction too short, trying OCR fallback")
                if self.ocr_processor and self.ocr_processor.available():
                    return self._process_with_ocr(
                        pdf_path, document_id, document_name, document_type, max_chunk_chars
                    )
            
            doc = chunk_by_sections(
                full_text=full_text,
                document_id=document_id,
                document_name=document_name,
                document_type=document_type,
                max_chunk_tokens=CHUNK_SIZE,
                overlap_tokens=CHUNK_OVERLAP,
            )
            doc.processing_info["method"] = "pymupdf4llm"
            return doc
            
        except Exception as e:
            logger.error(f"pymupdf4llm failed: {e}. Trying OCR fallback")
            if self.ocr_processor and self.ocr_processor.available():
                return self._process_with_ocr(
                    pdf_path, document_id, document_name, document_type, max_chunk_chars
                )
            raise
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get PDF page count."""
        doc = fitz.open(pdf_path)
        try:
            return doc.page_count
        finally:
            doc.close()


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = False) -> str:
    """
    Extract text from PDF with optional OCR.
    
    Args:
        pdf_path: Path to PDF
        use_ocr: Force OCR usage
        
    Returns:
        Extracted text
    """
    if use_ocr and OCR_AVAILABLE:
        ocr = OCRProcessor()
        return ocr.extract_text_from_pdf(pdf_path)
    
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
        if md_text and len(md_text.strip()) >= 100:
            return md_text
        logger.warning("pymupdf4llm extraction too short, falling back to basic extraction")
    except Exception as e:
        logger.warning(f"pymupdf4llm failed: {e}")
    
    return _basic_extract(pdf_path)


def _basic_extract(pdf_path: str) -> str:
    """Basic text extraction with pymupdf."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page_num, page in enumerate(doc, 1):
        text_parts.append(f"\n[Page {page_num}]\n{page.get_text('text')}")
    doc.close()
    return "\n".join(text_parts)


def extract_pages_with_numbers(pdf_path: str) -> list[dict]:
    """Extract text page by page with page numbers."""
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, 1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page_number": page_num, "text": text.strip()})
    doc.close()
    return pages


def chunk_by_sections(
    full_text: str,
    document_id: str,
    document_name: str,
    document_type: str,
    max_chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> ProcessedDocument:
    """
    Split document text into chunks by section headings.

    Implements dual-index chunking strategy:
    - Section Overview (level 1-2 headings): For hierarchical routing
    - Design Rule (detailed content): For detailed retrieval
    - Table/Equation: For data lookup

    Args:
        full_text: Full document text
        document_id: Document identifier
        document_name: Document display name
        document_type: Document type
        max_chunk_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap tokens between chunks

    Returns:
        ProcessedDocument with chunks
    """
    clause_pattern = re.compile(r"^(\d+(?:\.\d+)*)")
    cross_ref_pattern = re.compile(r"Clause\s+(\d+(?:\.\d+)+)", re.IGNORECASE)

    def detect_regulatory_strength(text: str) -> str:
        text_lower = text.lower()
        if "shall" in text_lower or "must" in text_lower:
            return "mandatory"
        if "should" in text_lower:
            return "advisory"
        return "informative"

    def extract_cross_references(text: str) -> list[str]:
        refs = cross_ref_pattern.findall(text or "")
        return list(dict.fromkeys(refs))

    def extract_clause_id(title: str) -> str:
        match = clause_pattern.match((title or "").strip().lstrip("#").strip())
        return match.group(1) if match else "unknown"

    def infer_page_number(title: str):
        m = re.search(r"Page\s+(\d+)", title or "", re.IGNORECASE)
        return int(m.group(1)) if m else None

    def classify_content_type(title: str, text: str, level: int) -> str:
        """
        Classify chunk content type for dual-index strategy.

        Returns:
            - "section_overview": Top-level headings (level 1-2) for Section Index
            - "design_rule": Detailed clause content for Rule Index
            - "table": Table content
            - "equation": Equation content
            - "definition": Definition content
        """
        title_lower = (title or "").lower()
        text_lower = (text or "").lower()

        # Check for tables
        if "table" in title_lower or ("table" in text_lower and len(text.splitlines()) < 20):
            return "table"

        # Check for equations
        if "equation" in title_lower or "formula" in title_lower:
            return "equation"

        # Check for definitions
        if "definition" in title_lower or "defined as" in text_lower:
            return "definition"

        # Section overview: level 1-2 headings (e.g., "6 Design", "6.1 General")
        if level <= 2:
            return "section_overview"

        # Default: design rule for Rule Index
        return "design_rule"

    def get_target_index(content_type: str) -> str:
        """
        Determine which index this chunk belongs to.

        Returns:
            - "section": For Section Index
            - "rule": For Rule Index
        """
        if content_type == "section_overview":
            return "section"
        return "rule"
    
    heading_pattern = r"^(#{1,4})\s+(.+)$"
    lines = full_text.split("\n")
    sections = []
    current_section_title = "Preamble"
    current_section_level = 0
    current_text_lines = []
    
    for line in lines:
        heading_match = re.match(heading_pattern, line, re.MULTILINE)
        if heading_match:
            if current_text_lines:
                section_text = "\n".join(current_text_lines).strip()
                if section_text:
                    sections.append(
                        {"title": current_section_title, "level": current_section_level, "text": section_text}
                    )
            current_section_level = len(heading_match.group(1))
            current_section_title = heading_match.group(2).strip()
            current_text_lines = []
        else:
            current_text_lines.append(line)
    
    if current_text_lines:
        section_text = "\n".join(current_text_lines).strip()
        if section_text:
            sections.append({"title": current_section_title, "level": current_section_level, "text": section_text})
    
    if len(sections) <= 1:
        sections = _split_by_pages_or_size(full_text, max_chunk_tokens)
    
    doc = ProcessedDocument(document_id=document_id, document_name=document_name, document_type=document_type)
    for section in sections:
        section_text = section["text"]
        section_title = section.get("title", "")
        section_level = section.get("level", 0)
        estimated_tokens = len(section_text.split()) * 1.3

        # Classify content type for dual-index strategy
        content_type = classify_content_type(section_title, section_text, section_level)
        target_index = get_target_index(content_type)

        if estimated_tokens <= max_chunk_tokens:
            clause_id = extract_clause_id(section.get("title", ""))
            page_no = infer_page_number(section.get("title", ""))
            doc.chunks.append(
                DocumentChunk(
                    text=section_text,
                    metadata={
                        "document_id": document_id,
                        "document_name": document_name,
                        "document_type": document_type,
                        "clause_id": clause_id,
                        "clause_title": section.get("title", ""),
                        "content_type": content_type,
                        "target_index": target_index,  # "section" or "rule"
                        "section_title": section.get("title", ""),
                        "section_level": section.get("level", 0),
                        "hierarchy_level": section.get("level", 0),
                        "page_no": page_no,
                        "regulatory_strength": detect_regulatory_strength(section_text),
                        "cross_references": extract_cross_references(section_text),
                        "char_count": len(section_text),
                    },
                )
            )
        else:
            sub_chunks = _split_large_section(
                section_text,
                max_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
            for i, sub_text in enumerate(sub_chunks):
                clause_id = extract_clause_id(section.get("title", ""))
                page_no = infer_page_number(section.get("title", ""))
                # Sub-chunks are always design rules (detailed content)
                sub_content_type = "design_rule"
                doc.chunks.append(
                    DocumentChunk(
                        text=sub_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": document_name,
                            "document_type": document_type,
                            "clause_id": clause_id,
                            "clause_title": section.get("title", ""),
                            "content_type": sub_content_type,
                            "target_index": "rule",  # Sub-chunks go to Rule Index
                            "section_title": section.get("title", ""),
                            "section_level": section.get("level", 0),
                            "hierarchy_level": section.get("level", 0),
                            "page_no": page_no,
                            "regulatory_strength": detect_regulatory_strength(sub_text),
                            "cross_references": extract_cross_references(sub_text),
                            "char_count": len(sub_text),
                            "sub_chunk_index": i,
                        },
                    )
                )
    
    logger.info(f"Processed '{document_name}': {len(doc.chunks)} chunks")
    return doc


def _split_large_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split a large section into overlapping chunks by paragraphs."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk_parts = []
    current_token_count = 0.0
    for para in paragraphs:
        para_tokens = len(para.split()) * 1.3
        if current_token_count + para_tokens > max_tokens and current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))
            overlap_parts = current_chunk_parts[-1:] if overlap_tokens > 0 else []
            current_chunk_parts = overlap_parts
            current_token_count = sum(len(p.split()) * 1.3 for p in current_chunk_parts)
        current_chunk_parts.append(para)
        current_token_count += para_tokens
    if current_chunk_parts:
        chunks.append("\n\n".join(current_chunk_parts))
    return chunks


def _split_by_pages_or_size(text: str, max_tokens: int) -> list[dict]:
    """Fallback splitting when no headings are found."""
    page_pattern = r"\[Page (\d+)\]"
    page_splits = re.split(page_pattern, text)
    if len(page_splits) > 1:
        sections = []
        for i in range(1, len(page_splits), 2):
            page_num = page_splits[i]
            page_text = page_splits[i + 1] if i + 1 < len(page_splits) else ""
            sections.append({"title": f"Page {page_num}", "level": 0, "text": page_text.strip()})
        return sections
    
    words = text.split()
    chunk_word_count = int(max_tokens / 1.3)
    sections = []
    for i in range(0, len(words), chunk_word_count):
        chunk_text = " ".join(words[i : i + chunk_word_count])
        sections.append({"title": f"Chunk {i // chunk_word_count + 1}", "level": 0, "text": chunk_text})
    return sections


def parse_pdf_with_structure(
    pdf_path: str,
    document_id: str,
    document_name: str,
    document_type: str,
    max_chunk_chars: int = CHUNK_MAX_CHARS,
    force_ocr: bool = False,
) -> ProcessedDocument:
    """
    Parse PDF with structure-aware chunking using hybrid approach.
    
    This is the main entry point for PDF processing.
    
    Args:
        pdf_path: Path to PDF file
        document_id: Document identifier
        document_name: Document display name
        document_type: Document type (code, manual, report, etc.)
        max_chunk_chars: Maximum characters per chunk
        force_ocr: Force OCR processing
        
    Returns:
        ProcessedDocument with structured chunks
    """
    processor = HybridPDFProcessor()
    return processor.process(
        pdf_path=pdf_path,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        max_chunk_chars=max_chunk_chars,
        force_ocr=force_ocr,
    )
