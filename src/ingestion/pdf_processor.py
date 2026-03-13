import re
from dataclasses import dataclass, field
import fitz
import pymupdf4llm
from loguru import logger
from config.settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_MAX_CHARS,
    USE_STRUCTURE_ASSEMBLY,
    DOCLING_MAX_PAGES,
)

try:
    from docling.document_converter import DocumentConverter
except Exception:
    DocumentConverter = None


@dataclass
class DocumentChunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ProcessedDocument:
    document_id: str
    document_name: str
    document_type: str
    chunks: list[DocumentChunk] = field(default_factory=list)


class CoPChunker:
    def __init__(self):
        self.converter = DocumentConverter() if DocumentConverter is not None else None
        self.clause_pattern = re.compile(r"^(\d+(?:\.\d+)*)")
        self.cross_ref_pattern = re.compile(r"Clause\s+(\d+(?:\.\d+)+)", re.IGNORECASE)

    def available(self) -> bool:
        return self.converter is not None

    def parse_and_chunk(
        self,
        pdf_path: str,
        document_id: str,
        document_name: str,
        document_type: str,
        max_chunk_chars: int = 1500,
    ) -> ProcessedDocument:
        if self.converter is None:
            raise RuntimeError("Docling is not available")
        result = self.converter.convert(pdf_path)
        doc = result.document
        processed = ProcessedDocument(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
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
                pg = self._item_page(item)
                current_metadata.update(
                    {
                        "clause_id": clause_id,
                        "clause_title": header_text,
                        "hierarchy_level": level,
                        "content_type": "clause_text",
                        "page_no": pg,
                        "page_number": pg,
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
                pg = self._item_page(item)
                table_meta.update(
                    {
                        "content_type": "table",
                        "page_no": pg,
                        "page_number": pg,
                        "table_caption": self._table_caption(item),
                    }
                )
                table_struct = self._table_to_dict(item)
                if table_struct is not None:
                    table_meta["table_struct"] = table_struct
                table_meta.setdefault("sub_chunk_index", 0)
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
                    pg = self._item_page(item) or current_metadata.get("page_no")
                    current_metadata["page_no"] = pg
                    current_metadata["page_number"] = pg
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
        if prov is None:
            return None
        # Docling: prov is a list of ProvenanceItem, each has page_no
        try:
            if hasattr(prov, "__getitem__") and len(prov) > 0:
                first = prov[0]
                return getattr(first, "page_no", None)
            pages = getattr(prov, "pages", None)
            if pages and len(pages) > 0:
                return getattr(pages[0], "page_no", None)
        except (IndexError, TypeError):
            pass
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
        pg = current_metadata.get("page_no") or current_metadata.get("page_number")
        if len(text) <= max_chunk_chars:
            metadata = dict(current_metadata)
            if pg is not None and "page_number" not in metadata:
                metadata["page_number"] = pg
            metadata.setdefault("sub_chunk_index", 0)
            metadata["regulatory_strength"] = self._detect_regulatory_strength(text)
            metadata["cross_references"] = self._extract_cross_references(text)
            metadata["char_count"] = len(text)
            processed.chunks.append(DocumentChunk(text=text, metadata=metadata))
        else:
            for idx, sub_text in enumerate(self._split_large_text_by_paragraph(text, max_chunk_chars)):
                metadata = dict(current_metadata)
                metadata["sub_chunk_index"] = idx
                if pg is not None and "page_number" not in metadata:
                    metadata["page_number"] = pg
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


def extract_text_from_pdf(pdf_path: str) -> str:
    logger.info(f"Extracting text from: {pdf_path}")
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path, page_chunks=False)
        if md_text and len(md_text.strip()) >= 100:
            return md_text
        logger.warning("pymupdf4llm extraction too short, falling back to basic extraction")
        return _basic_extract(pdf_path)
    except Exception as e:
        logger.warning(f"pymupdf4llm failed, fallback extraction used: {e}")
        return _basic_extract(pdf_path)


def _basic_extract(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_parts = []
    for page_num, page in enumerate(doc, 1):
        text_parts.append(f"\n[Page {page_num}]\n{page.get_text('text')}")
    doc.close()
    return "\n".join(text_parts)


def extract_pages_with_numbers(pdf_path: str) -> list[dict]:
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
        estimated_tokens = len(section_text.split()) * 1.3
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
                        "content_type": "clause_text",
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
                doc.chunks.append(
                    DocumentChunk(
                        text=sub_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": document_name,
                            "document_type": document_type,
                            "clause_id": clause_id,
                            "clause_title": section.get("title", ""),
                            "content_type": "clause_text",
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


def parse_pdf_with_structure(
    pdf_path: str,
    document_id: str,
    document_name: str,
    document_type: str,
    max_chunk_chars: int = CHUNK_MAX_CHARS,
) -> ProcessedDocument:
    chunker = CoPChunker()
    if USE_STRUCTURE_ASSEMBLY and chunker.available():
        page_count = _get_pdf_page_count(pdf_path)
        if page_count > DOCLING_MAX_PAGES:
            logger.warning(
                f"Skipping Docling structure chunking for '{document_name}' "
                f"because page count {page_count} exceeds DOCLING_MAX_PAGES={DOCLING_MAX_PAGES}. "
                "Using fallback markdown/page chunking."
            )
        else:
            try:
                doc = chunker.parse_and_chunk(
                    pdf_path=pdf_path,
                    document_id=document_id,
                    document_name=document_name,
                    document_type=document_type,
                    max_chunk_chars=max_chunk_chars,
                )
                if doc.chunks:
                    logger.info(f"Docling structure chunking used for '{document_name}' with {len(doc.chunks)} chunks")
                    return doc
            except Exception as e:
                logger.warning(f"Docling structure chunking failed; fallback to markdown chunking: {e}")
    full_text = extract_text_from_pdf(pdf_path)
    return chunk_by_sections(
        full_text=full_text,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
        max_chunk_tokens=CHUNK_SIZE,
        overlap_tokens=CHUNK_OVERLAP,
    )


def _get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _split_large_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
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
