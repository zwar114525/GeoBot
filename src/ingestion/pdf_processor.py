import re
from dataclasses import dataclass, field
import fitz
import pymupdf4llm
from loguru import logger


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
            doc.chunks.append(
                DocumentChunk(
                    text=section_text,
                    metadata={
                        "document_id": document_id,
                        "document_name": document_name,
                        "document_type": document_type,
                        "section_title": section.get("title", ""),
                        "section_level": section.get("level", 0),
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
                doc.chunks.append(
                    DocumentChunk(
                        text=sub_text,
                        metadata={
                            "document_id": document_id,
                            "document_name": document_name,
                            "document_type": document_type,
                            "section_title": section.get("title", ""),
                            "section_level": section.get("level", 0),
                            "sub_chunk_index": i,
                        },
                    )
                )
    logger.info(f"Processed '{document_name}': {len(doc.chunks)} chunks")
    return doc


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
