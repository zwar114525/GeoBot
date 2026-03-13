import sys
from pathlib import Path
from typing import List, Optional
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_processor import parse_pdf_with_structure, DocumentChunk
from src.vectordb.qdrant_store import GeoVectorStore


def save_chunks_to_text_file(
    chunks: List[DocumentChunk],
    document_name: str,
    output_dir: str = None,
) -> str:
    """
    Save parsed chunks to a text file before embedding.
    Each chunk is formatted with metadata header and content.

    Returns:
        Path to the saved file
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "chunked_files"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize document name for filename
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in document_name)
    safe_name = safe_name.strip().replace(" ", "_")
    output_path = output_dir / f"{safe_name}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Document: {document_name}\n")
        f.write(f"Total Chunks: {len(chunks)}\n")
        f.write("=" * 80 + "\n\n")

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            target = meta.get("target_index", "N/A")
            page = meta.get("page_number") or meta.get("page_no", "N/A")
            clause = meta.get("clause_id", "N/A")
            title = meta.get("clause_title", "") or meta.get("section_title", "")
            content_type = meta.get("content_type", "N/A")

            f.write(f"{'─' * 80}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'─' * 80}\n")
            f.write(f"Index: {target} | Page: {page} | Clause: {clause} | Type: {content_type}\n")
            if title:
                f.write(f"Title: {title}\n")
            f.write(f"{'─' * 80}\n\n")
            f.write(chunk.text.strip())
            f.write("\n\n")

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")
    return str(output_path)

_INGEST_LOG_SINK_ID = None


def setup_ingestion_logging():
    global _INGEST_LOG_SINK_ID
    if _INGEST_LOG_SINK_ID is not None:
        return
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "ingestion.log"
    _INGEST_LOG_SINK_ID = logger.add(
        str(log_path),
        level="INFO",
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
    )


def ingest_document(
    pdf_path: str,
    document_id: str,
    document_name: str,
    document_type: str = "code",
    use_local_db: bool = False,
) -> int:
    setup_ingestion_logging()
    logger.info(f"Step 1/3: Parsing and structure-aware chunking from {pdf_path}")
    processed_doc = parse_pdf_with_structure(
        pdf_path=pdf_path,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
    )
    if not processed_doc.chunks:
        logger.error(f"Failed to extract meaningful chunks from {pdf_path}")
        return 0
    logger.info("Step 2/3: Chunking complete")

    # Save chunks to text file for inspection
    saved_path = save_chunks_to_text_file(processed_doc.chunks, document_name)
    logger.info(f"Chunks saved to: {saved_path}")

    logger.info("Step 3/3: Embedding and storing in vector database")
    store = GeoVectorStore(use_local=use_local_db)
    num_stored = store.add_document(processed_doc)
    logger.info(f"Successfully ingested '{document_name}': {num_stored} chunks")
    return num_stored


def ingest_directory(directory: str, document_type: str = "code", use_local_db: bool = False):
    setup_ingestion_logging()
    dir_path = Path(directory)
    pdf_files = list(dir_path.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
    for pdf_file in pdf_files:
        doc_id = pdf_file.stem.lower().replace(" ", "_")
        doc_name = pdf_file.stem
        try:
            ingest_document(
                pdf_path=str(pdf_file),
                document_id=doc_id,
                document_name=doc_name,
                document_type=document_type,
                use_local_db=use_local_db,
            )
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_file.name}: {e}")


def ingest_document_dual_index(
    pdf_path: str,
    document_id: str,
    document_name: str,
    document_type: str = "code",
    use_local_db: bool = False,
) -> dict:
    """
    Ingest document using dual-index strategy.

    Distributes chunks to:
    - Section Index: Section overviews (level 1-2 headings)
    - Rule Index: Detailed content (tables, equations, rules)

    Returns:
        dict with section_count and rule_count
    """
    setup_ingestion_logging()
    logger.info(f"Dual-index ingestion: {pdf_path}")

    # Step 1: Parse PDF using existing pipeline (Docling when available, pymupdf4llm fallback)
    logger.info("Step 1/4: Parsing PDF (Docling/pymupdf4llm)")
    processed_doc = parse_pdf_with_structure(
        pdf_path=pdf_path,
        document_id=document_id,
        document_name=document_name,
        document_type=document_type,
    )

    # Step 2: Apply dual-index classification to each chunk
    logger.info("Step 2/4: Classifying chunks for dual-index")
    for chunk in processed_doc.chunks:
        content_type = chunk.metadata.get("content_type", "clause_text")
        hierarchy_level = chunk.metadata.get("hierarchy_level", 2)
        if content_type in ("table", "equation", "definition"):
            chunk.metadata["target_index"] = "rule"
        elif hierarchy_level <= 2:
            chunk.metadata["target_index"] = "section"
        else:
            chunk.metadata["target_index"] = "rule"

    if not processed_doc.chunks:
        logger.error(f"Failed to extract chunks from {pdf_path}")
        return {"section_count": 0, "rule_count": 0}

    # Save chunks to text file for inspection (before splitting)
    saved_path = save_chunks_to_text_file(processed_doc.chunks, document_name)
    logger.info(f"Chunks saved to: {saved_path}")

    # Separate chunks by target index
    section_chunks = []
    rule_chunks = []
    for chunk in processed_doc.chunks:
        target = chunk.metadata.get("target_index", "rule")
        if target == "section":
            section_chunks.append(chunk)
        else:
            rule_chunks.append(chunk)

    logger.info(f"Step 3/4: Distributing to indices - Section: {len(section_chunks)}, Rule: {len(rule_chunks)}")

    # Step 3: Create dual indices and store
    store = GeoVectorStore(use_local=use_local_db)
    store.create_section_index()
    store.create_rule_index()

    section_store = store.get_section_store()
    rule_store = store.get_rule_store()

    # Store section chunks
    section_count = 0
    if section_chunks:
        from src.ingestion.pdf_processor import ProcessedDocument
        section_doc = ProcessedDocument(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            chunks=section_chunks
        )
        section_count = section_store.add_document(section_doc)

    # Store rule chunks
    rule_count = 0
    if rule_chunks:
        from src.ingestion.pdf_processor import ProcessedDocument
        rule_doc = ProcessedDocument(
            document_id=document_id,
            document_name=document_name,
            document_type=document_type,
            chunks=rule_chunks
        )
        rule_count = rule_store.add_document(rule_doc)

    logger.info(f"Step 4/4: Dual-index ingestion complete - Section: {section_count}, Rule: {rule_count}")
    return {"section_count": section_count, "rule_count": rule_count}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into vector DB")
    subparsers = parser.add_subparsers(dest="command")

    single_parser = subparsers.add_parser("single")
    single_parser.add_argument("--path", required=True, help="Path to PDF")
    single_parser.add_argument("--id", required=True, help="Document ID")
    single_parser.add_argument("--name", required=True, help="Document name")
    single_parser.add_argument("--type", default="code", choices=["code", "manual", "report", "guideline", "drawing"])
    single_parser.add_argument("--local-db", action="store_true")
    single_parser.add_argument("--dual-index", action="store_true", help="Use dual-index strategy")

    dir_parser = subparsers.add_parser("directory")
    dir_parser.add_argument("--path", required=True)
    dir_parser.add_argument("--type", default="code")
    dir_parser.add_argument("--local-db", action="store_true")
    dir_parser.add_argument("--dual-index", action="store_true", help="Use dual-index strategy")

    args = parser.parse_args()
    if args.command == "single":
        if args.dual_index:
            result = ingest_document_dual_index(args.path, args.id, args.name, args.type, args.local_db)
            print(f"Dual-index ingestion: {result}")
        else:
            ingest_document(args.path, args.id, args.name, args.type, args.local_db)
    elif args.command == "directory":
        if args.dual_index:
            dir_path = Path(args.path)
            for pdf_file in dir_path.glob("*.pdf"):
                doc_id = pdf_file.stem.lower().replace(" ", "_")
                doc_name = pdf_file.stem
                try:
                    result = ingest_document_dual_index(str(pdf_file), doc_id, doc_name, args.type, args.local_db)
                    print(f"{pdf_file.name}: {result}")
                except Exception as e:
                    print(f"Failed: {pdf_file.name}: {e}")
        else:
            ingest_directory(args.path, args.type, args.local_db)
    else:
        parser.print_help()
