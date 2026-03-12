import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingestion.pdf_processor import parse_pdf_with_structure
from src.vectordb.qdrant_store import GeoVectorStore

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

    dir_parser = subparsers.add_parser("directory")
    dir_parser.add_argument("--path", required=True)
    dir_parser.add_argument("--type", default="code")
    dir_parser.add_argument("--local-db", action="store_true")

    args = parser.parse_args()
    if args.command == "single":
        ingest_document(args.path, args.id, args.name, args.type, args.local_db)
    elif args.command == "directory":
        ingest_directory(args.path, args.type, args.local_db)
    else:
        parser.print_help()
