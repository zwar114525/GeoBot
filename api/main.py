"""
FastAPI backend for GeoBot.
Provides REST API endpoints for programmatic access.
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from src.agents.qa_agent import QAAgent
from src.agents.designer_agent import DesignerAgent
from src.agents.validator_agent import ValidatorAgent
from src.ingestion.ingest import ingest_document
from src.utils.llm_client import get_runtime_llm_config, set_runtime_llm_config


# Initialize agents (singleton)
_qa_agent = None
_validator_agent = None


def get_qa_agent():
    global _qa_agent
    if _qa_agent is None:
        _qa_agent = QAAgent(use_local_db=True)
    return _qa_agent


def get_validator_agent():
    global _validator_agent
    if _validator_agent is None:
        _validator_agent = ValidatorAgent(use_local_db=True)
    return _validator_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _qa_agent, _validator_agent
    # Startup
    _qa_agent = QAAgent(use_local_db=True)
    _validator_agent = ValidatorAgent(use_local_db=True)
    yield
    # Shutdown
    pass


app = FastAPI(
    title="GeoBot API",
    description="Geotechnical AI Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AskQuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    clause_id: Optional[str] = None
    content_type: Optional[str] = None
    regulatory_strength: Optional[str] = None
    equation_mode: bool = False
    top_k: int = 8


class AskQuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    query: str


class ValidateReportResponse(BaseModel):
    document_name: str
    overall_status: str
    pass_count: int
    fail_count: int
    warning_count: int
    summary: str
    checks: List[Dict[str, Any]]


class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]


class IngestDocumentResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_created: int
    success: bool


@app.get("/")
async def root():
    return {"message": "GeoBot API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/qa/ask", response_model=AskQuestionResponse)
async def ask_question(request: AskQuestionRequest):
    """Ask a question to the knowledge base."""
    agent = get_qa_agent()

    try:
        result = agent.ask(
            question=request.question,
            document_id=request.document_id,
            clause_id=request.clause_id,
            content_type=request.content_type,
            regulatory_strength=request.regulatory_strength,
            equation_mode=request.equation_mode,
            top_k=request.top_k,
        )

        return AskQuestionResponse(
            answer=result["answer"],
            sources=result["sources"],
            query=result["query"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qa/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all documents in the knowledge base."""
    agent = get_qa_agent()

    try:
        docs = agent.list_knowledge_base()
        return DocumentListResponse(documents=docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/ingest", response_model=IngestDocumentResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
    document_type: str = "code",
):
    """Ingest a PDF document into the knowledge base."""

    # Save uploaded file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        doc_id = document_id or document_name or file.filename.replace(".pdf", "").lower().replace(" ", "_")
        doc_name = document_name or file.filename.replace(".pdf", "")

        chunks = ingest_document(
            pdf_path=tmp_path,
            document_id=doc_id,
            document_name=doc_name,
            document_type=document_type,
            use_local_db=True,
        )

        return IngestDocumentResponse(
            document_id=doc_id,
            document_name=doc_name,
            chunks_created=chunks,
            success=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/validate", response_model=ValidateReportResponse)
async def validate_report(file: UploadFile = File(...)):
    """Validate a geotechnical report PDF."""
    agent = get_validator_agent()

    # Save uploaded file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = agent.validate_report(tmp_path)

        return ValidateReportResponse(
            document_name=report.document_name,
            overall_status=report.overall_status,
            pass_count=report.pass_count,
            fail_count=report.fail_count,
            warning_count=report.warning_count,
            summary=report.summary,
            checks=[
                {
                    "category": c.category,
                    "item": c.item,
                    "status": c.status,
                    "details": c.details,
                    "severity": c.severity,
                }
                for c in report.checks
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/config/llm")
async def get_llm_config():
    """Get current LLM configuration."""
    return get_runtime_llm_config()


@app.post("/config/llm")
async def update_llm_config(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
):
    """Update LLM configuration."""
    set_runtime_llm_config(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )
    return {"success": True}


@app.get("/skills")
async def list_skills():
    """List all available calculation skills."""
    from src.skills.catalog import SkillCatalog

    catalog = SkillCatalog()
    return catalog.list_all()


if __name__ == "__main__":
    port = int(os.getenv("GEOBOT_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
