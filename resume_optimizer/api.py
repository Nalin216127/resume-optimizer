"""FastAPI backend.

Endpoints are plain ``def`` so FastAPI runs the blocking graph/LLM calls in its
threadpool instead of on the event loop.
"""
from __future__ import annotations

import logging
import secrets
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .schemas import FeedbackRequest, KnowledgeIngestRequest, SessionView, StartSessionRequest
from .textutils import file_to_text
from .service import ResumeService, SessionNotFound

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
settings = get_settings()
_service: Optional[ResumeService] = None


def get_service() -> ResumeService:
    if _service is None:
        raise HTTPException(503, "Service is starting")
    return _service


def require_auth(x_api_key: Optional[str] = Header(default=None)) -> None:
    if settings.api_auth_token and not secrets.compare_digest(x_api_key or "", settings.api_auth_token):
        raise HTTPException(401, "Invalid or missing X-API-Key")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _service
    _service = ResumeService(settings)
    yield


app = FastAPI(title="Multi-Agent Resume Optimizer", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/healthz", tags=["ops"])
def healthz():
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz(svc: ResumeService = Depends(get_service)):
    if not svc.store.healthy():
        raise HTTPException(503, "Vector store unavailable")
    return {"status": "ready", "llm": settings.llm_provider, "vector_backend": settings.vector_backend,
            "knowledge_docs": svc.store.count()}


@app.get("/api/v1/providers", dependencies=[Depends(require_auth)], tags=["ops"])
def provider_status(every_model: bool = False, svc: ResumeService = Depends(get_service)):
    """Report which LLM providers answer. ``every_model=true`` tests each configured model (one request each)."""
    return svc.provider_status(every_model)


@app.post("/api/v1/sessions", response_model=SessionView, dependencies=[Depends(require_auth)], tags=["sessions"])
def start_session(req: StartSessionRequest, svc: ResumeService = Depends(get_service)):
    """Run Supervisor → Context → Writer ⇄ Reviewer until the draft needs human approval."""
    return svc.start(req)


@app.get("/api/v1/sessions/{session_id}", response_model=SessionView, dependencies=[Depends(require_auth)],
         tags=["sessions"])
def get_session(session_id: str, svc: ResumeService = Depends(get_service)):
    try:
        return svc.get(session_id)
    except SessionNotFound:
        raise HTTPException(404, "Session not found")


@app.post("/api/v1/sessions/{session_id}/feedback", response_model=SessionView,
          dependencies=[Depends(require_auth)], tags=["sessions"])
def submit_feedback(session_id: str, req: FeedbackRequest, svc: ResumeService = Depends(get_service)):
    """Human-in-the-loop: approve the draft, or send it back to the Writer with feedback."""
    try:
        return svc.feedback(session_id, req)
    except SessionNotFound:
        raise HTTPException(404, "Session not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/sessions/{session_id}/retry", response_model=SessionView,
          dependencies=[Depends(require_auth)], tags=["sessions"])
def retry_session(session_id: str, svc: ResumeService = Depends(get_service)):
    """Continue a failed run (e.g. after an LLM rate limit) from the last completed agent."""
    try:
        return svc.retry(session_id)
    except SessionNotFound:
        raise HTTPException(404, "Session not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@app.post("/api/v1/knowledge", dependencies=[Depends(require_auth)], tags=["knowledge"])
def ingest_knowledge(req: KnowledgeIngestRequest, svc: ResumeService = Depends(get_service)):
    """Add guidelines, exemplar bullets or role profiles to the Milvus knowledge base."""
    return {"inserted": svc.add_knowledge(req.documents)}


@app.post("/api/v1/extract-text", dependencies=[Depends(require_auth)], tags=["utils"])
async def extract_text(file: UploadFile = File(...)):
    """Turn an uploaded PDF / TXT / MD resume into plain text."""
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(413, "File larger than 5 MB")
    try:
        return {"text": file_to_text(file.filename or "", data)}
    except ValueError as exc:
        raise HTTPException(422, str(exc))
