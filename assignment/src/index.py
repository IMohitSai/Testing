from __future__ import annotations
from fastapi import UploadFile, File
from pypdf import PdfReader
import docx
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .agent import AgentResponse, ChatMessage, Mode, build_orchestrator
from .settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("placementsprint.api")


class ChatRequest(BaseModel):
    mode: Mode = Field(default="auto")
    messages: list[ChatMessage] = Field(min_length=1, max_length=30)


class ApiError(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Vercel recommends lifespan for startup logic. :contentReference[oaicite:6]{index=6}
    settings = Settings()
    app.state.settings = settings
    app.state.orchestrator = build_orchestrator(settings)
    logger.info("startup ok (model=%s fallback=%s)", settings.openrouter_model, settings.openrouter_fallback_model)
    yield
    logger.info("shutdown")

MAX_RESUME_BYTES = 2 * 1024 * 1024  # 2 MB
ALLOWED_RESUME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

def _clean_text(s: str) -> str:
    s = s.replace("\x00", "").strip()
    # Keep it bounded so it doesn't explode tokens
    if len(s) > 12000:
        s = s[:12000] + "\n\n[Truncated resume text to 12k chars]"
    return s

def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages[:10]:  # hard cap pages
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)

def _extract_docx_text(data: bytes) -> str:
    doc = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    for p in doc.paragraphs[:400]:  # cap paragraphs
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-vercel-id") or request.headers.get("x-request-id")
    logger.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ApiError(
            error="internal_error",
            detail="Something went wrong. Check logs and try again.",
            request_id=request_id,
        ).model_dump(),
    )


@app.get("/", include_in_schema=False)
async def root():
    # Static assets live in public/** and are served automatically on Vercel. :contentReference[oaicite:7]{index=7}
    return RedirectResponse(url="/index.html", status_code=307)

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    t0 = time.time()
    content_type = file.content_type or ""

    if content_type not in ALLOWED_RESUME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PDF or DOCX resume.",
        )

    data = await file.read()
    if not data or len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 2MB.")

    try:
        kind = ALLOWED_RESUME_TYPES[content_type]
        if kind == "pdf":
            text = _extract_pdf_text(data)
        else:
            text = _extract_docx_text(data)

        text = _clean_text(text)
        if len(text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail="Could not extract enough text from this file. Try another PDF/DOCX (non-scanned).",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("resume extraction failed")
        raise HTTPException(status_code=422, detail="Failed to parse resume file.")

    dt_ms = int((time.time() - t0) * 1000)
    logger.info("resume uploaded kind=%s bytes=%s ms=%s", kind, len(data), dt_ms)

    return {
        "ok": True,
        "filename": file.filename,
        "content_type": content_type,
        "text": text,
        "chars": len(text),
    }

@app.get("/api/health")
async def health():
    return {"ok": True}


@app.post("/api/chat", response_model=AgentResponse)
async def chat(req: ChatRequest, request: Request):
    t0 = time.time()
    request_id = request.headers.get("x-vercel-id") or request.headers.get("x-request-id")

    # Basic sanity checks
    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be role='user'.")

    # Lightweight anti-abuse: reject absurd payloads
    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > 24000:
        raise HTTPException(status_code=413, detail="Message history too large. Keep it shorter.")

    orch = app.state.orchestrator
    try:
        out = await orch.respond(req.messages, req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("chat failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail="Model/provider error. Try again.") from e

    dt_ms = int((time.time() - t0) * 1000)
    logger.info("chat ok request_id=%s mode=%s ms=%s", request_id, req.mode, dt_ms)
    return out
from fastapi.staticfiles import StaticFiles
if os.getenv("VERCEL") != "1":
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
#app.mount("/", StaticFiles(directory="public", html=True), name="static")
