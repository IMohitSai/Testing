from __future__ import annotations

import io
import logging
import os
import time
import asyncio
from pathlib import Path

import docx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader

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


app = FastAPI()

# ---- Lazy singletons (safe for Vercel cold starts) ----
_settings: Settings | None = None
_orch = None
_orch_lock = asyncio.Lock()


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


async def get_orchestrator():
    global _orch
    if _orch is not None:
        return _orch
    async with _orch_lock:
        if _orch is None:
            s = get_settings()
            _orch = build_orchestrator(s)
            logger.info("orchestrator ready (model=%s fallback=%s)", s.openrouter_model, s.openrouter_fallback_model)
    return _orch


# ---- Error handling ----
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


# ---- API routes ----
@app.get("/api", include_in_schema=False)
async def api_index():
    return {
        "routes": ["/api/health", "/api/upload_resume", "/api/chat"],
        "note": "POST /api/upload_resume expects multipart/form-data with field name 'file'.",
    }


@app.get("/api/health")
async def health():
    s = get_settings()
    return {
        "ok": True,
        "vercel": os.getenv("VERCEL", "0"),
        "has_openrouter_api_key": bool(s.openrouter_api_key),
        "openrouter_model": s.openrouter_model,
        "openrouter_fallback_model": s.openrouter_fallback_model,
    }


MAX_RESUME_BYTES = 2 * 1024 * 1024  # 2 MB
ALLOWED_RESUME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}


def _clean_text(s: str) -> str:
    s = s.replace("\x00", "").strip()
    if len(s) > 12000:
        s = s[:12000] + "\n\n[Truncated resume text to 12k chars]"
    return s


def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages[:10]:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)


def _extract_docx_text(data: bytes) -> str:
    doc = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    for p in doc.paragraphs[:400]:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)


@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    t0 = time.time()
    content_type = file.content_type or ""

    if content_type not in ALLOWED_RESUME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a PDF or DOCX resume.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Max 2MB.")

    try:
        kind = ALLOWED_RESUME_TYPES[content_type]
        text = _extract_pdf_text(data) if kind == "pdf" else _extract_docx_text(data)
        text = _clean_text(text)
        if len(text.strip()) < 50:
            raise HTTPException(
                status_code=422,
                detail="Could not extract enough text. Try a non-scanned PDF/DOCX.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("resume extraction failed")
        raise HTTPException(status_code=422, detail="Failed to parse resume file.")

    dt_ms = int((time.time() - t0) * 1000)
    logger.info("resume uploaded kind=%s bytes=%s ms=%s", ALLOWED_RESUME_TYPES[content_type], len(data), dt_ms)

    return {
        "ok": True,
        "filename": file.filename,
        "content_type": content_type,
        "text": text,
        "chars": len(text),
    }


@app.post("/api/chat", response_model=AgentResponse)
async def chat(req: ChatRequest, request: Request):
    t0 = time.time()
    request_id = request.headers.get("x-vercel-id") or request.headers.get("x-request-id")

    if req.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Last message must be role='user'.")

    total_chars = sum(len(m.content) for m in req.messages)
    if total_chars > 24000:
        raise HTTPException(status_code=413, detail="Message history too large. Keep it shorter.")

    try:
        orch = await get_orchestrator()
        out = await orch.respond(req.messages, req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        # Provider/model/privacy errors should not crash the function
        logger.exception("chat failed request_id=%s", request_id)
        raise HTTPException(status_code=502, detail="Model/provider error. Try again or switch model.") from e

    dt_ms = int((time.time() - t0) * 1000)
    logger.info("chat ok request_id=%s mode=%s ms=%s", request_id, req.mode, dt_ms)
    return out


# ---- Serve OLD frontend at `/` from public/ (Vercel + local) ----
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
PUBLIC_DIR = BASE_DIR / "public"
app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
