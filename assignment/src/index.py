import io
import os
import time
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pypdf import PdfReader
import docx

from src.settings import settings
from src.agent import Orchestrator

logger = logging.getLogger("placementsprint.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)

app = FastAPI(title="PlacementSprint")


# ---------------------------
# Lazy orchestrator (prevents import-time crash)
# ---------------------------
_orch: Orchestrator | None = None

def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator(settings)
    return _orch


# ---------------------------
# Schemas
# ---------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    mode: Literal["auto", "plan", "resume", "interview"] = "auto"
    messages: list[ChatMessage] = Field(min_length=1)

    def ensure_last_user(self) -> None:
        if self.messages[-1].role != "user":
            raise ValueError("Last message must be role='user'.")


# ---------------------------
# Health
# ---------------------------
@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "service": "placementsprint",
        "model": settings.openrouter_model,
        "fallback_model": settings.openrouter_fallback_model,
        "has_api_key": bool(settings.openrouter_api_key),
    }


# ---------------------------
# Resume Upload
# ---------------------------
MAX_RESUME_BYTES = 2 * 1024 * 1024  # 2MB
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
    for page in reader.pages[:12]:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)

def _extract_docx_text(data: bytes) -> str:
    d = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    for p in d.paragraphs[:500]:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    t0 = time.time()
    content_type = (file.content_type or "").strip()

    if content_type not in ALLOWED_RESUME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload PDF or DOCX.")

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
                detail="Could not extract enough text. If PDF is scanned, upload DOCX or text-based PDF.",
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("resume parsing failed")
        raise HTTPException(status_code=422, detail="Failed to parse resume.")

    ms = int((time.time() - t0) * 1000)
    logger.info("resume uploaded kind=%s bytes=%s ms=%s", kind, len(data), ms)

    return {
        "ok": True,
        "filename": file.filename,
        "content_type": content_type,
        "text": text,
        "chars": len(text),
    }


# ---------------------------
# Chat
# ---------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        req.ensure_last_user()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    vercel_id = request.headers.get("x-vercel-id")
    logger.info("chat start vercel_id=%s mode=%s messages=%d", vercel_id, req.mode, len(req.messages))

    try:
        orch = get_orchestrator()
    except ValueError as e:
        # Missing env vars like OPENROUTER_API_KEY
        raise HTTPException(status_code=500, detail=str(e))

    try:
        out = await orch.respond(req.messages, req.mode)
        return out
    except Exception:
        logger.exception("chat failed vercel_id=%s", vercel_id)
        raise HTTPException(status_code=502, detail="Failed to generate response.")


# ---------------------------
# Local dev only: serve static UI
# Vercel serves `public/` automatically.
# ---------------------------
if os.getenv("VERCEL") != "1":
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
