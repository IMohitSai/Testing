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
from pydantic_ai.exceptions import ModelHTTPError


# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger("placementsprint.api")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)

app = FastAPI(title="PlacementSprint")

orch = Orchestrator(settings)


# ---------------------------
# Schemas
# ---------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    mode: Literal["auto", "plan", "resume", "interview"] = "auto"
    messages: list[ChatMessage] = Field(min_length=1)

    def last_user_message(self) -> str:
        # enforce "last must be user" (your UI expects this)
        if self.messages[-1].role != "user":
            raise ValueError("Last message must be role='user'.")
        return self.messages[-1].content


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
    }


# ---------------------------
# Resume upload helpers
# ---------------------------
MAX_RESUME_BYTES = 2 * 1024 * 1024  # 2 MB
ALLOWED_RESUME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

def _clean_text(s: str) -> str:
    s = s.replace("\x00", "").strip()
    # keep bounded to avoid huge token usage
    if len(s) > 12000:
        s = s[:12000] + "\n\n[Truncated resume text to 12k chars]"
    return s

def _extract_pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    # cap pages for safety
    for page in reader.pages[:12]:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n\n".join(parts)

def _extract_docx_text(data: bytes) -> str:
    d = docx.Document(io.BytesIO(data))
    parts: list[str] = []
    # cap paragraphs for safety
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
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a PDF or DOCX resume.")

    data = await file.read()
    if not data:
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
                detail="Could not extract enough text. If this is a scanned PDF, export as text PDF or upload DOCX.",
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


# ---------------------------
# Chat endpoint
# ---------------------------
@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    vercel_id = request.headers.get("x-vercel-id")
    try:
        _ = req.last_user_message()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    logger.info("chat start vercel_id=%s mode=%s messages=%d", vercel_id, req.mode, len(req.messages))

    try:
        out = await orch.respond(req.messages, req.mode)
        return out
    except ModelHTTPError as e:
        # surface OpenRouter failures clearly in logs + UI
        logger.error("openrouter failed vercel_id=%s status=%s body=%s", vercel_id, e.status_code, e.body)
        msg = e.body.get("message") if isinstance(e.body, dict) else str(e.body)
        raise HTTPException(status_code=502, detail=f"OpenRouter error {e.status_code}: {msg}")
    except Exception:
        logger.exception("chat failed vercel_id=%s", vercel_id)
        raise HTTPException(status_code=502, detail="Backend error while generating response.")


# ---------------------------
# Local dev only: serve static UI
# On Vercel, /public is served by the platform (not FastAPI).
# ---------------------------
if os.getenv("VERCEL") != "1":
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
