import io
import logging
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
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

# ✅ Lazy init so import never crashes the function
_orch: Orchestrator | None = None
def get_orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        _orch = Orchestrator(settings)
    return _orch


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    mode: Literal["auto", "plan", "resume", "interview"] = "auto"
    messages: list[ChatMessage] = Field(min_length=1)

    def ensure_last_user(self):
        if self.messages[-1].role != "user":
            raise ValueError("Last message must be role='user'.")


@app.get("/api")
@app.get("/api/")
async def api_root():
    # prevents confusing {"detail":"Not Found"} when you open /api in browser
    return {"ok": True, "routes": ["/api/health", "/api/upload_resume", "/api/chat"]}


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "model": settings.openrouter_model,
        "fallback_model": settings.openrouter_fallback_model,
        "has_api_key": bool(settings.openrouter_api_key),
    }


MAX_RESUME_BYTES = 2 * 1024 * 1024
ALLOWED = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

def _clean_text(s: str) -> str:
    s = s.replace("\x00", "").strip()
    if len(s) > 12000:
        s = s[:12000] + "\n\n[Truncated]"
    return s

def _pdf_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages[:12]:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts)

def _docx_text(data: bytes) -> str:
    d = docx.Document(io.BytesIO(data))
    parts = []
    for p in d.paragraphs[:500]:
        t = (p.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)

@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    t0 = time.time()
    ctype = (file.content_type or "").strip()

    if ctype not in ALLOWED:
        raise HTTPException(415, "Unsupported file type. Upload PDF or DOCX.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_RESUME_BYTES:
        raise HTTPException(413, "File too large. Max 2MB.")

    try:
        kind = ALLOWED[ctype]
        text = _pdf_text(data) if kind == "pdf" else _docx_text(data)
        text = _clean_text(text)
        if len(text.strip()) < 50:
            raise HTTPException(422, "Could not extract enough text (scanned PDF?).")
    except HTTPException:
        raise
    except Exception:
        logger.exception("resume parse failed")
        raise HTTPException(422, "Failed to parse resume.")

    logger.info("resume uploaded kind=%s bytes=%s ms=%s", kind, len(data), int((time.time()-t0)*1000))
    return {"ok": True, "text": text, "chars": len(text)}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        req.ensure_last_user()
    except ValueError as e:
        raise HTTPException(400, str(e))

    vercel_id = request.headers.get("x-vercel-id")
    logger.info("chat start vercel_id=%s mode=%s messages=%d", vercel_id, req.mode, len(req.messages))

    try:
        orch = get_orchestrator()
    except Exception as e:
        raise HTTPException(500, f"Backend misconfigured: {e}")

    try:
        out = await orch.respond(req.messages, req.mode)
        return out
    except Exception:
        logger.exception("chat failed vercel_id=%s", vercel_id)
        raise HTTPException(502, "LLM call failed. Check OpenRouter model/privacy settings.")
