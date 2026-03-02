#  Fast_api.py
import os
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from mongodb_handler import (
    generate_session_id,
    async_get_session_history,
    get_session_history,
)
from Orchestrator import run_orchestrator
from IPC_RAG_Pipeline import (
    process_uploaded_pdf,
    get_upload_status,
    clear_upload,
)

#  APP
app = FastAPI(
    title       = "Multi-Agent System API",
    description = "IPC | Healthcare | Software agents",
    version     = "2.0.0",
)

UPLOAD_DIR = Path("uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB


#  GLOBAL ERROR HANDLER

#  SCHEMAS
class ChatRequest(BaseModel):
    question  : str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id : str
    question   : str
    answer     : str
    agents_used: list[str]


class HistoryResponse(BaseModel):
    session_id: str
    total     : int
    history   : list[dict]

#  ENDPOINT 1 — POST /upload

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed. Please upload a .pdf file.",
        )

    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max allowed size is 50MB.",
        )

    # Save PDF to disk
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(contents)
    print(f"\n[Upload] File saved: {save_path}")

    # Run full pipeline in background thread (blocking operation)
    try:
        result = await run_in_threadpool(
            process_uploaded_pdf,
            str(save_path),
            file.filename,
        )
    except Exception as e:
        # Clean up file if pipeline fails
        if save_path.exists():
            os.remove(save_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}",
        )

    return {
        "message"     : f" '{file.filename}' uploaded and processed successfully!",
        "filename"    : result["filename"],
        "total_pages" : result["total_pages"],
        "total_chunks": result["total_chunks"],
        "status"      : "ready",
        "info"        : "IPC Agent will now answer from this document + recent news.",
    }


#  ENDPOINT 2 — GET /upload/status  

@app.get("/upload/status")
async def upload_status():
    return get_upload_status()


# ENDPOINT 3 DELETE /upload/clear
@app.delete("/upload/clear")
async def clear_document():
    await run_in_threadpool(clear_upload)
    return {
        "message": " Document cleared successfully.",
        "status" : "No document uploaded.",
    }

#  ENDPOINT 4 — POST /chat
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id
    if not session_id or session_id.strip() in ["", "string"]:
        session_id = generate_session_id()

    print(f"\n[API] Question   : {request.question}")
    print(f"[API] Session ID : {session_id}")

    answer = await run_in_threadpool(
        run_orchestrator,
        request.question,
        session_id,
    )

    try:
        history     = await run_in_threadpool(get_session_history, session_id)
        agents_used = history[-1]["agents_used"] if history else []
    except Exception:
        agents_used = []

    return ChatResponse(
        session_id  = session_id,
        question    = request.question,
        answer      = answer,
        agents_used = agents_used,
    )

#  ENDPOINT 5 — GET /history/{session_id}
@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    records = await async_get_session_history(session_id)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No history for session: {session_id}",
        )
    return HistoryResponse(
        session_id = session_id,
        total      = len(records),
        history    = records,
    )


#  ENDPOINT 6 — GET /health

@app.get("/health")
async def health():
    """API health check."""
    status = get_upload_status()
    return {
        "status"         : "ok",
        "message"        : "Multi-Agent API is running!",
        "document_loaded": status["uploaded"],
        "document_name"  : status["filename"],
    }