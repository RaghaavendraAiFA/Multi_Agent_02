# Fast_api.py
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from mongodb_handler import (
    generate_session_id,
    async_get_session_history,   # async version for FastAPI
    get_session_history,         # sync version for agents_used lookup
)
from Orchestrator import run_orchestrator

# APP

app = FastAPI(
    title="Multi-Agent System API",
    description="IPC | Healthcare | Software agents powered by LangGraph Multi Agent Orchestrator. Ask questions, get answers, and track conversation history",
    version="1.0.0",
)

# SCHEMAS

class ChatRequest(BaseModel):
    question:   str
    session_id: str | None = None   # optional — new session created if None


class ChatResponse(BaseModel):
    session_id:  str
    question:    str
    answer:      str
    agents_used: list[str]


class HistoryResponse(BaseModel):
    session_id: str
    total:      int
    history:    list[dict]


# ENDPOINT 1 — POST /chat

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Use provided session_id or generate new one
    # "string" is Swagger UI default placeholder — treat as new session
    session_id = request.session_id
    if not session_id or session_id.strip() == "" or session_id == "string":
        session_id = generate_session_id()

    print(f"\n[API] Question   : {request.question}")
    print(f"[API] Session ID : {session_id}")

    # run_orchestrator is sync → run in background thread
    answer = await run_in_threadpool(run_orchestrator, request.question, session_id)

    # Fetch agents_used from MongoDB (sync — already in threadpool context)
    history     = await run_in_threadpool(get_session_history, session_id)
    agents_used = history[-1]["agents_used"] if history else []

    print(f"[API] Agents used: {agents_used}")
    print(f"[API] Answer     : {answer[:80]}...")

    return ChatResponse(
        session_id  = session_id,
        question    = request.question,
        answer      = answer,
        agents_used = agents_used,
    )


#  ENDPOINT 2 — GET /history/{session_id}
@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    # Direct await — motor is natively async, no threadpool needed
    records = await async_get_session_history(session_id)

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No history found for session: {session_id}"
        )

    return HistoryResponse(
        session_id = session_id,
        total      = len(records),
        history    = records,
    )

@app.get("/health")
async def health():
    """Check if API is running."""
    return {"status": "ok", "message": "Multi-Agent API is running!"}


