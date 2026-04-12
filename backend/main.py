"""
main.py — FastAPI server for MedRAG
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import subprocess
import sys
import os

from agents import run_pipeline

app = FastAPI(title="MedRAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_history: list[ChatMessage] = []
    use_critic: bool = True

class SourceDoc(BaseModel):
    doc_id: str
    title: str
    source: str
    source_type: str
    url: str
    date: str
    tags: list[str]

class ChatResponse(BaseModel):
    query_type: str
    sub_queries: list[str]
    retrieved_docs: list[SourceDoc]
    reasoning_trace: str
    final_answer: str
    critique: Optional[str]
    verified_answer: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "qwen3:8b (Ollama)"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        history = [{"role": m.role, "content": m.content} for m in req.conversation_history]
        result = run_pipeline(
            query=req.query,
            conversation_history=history,
            use_critic=req.use_critic,
        )
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
def rebuild_index():
    """Admin endpoint to rebuild the FAISS + BM25 index."""
    try:
        indexer_path = os.path.join(os.path.dirname(__file__), "indexer.py")
        result = subprocess.run(
            [sys.executable, indexer_path],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return {"status": "index rebuilt", "output": result.stdout[-500:]}
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Indexing timed out")
