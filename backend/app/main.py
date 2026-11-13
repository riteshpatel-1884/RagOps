import importlib
import logging
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()  # picks up GROQ_API_KEY etc. from backend/.env

log = logging.getLogger("ragops")

app = FastAPI(title="RAGOPS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    version: str = "v2"  # "v1" | "v2"


# version -> module that exposes ask(question) -> dict
AGENT_MODULES = {
    "v1": "app.versions.v1.RAG_1.agent",  # static 2000-char chunks, vector top 3
    "v2": "app.versions.v2.RAG_2.agent",  # static 1000-char chunks, vector top 4
    # "v3": "app.versions.v3.RAG_3.agent",  # add when v3 exists
}

_agents = {}
_lock = threading.Lock()


def _get_agent(version: str):
    """Import a version lazily. Importing builds (or reuses) its Qdrant index,
    so we do it once, on first use."""
    with _lock:
        if version not in _agents:
            log.info("Loading %s ...", version)
            _agents[version] = importlib.import_module(AGENT_MODULES[version])
        return _agents[version]


def run_pipeline(version: str, question: str):
    """Return (answer, chunks) where chunks is a list of retrieved text strings."""
    result = _get_agent(version).ask(question)
    answer = result.get("answer", "")
    chunks = result.get("retrieved_chunks") or []
    if not chunks and result.get("context"):
        chunks = [result["context"]]
    return answer, chunks


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    if req.version not in AGENT_MODULES:
        raise HTTPException(status_code=400, detail="version must be one of: " + ", ".join(AGENT_MODULES))
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is required")
    try:
        answer, chunks = run_pipeline(req.version, req.question)
    except Exception as e:
        log.exception("pipeline %s failed", req.version)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"answer": answer, "sources": chunks}


# cd D:\ragops\backend
# uvicorn app.main:app --reload --port 8000