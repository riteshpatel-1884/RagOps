"""Shared settings for every experiment version (v1, v2, v3, ...).

Datasets and results stay inside each version folder; only the metric CODE is shared.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# toolkit -> versions -> app -> backend
VERSIONS_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parents[3]

# ---- Corpus: EVERY *.pdf in this folder is indexed (resume + NEXORA handbook, ...).
PDF_DIR = BACKEND_DIR / "pdfs"
MAX_QUESTIONS = 60   # size limit of the evaluation dataset

# ---- Vector database (Qdrant). Start the server once:
#   docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant
# Override in backend/.env:  QDRANT_URL=http://localhost:6333   (or ":memory:" for a throw-away in-process DB)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
# BGE models are trained to see this prefix in FRONT OF QUERIES (documents stay as they are, so no re-indexing).
# Off by default so existing results stay comparable. Try it:  $env:USE_QUERY_INSTRUCTION="1"
# Words removed from the SEARCH query only (the LLM still sees the full question) are set PER VERSION below
# ("strip"), so nothing in .env or the shell can silently change a version. Do NOT set QUERY_STRIP_TERMS any more.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
USE_QUERY_INSTRUCTION = os.getenv("USE_QUERY_INSTRUCTION", "0") == "1"

# Register a new version here when you add one. "strip" = words removed from the SEARCH query only.
# v3 = merged chunks + the company name removed from the SEARCH query (the LLM still sees the full question).
VERSIONS = {
    "v1":  {"n": 1, "name": "experiment_v1_k3",               "chunking": "raw PDF blocks",                      "collection": "ragops_v1", "strip": []},
    "v2":  {"n": 2, "name": "experiment_v2_k1",               "chunking": "raw PDF blocks",                      "collection": "ragops_v2", "strip": []},
    "v3":  {"n": 3, "name": "experiment_v3_k3_merged",        "chunking": "merged blocks (700 chars, overlap 1) + company name dropped from the search query", "collection": "ragops_v3", "strip": ["Nexora"]},
}

# ---- Judge (LLM-as-a-judge). Keep it FIXED across versions or scores are not comparable.
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_THRESHOLD = 0.7
JUDGE_METRICS = ["answer_relevancy", "faithfulness", "context_relevancy", "correctness", "completeness"]
DEFAULT_CONTEXT_SOURCE = "raw"   # "raw" = chunks the retriever returned, "llm" = the context text the LLM wrote

# ---- Retrieval evaluation depths (Recall@k / MRR@k / Precision@k are reported at each)
RETRIEVAL_DEPTHS = (1, 3, 5)

# ---- Cost of the GENERATOR model, USD per 1M tokens.
# Groq gpt-oss-120b on-demand list price when this was written. Verify at https://groq.com/pricing
GENERATOR_PRICE_PER_M = {"input": 0.15, "output": 0.60}

# ---- Statistics
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42

UNANSWERABLE = "unanswerable"


def paths(version: str) -> dict:
    if version not in VERSIONS:
        raise SystemExit(f"Unknown version {version!r}. Known: {', '.join(VERSIONS)} (see toolkit/config.py)")
    n = VERSIONS[version]["n"]
    d = VERSIONS_DIR / version
    return {
        "dir": d,
        "dataset": d / "eval_dataset.jsonl",
        "results": d / f"results{n}.jsonl",
        "evaluated": d / f"evaluated_results{n}.jsonl",
        "retrieval": d / f"retrieval_results{n}.jsonl",
        "experiment": d / f"experiment_{version}.json",
    }


def rag_module(version: str, name: str) -> str:
    """Dotted import path of a module inside the version's RAG package, e.g. agent / indexing."""
    n = VERSIONS[version]["n"]
    return f"app.versions.{version}.RAG_{n}.{name}"