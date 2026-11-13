"""Single place for every setting. Edit VERSIONS to change what a version does."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[3]          # .../backend
VERSIONS_DIR = Path(__file__).resolve().parents[1]         # .../backend/app/versions
load_dotenv(BACKEND_DIR / ".env")

PDF_DIR = BACKEND_DIR / "pdfs"                             # every *.pdf in here is indexed (drop new PDFs here)
CACHE_DIR = BACKEND_DIR / ".cache"                         # embedding cache + FlashRank model files

# ---- Keys / services (backend/.env) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")   # use ":memory:" for a throw-away local index
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# ---- Models (nothing is downloaded: embeddings are API calls) ----
LLM_MODEL = "openai/gpt-oss-120b"                          # answer generator (Groq)
JUDGE_MODEL = "openai/gpt-oss-120b"                        # DeepEval judge (keep FIXED for all versions)
JUDGE_THRESHOLD = 0.7
# If Google lists the model as "gemini-embedding-2-preview", set GEMINI_EMBEDDING_2_MODEL in .env.
EMBEDDING_1 = os.getenv("GEMINI_EMBEDDING_1_MODEL", "gemini-embedding-001")
EMBEDDING_2 = os.getenv("GEMINI_EMBEDDING_2_MODEL", "gemini-embedding-2")
EMBED_BATCH = 50
EMBED_PAUSE = float(os.getenv("EMBED_PAUSE", "0.4"))       # seconds between embedding batches (free-tier friendly)

# Reranker: "flashrank" (tiny CPU model, ~4 MB, downloaded once) or "llm" (Groq ranks the passages, no download).
RERANKER = os.getenv("RERANKER", "flashrank")
FLASHRANK_MODEL = os.getenv("FLASHRANK_MODEL", "ms-marco-TinyBERT-L-2-v2")

# ---- The three experiments. 1 token is about 4 characters, so 500 tokens = 2000 chars, 100 tokens = 400 chars.
VERSIONS = {
    "v1": {
        "n": 1, "name": "experiment_v1_baseline", "label": "Baseline",
        "chunking": {"strategy": "static", "chunk_chars": 2000, "overlap_chars": 200},
        "embedding": {"model": EMBEDDING_1, "dim": 768},
        "retrieval": {"mode": "vector", "top_k": 3},
    },
        "v2": {
        "n": 2, "name": "experiment_v2_smaller_chunks", "label": "Smaller chunks (1000 chars), top 4",
        "chunking": {"strategy": "static", "chunk_chars": 1000, "overlap_chars": 150},
        "embedding": {"model": EMBEDDING_1, "dim": 768},
        "retrieval": {"mode": "vector", "top_k": 4},
    },
    
}

# ---- Evaluation ----
UNANSWERABLE = "unanswerable"
JUDGE_METRICS = ["answer_relevancy", "faithfulness", "context_relevancy", "context_precision", "context_recall", "correctness"]
HIT_K = 3
GENERATOR_PRICE_PER_M = {"input": 0.15, "output": 0.60}     # USD per 1M tokens, Groq gpt-oss-120b list price; verify at groq.com/pricing
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42


def paths(version: str) -> dict:
    if version not in VERSIONS:
        raise SystemExit(f"Unknown version {version!r}. Known: {', '.join(VERSIONS)}")
    n = VERSIONS[version]["n"]
    return {
        "dataset": VERSIONS_DIR / "eval_dataset.jsonl",     # ONE dataset shared by all versions
        "results": VERSIONS_DIR / version / f"results{n}.jsonl",
        "evaluated": VERSIONS_DIR / version / f"evaluated_results{n}.jsonl",
        "experiment": VERSIONS_DIR / version / f"experiment_{version}.json",
    }


def agent_module(version: str) -> str:
    return f"app.versions.{version}.RAG_{VERSIONS[version]['n']}.agent"
