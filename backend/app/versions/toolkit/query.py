"""Query preparation shared by the RAG pipeline (rag_core) and the retrieval evaluation."""
import re

from .config import QUERY_STRIP_TERMS, USE_QUERY_INSTRUCTION


def strip_terms(question: str, terms) -> str:
    """Remove words (and their possessive) from the question, e.g. "at Nexora" -> "at"."""
    q = question
    for t in terms:
        q = re.sub(rf"\b{re.escape(t)}(?:'s)?\b", "", q, flags=re.I)
    q = re.sub(r"\s{2,}", " ", q).replace(" ?", "?").strip()
    return q or question                      # never search with an empty query


def prepare_query(question: str, terms=None) -> str:
    """The text that is sent to the vector search (the LLM always receives the original question)."""
    return strip_terms(question, QUERY_STRIP_TERMS if terms is None else terms)


def options_tag() -> str:
    """Identifies the active retrieval options; stored in every result row so runs are never silently mixed."""
    parts = []
    if QUERY_STRIP_TERMS:
        parts.append("strip=" + ",".join(sorted(t.lower() for t in QUERY_STRIP_TERMS)))
    if USE_QUERY_INSTRUCTION:
        parts.append("bge_instruction")
    return ";".join(parts)
