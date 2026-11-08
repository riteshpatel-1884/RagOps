"""Query preparation shared by the RAG pipeline (rag_core) and the retrieval evaluation."""
import re

from .config import USE_QUERY_INSTRUCTION


def strip_terms(question: str, terms) -> str:
    """Remove words (and their possessive) from the question, e.g. "at Nexora" -> "at"."""
    q = question
    for t in terms or []:
        q = re.sub(rf"\b{re.escape(t)}(?:'s)?\b", "", q, flags=re.I)
    q = re.sub(r"\s{2,}", " ", q).replace(" ?", "?").strip()
    return q or question                      # never search with an empty query


def prepare_query(question: str, terms=None) -> str:
    """The text that is sent to the vector search (the LLM always receives the original question)."""
    return strip_terms(question, terms)


def options_tag(terms=None) -> str:
    """Identifies the retrieval options of a version; stored in every result row so runs are never silently mixed."""
    parts = []
    if terms:
        parts.append("strip=" + ",".join(sorted(t.lower() for t in terms)))
    if USE_QUERY_INSTRUCTION:
        parts.append("bge_instruction")
    return ";".join(parts)