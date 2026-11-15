"""Generation: streams the answer from Groq so time-to-first-token (TTFT) can be measured."""
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from .config import LLM_MODEL

NOT_FOUND = "I could not find this in the handbook."

SYSTEM_PROMPT = f"""You answer questions about a company employee handbook, using ONLY the context provided.
- Answer concisely but completely: include every relevant number, date, name, list item and condition.
- When a question has several parts, answer every part.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND}" Do not guess and do not use outside knowledge.
- Mention the page in brackets, for example [p. 12], when you state a fact."""

llm = ChatGroq(model=LLM_MODEL, temperature=0)


def format_context(contexts):
    return "\n\n---\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in contexts)


def make_ask(retriever):
    def ask(question: str) -> dict:
        t_start = time.perf_counter()
        contexts, timings = retriever.retrieve(question)
        t_retrieved = time.perf_counter()

        messages = [SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=f"Context:\n{format_context(contexts)}\n\nQuestion: {question}")]
        parts, first, usage = [], None, None
        for chunk in llm.stream(messages):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text and first is None:
                first = time.perf_counter()
            parts.append(text)
            if getattr(chunk, "usage_metadata", None):
                usage = chunk.usage_metadata
        t_end = time.perf_counter()

        answer = "".join(parts).strip()
        in_tok = (usage or {}).get("input_tokens") or sum(len(c["text"]) for c in contexts) // 4
        out_tok = (usage or {}).get("output_tokens") or len(answer) // 4
        return {
            "answer": answer,
            "retrieved_chunks": [c["text"] for c in contexts],
            "retrieved_pages": [c["page"] for c in contexts],
            "retrieval_ms": (t_retrieved - t_start) * 1000,
            "ttft_ms": ((first or t_end) - t_start) * 1000,      # question received -> first answer token
            "total_ms": (t_end - t_start) * 1000,
            "retrieval_detail": timings,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        }

    return ask
