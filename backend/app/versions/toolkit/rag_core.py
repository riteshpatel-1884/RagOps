"""Generation code shared by v1, v2 and v3 (identical prompt and model, so only retrieval differs)."""
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

try:  # collects token usage for every LLM call (langchain-core >= 0.3.49)
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:  # pragma: no cover
    UsageMetadataCallbackHandler = None

load_dotenv()


class Answer(BaseModel):
    answer: str = Field(description="The final answer to the user's question.")
    context: str = Field(description="The relevant retrieved context used to generate the answer. "
                                     "Include the supporting information from the provided context.")


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
structured_llm = llm.with_structured_output(Answer, method="json_mode")

SYSTEM_PROMPT = """
You are a factual RAG assistant.

Answer the user's question using ONLY the retrieved context.

You must return TWO fields:

1. answer:
   - Provide a direct answer to the user's question.
   - Use only information supported by the retrieved context.
   - Do not invent facts.

2. context:
   - Return the relevant information from the retrieved context
     that you actually used to generate the answer.
   - Preserve the original facts, numbers, dates and names.
   - Do not introduce information from outside the retrieved context.

Additional rules:
- Never combine unrelated information.
- If the answer is unavailable, explicitly say so.
- Return both fields as strings.
- Return valid JSON with exactly these keys:
  "answer" and "context".
- Do not return Markdown outside JSON.
"""


def _token_counts(handler):
    if handler is None:
        return 0, 0
    usage = getattr(handler, "usage_metadata", None) or {}
    tin = sum(int(u.get("input_tokens", 0) or 0) for u in usage.values())
    tout = sum(int(u.get("output_tokens", 0) or 0) for u in usage.values())
    return tin, tout


def format_context(docs) -> str:
    """Each chunk is labelled with its document and page so the model knows which file a fact comes from."""
    parts = []
    for d in docs:
        m = d.metadata or {}
        parts.append(f"[Source: {m.get('source', 'unknown')}, page {m.get('page', '?')}]\n{d.page_content}")
    return "\n\n".join(parts)


def make_ask(vector_store, k: int):
    def ask(question: str) -> dict:
        t_start = time.perf_counter()
        docs = vector_store.similarity_search(question, k=k)               # 1. retrieve from Qdrant
        t_retrieved = time.perf_counter()

        context = format_context(docs)                                      # 2. prepare context

        handler = UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler else None
        response = structured_llm.invoke(                                   # 3. generate
            [SystemMessage(content=SYSTEM_PROMPT),
             HumanMessage(content=f"""
            Retrieved Context:
            {context}

            User Question:
            {question}
            """)],
            config={"callbacks": [handler]} if handler else None)
        t_done = time.perf_counter()
        tokens_in, tokens_out = _token_counts(handler)

        return {
            "answer": response.answer,
            "context": response.context,
            "retrieved_chunks": [d.page_content for d in docs],             # raw retriever output
            "retrieved_sources": [(d.metadata or {}).get("source") for d in docs],
            "retrieved_pages": [(d.metadata or {}).get("page") for d in docs],
            "retriever_k": k,
            "retrieval_ms": (t_retrieved - t_start) * 1000,
            "generation_ms": (t_done - t_retrieved) * 1000,
            "total_ms": (t_done - t_start) * 1000,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
        }
    return ask
