# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.messages import SystemMessage, HumanMessage

# from .indexing import vector_store, RETRIEVER_K
# from .objects import Answer

# load_dotenv()

# llm = ChatGroq(
#     model="openai/gpt-oss-120b",
#     temperature=0
# )

# structured_llm = llm.with_structured_output(
#     Answer,
#     method="json_mode"
# )

# SYSTEM_PROMPT = """
# You are a factual RAG assistant.

# Answer the user's question using ONLY the retrieved context.

# You must return TWO fields:

# 1. answer:
#    - Provide a direct answer to the user's question.
#    - Use only information supported by the retrieved context.
#    - Do not invent facts.

# 2. context:
#    - Return the relevant information from the retrieved context
#      that you actually used to generate the answer.
#    - Preserve the original facts, numbers, dates and names.
#    - Do not introduce information from outside the retrieved context.

# Additional rules:
# - Never combine unrelated information.
# - If the answer is unavailable, explicitly say so.
# - Return both fields as strings.
# - Return valid JSON with exactly these keys:
#   "answer" and "context".
# - Do not return Markdown outside JSON.
# """


# def ask(question: str) -> dict:

#     # 1. Retrieve relevant chunks (k = 3 in v1)
#     docs = vector_store.similarity_search(
#         question,
#         k=RETRIEVER_K
#     )

#     # 2. Prepare context
#     context = "\n\n".join(
#         doc.page_content for doc in docs
#     )

#     # 3. Generate both answer and context
#     response = structured_llm.invoke([
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(
#             content=f"""
#             Retrieved Context:
#             {context}

#             User Question:
#             {question}
#             """
#         )
#     ])

#     # 4. Return LLM-generated structured response
#     return {
#         "answer": response.answer,
#         "context": response.context
#     }


# if __name__ == "__main__":

#     result = ask(
#         "What is Ritesh Patel's educational qualification?"
#     )

#     print("\nANSWER:")
#     print(result["answer"])

#     print("\nCONTEXT:")
#     print(result["context"])



import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage

from .indexing import vector_store, RETRIEVER_K
from .objects import Answer

try:  # collects token usage for every LLM call (langchain-core >= 0.3.49)
    from langchain_core.callbacks import UsageMetadataCallbackHandler
except ImportError:  # pragma: no cover
    UsageMetadataCallbackHandler = None

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    Answer,
    method="json_mode"
)

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
    """Sum input/output tokens over all calls recorded by the callback (0 if unavailable)."""
    if handler is None:
        return 0, 0
    usage = getattr(handler, "usage_metadata", None) or {}
    tin = sum(int(u.get("input_tokens", 0) or 0) for u in usage.values())
    tout = sum(int(u.get("output_tokens", 0) or 0) for u in usage.values())
    return tin, tout


def ask(question: str) -> dict:
    t_start = time.perf_counter()

    # 1. Retrieve relevant chunks (k = 3, raw PDF blocks)
    docs = vector_store.similarity_search(
        question,
        k=RETRIEVER_K
    )
    t_retrieved = time.perf_counter()

    # 2. Prepare context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # 3. Generate both answer and context (token usage is captured by the callback)
    handler = UsageMetadataCallbackHandler() if UsageMetadataCallbackHandler else None
    response = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"""
            Retrieved Context:
            {context}

            User Question:
            {question}
            """
            ),
        ],
        config={"callbacks": [handler]} if handler else None,
    )
    t_done = time.perf_counter()

    tokens_in, tokens_out = _token_counts(handler)

    # 4. Return the structured answer plus everything needed for evaluation
    return {
        "answer": response.answer,
        "context": response.context,
        "retrieved_chunks": [doc.page_content for doc in docs],   # raw retriever output
        "retriever_k": RETRIEVER_K,
        "retrieval_ms": (t_retrieved - t_start) * 1000,
        "generation_ms": (t_done - t_retrieved) * 1000,
        "total_ms": (t_done - t_start) * 1000,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
    }


if __name__ == "__main__":

    result = ask(
        "What is Ritesh Patel's educational qualification?"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nCONTEXT:")
    print(result["context"])

    print(
        f"\nretrieval {result['retrieval_ms']:.0f} ms | generation {result['generation_ms']:.0f} ms | "
        f"tokens in/out {result['input_tokens']}/{result['output_tokens']}"
    )
