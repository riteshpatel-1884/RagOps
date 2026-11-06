"""v2 agent: retrieval from this version's Qdrant collection + the shared generator."""
from app.versions.toolkit.rag_core import Answer, SYSTEM_PROMPT, llm, structured_llm, make_ask

from .indexing import vector_store, RETRIEVER_K

ask = make_ask(vector_store, RETRIEVER_K)


if __name__ == "__main__":
    result = ask("What is Ritesh Patel's educational qualification?")
    print("\nANSWER:\n" + result["answer"])
    print("\nSOURCES:", list(zip(result["retrieved_sources"], result["retrieved_pages"])))
    print(f"\nretrieval {result['retrieval_ms']:.0f} ms | generation {result['generation_ms']:.0f} ms | "
          f"tokens in/out {result['input_tokens']}/{result['output_tokens']}")
