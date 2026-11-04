from langchain.tools import tool

from .indexing import vector_store, RETRIEVER_K

last_retrieved_context: list[str] = []


@tool
def collecting_info(query: str) -> str:
    """Search the PDF knowledge base and return relevant information
    for the user's question."""
    global last_retrieved_context

    docs = vector_store.similarity_search(query, k=RETRIEVER_K)

    # Save the real retrieved chunks so ask() can return them as context
    last_retrieved_context = [doc.page_content for doc in docs]

    return "\n\n".join(last_retrieved_context)