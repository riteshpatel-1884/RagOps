from langchain.tools import tool

from .indexing import vector_store

last_retrieved_context: list[str] = []


@tool
def collecting_info(query: str) -> str:
    """Search the PDF knowledge base and return relevant information
    for the user's question."""
    global last_retrieved_context

    docs = vector_store.similarity_search(query, k=4)
    # (backend) PS D:\ragops\backend> python -m app.RAG_1.indexing
    # Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
    # Loading weights: 100%|███████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 306.08it/s]
    # indexed 26 chunks -> so it will gie best 4 as k = 4
    
    # Save the real retrieved chunks so ask() can return them as context
    last_retrieved_context = [doc.page_content for doc in docs]

    return "\n\n".join(last_retrieved_context)