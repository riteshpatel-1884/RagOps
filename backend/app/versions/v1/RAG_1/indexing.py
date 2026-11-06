"""v1: raw PDF blocks, k = 3.

The ONLY things that differ between v1, v2 and v3 are RETRIEVER_K and the chunking below.
Everything else (PDFs, embeddings, Qdrant, prompt, LLM) is shared through app/versions/toolkit.
"""
from app.versions.toolkit.corpus import load_blocks, describe
from app.versions.toolkit.vectorstore import build_vector_store

RETRIEVER_K = 3

blocks = load_blocks()                      # every PDF in backend/pdfs/, with source + page
chunks = blocks
vector_store = build_vector_store("ragops_v1", chunks)

print(f"v1: {len(chunks)} chunks ({describe(chunks)}) | retriever k = {RETRIEVER_K}")
