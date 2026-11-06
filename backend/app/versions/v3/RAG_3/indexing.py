"""v3: merged blocks, k = 3 (the ONLY change vs v1 is chunking).

The ONLY things that differ between v1, v2 and v3 are RETRIEVER_K and the chunking below.
Everything else (PDFs, embeddings, Qdrant, prompt, LLM) is shared through app/versions/toolkit.
"""
from app.versions.toolkit.corpus import load_blocks, merge_blocks, describe
from app.versions.toolkit.vectorstore import build_vector_store

RETRIEVER_K = 3
MAX_CHARS = 700   # target max size of a merged chunk
OVERLAP = 1       # trailing blocks repeated in the next chunk

blocks = load_blocks()                      # every PDF in backend/pdfs/, with source + page
chunks = merge_blocks(blocks, max_chars=MAX_CHARS, overlap=OVERLAP)
vector_store = build_vector_store("ragops_v3", chunks)

print(f"v3: {len(chunks)} chunks ({describe(chunks)}) | retriever k = {RETRIEVER_K}")
