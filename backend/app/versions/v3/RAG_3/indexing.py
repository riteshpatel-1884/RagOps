"""v3: merged blocks + company name removed from the search query, k = 3.

What differs from v1/v2: the chunking below and STRIP_TERMS.
Everything else (PDFs, embeddings, Qdrant, prompt, LLM) is shared through app/versions/toolkit.
"""
from app.versions.toolkit.config import VERSIONS
from app.versions.toolkit.corpus import load_blocks, merge_blocks, describe
from app.versions.toolkit.vectorstore import build_vector_store

RETRIEVER_K = 3
MAX_CHARS = 700   # target max size of a merged chunk
OVERLAP = 1       # trailing blocks repeated in the next chunk
STRIP_TERMS = VERSIONS["v3"]["strip"]   # words removed from the SEARCH query only (set in toolkit/config.py)

blocks = load_blocks()                      # every PDF in backend/pdfs/, with source + page
chunks = merge_blocks(blocks, max_chars=MAX_CHARS, overlap=OVERLAP)
vector_store = build_vector_store("ragops_v3", chunks)

print(f"v3: {len(chunks)} chunks ({describe(chunks)}) | retriever k = {RETRIEVER_K} | strip {STRIP_TERMS}")