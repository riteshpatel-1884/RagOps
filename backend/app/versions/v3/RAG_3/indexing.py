from pathlib import Path
from dotenv import load_dotenv
from .file_reader import extract_paragraphs, merge_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# v3: same k as v1. The ONLY change vs v1 is chunking (merged blocks).
RETRIEVER_K = 3
MAX_CHARS = 700   # target max size of a merged chunk
OVERLAP = 1       # number of trailing blocks repeated in the next chunk

# RAG_3 -> v3 -> versions -> app -> backend
PDF_PATH = Path(__file__).resolve().parents[4] / "pdfs" / "ritesh_AI.pdf"

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = InMemoryVectorStore(embedding)

paragraphs = extract_paragraphs(str(PDF_PATH))
chunks = merge_paragraphs(paragraphs, max_chars=MAX_CHARS, overlap=OVERLAP)
vector_store.add_texts(texts=chunks)
print(
    f"indexed {len(chunks)} merged chunks from {len(paragraphs)} blocks "
    f"(retriever k = {RETRIEVER_K}, max_chars = {MAX_CHARS}, overlap = {OVERLAP})"
)
