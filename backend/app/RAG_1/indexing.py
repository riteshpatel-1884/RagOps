from pathlib import Path
from dotenv import load_dotenv
from .file_reader import extract_paragraphs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PDF_PATH = Path(__file__).resolve().parents[2] / "pdfs" / "ritesh_AI.pdf"

embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = InMemoryVectorStore(embedding)

chunks = extract_paragraphs(str(PDF_PATH))
vector_store.add_texts(texts=chunks)
print(f"indexed {len(chunks)} chunks")